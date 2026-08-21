import torch
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable

# k = 8
class ActFn(Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        ctx.save_for_backward(x, alpha)
        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        y = torch.clamp(x, min = 0, max = alpha.item())
        scale = (2**k - 1) / alpha
        y_q = torch.round( y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
        lower_bound      = x < 0
        upper_bound      = x > alpha
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None

def quantize_k(r_i, k):
    scale = (2**k - 1)
    r_o = torch.round( scale * r_i ) / scale
    return r_o

class DoReFaQuant(Function):
    @staticmethod
    def forward(ctx, r_i, k):
        tanh = torch.tanh(r_i).float()
        # scale = 2**k - 1.
        # quantize_k = torch.round( scale * (tanh / 2*torch.abs(tanh).max() + 0.5 ) ) / scale
        r_o = 2*quantize_k( tanh / (2*torch.max(torch.abs(tanh)).detach()) + 0.5 , k) - 1
        # r_o = 2 * quantize_k - 1.
        return r_o

    @staticmethod
    def backward(ctx, dLdr_o):
        # due to STE, dr_o / d_r_i = 1 according to formula (5)
        return dLdr_o, None


class Conv2d(nn.Conv2d):
    def __init__(self, in_places, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False, bitwidth=8):
        super(Conv2d, self).__init__(
            in_places, out_planes, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth
        self.unlearn = False
        self.quant_unlearn = False

    def forward(self, x):
        vhat = self.quantize(self.weight, self.bitwidth)
        if self.unlearn:
            vhat += self.quant_ul_influence
        if self.quant_unlearn:
            vhat = self.quantize(vhat, self.bitwidth)
        y = F.conv2d(x, vhat, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, bitwidth = 8):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.quantize = DoReFaQuant.apply
        self.bitwidth = bitwidth
        self.unlearn = False
        self.quant_unlearn = False

    def forward(self, x):
        vhat = self.quantize(self.weight, self.bitwidth)
        if self.unlearn:
            vhat += self.quant_ul_influence
        if self.quant_unlearn:
            vhat = self.quantize(vhat, self.bitwidth)

        y = F.linear(x, vhat, self.bias)
        return y
	

# PACT Activation Function
# PACT: Parameterized Clipping Activation for Post-Training quantization
# https://arxiv.org/abs/2004.00288
# This implementation is based on the paper and the original code

class PACTActivation(nn.Module):
    def __init__(self, bit_width=4, init_clip=6.0):
        super(PACTActivation, self).__init__()
        # Initialize alpha as a learnable scalar
        self.alpha = nn.Parameter(torch.tensor(init_clip))
      #  self.alpha = init_clip

        self.bit_width = bit_width
        self.act = ActFn.apply

    def forward(self, x):
        return self.act(x, self.alpha, self.bit_width)
	

class PACTResnet(nn.Module):
	
    def __init__(self, model, bit_width=4, init_clip=10., act_bit_width=None, weight_bit_width=None):
        super(PACTResnet, self).__init__()
        self.model = model
        # Backwards compatible: if act/weight bit-width not provided, fall back to bit_width.
        self.act_bit_width = bit_width if act_bit_width is None else act_bit_width
        self.weight_bit_width = bit_width if weight_bit_width is None else weight_bit_width
        self.init_clip = init_clip
        self.args = model.args

        # Relace the ReLU activation function with PACTActivation
        self._replace_relu_with_pact(self.model, self.act_bit_width, init_clip)

        # Prepare the model for quantization
        self._prepare_quantization(self.model)
      #  self._convert_to_quantized(self.model)
        self._copy_methods(model)

        # Define quantization Stubs
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()

    def _copy_methods(self, original_model):
        """Copies methods from the original model to this PACTResnet instance."""
        for name in dir(original_model):
            if not name.startswith("__") and not hasattr(self, name) and callable(getattr(original_model, name)):
                setattr(self, name, getattr(original_model, name).__get__(self, PACTResnet))

    def _replace_relu_with_pact(self, module, bit_width, init_clip):
        """
        Recursively replace ReLU activations with PACT activations
        :param module: nn.Module with ReLU in its children
        :param bit_width: bit width for quantization
        :param init_clip: initial value for the alpha parameter
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.ReLU):
                setattr(module, name, PACTActivation(bit_width=bit_width, init_clip=init_clip))
            elif isinstance(child_module, nn.ReLU6):
                setattr(module, name, PACTActivation(bit_width=bit_width, init_clip=6.0))
            else:
                self._replace_relu_with_pact(child_module, bit_width, init_clip)

    def _prepare_quantization(self, m):
        for name, module in m.named_children():
            if isinstance(module, torch.nn.Conv2d):
             #   bit_width = 8 if name == 'conv1' else self.bit_width
                conv = Conv2d(
                    module.in_channels, module.out_channels,
                    module.kernel_size, module.stride, module.padding,
                    groups=module.groups, dilation=module.dilation,
                    bias=module.bias is not None, bitwidth=self.weight_bit_width,
                )
                conv.weight.data = module.weight.data
                if module.bias is not None:
                    conv.bias.data = module.bias.data
                setattr(m, name, conv)
                
            elif isinstance(module, torch.nn.Linear):
                linear = Linear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None, bitwidth=self.weight_bit_width,
                )
                linear.weight.data = module.weight.data
                if module.bias is not None:
                    linear.bias.data = module.bias.data
                setattr(m, name, linear)

            self._prepare_quantization(module)

    def check_quantization(self):
        """Quick check that quantization was applied."""
        quantized_count = 0
        total_count = 0
        
        def count_modules(module):
            nonlocal quantized_count, total_count
            for name, child in module.named_children():
                total_count += 1
                if isinstance(child, (Conv2d, Linear)):
                    quantized_count += 1
                    print(f"✅ quantized layer: {name}, bitwidth: {child.bitwidth}")
                count_modules(child)
        
        count_modules(self.model)
        print(f"quantization summary: {quantized_count}/{total_count} layers quantized")
        return quantized_count > 0

    def check_forward_quantization(self, test_input):
        """Check that quantization actually happens during the forward pass."""
        print("=== checking forward-pass quantization ===")
        
        def check_module_quantization(module, name=""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, (Conv2d, Linear)):
                    print(f"\n🔍 checking layer: {full_name}")
                    
                    # Fetch the original weights
                    original_weight = child.weight.data.clone()
                    print(f"   original weight range: [{original_weight.min().item():.6f}, {original_weight.max().item():.6f}]")
                    print(f"   original weight unique values: {len(torch.unique(original_weight))}")
                    
                    # Simulate the quantization step
                    quantized_weight = child.quantize(original_weight, child.bitwidth)
                    print(f"   quantized weight range: [{quantized_weight.min().item():.6f}, {quantized_weight.max().item():.6f}]")
                    print(f"   quantized weight unique values: {len(torch.unique(quantized_weight))}")
                    
                    # Check that quantization actually took effect
                    if len(torch.unique(quantized_weight)) <= 2**child.bitwidth:
                        print(f"   ✅ forward quantization OK (unique values ≤ {2**child.bitwidth})")
                    else:
                        print(f"   ❌ forward quantization failed (unique values > {2**child.bitwidth})")
                    
                    # Check whether the weights actually changed
                    weight_changed = not torch.allclose(original_weight, quantized_weight, atol=1e-6)
                    if weight_changed:
                        print(f"   ✅ weights changed after quantization")
                    else:
                        print(f"   ⚠️  weights unchanged after quantization")
                
                check_module_quantization(child, full_name)
        
        check_module_quantization(self.model)
        
        # Test an actual forward pass
        print(f"\n🧪 testing an actual forward pass")
        self.eval()
        
        # Make sure model and input live on the same device
        model_device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        test_input = test_input.to(model_device)
        
        with torch.no_grad():
            try:
                output = self(test_input)
                print(f"   ✅ forward pass succeeded, output shape: {output.shape}")
                return True
            except Exception as e:
                print(f"   ❌ forward pass failed: {e}")
                return False

    def convert_to_quantized(self):
        return self._convert_to_quantized(self.model)

    def _convert_to_quantized(self, m):
        # for name, module in self.named_modules():
        for name, module in m.named_children():
            if isinstance(module, Conv2d):
                conv = torch.nn.Conv2d(
                    module.in_channels, module.out_channels,
                    module.kernel_size, module.stride, module.padding,
                    bias=module.bias is not None, groups=module.groups,
                    dilation=module.dilation
                )
                conv.weight.data = module.quantize(module.weight, self.bit_width)
                setattr(m, name, conv)
            if isinstance(module, Linear):
                linear = torch.nn.Linear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None
                )
                linear.weight.data = module.quantize(module.weight, self.bit_width)
                setattr(m, name, linear)
            else:
                self._convert_to_quantized(module)

    def forward(self, x):
        x = self.model(x)
        return x

    def inner_out(self, x, layer=4):
        return self.model.inner_out(x, layer) # in resnet_bsq.py
				


if __name__ == '__main__':
	# Test backpropagation
    x = torch.tensor([1.5, 3.2, 6.0], requires_grad=True)  # Input tensor

    pact = PACTActivation(bit_width=4, init_clip=4.0)  # PACT Layer
    out = pact(x)
    print(f'PACT output: {out}')  # Output tensor
    # Check gradients
    out.sum().backward()  # Backpropagate

    print(f"Gradients w.r.t x: {x.grad}")  # Should be nonzero for 0 <= x <= alpha
    print(f"Gradient w.r.t α: {pact.alpha.grad}")  # Should be nonzero if x was clipped
