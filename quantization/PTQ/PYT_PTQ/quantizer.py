import torch
import torch.nn as nn
import torch.nn.functional as F

from .observer import OmseObserver

class Quantizer(nn.Module):
    def __init__(self, bit, observer, ptq, sign=False):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
    
    def update_qparams(self, tensor):
        raise NotImplementedError
    
    def forward(self, tensor):
        if self.training or self.ptq:
            self.observer(tensor)
            self.update_qparams(tensor)
        quant_tensor = (torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point
        fake_quant_tensor = (quant_tensor - self.zero_point) * self.scale

        return fake_quant_tensor


class AsymmetricQuantizer(Quantizer):
    def __init__(self, bit, observer, ptq, sign=False, device='cpu'):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
        self.device = device

        if self.observer.level == "L":
            self.register_buffer("scale", torch.ones((1), dtype=torch.float32, device=device))
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32, device=device))
        elif self.observer.level == "C":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32, device=device),
            )
        elif self.observer.level == "FC":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1), dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1), dtype=torch.float32, device=device),
            )
        # self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
        # self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))

        self.register_buffer("quant_min",
                              torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32, device=device),
                            )

        self.register_buffer("quant_max",
                              torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32, device=device),
                            )
        self.register_buffer("eps", 
                              torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32, device=device)
                            )
    
    def update_qparams(self, input):
        scale = (self.observer.max_val.to(self.device) - self.observer.min_val.to(self.device)) / (self.quant_max - self.quant_min)
        zero_point = (torch.round(self.quant_min - self.observer.min_val.to(self.device) / scale) - (self.quant_min - self.observer.min_val.to(self.device) / scale)).detach() + \
                    (self.quant_min - self.observer.min_val.to(self.device) / scale)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


class AdaRoundQuantizer(Quantizer):

    def __init__(self, bit, observer, ptq, sign=False, round_mode='learned_hard_sigmoid', device='cpu'):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
        self.round_mode = round_mode
        self.alpha = None
        self.ada_init = None
        self.soft_targets = True
        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.device = device

        
        # self.init_alpha(x=weight_tensor.clone())
        if self.observer.level == "L":
            self.register_buffer("scale", torch.ones((1), dtype=torch.float32, device=device))
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32, device=device))
        elif self.observer.level == "C":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32, device=device),
            )
        elif self.observer.level == "FC":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1), dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1), dtype=torch.float32, device=device),
            )
        self.register_buffer("quant_min",
                              torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32, device=device),
                            )

        self.register_buffer("quant_max",
                              torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32, device=device),
                            )
        self.register_buffer("eps", 
                              torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32, device=device)
                            )

    
    def update_qparams(self, inputs):
        scale = (self.observer.max_val.to(self.device) - self.observer.min_val.to(self.device)) / (self.quant_max - self.quant_min)
        zero_point = (torch.round(self.quant_min - self.observer.min_val.to(self.device) / scale) - (self.quant_min - self.observer.min_val.to(self.device) / scale)).detach() + \
                    (self.quant_min - self.observer.min_val.to(self.device) / scale)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

    def forward(self, tensor):
        if self.training or self.ptq:
            self.observer(tensor)
            self.update_qparams(tensor)

        # print('self.ada_init', self.ada_init)
        if not self.ada_init:
            self.init_alpha(tensor.clone())
            self.ada_init = True
        # quant_tensor = (torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point
        # fake_quant_tensor = (quant_tensor - self.zero_point) * self.scale


        quant_tensor = self.quant(tensor)
        fake_quant_tensor = self.dequantize(quant_tensor)

        return fake_quant_tensor

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        # range_shape = self.get_reshape_range(inputs)
        # scale = scale.reshape(range_shape)
        # zero_point = zero_point.reshape(range_shape)

        if self.round_mode == 'nearest':
            x_int = torch.round(inputs / scale)
        # elif self.round_mode == 'nearest_ste':
        #     x_int = round_ste(inputs / scale)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(inputs / scale)
            rest = (inputs / scale) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            print('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(inputs / scale)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                # print('test test test')
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')

        outputs = x_int + zero_point
        outputs = outputs.round().clamp(self.quant_min,
                                        self.quant_max)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        # range_shape = self.get_reshape_range(inputs)
        # scale = scale.reshape(range_shape)
        # zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):

        scale = self.scale
        # range_shape = self.get_reshape_range(x)
        # scale = scale.reshape(range_shape)
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            # print('Init alpha to be FP32')
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
