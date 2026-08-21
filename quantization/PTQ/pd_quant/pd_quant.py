import copy
import torch
import torch.nn as nn

from .quant_block import specials, BaseQuantBlock
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from .fold_bn import search_fold_and_remove_bn
from .set_weight_quantize_params import set_weight_quantize_params
from .layer_recon import layer_reconstruction
from .block_recon import block_reconstruction



class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, is_fusing=True):
        super().__init__()
        if is_fusing:
            search_fold_and_remove_bn(model)
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            self.model = model
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_module_refactor_wo_fuse(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, nn.BatchNorm2d):
                if prev_quantmodule is not None:
                    prev_quantmodule.norm_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
            
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor_wo_fuse(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        'the image input has been in 0~255, set the last layer\'s input to 8-bit'
        a_list[-2].bitwidth_refactor(8)
        # a_list[0].bitwidth_refactor(8)

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True


def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]


# def set_weight_quantize_params(model):
#     for module in model.modules():
#         if isinstance(module, QuantModule):
#             module.weight_quantizer.set_inited(False)
#             '''caculate the step size and zero point for weight quantizer'''
#             module.weight_quantizer(module.weight)
#             module.weight_quantizer.set_inited(True)


def pd_quant(fp_model, data,args):
    n_bits_w = args["w_bit"]
    n_bits_a = args["a_bit"]
    channel_wise = True
    init_wmode = 'mse'
    init_amode = 'mse'
    prob = 0.5
    iters_w = 20000
    weight = 0.01
    b_start = 20
    b_end = 2
    warmup = 0.2
    lr = 4e-5
    input_prob = 0.5
    keep_cpu = False
    lamb_r = 0.1
    T = 4.0
    bn_lr = 1e-3
    lamb_c = 0.02
    device = args['device']
    cnn = copy.deepcopy(fp_model)
    cnn.eval().to(device)
    _fp_model = copy.deepcopy(cnn)
    _fp_model.eval().to(device)
    wq_params = {'n_bits': n_bits_w, 'channel_wise': channel_wise, 'scale_method': init_wmode}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': init_amode,
                 'leaf_param': True, 'prob': prob}
    
    _fp_model = QuantModel(_fp_model, weight_quant_params=wq_params, act_quant_params=aq_params)
    _fp_model.eval().to(device)
    _fp_model.set_quant_state(False, False)

    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.eval().to(device)
    qnn.disable_network_output_quantization()
    # print('the quantized model is below:')
    # print(qnn)

    train_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    cali_data, cali_target = get_train_samples(train_loader, num_samples=1024)
    # cali_data = cali_data.to(device)
    next(qnn.parameters())

        # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=iters_w, weight=weight,
                b_range=(b_start, b_end), warmup=warmup, opt_mode='mse',
                lr=lr, input_prob=input_prob, keep_gpu=not keep_cpu, 
                lamb_r=lamb_r, T=T, bn_lr=bn_lr, lamb_c=lamb_c, device=device)


    '''init weight quantizer'''
    set_weight_quantize_params(qnn)

    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            layer_reconstruction(qnn, _fp_model, module, fp_module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(qnn, _fp_model, module, fp_module, **kwargs)
        else:
            raise NotImplementedError
    def recon_model(model: nn.Module, fp: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp.named_children()):
            print('Reconstruction for layer {}'.format(name))
            if isinstance(module, QuantModule):
            #     print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
            #     print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
            #     print('Reconstruction for layer {}'.format(name))
                recon_model(module, fp_module)
    # Start calibration
    recon_model(qnn, _fp_model)
    print('!!!!!!!!!!! finished the iteration of reconstruction !!!!!!!!!!!')
    qnn.set_quant_state(weight_quant=True, act_quant=True)

    return qnn
    # print('Full quantization (W{}A{}) accuracy: {}'.format(n_bits_w, n_bits_a,
                                                        #    validate_model(test_loader, qnn)))