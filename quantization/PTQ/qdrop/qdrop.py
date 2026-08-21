import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
from torchvision.models.resnet import BasicBlock
from .observer import ObserverBase
from .fake_quant import QuantizeBase
from .quantized_module import QuantizedLayer, QuantizedBlock, Quantizer
from .recon import reconstruction


def _get_cali_data(train_loader, num_samples):
    cali_data = []
    for batch in train_loader:
        cali_data.append(batch[0])
        if len(cali_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(cali_data, dim=0)[:num_samples]


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """

    def __init__(self, org_module: BasicBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu, w_qconfig, a_qconfig)
        self.conv2 = QuantizedLayer(org_module.conv2, None, w_qconfig, a_qconfig, qoutput=False)
        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = copy.deepcopy(org_module.relu)
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


specials = {
    BasicBlock: QuantBasicBlock
}


def quantize_model(model, config_quant,args):
    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        while (st < ed):
            tmp_qoutput = qoutput if st == ed - 1 else True
            name, child_module = childs[st][0], childs[st][1]
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1

    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)
    a_list[-1].set_bit(8)
    # print('finish quantize model:\n{}'.format(str(model)))
    return model

def get_config_quant(args):
    config_quant_dict = {
        'a_qconfig': {
            'quantizer': 'FixedFakeQuantize',
            'observer': 'MinMaxObserver',
            'bit': 4,
            'symmetric': False,
            'ch_axis': -1,
        },
        'w_qconfig': {
            'quantizer': 'AdaRoundFakeQuantize',
            'observer': 'MSEObserver',
            'bit': 4,
            'symmetric': False,
            'ch_axis': 0,
        },
        'recon': {
            'batch_size': 32,
            'scale_lr': 4e-5,
            'warm_up': 0.2,
            'weight': 0.01,
            'iters': 20000,
            'b_range': [20, 2],
            'keep_gpu': True,
            'round_mode': 'learned_hard_sigmoid',
            'drop_prob': 0.5,
        }
    }


    config_quant_dict_MInA = {
        'a_qconfig': {
            'quantizer': 'FakeQuantize',
            'observer': 'AvgMinMaxObserver',
            'bit': 8,
            'symmetric': False,
            'ch_axis': -1,
        },
        'w_qconfig': {
            'quantizer': 'AdaRoundFakeQuantize',
            'observer': 'MSEObserver',
            'bit': 6,
            'symmetric': False,
            'ch_axis': 0,
        },
        'recon': {
            'batch_size': 64,
            'scale_lr': 2e-5,
            'warm_up': 0.5,
            'weight': 0.01,
            'iters': 5000,
            'b_range': [20, 2],
            'keep_gpu': True,
            'round_mode': 'learned_hard_sigmoid',
            'drop_prob': 0.1,
        }
    }
    if args['attack_method']=='MInv':
        return config_quant_dict_MInA

    else:

        return config_quant_dict

def disable_all(model):
    # print('Disable observer and disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            # print('Disable observer and disable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    # print('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                # print('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            # print('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_quantization(model, quantizer_type='fake_quant'):
    # print('Disable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                # print('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            # print('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    else:
        return d




def qdrop(model, data,args,label_key=None):
    w_bit=args['w_bit']
    a_bit=args['a_bit']
    device=args['device']
    quant_model = copy.deepcopy(model)
    config_quant_dict = get_config_quant(args)
    config_quant = _dict_to_namespace(config_quant_dict)
    config_quant.w_qconfig.bit = w_bit
    config_quant.a_qconfig.bit = a_bit

    train_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    cali_data = _get_cali_data(train_loader, 1024)
    print('cali_data shape:', cali_data.shape)

    search_fold_and_remove_bn(quant_model)
    # if hasattr(quant_model, 'quant'):
    quant_model = quantize_model(quant_model, config_quant,args)
    print('Finish converting model')

    quant_model.to(device)
    quant_model.eval()
    fp_model = copy.deepcopy(quant_model)
    disable_all(fp_model)
    for name, module in quant_model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    with torch.no_grad():
        st = time.time()
        print('start calibration for activation quantizer')
        enable_calibration_woquantization(quant_model, quantizer_type='act_fake_quant')
        quant_model(cali_data[: 256].to(device))
        print('start calibration for weight quantizer')
        enable_calibration_woquantization(quant_model, quantizer_type='weight_fake_quant')
        # Use moderate samples for weight calibration to balance speed and stability
        quant_model(cali_data[: 256].to(device))
        ed = time.time()
        print('the calibration time is {}'.format(ed - st))

    if hasattr(config_quant, 'recon'):
        enable_quantization(quant_model)

        def recon_model(module: nn.Module, fp_module: nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    # print('begin reconstruction for module:\n{}'.format(str(child_module)))
                    reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data,
                                   config_quant.recon)
                else:
                    recon_model(child_module, getattr(fp_module, name))

        # Start reconstruction
        recon_model(quant_model, fp_model)
    enable_quantization(quant_model)
    return quant_model
    # imagenet_utils.validate_model(val_loader, model)
