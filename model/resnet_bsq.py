from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.nn as nn
import math
import numpy as np

class PACTFunction(torch.autograd.Function):
    """
    Parametrized Clipping Activation Function
    https://arxiv.org/pdf/1805.06085.pdf
    Code from https://github.com/obilaniu/GradOverride
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.clamp(min=0.0).min(alpha)

    @staticmethod
    def backward(ctx, dLdy):
        x, alpha = ctx.saved_variables
        lt0 = x < 0
        gta = x > alpha
        gi = 1.0 - lt0.float() - gta.float()
        dLdx = dLdy * gi
        dLdalpha = torch.sum(dLdy * x.ge(alpha).float())
        return dLdx, dLdalpha


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit):
        if bit == 0:
            # No quantization
            act = x
        else:
            S = torch.max(torch.abs(x))
            if S == 0:
                act = x * 0
            else:
                step = 2 ** (bit) - 1
                R = torch.round(torch.abs(x) * step / S) / step
                act = S * R * torch.sign(x)
        return act

    @staticmethod
    def backward(ctx, g):
        return g, None


class PACT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return PACTFunction.apply(x, self.alpha)  #


__all__ = ['resnet_bsq.py']


# def conv3x3(in_planes, out_planes, stride=1):
#    "3x3 convolution with padding"
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                     padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1, Nbits=4, bin=False):
    "3x3 convolution with padding"
    return BitConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, Nbits=Nbits, bin=bin)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, Nbits=4, act_bit=4, bin=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, Nbits=Nbits, bin=bin)
        self.bn2 = nn.BatchNorm2d(planes)
        if act_bit > 3:
            self.relu1 = nn.ReLU6(inplace=True)
            self.relu2 = nn.ReLU6(inplace=True)
        else:
            self.relu1 = PACT()
            self.relu2 = PACT()
        self.downsample = downsample
        self.stride = stride
        self.act_bit = act_bit

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = STE.apply(out, self.act_bit)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        out = STE.apply(out, self.act_bit)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth=20, num_classes=1000, block_name='BasicBlock', Nbits=4, act_bit=4, bin=False):
        super(ResNet, self).__init__()
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = BitConv2d(3, 16, kernel_size=3, padding=1,
                               bias=False, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.layer2 = self._make_layer(block, 32, n, stride=2, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.layer3 = self._make_layer(block, 64, n, stride=2, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = BitLinear(64 * block.expansion, num_classes, Nbits=Nbits, bin=bin)

        for m in self.modules():
            if isinstance(m, BitConv2d):
                if m.bin:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    ini_w = torch.full_like(m.pweight[..., 0], 0)
                    ini_w.normal_(0, math.sqrt(2. / n))
                    m.ini2bit(ini_w)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, Nbits=4, act_bit=4, bin=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, Nbits=Nbits, act_bit=act_bit, bin=bin))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, Nbits=Nbits, act_bit=act_bit, bin=bin))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if (isinstance(m, BitLinear) or isinstance(m, BitConv2d)) and 'downsample' not in name:
                if m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape) / m.Nbits
                    if m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape) / m.bNbits
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape)
        return N

    def total_bit(self):
        N = 0
        for name, m in self.named_modules():
            if (isinstance(m, BitLinear) or isinstance(m, BitConv2d)) and 'downsample' not in name:
                if m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape)
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape) * m.Nbits
                    if m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape) * m.bNbits
        return N

    def get_Nbits(self):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                Nbit_dict[name] = [m.Nbits, 0]
                if m.pbias is not None or m.bias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
        return Nbit_dict

    def set_Nbits(self, Nbit_dict):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                N = Nbit_dict[name]
                N0 = N[0]
                N1 = N[1]
                ex = np.arange(N0 - 1, -1, -1)
                m.exps = torch.Tensor((2 ** ex) / (2 ** (N0) - 1)).float()
                m.Nbits = N0
                if N1:
                    ex = np.arange(N1 - 1, -1, -1)
                    m.bexps = torch.Tensor((2 ** ex) / (2 ** (N1) - 1)).float()
                    m.bNbits = N1
                if m.bin:
                    m.pweight.data = m.pweight.data[..., 0:N0]
                    m.nweight.data = m.nweight.data[..., 0:N0]
                    if N1:
                        m.pbias.data = m.pbias.data[..., 0:N1]
                        m.nbias.data = m.nbias.data[..., 0:N1]

    def set_zero(self):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                weight = m.pweight.data - m.nweight.data
                if m.Nbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                    m.zero = True
                else:
                    m.zero = False
                if m.pbias is not None:
                    weight = m.pbias.data - m.nbias.data
                    if m.bNbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                        m.bzero = True
                    else:
                        m.bzero = False

    def pruning(self, threshold=1.0,
                drop=True):  # Use drop to control whether 0 bit after pruning will be removed, 0 bit before pruning will always be removed
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                if m.Nbits > 1:
                    # Remove MSB
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = nonz_weight / total_weight
                    N = m.Nbits
                    N0 = m.Nbits
                    pweight = m.pweight.data
                    nweight = m.nweight.data
                    for i in range(N):
                        if nonz_weight[i] == 0:
                            m.pweight.data = pweight[..., i + 1:N]
                            m.nweight.data = nweight[..., i + 1:N]
                            m.Nbits -= 1
                            if m.Nbits == 1:
                                break
                        elif nonz_weight[i] < threshold:  # set MSB to 0, remove MSB if "drop"
                            if drop:
                                m.pweight.data = pweight[..., i + 1:N] + pweight[..., i].unsqueeze(-1)
                                m.nweight.data = nweight[..., i + 1:N] + nweight[..., i].unsqueeze(-1)
                                m.Nbits -= 1
                                if m.Nbits == 1:
                                    break
                            else:
                                m.pweight.data = pweight[..., i:N] + pweight[..., i].unsqueeze(-1)
                                m.nweight.data = nweight[..., i:N] + nweight[..., i].unsqueeze(-1)
                                m.pweight.data[..., 0] = 0.
                                m.nweight.data[..., 0] = 0.
                            m.pweight.data = torch.where(m.pweight.data < 1, m.pweight.data,
                                                         torch.full_like(m.pweight.data, 1.))
                            m.nweight.data = torch.where(m.nweight.data < 1, m.nweight.data,
                                                         torch.full_like(m.nweight.data, 1.))
                        else:
                            break
                    # Remove LSB
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = nonz_weight / total_weight
                    N = m.Nbits
                    pweight = m.pweight.data
                    nweight = m.nweight.data
                    if m.Nbits > 1:
                        for i in range(N):
                            if nonz_weight[N - 1 - i] <= threshold:
                                m.pweight.data = pweight[..., 0:N - 1 - i]
                                m.nweight.data = nweight[..., 0:N - 1 - i]
                                m.Nbits -= 1
                                m.scale.data = m.scale.data * 2
                                if m.Nbits == 1:
                                    break
                            else:
                                break
                    # Reset exps
                    N = m.Nbits
                    ex = np.arange(N - 1, -1, -1)
                    m.exps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.scale.data = m.scale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    ## Match the shape of grad to data
                    if m.pweight.grad is not None:
                        m.pweight.grad.data = m.pweight.grad.data[..., 0:N]
                        m.nweight.grad.data = m.nweight.grad.data[..., 0:N]
                # For bias
                if m.pbias is not None and m.bNbits > 1:
                    # Remove MSB
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = nonz_weight / total_weight
                    N = m.bNbits
                    N0 = m.bNbits
                    pweight = m.pbias.data
                    nweight = m.nbias.data
                    for i in range(N):
                        if nonz_weight[i] == 0:
                            m.pbias.data = pweight[..., i + 1:N]
                            m.nbias.data = nweight[..., i + 1:N]
                            m.bNbits -= 1
                            if m.bNbits == 1:
                                break
                        elif nonz_weight[i] < threshold:
                            m.pbias.data = pweight[..., i + 1:N] + pweight[..., i].unsqueeze(-1)
                            m.nbias.data = nweight[..., i + 1:N] + nweight[..., i].unsqueeze(-1)
                            m.pbias.data = torch.where(m.pbias.data < 1, m.pbias.data,
                                                       torch.full_like(m.pbias.data, 1.))
                            m.nbias.data = torch.where(m.nbias.data < 1, m.nbias.data,
                                                       torch.full_like(m.nbias.data, 1.))
                            m.bNbits -= 1
                            if m.bNbits == 1:
                                break
                        else:
                            break
                    # Remove LSB
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = nonz_weight / total_weight
                    if m.bNbits > 1:
                        N = m.bNbits
                        pweight = m.pbias.data
                        nweight = m.nbias.data
                        for i in range(N):
                            if nonz_weight[N - 1 - i] <= threshold:
                                m.pbias.data = pweight[..., 0:N - 1 - i]
                                m.nbias.data = nweight[..., 0:N - 1 - i]
                                m.bNbits -= 1
                                m.biasscale.data = m.biasscale.data * 2
                                if m.bNbits == 1:
                                    break
                            else:
                                break
                    # Reset exps
                    N = m.bNbits
                    ex = np.arange(N - 1, -1, -1)
                    m.bexps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.biasscale.data = m.biasscale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    ## Match the shape of grad to data
                    if m.pbias.grad is not None:
                        m.pbias.grad.data = m.pbias.grad.data[..., 0:N]
                        m.nbias.grad.data = m.nbias.grad.data[..., 0:N]
                if m.pbias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
                else:
                    Nbit_dict[name] = [m.Nbits, 0]
        return Nbit_dict


#resnet20

def resnet_bsq(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def resnet_bsq_dropout(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet_dropout(**kwargs)



import math

import numpy as np
import torch
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


############################################################################################################################################################

## Straight Through Estimator, modified from https://github.com/zjysteven/bitslice_sparsity/blob/master/mnist/pretrain.py#L181

############################################################################################################################################################

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit):
        if w is None:
            return None
        if bit == 0:
            weight = w * 0
        else:
            S = torch.max(torch.abs(w))
            if S == 0:
                weight = w * 0
            else:
                step = 2 ** (bit) - 1
                R = torch.round(torch.abs(w) * step / S) / step
                weight = S * R * torch.sign(w)
        return weight

    @staticmethod
    def backward(ctx, g):
        return g, None


class bit_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit, zero):
        if w is None:
            return None
        if zero or bit == 0:
            weight = w * 0
        else:
            w = torch.where(w > 1, torch.full_like(w, 1), w)
            w = torch.where(w < -1, torch.full_like(w, -1), w)
            step = 2 ** (bit) - 1
            R = torch.round(torch.abs(w) * step) / step
            weight = R * torch.sign(w)
        return weight

    @staticmethod
    def backward(ctx, g):
        return g, None, None


############################################################################################################################################################

## Fully-connected Layer, modified from https://github.com/mightydeveloper/Deep-Compression-PyTorch/blob/master/net/prune.py

############################################################################################################################################################

class BitLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """

    def __init__(self, in_features, out_features, Nbits=8, bias=True, bin=False):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Nbits = Nbits
        ex = np.arange(Nbits - 1, -1, -1)
        self.exps = torch.Tensor((2 ** ex) / (2 ** (self.Nbits) - 1)).float()
        self.bNbits = Nbits
        self.bexps = torch.Tensor((2 ** ex) / (2 ** (self.bNbits) - 1)).float()
        self.bin = bin
        self.total_weight = out_features * in_features
        self.total_bias = out_features
        self.zero = False
        self.bzero = False
        self.ft = False
        if self.bin:
            self.pweight = Parameter(torch.Tensor(out_features, in_features, Nbits))
            self.nweight = Parameter(torch.Tensor(out_features, in_features, Nbits))
            self.scale = Parameter(torch.Tensor(1))
            if bias:
                self.pbias = Parameter(torch.Tensor(out_features, Nbits))
                self.nbias = Parameter(torch.Tensor(out_features, Nbits))
                self.biasscale = Parameter(torch.Tensor(1))
            else:
                self.register_parameter('pbias', None)
                self.register_parameter('nbias', None)
                self.register_parameter('biasscale', None)
            self.bin_reset_parameters()
            # book keeping
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            # book keeping
            self.register_parameter('pweight', None)
            self.register_parameter('nweight', None)
            self.register_parameter('scale', None)
            self.register_parameter('pbias', None)
            self.register_parameter('nbias', None)
            self.register_parameter('biasscale', None)

    def reset_parameters(self):
        # For float model
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def ini2bit(self, ini, b=False, ft=False):
        # For binary model
        if ft:
            S = 1.0
        else:
            S = torch.max(torch.abs(ini))
        if S == 0:
            if b:
                self.pbias.data.fill_(0)
                self.nbias.data.fill_(0)
            else:
                self.pweight.data.fill_(0)
                self.nweight.data.fill_(0)
            return

        inip = torch.where(ini > 0, ini, torch.full_like(ini, 0))
        inin = torch.where(ini <= 0, -ini, torch.full_like(ini, 0))
        if b:
            step = 2 ** (self.bNbits) - 1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.biasscale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.bNbits):
                ex = 2 ** (self.bNbits - i - 1)
                self.pbias.data[:, i] = torch.floor(Rp / ex)
                self.nbias.data[:, i] = torch.floor(Rn / ex)
                Rp = Rp - torch.floor(Rp / ex) * ex
                Rn = Rn - torch.floor(Rn / ex) * ex
        else:
            step = 2 ** (self.Nbits) - 1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.scale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.Nbits):
                ex = 2 ** (self.Nbits - i - 1)
                self.pweight.data[..., i] = torch.floor(Rp / ex)
                self.nweight.data[..., i] = torch.floor(Rn / ex)
                Rp = Rp - torch.floor(Rp / ex) * ex
                Rn = Rn - torch.floor(Rn / ex) * ex

    def bin_reset_parameters(self):
        # For binary model
        stdv = 1. / math.sqrt(self.pweight.size(1))
        ini_w = torch.Tensor(self.out_features, self.in_features).uniform_(-stdv, stdv)
        self.ini2bit(ini_w)
        if self.pbias is not None:
            ini_b = torch.Tensor(self.out_features).uniform_(-stdv, stdv)
            self.ini2bit(ini_b, b=True)

    def to_bin(self):
        if self.bin:
            return
        else:
            self.bin = True
            self.ft = False
            if self.Nbits == 0:
                self.Nbits = 1
                self.weight.data = self.weight.data * 0
                self.zero = True
            if self.bNbits == 0:
                self.bNbits = 1
                self.bias.data = self.bias.data * 0
                self.bzero = True
            self.pweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
            self.nweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
            self.scale = Parameter(self.weight.data.new_zeros(1))
            self.ini2bit(self.weight.data)
            self.weight = None
            if self.bias is not None:
                self.pbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
                self.nbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
                self.biasscale = Parameter(self.bias.data.new_zeros(1))
                self.ini2bit(self.bias.data, b=True)
                self.bias = None

    def to_float(self):
        if self.bin:
            self.bin = False
            self.ft = False
            self.weight = Parameter(self.pweight.data.new_zeros(self.out_features, self.in_features))
            dev = self.pweight.device
            weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight, dim=2) * self.scale
            if self.Nbits == 1 and (np.count_nonzero(self.weight.data.cpu().numpy()) == 0):
                self.Nbits = 0
            self.pweight = None
            self.nweight = None
            self.scale = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias, dim=1) * self.biasscale
                if self.bNbits == 1 and (np.count_nonzero(self.bias.data.cpu().numpy()) == 0):
                    self.bNbits = 0
                self.pbias = None
                self.nbias = None
                self.biasscale = None
        else:
            return

    def finetune_to_bit(self):
        self.ft = False
        self.bin = True
        if self.Nbits == 0:
            self.Nbits = 1
            self.weight.data = self.weight.data * 0
            self.zero = True
        if self.bNbits == 0:
            self.bNbits = 1
            self.bias.data = self.bias.data * 0
            self.bzero = True
        self.pweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
        self.nweight = Parameter(self.weight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
        w = self.weight.data
        w = torch.where(w > 1, torch.full_like(w, 1), w)
        w = torch.where(w < -1, torch.full_like(w, -1), w)
        self.ini2bit(w, ft=True)
        self.weight = None
        if self.bias is not None:
            self.pbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            self.nbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            w = self.bias.data
            w = torch.where(w > 1, torch.full_like(w, 1), w)
            w = torch.where(w < -1, torch.full_like(w, -1), w)
            self.ini2bit(w, b=True, ft=True)
            self.bias = None

    def to_finetune(self):
        if self.bin:
            self.bin = False
            self.ft = True
            self.weight = Parameter(self.pweight.data.new_zeros(self.out_features, self.in_features))
            dev = self.pweight.device
            weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight, dim=2)
            if self.Nbits == 1 and (np.count_nonzero(self.weight.data.cpu().numpy()) == 0):
                self.Nbits = 0
            self.pweight = None
            self.nweight = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias, dim=1)
                if self.bNbits == 1 and (np.count_nonzero(self.bias.data.cpu().numpy()) == 0):
                    self.bNbits = 0
                self.pbias = None
                self.nbias = None
        else:
            return

    def forward(self, input):
        if self.bin:
            dev = self.pweight.device
            weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
            weight = bit_STE.apply(torch.sum(weight, dim=2), self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
                bias = bit_STE.apply(torch.sum(bias, dim=1), self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return F.linear(input, weight, bias)
        elif self.ft:
            weight = bit_STE.apply(self.weight, self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = bit_STE.apply(self.bias, self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, self.weight, self.bias)
            # return F.linear(input, STE.apply(self.weight, self.Nbits), STE.apply(self.bias,self.bNbits))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def quant(self, maxbit=8):
        # For binary model
        ## Re-quantize the binary part of the weight, keep scale unchanged

        dev = self.pweight.device
        ## Quantize weight
        weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
        weight = torch.sum(weight, dim=2)
        inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
        inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
        step = 2 ** (self.Nbits) - 1
        Rp = torch.round(inip * step)
        Rn = torch.round(inin * step)

        ## Increase layer precision if the binary part is out of range
        if self.Nbits >= maxbit:
            Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
            Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
        elif torch.max(torch.cat((Rp, Rn), 0)) > step:
            self.Nbits = self.Nbits + 1
            self.pweight = Parameter(self.pweight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
            self.nweight = Parameter(self.nweight.data.new_zeros(self.out_features, self.in_features, self.Nbits))
            N = self.Nbits
            ex = np.arange(N - 1, -1, -1)
            self.exps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
            self.scale.data = self.scale.data * (2 ** (N) - 1) / (2 ** (N - 1) - 1)

        for i in range(self.Nbits):
            ex = 2 ** (self.Nbits - i - 1)
            self.pweight.data[..., i] = torch.floor(Rp / ex)
            self.nweight.data[..., i] = torch.floor(Rn / ex)
            Rp = Rp - torch.floor(Rp / ex) * ex
            Rn = Rn - torch.floor(Rn / ex) * ex
        ## Quantize bias
        if self.pbias is not None:
            weight = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
            weight = torch.sum(weight, dim=1)
            inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
            inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
            step = 2 ** (self.bNbits) - 1
            Rp = torch.round(inip * step)
            Rn = torch.round(inin * step)

            ## Increase layer precision if the binary part is out of range
            if self.bNbits >= maxbit:
                Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
                Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
            elif torch.max(torch.cat((Rp, Rn), 0)) > step:
                self.bNbits = self.bNbits + 1
                self.pbias = Parameter(self.pbias.data.new_zeros(self.out_features, self.bNbits))
                self.nbias = Parameter(self.nbias.data.new_zeros(self.out_features, self.bNbits))
                N = self.bNbits
                ex = np.arange(N - 1, -1, -1)
                self.bexps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                self.biasscale.data = self.biasscale.data * (2 ** (N) - 1) / (2 ** (N - 1) - 1)

            for i in range(self.bNbits):
                ex = 2 ** (self.bNbits - i - 1)
                self.pbias.data[:, i] = torch.floor(Rp / ex)
                self.nbias.data[:, i] = torch.floor(Rn / ex)
                Rp = Rp - torch.floor(Rp / ex) * ex
                Rn = Rn - torch.floor(Rn / ex) * ex

    def print_stat(self):
        # For binary model
        weight = self.pweight.data.cpu().numpy() - self.nweight.data.cpu().numpy()
        total_weight = np.prod(weight.shape) / self.Nbits
        nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(self.Nbits)]
        print('Weight: ' + np.array2string(nonz_weight / total_weight, separator='%, ',
                                           formatter={'float_kind': lambda x: "%6.2f" % x}).strip('[]') + '%')
        if self.pbias is not None:
            bia = self.pbias.data.cpu().numpy() - self.nbias.data.cpu().numpy()
            total_weight = np.prod(bia.shape) / self.bNbits
            nonz_weight = [np.count_nonzero(bia[:, i]) * 100 for i in range(self.bNbits)]
            print('Bias:   ' + np.array2string(nonz_weight / total_weight, separator='%, ',
                                               formatter={'float_kind': lambda x: "%6.2f" % x}).strip('[]') + '%')

    def L1reg(self, reg):
        # For binary model
        param = torch.cat((self.pweight, self.nweight), 0)
        total_weight = self.Nbits * self.total_weight
        reg += torch.sum(torch.sqrt(1e-8 + torch.sum(param ** 2, (0, 1)))) * total_weight
        if self.pbias is not None:
            param = torch.cat((self.pbias, self.nbias), 0)
            total_bias = self.total_bias * self.bNbits
            reg += total_bias * torch.sum(torch.sqrt(1e-8 + torch.sum(param ** 2, 0)))
        return reg


############################################################################################################################################################

## Convolutional Layer, modified from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html

############################################################################################################################################################

class Bit_ConvNd(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'Nbits']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, Nbits=8, bin=False):
        super(Bit_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.Nbits = Nbits
        ex = np.arange(Nbits - 1, -1, -1)
        self.exps = torch.Tensor((2 ** ex) / (2 ** (self.Nbits) - 1)).float()
        self.bNbits = Nbits
        self.bexps = torch.Tensor((2 ** ex) / (2 ** (self.bNbits) - 1)).float()
        self.zero = False
        self.bzero = False
        self.ft = False
        self.bin = bin
        if self.bin:
            if transposed:
                self.pweight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size, Nbits))
                self.nweight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size, Nbits))
                self.scale = Parameter(torch.Tensor(1))
            else:
                self.pweight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size, Nbits))
                self.nweight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size, Nbits))
                self.scale = Parameter(torch.Tensor(1))

            if bias:
                self.pbias = Parameter(torch.Tensor(out_channels, Nbits))
                self.nbias = Parameter(torch.Tensor(out_channels, Nbits))
                self.biasscale = Parameter(torch.Tensor(1))
            else:
                self.register_parameter('pbias', None)
                self.register_parameter('nbias', None)
                self.register_parameter('biasscale', None)
            self.bin_reset_parameters()
            # book keeping
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            if transposed:
                self.weight = Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            # book keeping
            self.register_parameter('pweight', None)
            self.register_parameter('nweight', None)
            self.register_parameter('scale', None)
            self.register_parameter('pbias', None)
            self.register_parameter('nbias', None)
            self.register_parameter('biasscale', None)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def ini2bit(self, ini, b=False, ft=False):
        # For binary model
        if ft:
            S = 1.0
        else:
            S = torch.max(torch.abs(ini))
        if S == 0:
            if b:
                self.pbias.data.fill_(0)
                self.nbias.data.fill_(0)
            else:
                self.pweight.data.fill_(0)
                self.nweight.data.fill_(0)
            return

        inip = torch.where(ini > 0, ini, torch.full_like(ini, 0))
        inin = torch.where(ini <= 0, -ini, torch.full_like(ini, 0))
        if b:
            step = 2 ** (self.bNbits) - 1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.biasscale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.bNbits):
                ex = 2 ** (self.bNbits - i - 1)
                self.pbias.data[:, i] = torch.floor(Rp / ex)
                self.nbias.data[:, i] = torch.floor(Rn / ex)
                Rp = Rp - torch.floor(Rp / ex) * ex
                Rn = Rn - torch.floor(Rn / ex) * ex
        else:
            step = 2 ** (self.Nbits) - 1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.scale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.Nbits):
                ex = 2 ** (self.Nbits - i - 1)
                self.pweight.data[..., i] = torch.floor(Rp / ex)
                self.nweight.data[..., i] = torch.floor(Rn / ex)
                Rp = Rp - torch.floor(Rp / ex) * ex
                Rn = Rn - torch.floor(Rn / ex) * ex

    def bin_reset_parameters(self):
        ini_w = torch.full_like(self.pweight[..., 0], 0)
        init.kaiming_uniform_(ini_w, a=math.sqrt(5))
        self.ini2bit(ini_w)
        if self.pbias is not None:
            # stdv = 1. / math.sqrt(self.pweight.size(1))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.pweight)
            stdv = 1 / math.sqrt(fan_in)
            ini_b = torch.Tensor(self.out_channels).uniform_(-stdv, stdv)
            self.ini2bit(ini_b, b=True)

    def to_bin(self):
        if self.bin:
            return
        else:
            self.bin = True
            self.ft = False
            if self.Nbits == 0:
                self.Nbits = 1
                self.weight.data = self.weight.data * 0
                self.zero = True
            if self.bNbits == 0:
                self.bNbits = 1
                self.bias.data = self.bias.data * 0
                self.bzero = True
            if self.transposed:
                self.pweight = Parameter(
                    self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size,
                                               self.Nbits))
                self.nweight = Parameter(
                    self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size,
                                               self.Nbits))
            else:
                self.pweight = Parameter(
                    self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size,
                                               self.Nbits))
                self.nweight = Parameter(
                    self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size,
                                               self.Nbits))
            self.scale = Parameter(self.weight.data.new_zeros(1))
            # print(self.weight.data.shape)
            self.ini2bit(self.weight.data)
            self.weight = None
            if self.bias is not None:
                self.pbias = Parameter(self.bias.data.new_zeros(self.out_channels, self.bNbits))
                self.nbias = Parameter(self.bias.data.new_zeros(self.out_channels, self.bNbits))
                self.biasscale = Parameter(self.bias.data.new_zeros(1))
                self.ini2bit(self.bias.data, b=True)
                self.bias = None

    def to_float(self):
        if self.bin:
            self.bin = False
            self.ft = False
            if self.transposed:
                self.weight = Parameter(
                    self.pweight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size))
            else:
                self.weight = Parameter(
                    self.pweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
            dev = self.pweight.device
            weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight, dim=-1) * self.scale
            if self.Nbits == 1 and (np.count_nonzero(self.weight.data.cpu().numpy()) == 0):
                self.Nbits = 0
            self.pweight = None
            self.nweight = None
            self.scale = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias, dim=1) * self.biasscale
                if self.bNbits == 1 and np.count_nonzero(self.bias.data.cpu().numpy()) == 0:
                    self.bNbits = 0
                self.pbias = None
                self.nbias = None
                self.biasscale = None
        else:
            return

    def finetune_to_bit(self):
        self.ft = False
        self.bin = True
        if self.Nbits == 0:
            self.Nbits = 1
            self.weight.data = self.weight.data * 0
            self.zero = True
        if self.bNbits == 0:
            self.bNbits = 1
            self.bias.data = self.bias.data * 0
            self.bzero = True
        if self.transposed:
            self.pweight = Parameter(
                self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size,
                                           self.Nbits))
            self.nweight = Parameter(
                self.weight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size,
                                           self.Nbits))
        else:
            self.pweight = Parameter(
                self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size,
                                           self.Nbits))
            self.nweight = Parameter(
                self.weight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size,
                                           self.Nbits))
        w = self.weight.data
        w = torch.where(w > 1, torch.full_like(w, 1), w)
        w = torch.where(w < -1, torch.full_like(w, -1), w)
        self.ini2bit(w, ft=True)
        self.weight = None
        if self.bias is not None:
            self.pbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            self.nbias = Parameter(self.bias.data.new_zeros(self.out_features, self.bNbits))
            w = self.bias.data
            w = torch.where(w > 1, torch.full_like(w, 1), w)
            w = torch.where(w < -1, torch.full_like(w, -1), w)
            self.ini2bit(w, b=True, ft=True)
            self.bias = None

    def to_finetune(self):
        if self.bin:
            self.bin = False
            self.ft = True
            if self.transposed:
                self.weight = Parameter(
                    self.pweight.data.new_zeros(self.in_channels, self.out_channels // self.groups, *self.kernel_size))
            else:
                self.weight = Parameter(
                    self.pweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
            dev = self.pweight.device
            weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
            self.weight.data = torch.sum(weight, dim=-1)
            if self.Nbits == 1 and (np.count_nonzero(self.weight.data.cpu().numpy()) == 0):
                self.Nbits = 0
            self.pweight = None
            self.nweight = None
            if self.pbias is not None:
                self.bias = Parameter(self.pbias.data.new_zeros(self.out_features))
                bias = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
                self.bias.data = torch.sum(bias, dim=1)
                if self.bNbits == 1 and (np.count_nonzero(self.bias.data.cpu().numpy()) == 0):
                    self.bNbits = 0
                self.pbias = None
                self.nbias = None
        else:
            return

    def print_stat(self):
        weight = self.pweight.data.cpu().numpy() - self.nweight.data.cpu().numpy()
        total_weight = np.prod(weight.shape) / self.Nbits
        nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(self.Nbits)]
        # print(self.scale.data.cpu().numpy())
        print('Weight: ' + np.array2string(nonz_weight / total_weight, separator='%, ',
                                           formatter={'float_kind': lambda x: "%6.2f" % x}).strip('[]') + '%')
        if self.pbias is not None:
            bia = self.pbias.data.cpu().numpy() - self.nbias.data.cpu().numpy()
            total_weight = np.prod(bia.shape) / self.bNbits
            nonz_weight = [np.count_nonzero(bia[:, i]) * 100 for i in range(self.bNbits)]
            print('Bias:   ' + np.array2string(nonz_weight / total_weight, separator='%, ',
                                               formatter={'float_kind': lambda x: "%6.2f" % x}).strip('[]') + '%')

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(Bit_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class BitConv2d(Bit_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', Nbits=8, bin=False):

        self.total_weight = (in_channels // groups) * out_channels * kernel_size * kernel_size
        self.total_bias = out_channels
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BitConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, Nbits, bin)

    def quant(self, maxbit=10):
        # For binary model
        ## Re-quantize the binary part of the weight, keep scale unchanged

        dev = self.pweight.device
        ## Quantize weight
        weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
        weight = torch.sum(weight, dim=4)
        inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
        inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
        step = 2 ** (self.Nbits) - 1
        Rp = torch.round(inip * step)
        Rn = torch.round(inin * step)

        ## Increase layer precision if the binary part is out of range
        if self.Nbits >= maxbit:
            Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
            Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
        elif torch.max(torch.cat((Rp, Rn), 0)) > step:
            self.Nbits = self.Nbits + 1
            self.pweight = Parameter(
                self.pweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size,
                                            self.Nbits))
            self.nweight = Parameter(
                self.nweight.data.new_zeros(self.out_channels, self.in_channels // self.groups, *self.kernel_size,
                                            self.Nbits))
            N = self.Nbits
            ex = np.arange(N - 1, -1, -1)
            self.exps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
            self.scale.data = self.scale.data * (2 ** (N) - 1) / (2 ** (N - 1) - 1)

        for i in range(self.Nbits):
            ex = 2 ** (self.Nbits - i - 1)
            self.pweight.data[..., i] = torch.floor(Rp / ex)
            self.nweight.data[..., i] = torch.floor(Rn / ex)
            Rp = Rp - torch.floor(Rp / ex) * ex
            Rn = Rn - torch.floor(Rn / ex) * ex
        ## Quantize bias
        if self.pbias is not None:
            weight = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
            weight = torch.sum(weight, dim=1)
            inip = torch.where(weight > 0, weight, torch.full_like(weight, 0))
            inin = torch.where(weight <= 0, -weight, torch.full_like(weight, 0))
            step = 2 ** (self.bNbits) - 1
            Rp = torch.round(inip * step)
            Rn = torch.round(inin * step)

            ## Increase layer precision if the binary part is out of range
            if self.bNbits >= maxbit:
                Rp = torch.where(Rp > step, torch.full_like(Rp, step), Rp)
                Rn = torch.where(Rn > step, torch.full_like(Rn, step), Rn)
            elif torch.max(torch.cat((Rp, Rn), 0)) > step:
                self.Nbits = self.Nbits + 1
                self.pbias = Parameter(self.pbias.data.new_zeros(self.out_channels, self.Nbits))
                self.nbias = Parameter(self.nbias.data.new_zeros(self.out_channels, self.Nbits))
                N = self.bNbits
                ex = np.arange(N - 1, -1, -1)
                self.bexps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                self.biasscale.data = self.biasscale.data * (2 ** (N) - 1) / (2 ** (N - 1) - 1)

            for i in range(self.bNbits):
                ex = 2 ** (self.bNbits - i - 1)
                self.pbias.data[:, i] = torch.floor(Rp / ex)
                self.nbias.data[:, i] = torch.floor(Rn / ex)
                Rp = Rp - torch.floor(Rp / ex) * ex
                Rn = Rn - torch.floor(Rn / ex) * ex

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        if self.bin:
            dev = self.pweight.device
            weight = torch.mul((self.pweight - self.nweight), self.exps.to(dev))
            weight = bit_STE.apply(torch.sum(weight, dim=4), self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = torch.mul((self.pbias - self.nbias), self.bexps.to(dev))
                bias = bit_STE.apply(torch.sum(bias, dim=1), self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        elif self.ft:
            weight = bit_STE.apply(self.weight, self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = bit_STE.apply(self.bias, self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        else:
            return self.conv2d_forward(input, self.weight, self.bias)
            # return self.conv2d_forward(input, STE.apply(self.weight, self.Nbits), self.bias)

    def L1reg(self, reg):
        param = torch.cat((self.pweight, self.nweight), 0)
        total_weight = self.total_weight * self.Nbits
        reg += torch.sum(torch.sqrt(1e-8 + torch.sum(param ** 2, (0, 1, 2, 3)))) * total_weight
        if self.pbias is not None:
            param = torch.cat((self.pbias, self.nbias), 0)
            reg += torch.sum(torch.sqrt(1e-8 + torch.sum(param ** 2, 0)))
        return reg


class ResNet_dropout(nn.Module):

    def __init__(self, depth=20, num_classes=1000, block_name='BasicBlock', Nbits=4, act_bit=4, bin=False, dropout_rate=0.1):
        super(ResNet_dropout, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock_dropout
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck_dropout
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = BitConv2d(3, 16, kernel_size=3, padding=1,
                               bias=False, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.input_dropout = nn.Dropout2d(dropout_rate)
        self.layer1 = self._make_layer(block, 16, n, Nbits=Nbits, act_bit=act_bit, bin=bin, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 32, n, stride=2, Nbits=Nbits, act_bit=act_bit, bin=bin, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 64, n, stride=2, Nbits=Nbits, act_bit=act_bit, bin=bin, dropout_rate=dropout_rate)
        self.avgpool = nn.AvgPool2d(8)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = BitLinear(64 * block.expansion, num_classes, Nbits=Nbits, bin=bin)
        self.fc_dropout = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, BitConv2d):
                if m.bin:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    ini_w = torch.full_like(m.pweight[..., 0], 0)
                    ini_w.normal_(0, math.sqrt(2. / n))
                    m.ini2bit(ini_w)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, Nbits=4, act_bit=4, bin=False, dropout_rate=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, Nbits=Nbits, act_bit=act_bit, bin=bin, dropout_rate=dropout_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, Nbits=Nbits, act_bit=act_bit, bin=bin, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        x = self.input_dropout(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc_dropout(x)

        return x

    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if (isinstance(m, BitLinear) or isinstance(m, BitConv2d)) and 'downsample' not in name:
                if m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape) / m.Nbits
                    if m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape) / m.bNbits
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape)
        return N

    def total_bit(self):
        N = 0
        for name, m in self.named_modules():
            if (isinstance(m, BitLinear) or isinstance(m, BitConv2d)) and 'downsample' not in name:
                if m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape)
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape) * m.Nbits
                    if m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape) * m.bNbits
        return N

    def get_Nbits(self):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                Nbit_dict[name] = [m.Nbits, 0]
                if m.pbias is not None or m.bias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
        return Nbit_dict

    def set_Nbits(self, Nbit_dict):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                N = Nbit_dict[name]
                N0 = N[0]
                N1 = N[1]
                ex = np.arange(N0 - 1, -1, -1)
                m.exps = torch.Tensor((2 ** ex) / (2 ** (N0) - 1)).float()
                m.Nbits = N0
                if N1:
                    ex = np.arange(N1 - 1, -1, -1)
                    m.bexps = torch.Tensor((2 ** ex) / (2 ** (N1) - 1)).float()
                    m.bNbits = N1
                if m.bin:
                    m.pweight.data = m.pweight.data[..., 0:N0]
                    m.nweight.data = m.nweight.data[..., 0:N0]
                    if N1:
                        m.pbias.data = m.pbias.data[..., 0:N1]
                        m.nbias.data = m.nbias.data[..., 0:N1]

    def set_zero(self):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                weight = m.pweight.data - m.nweight.data
                if m.Nbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                    m.zero = True
                else:
                    m.zero = False
                if m.pbias is not None:
                    weight = m.pbias.data - m.nbias.data
                    if m.bNbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                        m.bzero = True
                    else:
                        m.bzero = False

    def pruning(self, threshold=1.0,
                drop=True):  # Use drop to control whether 0 bit after pruning will be removed, 0 bit before pruning will always be removed
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                if m.Nbits > 1:
                    # Remove MSB
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = nonz_weight / total_weight
                    N = m.Nbits
                    N0 = m.Nbits
                    pweight = m.pweight.data
                    nweight = m.nweight.data
                    for i in range(N):
                        if nonz_weight[i] == 0:
                            m.pweight.data = pweight[..., i + 1:N]
                            m.nweight.data = nweight[..., i + 1:N]
                            m.Nbits -= 1
                            if m.Nbits == 1:
                                break
                        elif nonz_weight[i] < threshold:  # set MSB to 0, remove MSB if "drop"
                            if drop:
                                m.pweight.data = pweight[..., i + 1:N] + pweight[..., i].unsqueeze(-1)
                                m.nweight.data = nweight[..., i + 1:N] + nweight[..., i].unsqueeze(-1)
                                m.Nbits -= 1
                                if m.Nbits == 1:
                                    break
                            else:
                                m.pweight.data = pweight[..., i:N] + pweight[..., i].unsqueeze(-1)
                                m.nweight.data = nweight[..., i:N] + nweight[..., i].unsqueeze(-1)
                                m.pweight.data[..., 0] = 0.
                                m.nweight.data[..., 0] = 0.
                            m.pweight.data = torch.where(m.pweight.data < 1, m.pweight.data,
                                                         torch.full_like(m.pweight.data, 1.))
                            m.nweight.data = torch.where(m.nweight.data < 1, m.nweight.data,
                                                         torch.full_like(m.nweight.data, 1.))
                        else:
                            break
                    # Remove LSB
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = nonz_weight / total_weight
                    N = m.Nbits
                    pweight = m.pweight.data
                    nweight = m.nweight.data
                    if m.Nbits > 1:
                        for i in range(N):
                            if nonz_weight[N - 1 - i] <= threshold:
                                m.pweight.data = pweight[..., 0:N - 1 - i]
                                m.nweight.data = nweight[..., 0:N - 1 - i]
                                m.Nbits -= 1
                                m.scale.data = m.scale.data * 2
                                if m.Nbits == 1:
                                    break
                            else:
                                break
                    # Reset exps
                    N = m.Nbits
                    ex = np.arange(N - 1, -1, -1)
                    m.exps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.scale.data = m.scale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    ## Match the shape of grad to data
                    if m.pweight.grad is not None:
                        m.pweight.grad.data = m.pweight.grad.data[..., 0:N]
                        m.nweight.grad.data = m.nweight.grad.data[..., 0:N]
                # For bias
                if m.pbias is not None and m.bNbits > 1:
                    # Remove MSB
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = nonz_weight / total_weight
                    N = m.bNbits
                    N0 = m.bNbits
                    pweight = m.pbias.data
                    nweight = m.nbias.data
                    for i in range(N):
                        if nonz_weight[i] == 0:
                            m.pbias.data = pweight[..., i + 1:N]
                            m.nbias.data = nweight[..., i + 1:N]
                            m.bNbits -= 1
                            if m.bNbits == 1:
                                break
                        elif nonz_weight[i] < threshold:
                            m.pbias.data = pweight[..., i + 1:N] + pweight[..., i].unsqueeze(-1)
                            m.nbias.data = nweight[..., i + 1:N] + nweight[..., i].unsqueeze(-1)
                            m.pbias.data = torch.where(m.pbias.data < 1, m.pbias.data,
                                                       torch.full_like(m.pbias.data, 1.))
                            m.nbias.data = torch.where(m.nbias.data < 1, m.nbias.data,
                                                       torch.full_like(m.nbias.data, 1.))
                            m.bNbits -= 1
                            if m.bNbits == 1:
                                break
                        else:
                            break
                    # Remove LSB
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = nonz_weight / total_weight
                    if m.bNbits > 1:
                        N = m.bNbits
                        pweight = m.pbias.data
                        nweight = m.nbias.data
                        for i in range(N):
                            if nonz_weight[N - 1 - i] <= threshold:
                                m.pbias.data = pweight[..., 0:N - 1 - i]
                                m.nbias.data = nweight[..., 0:N - 1 - i]
                                m.bNbits -= 1
                                m.biasscale.data = m.biasscale.data * 2
                                if m.bNbits == 1:
                                    break
                            else:
                                break
                    # Reset exps
                    N = m.bNbits
                    ex = np.arange(N - 1, -1, -1)
                    m.bexps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.biasscale.data = m.biasscale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    ## Match the shape of grad to data
                    if m.pbias.grad is not None:
                        m.pbias.grad.data = m.pbias.grad.data[..., 0:N]
                        m.nbias.grad.data = m.nbias.grad.data[..., 0:N]
                if m.pbias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
                else:
                    Nbit_dict[name] = [m.Nbits, 0]
        return Nbit_dict


class BasicBlock_dropout(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, Nbits=4, act_bit=4, bin=False, dropout_rate=0.1):
        super(BasicBlock_dropout, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, Nbits=Nbits, bin=bin)
        self.bn2 = nn.BatchNorm2d(planes)
        if act_bit > 3:
            self.relu1 = nn.ReLU6(inplace=True)
            self.relu2 = nn.ReLU6(inplace=True)
        else:
            self.relu1 = PACT()
            self.relu2 = PACT()
        self.downsample = downsample
        self.stride = stride
        self.act_bit = act_bit
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = STE.apply(out, self.act_bit)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        out = self.dropout(out)

        out = STE.apply(out, self.act_bit)

        return out


class Bottleneck_dropout(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.1):
        super(Bottleneck_dropout, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out







