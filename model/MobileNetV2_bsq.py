"""BSQ-compatible MobileNetV2 (torchvision backbone with BitConv2d / BitLinear)."""

import numpy as np
import torch
import torch.nn as nn

from model.MobileNetV2 import mobilenetv2
from model.resnet_bsq import BitConv2d, BitLinear


def _scalar(param):
    if isinstance(param, tuple):
        return param[0]
    return param


def _replace_with_bit_layers(module: nn.Module, Nbits: int, bin: bool) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            bit_conv = BitConv2d(
                child.in_channels,
                child.out_channels,
                _scalar(child.kernel_size),
                stride=_scalar(child.stride),
                padding=_scalar(child.padding),
                dilation=_scalar(child.dilation),
                groups=child.groups,
                bias=child.bias is not None,
                Nbits=Nbits,
                bin=bin,
            )
            if not bin:
                bit_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    bit_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, bit_conv)
        elif isinstance(child, nn.Linear):
            bit_lin = BitLinear(
                child.in_features,
                child.out_features,
                Nbits=Nbits,
                bias=child.bias is not None,
                bin=bin,
            )
            if not bin:
                bit_lin.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    bit_lin.bias.data.copy_(child.bias.data)
            setattr(module, name, bit_lin)
        else:
            _replace_with_bit_layers(child, Nbits=Nbits, bin=bin)


class MobileNetV2_bsq(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, Nbits=8, bin=False):
        super().__init__()
        self.net = mobilenetv2(num_classes=num_classes, in_channels=in_channels)
        _replace_with_bit_layers(self.net, Nbits=Nbits, bin=bin)

    def forward(self, x):
        return self.net(x)

    def to_bin(self):
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                m.to_bin()

    def to_float(self):
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                m.to_float()

    def finetune_to_bit(self):
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                m.finetune_to_bit()

    def set_zero(self):
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                m.bzero = m.Nbits > 1

    def pruning(self, threshold=1.0, drop=True):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                if m.Nbits > 1:
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight

                    N = m.Nbits
                    N0 = m.Nbits
                    pweight = m.pweight.data
                    nweight = m.nweight.data

                    for i in range(N):
                        if nonz_weight[i] == 0:
                            m.pweight.data = pweight[..., i + 1 : N]
                            m.nweight.data = nweight[..., i + 1 : N]
                            m.Nbits -= 1
                            if m.Nbits == 1:
                                break
                        elif nonz_weight[i] < threshold:
                            if drop:
                                m.pweight.data = pweight[..., i + 1 : N] + pweight[..., i].unsqueeze(-1)
                                m.nweight.data = nweight[..., i + 1 : N] + nweight[..., i].unsqueeze(-1)
                                m.Nbits -= 1
                                if m.Nbits == 1:
                                    break
                            else:
                                m.pweight.data = pweight[..., i:N] + pweight[..., i].unsqueeze(-1)
                                m.nweight.data = nweight[..., i:N] + nweight[..., i].unsqueeze(-1)
                                m.pweight.data[..., 0] = 0.0
                                m.nweight.data[..., 0] = 0.0
                            m.pweight.data = torch.where(
                                m.pweight.data < 1, m.pweight.data, torch.full_like(m.pweight.data, 1.0)
                            )
                            m.nweight.data = torch.where(
                                m.nweight.data < 1, m.nweight.data, torch.full_like(m.nweight.data, 1.0)
                            )
                        else:
                            break

                    if hasattr(m, "exps") and m.Nbits != N0:
                        ex = np.arange(m.Nbits - 1, -1, -1)
                        m.exps = torch.Tensor((2**ex) / (2 ** (m.Nbits) - 1)).float().to(m.pweight.device)

                    Nbit_dict[name] = m.Nbits
        return Nbit_dict

    def total_param(self):
        N = 0
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                if getattr(m, "bin", False) and m.pweight is not None:
                    N += m.pweight.numel() + m.nweight.numel()
                    if m.pbias is not None:
                        N += m.pbias.numel() + m.nbias.numel()
                else:
                    N += m.weight.numel()
                    if m.bias is not None:
                        N += m.bias.numel()
        return N

    def total_bit(self):
        N = 0
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                if getattr(m, "bin", False) and m.pweight is not None:
                    N += m.pweight.numel() + m.nweight.numel()
                    if m.pbias is not None:
                        N += m.pbias.numel() + m.nbias.numel()
                else:
                    N += m.weight.numel() * m.Nbits
                    if m.bias is not None:
                        N += m.bias.numel() * m.bNbits
        return N
