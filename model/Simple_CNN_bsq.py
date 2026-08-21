import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet_bsq import BitConv2d, BitLinear


class Simple_CNN_Tanh_bsq(nn.Module):
    """
    BSQ-compatible version of `Simple_CNN_Tanh` (used by net_name == 'simple_cnn').

    Key differences vs the float model:
    - Conv/Linear layers are replaced with BitConv2d / BitLinear.
    - Exposes BSQ-required methods: pruning / total_param / total_bit / to_bin / ...
    """

    def __init__(self, num_classes=100, in_channels=3, size=None, Nbits=8, bin=False, hidden=128):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.in_channels == 3:
            # Match `Simple_CNN_Tanh` defaults in `model/DNN.py`
            if size == "small":
                cfg = [16, 16, "M", 32, 32, "M", 64, "M"]
            else:
                cfg = [32, 32, "M", 64, 64, "M", 128, 128, "M"]
        else:
            if size == "small":
                cfg = [16, 16, "M", 32, 32]
            else:
                cfg = [64, "M", 64]

        layers = []
        c = self.in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(BitConv2d(c, v, kernel_size=3, stride=1, padding=1, Nbits=Nbits, bin=bin))
                layers.append(nn.Tanh())
                c = v

        self.features = nn.Sequential(*layers)

        # For 32x32 input with 3 pools => 4x4 spatial.
        # If a different size is used, this may need adjustment.
        if self.in_channels == 3:
            self.classifier = nn.Sequential(
                BitLinear(c * 4 * 4, hidden, Nbits=Nbits, bin=bin),
                nn.Tanh(),
                BitLinear(hidden, self.num_classes, Nbits=Nbits, bin=bin),
            )
        else:
            self.classifier = BitLinear(c * 4 * 4, self.num_classes, Nbits=Nbits, bin=bin)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_last_hidden_layer(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if isinstance(self.classifier, nn.Sequential):
            # return the hidden layer output (before final logits)
            x = self.classifier[0](x)
            return x
        return x

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
                if m.Nbits > 1:
                    m.bzero = True
                else:
                    m.bzero = False

    def pruning(self, threshold=1.0, drop=True):
        """
        Copied from existing BSQ models: remove redundant/low-importance bit-slices.
        Returns a dict mapping module name -> updated Nbits.
        """
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
                N += m.pweight.numel() + m.nweight.numel()
                if getattr(m, "bias", None) is not None:
                    N += m.bias.numel()
        return N

    def total_bit(self):
        N = 0
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                N += (m.pweight.numel() + m.nweight.numel()) * m.Nbits
                if getattr(m, "bias", None) is not None:
                    N += m.bias.numel() * 32
        return N

