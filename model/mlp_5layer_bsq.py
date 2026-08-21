import numpy as np
import torch
import torch.nn as nn

from model.resnet_bsq import BitLinear


class MLP_5Layer_Tanh_bsq(nn.Module):
    """BSQ version of MLP_5Layer_Tanh: [2048, 1024, 512, 256, 100] + Tanh."""

    HIDDEN = (2048, 1024, 512, 256, 100)

    def __init__(self, input_dim: int, num_classes: int, Nbits: int = 8, bin: bool = False):
        super().__init__()
        self.input_fc = BitLinear(input_dim, self.HIDDEN[0], Nbits=Nbits, bin=bin)
        self.hidden_fc1 = BitLinear(self.HIDDEN[0], self.HIDDEN[1], Nbits=Nbits, bin=bin)
        self.hidden_fc2 = BitLinear(self.HIDDEN[1], self.HIDDEN[2], Nbits=Nbits, bin=bin)
        self.hidden_fc3 = BitLinear(self.HIDDEN[2], self.HIDDEN[3], Nbits=Nbits, bin=bin)
        self.hidden_fc4 = BitLinear(self.HIDDEN[3], self.HIDDEN[4], Nbits=Nbits, bin=bin)
        if num_classes != self.HIDDEN[-1]:
            self.output_fc = BitLinear(self.HIDDEN[-1], num_classes, Nbits=Nbits, bin=bin)
        else:
            self.output_fc = None

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        x = torch.tanh(self.input_fc(x))
        x = torch.tanh(self.hidden_fc1(x))
        x = torch.tanh(self.hidden_fc2(x))
        x = torch.tanh(self.hidden_fc3(x))
        x = self.hidden_fc4(x)
        if self.output_fc is not None:
            x = torch.tanh(x)
            x = self.output_fc(x)
        return x

    def to_bin(self):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.to_bin()

    def to_float(self):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.to_float()

    def finetune_to_bit(self):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.finetune_to_bit()

    def set_zero(self):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.bzero = m.Nbits > 1

    def pruning(self, threshold=1.0, drop=True):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
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
            if isinstance(m, BitLinear):
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
            if isinstance(m, BitLinear):
                if getattr(m, "bin", False) and m.pweight is not None:
                    N += m.pweight.numel() + m.nweight.numel()
                    if m.pbias is not None:
                        N += m.pbias.numel() + m.nbias.numel()
                else:
                    N += m.weight.numel() * m.Nbits
                    if m.bias is not None:
                        N += m.bias.numel() * m.bNbits
        return N
