import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.resnet_bsq import BitLinear
import os
from torchvision import datasets, transforms
from PIL import Image



class MLP_VIB_bsq(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=64, Nbits=8, bin=False):
        super().__init__()
        self.input_fc = BitLinear(input_dim, 3000, Nbits=Nbits, bin=bin)
        self.hidden_fc1 = BitLinear(3000, 1000, Nbits=Nbits, bin=bin)
        self.hidden_fc2 = BitLinear(1000, 512, Nbits=Nbits, bin=bin)
        self.hidden_fc3 = BitLinear(512, latent_dim * 2, Nbits=Nbits, bin=bin)
        self.classifier = BitLinear(latent_dim, output_dim, Nbits=Nbits, bin=bin)
        self.latent_dim = latent_dim

    def encode(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = F.relu(self.input_fc(x))
        h = F.relu(self.hidden_fc1(h))
        h = F.relu(self.hidden_fc2(h))
        h = self.hidden_fc3(h)
        mu, std = h[:, :self.latent_dim], h[:, self.latent_dim:]
        std = F.softplus(std - 5, beta=1)
        return mu, std

    def forward(self, x):
        mu, std = self.encode(x)
        eps = torch.randn_like(std)
        z = mu + std * eps
        output = self.classifier(z)
        return output

    def get_mu_std(self, x):
        return self.encode(x)

    def to_bin(self):
        self.input_fc.to_bin()
        self.hidden_fc1.to_bin()
        self.hidden_fc2.to_bin()
        self.hidden_fc3.to_bin()
        self.classifier.to_bin()

    def to_float(self):
        self.input_fc.to_float()
        self.hidden_fc1.to_float()
        self.hidden_fc2.to_float()
        self.hidden_fc3.to_float()
        self.classifier.to_float()

    def finetune_to_bit(self):
        self.input_fc.finetune_to_bit()
        self.hidden_fc1.finetune_to_bit()
        self.hidden_fc2.finetune_to_bit()
        self.hidden_fc3.finetune_to_bit()
        self.classifier.finetune_to_bit()

    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                if hasattr(m, 'bin') and m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape) / m.Nbits
                    if hasattr(m, 'pbias') and m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape) / m.bNbits
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if hasattr(m, 'bias') and m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape)
        return N

    def total_bit(self):
        N = 0
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                if hasattr(m, 'bin') and m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if hasattr(m, 'pbias') and m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape)
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape) * m.Nbits
                    if hasattr(m, 'bias') and m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape) * m.bNbits
        return N

    def get_Nbits(self):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                Nbit_dict[name] = [m.Nbits, 0]
                if hasattr(m, 'pbias') and m.pbias is not None or hasattr(m, 'bias') and m.bias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
        return Nbit_dict

    def set_Nbits(self, Nbit_dict):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
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
                if hasattr(m, 'bin') and m.bin:
                    m.pweight.data = m.pweight.data[..., 0:N0]
                    m.nweight.data = m.nweight.data[..., 0:N0]
                    if N1:
                        m.pbias.data = m.pbias.data[..., 0:N1]
                        m.nbias.data = m.nbias.data[..., 0:N1]

    def set_zero(self):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                weight = m.pweight.data - m.nweight.data
                if m.Nbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                    m.zero = True
                else:
                    m.zero = False
                if hasattr(m, 'pbias') and m.pbias is not None:
                    weight = m.pbias.data - m.nbias.data
                    if m.bNbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                        m.bzero = True
                    else:
                        m.bzero = False

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
                            m.pweight.data = pweight[..., i + 1:N]
                            m.nweight.data = nweight[..., i + 1:N]
                            m.Nbits -= 1
                            if m.Nbits == 1:
                                break
                        elif nonz_weight[i] < threshold:
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
                            m.pweight.data = torch.where(m.pweight.data < 1, m.pweight.data, torch.full_like(m.pweight.data, 1.))
                            m.nweight.data = torch.where(m.nweight.data < 1, m.nweight.data, torch.full_like(m.nweight.data, 1.))
                        else:
                            break
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                    N = m.Nbits
                    ex = np.arange(N - 1, -1, -1)
                    m.exps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.scale.data = m.scale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    if m.pweight.grad is not None:
                        m.pweight.grad.data = m.pweight.grad.data[..., 0:N]
                        m.nweight.grad.data = m.nweight.grad.data[..., 0:N]
                if hasattr(m, 'pbias') and m.pbias is not None and m.bNbits > 1:
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                            m.pbias.data = torch.where(m.pbias.data < 1, m.pbias.data, torch.full_like(m.pbias.data, 1.))
                            m.nbias.data = torch.where(m.nbias.data < 1, m.nbias.data, torch.full_like(m.nbias.data, 1.))
                            m.bNbits -= 1
                            if m.bNbits == 1:
                                break
                        else:
                            break
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                    N = m.bNbits
                    ex = np.arange(N - 1, -1, -1)
                    m.bexps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.biasscale.data = m.biasscale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    if m.pbias.grad is not None:
                        m.pbias.grad.data = m.pbias.grad.data[..., 0:N]
                        m.nbias.grad.data = m.nbias.grad.data[..., 0:N]
                if hasattr(m, 'pbias') and m.pbias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
                else:
                    Nbit_dict[name] = [m.Nbits, 0]
        return Nbit_dict



class MLP_bsq(nn.Module):
    def __init__(self, input_dim, output_dim, Nbits=8, bin=False):
        super().__init__()
        self.input_fc = BitLinear(input_dim, 3000, Nbits=Nbits, bin=bin)
        self.hidden_fc1 = BitLinear(3000, 1000, Nbits=Nbits, bin=bin)
        self.hidden_fc2 = BitLinear(1000, 512, Nbits=Nbits, bin=bin)
        self.hidden_fc3 = BitLinear(512, 64, Nbits=Nbits, bin=bin)
        self.output_fc = BitLinear(64, output_dim, Nbits=Nbits, bin=bin)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = F.relu(self.input_fc(x))
        h = F.relu(self.hidden_fc1(h))
        h = F.relu(self.hidden_fc2(h))
        h = F.relu(self.hidden_fc3(h))
        output = self.output_fc(h)
        return output

    def to_bin(self):
        self.input_fc.to_bin()
        self.hidden_fc1.to_bin()
        self.hidden_fc2.to_bin()
        self.hidden_fc3.to_bin()
        self.output_fc.to_bin()

    def to_float(self):
        self.input_fc.to_float()
        self.hidden_fc1.to_float()
        self.hidden_fc2.to_float()
        self.hidden_fc3.to_float()
        self.output_fc.to_float()

    def finetune_to_bit(self):
        self.input_fc.finetune_to_bit()
        self.hidden_fc1.finetune_to_bit()
        self.hidden_fc2.finetune_to_bit()
        self.hidden_fc3.finetune_to_bit()
        self.output_fc.finetune_to_bit()

    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                if hasattr(m, 'bin') and m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape) / m.Nbits
                    if hasattr(m, 'pbias') and m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape) / m.bNbits
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if hasattr(m, 'bias') and m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape)
        return N

    def total_bit(self):
        N = 0
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                if hasattr(m, 'bin') and m.bin:
                    param = m.pweight - m.nweight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if hasattr(m, 'pbias') and m.pbias is not None:
                        param = m.pbias - m.nbias
                        N += np.prod(param.data.cpu().numpy().shape)
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape) * m.Nbits
                    if hasattr(m, 'bias') and m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape) * m.bNbits
        return N

    def get_Nbits(self):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                Nbit_dict[name] = [m.Nbits, 0]
                if hasattr(m, 'pbias') and m.pbias is not None or hasattr(m, 'bias') and m.bias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
        return Nbit_dict

    def set_Nbits(self, Nbit_dict):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
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
                if hasattr(m, 'bin') and m.bin:
                    m.pweight.data = m.pweight.data[..., 0:N0]
                    m.nweight.data = m.nweight.data[..., 0:N0]
                    if N1:
                        m.pbias.data = m.pbias.data[..., 0:N1]
                        m.nbias.data = m.nbias.data[..., 0:N1]

    def set_zero(self):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                weight = m.pweight.data - m.nweight.data
                if m.Nbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                    m.zero = True
                else:
                    m.zero = False
                if hasattr(m, 'pbias') and m.pbias is not None:
                    weight = m.pbias.data - m.nbias.data
                    if m.bNbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                        m.bzero = True
                    else:
                        m.bzero = False

    def pruning(self, threshold=1.0, drop=True):
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
                            m.pweight.data = pweight[..., i + 1:N]
                            m.nweight.data = nweight[..., i + 1:N]
                            m.Nbits -= 1
                            if m.Nbits == 1:
                                break
                        elif nonz_weight[i] < threshold:
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
                            m.pweight.data = torch.where(m.pweight.data < 1, m.pweight.data, torch.full_like(m.pweight.data, 1.))
                            m.nweight.data = torch.where(m.nweight.data < 1, m.nweight.data, torch.full_like(m.nweight.data, 1.))
                        else:
                            break
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                    N = m.Nbits
                    ex = np.arange(N - 1, -1, -1)
                    m.exps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.scale.data = m.scale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    if m.pweight.grad is not None:
                        m.pweight.grad.data = m.pweight.grad.data[..., 0:N]
                        m.nweight.grad.data = m.nweight.grad.data[..., 0:N]
                if hasattr(m, 'pbias') and m.pbias is not None and m.bNbits > 1:
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                            m.pbias.data = torch.where(m.pbias.data < 1, m.pbias.data, torch.full_like(m.pbias.data, 1.))
                            m.nbias.data = torch.where(m.nbias.data < 1, m.nbias.data, torch.full_like(m.nbias.data, 1.))
                            m.bNbits -= 1
                            if m.bNbits == 1:
                                break
                        else:
                            break
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                    N = m.bNbits
                    ex = np.arange(N - 1, -1, -1)
                    m.bexps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.biasscale.data = m.biasscale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    if m.pbias.grad is not None:
                        m.pbias.grad.data = m.pbias.grad.data[..., 0:N]
                        m.nbias.grad.data = m.nbias.grad.data[..., 0:N] 


class MLP_bsq_dropout(nn.Module):
    def __init__(self, input_dim, output_dim, Nbits=8, bin=False, dropout_rate=0.1):
        super().__init__()
        self.input_fc = BitLinear(input_dim, 3000, Nbits=Nbits, bin=bin)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        self.hidden_fc1 = BitLinear(3000, 1000, Nbits=Nbits, bin=bin)
        self.hidden_dropout1 = nn.Dropout(dropout_rate)
        
        self.hidden_fc2 = BitLinear(1000, 512, Nbits=Nbits, bin=bin)
        self.hidden_dropout2 = nn.Dropout(dropout_rate)
        
        self.hidden_fc3 = BitLinear(512, 64, Nbits=Nbits, bin=bin)
        self.hidden_dropout3 = nn.Dropout(dropout_rate)
        
        self.output_fc = BitLinear(64, output_dim, Nbits=Nbits, bin=bin)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        h = F.relu(self.input_fc(x))
        h = self.input_dropout(h)
        
        h = F.relu(self.hidden_fc1(h))
        h = self.hidden_dropout1(h)
        
        h = F.relu(self.hidden_fc2(h))
        h = self.hidden_dropout2(h)
        
        h = F.relu(self.hidden_fc3(h))
        h = self.hidden_dropout3(h)
        
        output = self.output_fc(h)
        output = self.output_dropout(output)
        
        return output

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

    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
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
            if isinstance(m, BitLinear):
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
            if isinstance(m, BitLinear):
                Nbit_dict[name] = [m.Nbits, 0]
                if m.pbias is not None or m.bias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
        return Nbit_dict

    def set_Nbits(self, Nbit_dict):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
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
            if isinstance(m, BitLinear):
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

    def pruning(self, threshold=1.0, drop=True):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m, BitLinear):
                if m.Nbits > 1:
                    # Remove MSB
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
                            m.pweight.data = pweight[..., i + 1:N]
                            m.nweight.data = nweight[..., i + 1:N]
                            m.Nbits -= 1
                            if m.Nbits == 1:
                                break
                        elif nonz_weight[i] < threshold:
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
                            m.pweight.data = torch.where(m.pweight.data < 1, m.pweight.data, torch.full_like(m.pweight.data, 1.))
                            m.nweight.data = torch.where(m.nweight.data < 1, m.nweight.data, torch.full_like(m.nweight.data, 1.))
                        else:
                            break
                    weight = m.pweight.data.cpu().numpy() - m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.Nbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.Nbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                    N = m.Nbits
                    ex = np.arange(N - 1, -1, -1)
                    m.exps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.scale.data = m.scale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    if m.pweight.grad is not None:
                        m.pweight.grad.data = m.pweight.grad.data[..., 0:N]
                        m.nweight.grad.data = m.nweight.grad.data[..., 0:N]
                if hasattr(m, 'pbias') and m.pbias is not None and m.bNbits > 1:
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                            m.pbias.data = torch.where(m.pbias.data < 1, m.pbias.data, torch.full_like(m.pbias.data, 1.))
                            m.nbias.data = torch.where(m.nbias.data < 1, m.nbias.data, torch.full_like(m.nbias.data, 1.))
                            m.bNbits -= 1
                            if m.bNbits == 1:
                                break
                        else:
                            break
                    weight = m.pbias.data.cpu().numpy() - m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape) / m.bNbits
                    nonz_weight = [np.count_nonzero(weight[..., i]) * 100 for i in range(m.bNbits)]
                    nonz_weight = np.array(nonz_weight) / total_weight
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
                    N = m.bNbits
                    ex = np.arange(N - 1, -1, -1)
                    m.bexps = torch.Tensor((2 ** ex) / (2 ** (N) - 1)).float()
                    m.biasscale.data = m.biasscale.data * (2 ** (N) - 1) / (2 ** (N0) - 1)
                    if m.pbias.grad is not None:
                        m.pbias.grad.data = m.pbias.grad.data[..., 0:N]
                        m.nbias.grad.data = m.nbias.grad.data[..., 0:N]
                if m.pbias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
                else:
                    Nbit_dict[name] = [m.Nbits, 0]
        return Nbit_dict