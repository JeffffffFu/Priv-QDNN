import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_bsq import BitConv2d, BitLinear
import numpy as np

class CNN_MNIST_bsq(nn.Module):
    def __init__(self, num_classes=10, Nbits=8, bin=True, latent_dim=64):
        super(CNN_MNIST_bsq, self).__init__()
        self.out_features = num_classes
        self.latent_dim = latent_dim
        self.conv1 = BitConv2d(1, 128, 3, 1, 1, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = BitConv2d(128, 256, 3, 1, 1, Nbits=Nbits, bin=bin)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = BitConv2d(256, 512, 3, 1, 1, Nbits=Nbits, bin=bin)
        self.bn3 = nn.BatchNorm2d(512)
        self.mp = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # --- Fully-connected part ---
        self.dropout = nn.Dropout()
        self.fc1 = BitLinear(8192, 512, Nbits=Nbits, bin=bin)  # 32x32 after 3 maxpools: 512*4*4 = 8192
        self.fc2 = BitLinear(512, 64, Nbits=Nbits, bin=bin)
        # Add latent representation layer for VIB
        self.fc_encode = BitLinear(64, latent_dim * 2, Nbits=Nbits, bin=bin)
        # Output layer for standard forward
        self.fc3_standard = BitLinear(64, num_classes, Nbits=Nbits, bin=bin)
        # Output layer for VIB forward
        self.fc3 = BitLinear(latent_dim, num_classes, Nbits=Nbits, bin=bin)

    def encode(self, x):
        """Encode input to latent representation with mu and std"""
        # Forward through CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mp(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        x = x.view(-1, 8192)  # 32x32 after 3 maxpools: 512*4*4 = 8192  # 32x32 after 3 maxpools: 512*3*3 = 4608
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Encode to latent representation
        h = self.fc_encode(x)  # [batch, latent_dim * 2]
        mu, std = h[:, :self.latent_dim], h[:, self.latent_dim:]
        std = F.softplus(std - 5, beta=1)  # Ensure std is positive
        return mu, std
    def forward(self, x, use_vib=False):
        if use_vib:
            mu, std = self.encode(x)
            eps = torch.randn_like(std)
            z = mu + std * eps
            output = self.fc3(z)
        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.mp(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.mp(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.mp(x)
            x = x.view(-1, 8192)  # 32x32 after 3 maxpools: 512*4*4 = 8192
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            output = self.fc3_standard(x)
        return output
    
    def get_mu_std(self, x):
        """Get mu and std from latent representation"""
        return self.encode(x)

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
                            m.pweight.data = pweight[..., i + 1:N]
                            m.nweight.data = nweight[..., i + 1:N]
                            m.Nbits -= 1
                            if m.Nbits == 1:
                                break
                        elif nonz_weight[i] < threshold:
                            if drop:
                                m.pweight.data = pweight[..., i + 1:N]
                                m.nweight.data = nweight[..., i + 1:N]
                                m.Nbits -= 1
                                if m.Nbits == 1:
                                    break
                    
                    if hasattr(m, 'exps') and m.Nbits != N0:
                        ex = np.arange(m.Nbits - 1, -1, -1)
                        m.exps = torch.Tensor((2 ** ex) / (2 ** (m.Nbits) - 1)).float().to(m.pweight.device)
                    
                    Nbit_dict[name] = m.Nbits
        return Nbit_dict

    def total_param(self):
        N = 0
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                N += m.pweight.numel() + m.nweight.numel()
                if m.bias is not None:
                    N += m.bias.numel()
        return N

    def total_bit(self):
        N = 0
        for m in self.modules():
            if isinstance(m, (BitLinear, BitConv2d)):
                N += (m.pweight.numel() + m.nweight.numel()) * m.Nbits
                if m.bias is not None:
                    N += m.bias.numel() * 32
        return N

class CNN_UTKFace_bsq(nn.Module):
    def __init__(self, Nbits=4, bin=False):
        super(CNN_UTKFace_bsq, self).__init__()
        self.conv1 = BitConv2d(3, 16, 3, stride=1, padding=0, Nbits=Nbits, bin=bin)
        self.conv2 = BitConv2d(16, 32, 3, stride=1, padding=0, Nbits=Nbits, bin=bin)
        self.conv3 = BitConv2d(32, 64, 3, stride=1, padding=0, Nbits=Nbits, bin=bin)
        self.fc1 = BitLinear(64 * 26 * 26, 128, Nbits=Nbits, bin=bin)
        self.fc2 = BitLinear(128, 64, Nbits=Nbits, bin=bin)
        self.fc3 = BitLinear(64, 2, Nbits=Nbits, bin=bin)
        self.activation = F.relu
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = self.conv2(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = self.conv3(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = self.dropout(h)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc3(h)
        output = F.log_softmax(h, dim=1)
        return output

    def get_last_hidden_layer(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = self.conv2(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = self.conv3(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = self.activation(h)
        output = self.fc2(h)
        return output.detach().cpu().numpy()

    def to_bin(self):
        self.conv1.to_bin()
        self.conv2.to_bin()
        self.conv3.to_bin()
        self.fc1.to_bin()
        self.fc2.to_bin()
        self.fc3.to_bin()

    def to_float(self):
        self.conv1.to_float()
        self.conv2.to_float()
        self.conv3.to_float()
        self.fc1.to_float()
        self.fc2.to_float()
        self.fc3.to_float()

    def finetune_to_bit(self):
        self.conv1.finetune_to_bit()
        self.conv2.finetune_to_bit()
        self.conv3.finetune_to_bit()
        self.fc1.finetune_to_bit()
        self.fc2.finetune_to_bit()
        self.fc3.finetune_to_bit()

    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if (isinstance(m, BitLinear) or isinstance(m, BitConv2d)):
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
            if (isinstance(m, BitLinear) or isinstance(m, BitConv2d)):
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
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
                Nbit_dict[name] = [m.Nbits, 0]
                if hasattr(m, 'pbias') and m.pbias is not None or hasattr(m, 'bias') and m.bias is not None:
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
                if hasattr(m, 'bin') and m.bin:
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
                if hasattr(m, 'pbias') and m.pbias is not None:
                    weight = m.pbias.data - m.nbias.data
                    if m.bNbits == 1 and (np.count_nonzero(weight.cpu().numpy()) == 0):
                        m.bzero = True
                    else:
                        m.bzero = False

    def pruning(self, threshold=1.0, drop=True):
        for name, m in self.named_modules():
            if isinstance(m, BitLinear) or isinstance(m, BitConv2d):
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


class CNN_UTKFace_bsq_dropout(nn.Module):
    def __init__(self, Nbits=4, bin=False, dropout_rate=0.1):
        super(CNN_UTKFace_bsq_dropout, self).__init__()
        self.conv1 = BitConv2d(3, 16, 3, stride=1, padding=0, Nbits=Nbits, bin=bin)
        self.conv1_dropout = nn.Dropout2d(dropout_rate)
        
        self.conv2 = BitConv2d(16, 32, 3, stride=1, padding=0, Nbits=Nbits, bin=bin)
        self.conv2_dropout = nn.Dropout2d(dropout_rate)
        
        self.conv3 = BitConv2d(32, 64, 3, stride=1, padding=0, Nbits=Nbits, bin=bin)
        self.conv3_dropout = nn.Dropout2d(dropout_rate)
        
        self.fc1 = BitLinear(64 * 26 * 26, 128, Nbits=Nbits, bin=bin)
        self.fc1_dropout = nn.Dropout(dropout_rate)
        
        self.fc2 = BitLinear(128, 64, Nbits=Nbits, bin=bin)
        self.fc2_dropout = nn.Dropout(dropout_rate)
        
        self.fc3 = BitLinear(64, 2, Nbits=Nbits, bin=bin)
        self.fc3_dropout = nn.Dropout(dropout_rate)
        
        self.activation = F.relu

    def forward(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv1_dropout(h)
        h = F.max_pool2d(h, 2)
        
        h = self.conv2(h)
        h = self.activation(h)
        h = self.conv2_dropout(h)
        h = F.max_pool2d(h, 2)
        
        h = self.conv3(h)
        h = self.activation(h)
        h = self.conv3_dropout(h)
        h = F.max_pool2d(h, 2)
        
        h = torch.flatten(h, 1)
        
        h = self.fc1(h)
        h = self.activation(h)
        h = self.fc1_dropout(h)
        
        h = self.fc2(h)
        h = self.activation(h)
        h = self.fc2_dropout(h)
        
        h = self.fc3(h)
        h = self.fc3_dropout(h)
        output = F.log_softmax(h, dim=1)
        return output

    def get_last_hidden_layer(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv1_dropout(h)
        h = F.max_pool2d(h, 2)
        
        h = self.conv2(h)
        h = self.activation(h)
        h = self.conv2_dropout(h)
        h = F.max_pool2d(h, 2)
        
        h = self.conv3(h)
        h = self.activation(h)
        h = self.conv3_dropout(h)
        h = F.max_pool2d(h, 2)
        
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = self.activation(h)
        h = self.fc1_dropout(h)
        output = self.fc2(h)
        return output.detach().cpu().numpy()

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

    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if isinstance(m, (BitLinear, BitConv2d)):
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
            if isinstance(m, (BitLinear, BitConv2d)):
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
            if isinstance(m, (BitLinear, BitConv2d)):
                Nbit_dict[name] = [m.Nbits, 0]
                if m.pbias is not None or m.bias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
        return Nbit_dict

    def set_Nbits(self, Nbit_dict):
        for name, m in self.named_modules():
            if isinstance(m, (BitLinear, BitConv2d)):
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
            if isinstance(m, (BitLinear, BitConv2d)):
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
            if isinstance(m, (BitLinear, BitConv2d)):
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