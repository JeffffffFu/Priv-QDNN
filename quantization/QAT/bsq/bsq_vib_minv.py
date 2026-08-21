import copy
import os

import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import torch
import torch.optim as optim

from attack.utils import MLP2Layer
from dp.dp_optimizer import get_dp_optimizer
from model.resnet_bsq import BitConv2d, BitLinear
from quantization.PTQ.qdrop.qdrop import quantize_model
from torch.utils.data import TensorDataset


def bsq_vib(target_model, train_set,args,label_key=None):

    quantize_model =target_model

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.to_bin()

    Nbit_dict = quantize_model.pruning(threshold=0.0, drop=True)
    quantize_model=quantize_model.to(args['device'])

    optimizer = optim.Adam(quantize_model.parameters(), lr=args['lr'], weight_decay=1e-4)

    for epoch in range(args['num_epochs']):
        if epoch % 10 == 0:
            for name, module in quantize_model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    module.quant(maxbit=args['w_bit'])  # args.Nbits)
            Nbit_dict = quantize_model.pruning(threshold=0.0, drop=True)
            print('########Model after pruning########')
            for name, module in quantize_model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    print(name)
                    module.print_stat()
            ## Reset optimizer to fit changed variable dimension
            del optimizer
            optimizer = optim.Adam(quantize_model.parameters(), lr=args['lr'], weight_decay=1e-4)

            TP = quantize_model.total_param()
            TB = quantize_model.total_bit()
            Comp = (TP * 32) / TB
            print(' Compression rate after pruning [%d / %d]:  %.2f X' % (TP * 32, TB, Comp))

        train_with_IVB(quantize_model, train_loader, optimizer, args)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit=args['w_bit'])

    Nbit_dict = quantize_model.pruning(threshold=0.0)
    return quantize_model

def train_with_IVB(target_model, train_loader, optimizer, args):
    criterion = nn.CrossEntropyLoss()
    beta = 0.5
    for data, target in tqdm(train_loader):
        data = data.to(args['device'])
        target = target.to(torch.int64).to(args['device'])

        optimizer.zero_grad()
        output = target_model(data)
        cross_loss = criterion(output, target)
        mu, std = target_model.get_mu_std(data)
        info_loss = - 0.5 * (1 + 2 * (std + 1e-7).log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + beta * info_loss
        loss.backward()
        optimizer.step()
