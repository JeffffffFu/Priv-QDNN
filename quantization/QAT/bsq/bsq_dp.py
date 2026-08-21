import copy
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import torch
import torch.optim as optim

from dp.dp_optimizer import get_dp_optimizer
from model.resnet_bsq import BitConv2d, BitLinear
from quantization.PTQ.qdrop.qdrop import quantize_model
from torch.utils.data import TensorDataset


def bsq_dp(target_model, train_set,args,label_key=None):

    quantize_model =target_model

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.to_bin()

    Nbit_dict = quantize_model.pruning(threshold=0.0, drop=True)
    quantize_model=quantize_model.to(args['device'])

    optimizer = get_dp_optimizer(lr=args['lr'], C_t=1.0, sigma=args['sigma'], batch_size=args['batch_size'],
                                 model=quantize_model)

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
            optimizer = get_dp_optimizer(lr=args['lr'], C_t=1.0, sigma=args['sigma'], batch_size=args['batch_size'],
                                         model=quantize_model)

            TP = quantize_model.total_param()
            TB = quantize_model.total_bit()
            Comp = (TP * 32) / TB
            print(' Compression rate after pruning [%d / %d]:  %.2f X' % (TP * 32, TB, Comp))

        train_with_dp(quantize_model, train_loader, optimizer, args['device'], label_key)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit=args['w_bit'])

    Nbit_dict = quantize_model.pruning(threshold=0.0)
    return quantize_model


def train_with_dp(model, train_loader, optimizer, device, label_key=None):
    model=model.to(device)
    correct = 0
    for data, target in tqdm(train_loader):
        if isinstance(target, dict):
            if label_key is None:
                raise ValueError("label_key must be specified for multi-label datasets.")
            target = target[label_key]
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid, (X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape) == 2:
                output = torch.squeeze(output, 0)
            loss = F.cross_entropy(output, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp()