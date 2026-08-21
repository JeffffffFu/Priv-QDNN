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


def bsq_adv_aia(target_model, train_set,args,label_key=None):

    quantize_model =target_model

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.to_bin()

    Nbit_dict = quantize_model.pruning(threshold=0.0, drop=True)
    quantize_model=quantize_model.to(args['device'])


    # Initialize quantization
    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit=args['w_bit'])
    
    Nbit_dict = quantize_model.pruning(threshold=0.0, drop=True)
    print('########Model after pruning########')
    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            print(name)
            module.print_stat()

    TP = quantize_model.total_param()
    TB = quantize_model.total_bit()
    Comp = (TP * 32) / TB
    print(' Compression rate after pruning [%d / %d]:  %.2f X' % (TP * 32, TB, Comp))

    train_with_adversarial(quantize_model, train_loader, args, 'black', label_key)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit=args['w_bit'])

    Nbit_dict = quantize_model.pruning(threshold=0.0)
    return quantize_model


def train_with_adversarial(model,train_loader,args,method,label_key='None'):

    optimizer_remained = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    model.train()
    if method == 'black':
        clf = MLP2Layer(in_dim=2, out_dim=4, layer_list=[128, 64], device=args['device']).to(args['device'])

    else:
        clf = MLP2Layer(in_dim=64, out_dim=4, layer_list=[128, 64], device=args['device']).to(args['device'])
    optimizer_ascent = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    model_path = os.path.join(os.getcwd(), "pre_trained_model", f"AIA_attack_model_celebA_{method}.pth")
    clf.load_state_dict(torch.load(model_path, map_location=args['device']))


    for epoch in range(args['num_epochs']):
        refine_remained( model, train_loader, optimizer_remained, criterion, args,label_key)
        if epoch % 3==0:
            gradient_ascent(model,clf,train_loader, optimizer_ascent, criterion,args,method)


def gradient_ascent(target_model,clf, train_loader, optimizer, criterion, args,method):
    correct=0
    total = 0

    for data, target in train_loader:
        if isinstance(target, dict):
            target = target['sen_attr']
        data, target = data.to(args['device']), target.to(args['device'])
        optimizer.zero_grad()
        if method == 'black':
            output = target_model(data)

        else:
            output = target_model.get_last_hidden_layer(data)

        output=output.to(args['device'])
        pred_attack = clf(output)


        loss = -criterion(pred_attack, target)

        loss.backward()
        optimizer.step()

def refine_remained(model, train_loader, optimizer, criterion, args,label_key):
    correct=0
    total = 0
    for data, target in train_loader:
        if isinstance(target, dict):
            if label_key is None:
                raise ValueError("label_key must be specified for multi-label datasets.")
            target = target[label_key]

        data, target = data.to(args['device']), target.to(args['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
