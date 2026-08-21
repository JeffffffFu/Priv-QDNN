
"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

import copy
import os

import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F

from attack.utils import MLP2Layer
from data.load_data import get_data
from dp.dp_optimizer import get_dp_optimizer
from .q_model.q_CNN_UTKFace import q_cnn_utkface
from .q_model.q_CNN_MNIST import q_cnn_mnist
from .q_model.q_mlp import q_mlp
from .q_model.q_mlp_vib import q_mlp_vib
from .q_model.q_resnet import q_resnet18
from .quant_modules import *
from .resnet import QuantizedBasicBlock
import torch
import torch.optim as optim

from torch.utils.data import TensorDataset



def hawq_vib(target_model, train_set, args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)
    if args['net_name']=='resnet18' or args['net_name']=='resnet18_dp':
        quant_model = q_resnet18(target_model.model,train_set,args)
    elif args['net_name']=='CNN_UTKFace' or args['net_name']=='CNN_UTKFace_dp' or args['net_name']=='CNN_UTKFace_adversarial':
        quant_model = q_cnn_utkface(target_model.model,train_set,label_key,args)
    elif args['net_name']=='CNN_MNIST':
        quant_model = q_cnn_mnist(target_model.model,train_set,label_key,args)
    elif args['net_name'] == 'mlp' or args['net_name']=='mlp_dp':
        quant_model = q_mlp(target_model.model,train_set,args)
    elif args['net_name']=='mlp_vib':
        quant_model = q_mlp_vib(target_model.model,train_set,args)
    else:
        raise ValueError("hawq can not support this net")

    quant_model=quant_model.to(args['device'])

    quant_model.train()
    optimizer = optim.Adam(target_model.parameters(), lr=args['lr'], weight_decay=1e-4)

    for epoch in range(args['num_epochs']):
        train_with_IVB(quant_model, train_loader, optimizer, args)

    return quant_model


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
