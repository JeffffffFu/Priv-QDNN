
"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

import copy
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F

from data.load_data import get_data
from dp.dp_optimizer import get_dp_optimizer
from .q_model.q_CNN_UTKFace import q_cnn_utkface
from .q_model.q_CNN_MNIST import q_cnn_mnist
from .q_model.q_mlp import q_mlp
from .q_model.q_resnet import q_resnet18
from .quant_modules import *
from .resnet import QuantizedBasicBlock
import torch
import torch.optim as optim

from torch.utils.data import TensorDataset



def hawq_dp(target_model, train_set, args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)
    if args['net_name']=='resnet18' or args['net_name']=='resnet18_dp':
        quant_model = q_resnet18(target_model.model,train_set,args)
    elif args['net_name']=='CNN_UTKFace' or args['net_name']=='CNN_UTKFace_dp':
        quant_model = q_cnn_utkface(target_model.model,train_set,label_key,args)
    elif args['net_name']=='CNN_MNIST_dp':
        quant_model = q_cnn_mnist(target_model.model,train_set,label_key,args)
    elif args['net_name'] in ('mlp', 'mlp_dp', 'mlp_5layer', 'mlp_5layer_dp'):
        quant_model = q_mlp(target_model.model,train_set,args)
    else:
        raise ValueError("hawq can not support this net")

    quant_model=quant_model.to(args['device'])

    optimizer = get_dp_optimizer(lr=args['lr'], C_t=1.0, sigma=args['sigma'], batch_size=args['batch_size'],
                                 model=quant_model.model)

    criterion = nn.CrossEntropyLoss()

    quant_model.train()

    for epoch in range(args['num_epochs']):
        train_with_dp(quant_model, train_loader, optimizer, args['device'], label_key)

    return quant_model



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