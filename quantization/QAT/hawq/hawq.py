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
from .q_model.q_CNN_UTKFace import q_cnn_utkface
from .q_model.q_CNN_MNIST import q_cnn_mnist
from .q_model.q_mlp import q_mlp
from .q_model.q_simple_cnn import q_simple_cnn
from .q_model.q_resnet import q_resnet18
from .quant_modules import *
from .resnet import QuantizedBasicBlock
import torch
import torch.optim as optim




def hawq(target_model, train_set, args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)
    if args['net_name'] in (
        'resnet18',
        'resnet18_drop010',
        'resnet18_drop015',
        'resnet18_drop020',
        'resnet18_drop010_full',
        'resnet18_drop015_full',
        'resnet18_drop020_full',
        'resnet18_adversarial',
    ):
        quant_model = q_resnet18(target_model.model,train_set,args)
    elif args['net_name'] == 'mobilenetv2':
        quant_model = q_simple_cnn(target_model.model, train_set, args)
    elif args['net_name']=='CNN_UTKFace' or args['net_name']=='CNN_UTKFace_dropout':
        quant_model = q_cnn_utkface(target_model.model,train_set,label_key,args)
    elif args['net_name']=='CNN_MNIST' or args['net_name']=='CNN_MNIST_dp' or args['net_name']=='CNN_MNIST_adversarial' or args['net_name']=='CNN_MNIST_memorization' or args['net_name']=='CNN_MNIST_HSIC':
        quant_model = q_cnn_mnist(target_model.model,train_set,label_key,args)
    elif args['net_name'] == 'mlp' or args['net_name']=='mlp_dropout' or args['net_name'] == 'mlp_5layer' or args['net_name'].startswith('mlp_5layer_drop'):
        quant_model = q_mlp(target_model.model,train_set,args)
    elif args['net_name'] in ('simple_cnn', 'SimpleCNN'):
        quant_model = q_simple_cnn(target_model.model, train_set, args)
    else:
        raise ValueError("hawq can not support this net")

    quant_model=quant_model.to(args['device'])
    optimizer = optim.Adam(quant_model.parameters(), lr=0.001, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    quant_model.train()

    for epoch in range(args['num_epochs']):
        losses = []
        correct=0.

        for data, target in tqdm(train_loader):
            data = data.to(args['device'])

            if isinstance(target, dict):
                if label_key is None:
                    raise ValueError("label_key must be specified for multi-label datasets.")
                target = target[label_key]
            target = target.to(torch.int64).to(args['device'])

            optimizer.zero_grad()
            output = quant_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

        print(correct / len(train_loader.dataset))

    return quant_model

