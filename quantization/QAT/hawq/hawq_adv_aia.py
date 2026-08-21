
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
from .q_model.q_resnet import q_resnet18
from .quant_modules import *
from .resnet import QuantizedBasicBlock
import torch
import torch.optim as optim

from torch.utils.data import TensorDataset



def hawq_adv_aia(target_model, train_set, args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)
    if args['net_name']=='resnet18' or args['net_name']=='resnet18_dp':
        quant_model = q_resnet18(target_model.model,train_set,args)
    elif args['net_name']=='CNN_UTKFace' or args['net_name']=='CNN_UTKFace_dp' or args['net_name']=='CNN_UTKFace_adversarial':
        quant_model = q_cnn_utkface(target_model.model,train_set,label_key,args)
    elif args['net_name']=='CNN_MNIST':
        quant_model = q_cnn_mnist(target_model.model,train_set,label_key,args)
    elif args['net_name'] == 'mlp' or args['net_name']=='mlp_dp' or args['net_name']=='mlp_vib':
        quant_model = q_mlp(target_model.model,train_set,args)
    else:
        raise ValueError("hawq can not support this net")

    quant_model=quant_model.to(args['device'])



    quant_model.train()

    train_with_adversarial(quant_model, train_loader, args, 'black', label_key)


    return quant_model

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
      #  acc_u,acc_a=test_model(clf, model, test_loader, args, method, label_key)
     #   print(f"epoch:{epoch}| target model acc: {acc_u:.4f} | attack acc: {acc_a:.4f}")


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
