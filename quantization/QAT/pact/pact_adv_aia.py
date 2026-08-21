# import torchvision.models as models
from tqdm import tqdm

from attack.utils import MLP2Layer
from data.load_data import get_data
from data.prepare_data import split_dataset
from dp.compute_dp_sgd import apply_dp_sgd_analysis
from dp.dp_optimizer import get_dp_optimizer
from model.DNN import DNN
import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn

from quantization.QAT.pact.module import PACTResnet
import torch.optim as optim

from torch.utils.data import TensorDataset
import torch.nn.functional as F

def test_model_acc(model, test_loader, device,label_key=None):
    model.eval()

    model = model.to(device)
    correct = 0

    with torch.no_grad():

        for data, target in test_loader:
            data = data.to(device)
            # Handle multi-label or single-label
            if isinstance(target, dict):
                if label_key is None:
                    raise ValueError("label_key must be specified for multi-label datasets.")
                target = target[label_key]

            target = target.to(torch.int64).to(device)

            outputs = model(data).to(device)

            pred = outputs.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

        return correct / len(test_loader.dataset)


def pact_adv_aia(target_model,train_set,args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    quantized_model = PACTResnet(target_model, bit_width=args['w_bit'])

    quantized_model = quantized_model.to(args['device'])



    quantized_model.train()



    train_with_adversarial(quantized_model, train_loader, args, 'black', label_key)


    return quantized_model

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
