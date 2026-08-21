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


def pact_vib(target_model,train_set,args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    quantized_model = PACTResnet(target_model, bit_width=args['w_bit'])

    quantized_model = quantized_model.to(args['device'])



    quantized_model.train()

    optimizer = optim.Adam(target_model.parameters(), lr=args['lr'], weight_decay=1e-4)

    for epoch in range(args['num_epochs']):

        train_with_IVB(quantized_model,train_loader,optimizer,args)
        acc=test_model_acc(quantized_model, train_loader, args['device'],label_key=None)
        print(f'epoch:{epoch}|training acc:{acc}')

    return quantized_model

def train_with_IVB(target_model,train_loader,optimizer,args):

    criterion = nn.CrossEntropyLoss()
    beta=0.0000001
    for data, target in tqdm(train_loader):
        data = data.to(args['device'])
        target = target.to(torch.int64).to(args['device'])

        optimizer.zero_grad()
        output = target_model(data)
        cross_loss = criterion(output, target)
        mu,std=target_model.get_mu_std(data)
        info_loss = - 0.5 * (1 + 2 * (std + 1e-7).log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        loss = cross_loss + beta * info_loss
        loss.backward()
        optimizer.step()
