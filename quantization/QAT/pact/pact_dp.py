# import torchvision.models as models
from tqdm import tqdm

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


def pact_dp(target_model,train_set,args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    quantized_model = PACTResnet(target_model, bit_width=args['w_bit'])

    quantized_model = quantized_model.to(args['device'])



    quantized_model.train()

    optimizer = get_dp_optimizer(lr=args['lr'], C_t=1.0, sigma=args['sigma'], batch_size=args['batch_size'],
                                 model=target_model)

    for epoch in range(args['num_epochs']):


        train_with_dp(target_model, train_loader, optimizer, args['device'], label_key)
        train_loss, train_acc = validation(target_model, train_loader, args['device'], label_key)


        print(
            f'DP-epoches:{epoch} |'f' train_acc: ({train_acc:.2f}%)')

    return quantized_model


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

def validation(model, test_loader,device,label_key=None):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            if isinstance(target, dict):
                if label_key is None:
                    raise ValueError("label_key must be specified for multi-label datasets.")
                target = target[label_key]
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum')

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)
    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    return test_loss, test_acc