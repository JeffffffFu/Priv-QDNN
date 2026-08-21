# import torchvision.models as models
from tqdm import tqdm
import torch.optim as optim

from data.load_data import get_data
from data.prepare_data import split_dataset
from model.DNN import DNN
import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn

from quantization.QAT.pact.module import PACTResnet



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

def pact(target_model,train_set,args,label_key=None):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    # Use separate bit-widths for activation (PACT) vs weights.
    quantized_model = PACTResnet(
        target_model,
        act_bit_width=args.get('a_bit', args['w_bit']),
        weight_bit_width=args.get('w_bit', args['w_bit']),
    )
    


    quantized_model = quantized_model.to(args['device'])

    # QAT is usually a fine-tuning stage; a smaller LR is typically more stable.
    qat_lr = args.get('qat_lr', 1e-4)
    optimizer = optim.Adam(quantized_model.parameters(), lr=qat_lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    quantized_model.train()

    for epoch in range(args['num_epochs']):
        losses = []

        for data, target in tqdm(train_loader):
            data = data.to(args['device'])

            if isinstance(target, dict):
                if label_key is None:
                    raise ValueError("label_key must be specified for multi-label datasets.")
                target = target[label_key]
            target = target.to(torch.int64).to(args['device'])

            optimizer.zero_grad()
            output = quantized_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f' epoch:{epoch} | train acc:{round(test_model_acc(quantized_model,train_loader,args["device"],label_key), 4)}')

    return quantized_model

