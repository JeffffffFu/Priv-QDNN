import copy

import torch.ao.quantization as quantization
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from data.load_data import get_data
from quantization.PTQ.PYT_PTQ import utils
from quantization.PTQ.PYT_PTQ.PYT_PTQ import inplace_quantize_layers



def PYT_QAT(target_model, train_set, args,label_key=None):


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    quant_model=copy.deepcopy(target_model)

    inplace_quantize_layers(
        quant_model, args['w_bit'], args['device']
    )

    optimizer = optim.Adam(quant_model.parameters(), lr=0.001, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    quant_model.train()

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
            output = quant_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    inplace_quantize_layers(
        quant_model, args['w_bit'], args['device']
    )

    utils.enable_calibrate(quant_model)
    utils.calibrate(quant_model, train_loader,label_key, device=args['device'])
    utils.disable_calibrate(quant_model)

    return quant_model