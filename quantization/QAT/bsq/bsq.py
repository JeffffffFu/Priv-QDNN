import copy
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import torch
import torch.optim as optim

from model.resnet_bsq import BitConv2d, BitLinear
from quantization.PTQ.qdrop.qdrop import quantize_model


def bsq(target_model, train_set,args,label_key=None):

    quantize_model =target_model

    # Restored original BSQ defaults (no special-casing by net)
    threshold = 0.0
    prune_interval = 10
    drop = True
    if args['net_name']=='CNN_MNIST':
        args['num_epochs']=20

    if args['net_name']=='simple_cnn' :
    #and args['dataset_name']=='cinic10':
        #args['num_epochs']=10
        threshold = 25.0  # 25
        prune_interval = 4

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.to_bin()

    Nbit_dict = quantize_model.pruning(threshold=threshold, drop=drop)

    optimizer = optim.Adam(quantize_model.parameters(), lr=0.001, weight_decay=1e-4)



    print(f'xxxxx: {args["num_epochs"]}|{threshold}|{prune_interval}')
  
    for epoch in range(args['num_epochs']):
        if epoch % prune_interval == 0:
            for name, module in quantize_model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    module.quant(maxbit=args['w_bit'])  # args.Nbits)
            Nbit_dict = quantize_model.pruning(threshold=threshold, drop=drop)
            print('########Model after pruning########')
            for name, module in quantize_model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    print(name)
                    module.print_stat()
            ## Reset optimizer to fit changed variable dimension
            del optimizer
            optimizer = optim.Adam(quantize_model.parameters(), lr=0.001, weight_decay=1e-4)
            TP = quantize_model.total_param()
            TB = quantize_model.total_bit()
            Comp = (TP * 32) / TB
            print(' Compression rate after pruning [%d / %d]:  %.2f X' % (TP * 32, TB, Comp))

        quantize_model = train(quantize_model,train_set,optimizer,args,label_key)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit=args['w_bit'])

    Nbit_dict = quantize_model.pruning(threshold=threshold)
    return quantize_model


def train(quantize_model,train_set,optimizer,args,label_key=None):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)


    criterion = nn.CrossEntropyLoss()

    quantize_model.train()
    losses=[]
    for data, target in tqdm(train_loader):
        data = data.to(args['device'])

        if isinstance(target, dict):
            if label_key is None:
                raise ValueError("label_key must be specified for multi-label datasets.")
            target = target[label_key]
        target = target.to(torch.int64).to(args['device'])

        optimizer.zero_grad()
        output = quantize_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return quantize_model
