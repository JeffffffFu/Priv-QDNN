import copy
import os

import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import torch
import torch.optim as optim

from attack.utils import MLP2Layer, load_mia_attack_model
from dp.dp_optimizer import get_dp_optimizer
from model.resnet_bsq import BitConv2d, BitLinear
from quantization.PTQ.qdrop.qdrop import quantize_model
from torch.utils.data import TensorDataset



def bsq_adv_mia(target_model, set,args,label_key=None):

    quantize_model =target_model
    train_set=set[0]
    shadow_m=set[1]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args['batch_size'], shuffle=True)

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.to_bin()

    Nbit_dict = quantize_model.pruning(threshold=0.0, drop=True)
    quantize_model=quantize_model.to(args['device'])


    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit=args['w_bit'])
    
    Nbit_dict = quantize_model.pruning(threshold=0.0, drop=True)
    print('########Model after pruning########')
    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            print(name)
            module.print_stat()

    TP = quantize_model.total_param()
    TB = quantize_model.total_bit()
    Comp = (TP * 32) / TB
    print(' Compression rate after pruning [%d / %d]:  %.2f X' % (TP * 32, TB, Comp))

    train_with_adversarial_MIA(quantize_model, train_set, shadow_m, args, label_key='None')

    for name, module in quantize_model.named_modules():
        if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
            module.quant(maxbit=args['w_bit'])

    Nbit_dict = quantize_model.pruning(threshold=0.0)
    return quantize_model

def train_with_adversarial_MIA(model, target_m, shadow_m, args, label_key='None'):
    lambda_param = 6
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)

    number = int(0.2 * len(target_m))

    member_set = sample_target_samples(target_m, number, args['random'])
    non_member_set = sample_target_samples(shadow_m, number, args['random'])

    optimizer_target = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    attack_model = MLP2Layer(in_dim=1, out_dim=2, layer_list=[32, 16], device=torch.device(args['device']))
    load_mia_attack_model(attack_model, args['dataset_name'], args['device'])
    optimizer_attack = optim.Adam(attack_model.parameters(), lr=args['lr'], weight_decay=1e-4)

    for epoch in range(args['num_epochs']):
        train_mia_attack(attack_model, optimizer_attack, target_m, model, lambda_param, member_set, non_member_set,
                         args)
        train_target_model(model, train_loader, optimizer_target, criterion, lambda_param, attack_model, args)
        # train_acc = model.test_model_acc(train_loader, label_key)
        # test_acc = model.test_model_acc(test_loader, label_key)

        print(f'epoch:{epoch}')


def train_mia_attack(attack_model, optimizer, train_data, target_model, lambda_param, member_set, non_member_set, args):
    number = len(member_set)
    attack_model.train()
    member_loss = 0.0

    for i, idx in enumerate(member_set.indices):
        data, label = member_set.dataset[idx]
        posterior = target_model.predict_proba(data)

        prob_value = float(posterior[0][label])
        prob_input = torch.tensor([[prob_value]], dtype=torch.float32, device=args['device'])
        prob_member = attack_model(prob_input)
        member_loss += torch.log(prob_member[0][1])

    non_member_loss = 0.0

    for i, idx in enumerate(non_member_set.indices):
        data, label = non_member_set.dataset[idx]
        posterior = target_model.predict_proba(data)
        prob_value = float(posterior[0][label])
        prob_input = torch.tensor([[prob_value]], dtype=torch.float32, device=args['device'])
        prob_non_member = attack_model(prob_input)
        non_member_loss += torch.log(1 - prob_non_member[0][0])

    total_loss = - (lambda_param / (2 * number)) * (member_loss + non_member_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


def train_target_model(model, train_loader, optimizer, criterion, lambda_param, attack_model, args):
    model.train()
    for data, target in train_loader:
        # Handle dtypes correctly
        if isinstance(target, dict):
            if 'label_key' in args and args['label_key'] is not None:
                target = target[args['label_key']]
            else:
                raise ValueError("label_key must be specified for multi-label datasets.")

        data, target = data.to(args['device']), target.to(torch.int64).to(args['device'])
        optimizer.zero_grad()

        output = model(data)
        classification_loss = criterion(output, target)

        batch_size = output.size(0)

        # Get probabilities
        probs = torch.softmax(output, dim=1)

        # Probability of the corresponding class (batched)
        prob_values = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Build the attack-model input (batched)
        prob_input = prob_values.unsqueeze(1)  # [batch_size, 1]

        # Predict membership probability with the attack model (batched)
        prob_member = attack_model(prob_input)  # [batch_size, 2]

        # Compute the privacy loss (batched)
        # Membership probability (second column)
        member_probs = prob_member[:, 1]  # [batch_size]

        # Batched log(membership probability)
        privacy_loss = torch.log(member_probs + 1e-8).mean()

        total_loss = classification_loss + lambda_param * privacy_loss

        # Print detailed loss information for debugging

        total_loss.backward()
        optimizer.step()


import random
from torch.utils.data import DataLoader, Subset
def sample_target_samples(train_data, number, random_seed):
    if random_seed != 'None':
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    all_indices = list(range(len(train_data)))

    # Make sure the sample count does not exceed the available samples
    actual_number = min(number, len(all_indices))
    if actual_number != number:
        print(
            f"Warning: Requested {number} samples but only {len(all_indices)} available. Using {actual_number} samples.")

    random_indices = random.sample(all_indices, actual_number)

    selected_samples = Subset(train_data, random_indices)

    return selected_samples