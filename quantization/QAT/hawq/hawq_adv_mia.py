
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

from attack.utils import MLP2Layer, load_mia_attack_model
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



def hawq_adv_mia(target_model, set, args,label_key=None):

    train_set=set[0]
    shadow_m=set[1]

    if args['net_name']=='resnet18' or args['net_name']=='resnet18_dp' or args['net_name']=='resnet18_adversarial':
        quant_model = q_resnet18(target_model.model,train_set,args)
    elif args['net_name']=='CNN_UTKFace' or args['net_name']=='CNN_UTKFace_dp' or args['net_name']=='CNN_UTKFace_adversarial':
        quant_model = q_cnn_utkface(target_model.model,train_set,label_key,args)
    elif args['net_name']=='CNN_MNIST':
        quant_model = q_cnn_mnist(target_model.model,train_set,label_key,args)
    elif args['net_name'] in ('mlp', 'mlp_dp', 'mlp_vib', 'mlp_5layer', 'mlp_5layer_adversarial'):
        quant_model = q_mlp(target_model.model,train_set,args)
    else:
        raise ValueError("hawq can not support this net")

    quant_model=quant_model.to(args['device'])



    quant_model.train()

    train_with_adversarial_MIA(quant_model, train_set,shadow_m ,args,  label_key)

    return quant_model


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
        train_acc = model.test_model_acc(train_loader, label_key)
        # test_acc = model.test_model_acc(test_loader, label_key)

        print(f'epoch:{epoch} |  training acc: {train_acc}')


def train_mia_attack(attack_model, optimizer,train_data,target_model,lambda_param,member_set,non_member_set,args):
    number=len(member_set)
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
        prob_non_member  = attack_model(prob_input)
        non_member_loss += torch.log(1 - prob_non_member[0][0])

    total_loss = - (lambda_param / (2 * number)) * (member_loss + non_member_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


def train_target_model(model, train_loader, optimizer, criterion, lambda_param, attack_model, args):
    model.train()
    for data, target in train_loader:
        # Correctly handle data types
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

        # Get probability values for corresponding classes (batch form)
        prob_values = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Prepare attack model input (batch form)
        prob_input = prob_values.unsqueeze(1)  # [batch_size, 1]

        # Use attack model to predict membership probability (batch form)
        prob_member = attack_model(prob_input)  # [batch_size, 2]

        # Calculate privacy loss (batch form)
        # Get membership probability (second column)
        member_probs = prob_member[:, 1]  # [batch_size]

        # Calculate log(membership probability) in batch form
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

    # Ensure sampling number doesn't exceed available sample count
    actual_number = min(number, len(all_indices))
    if actual_number != number:
        print(
            f"Warning: Requested {number} samples but only {len(all_indices)} available. Using {actual_number} samples.")

    random_indices = random.sample(all_indices, actual_number)

    selected_samples = Subset(train_data, random_indices)

    return selected_samples