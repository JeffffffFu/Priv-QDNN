import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import argparse
import torch.optim as optim
from tqdm import tqdm

from attack.utils import shuffle_labels
from data.load_data import get_data, get_data_random_sample
from dp.dp_optimizer import get_dp_optimizer
from dp.get_max import get_noise_multiplier
from model.DNN import DNN
from quantization import get_PTQ_quantization
from training.utlis import sample_target_samples, save_output_MInv, print_model_parameters_and_size, compute_expression

from torch.utils.data import TensorDataset
import torch.nn.functional as F

def MInv_memorization(args):
    print(f"{args['attack_method']}-{args['net_name']}-{args['dataset_name']}-----")
    save_path = os.getcwd() + f"/result/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
    os.makedirs(save_path, exist_ok=True)
    number_of_samples_list = [2000,3000, 4000]
    number_of_samples_list = [5000]

    results = []

    for N in number_of_samples_list:
        P_list = []
        C_list = []
        for t in range(args['trials']):
            P = model_training_MInv_random_sample(args, N,t)
            C = compute_expression(N, P)
            P_list.append(P)
            C_list.append(C)
        P_avg = np.mean(P_list)
        C_avg = np.mean(C_list)
        results.append({'N': N, 'P_avg': P_avg, 'C_avg': C_avg})
    print(results)
    data_to_save = pd.DataFrame(results)
    data_to_save.to_csv(f"{save_path}/memorization_capabilities.csv", index=False)

def model_training_MInv_random_sample(args,N,t):
    print("dataset and net_name and method:",args['dataset_name'],args['net_name'],args['quantization'])
    train_data, test_data = get_data_random_sample(N,t)
    target_model = DNN(args)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    target_model.train_model(train_loader,test_loader)

    if args['quantization'] != 'Original':
        PTQ_quantizer = get_PTQ_quantization(args['quantization'],args['net_name'])
        if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args['quantization'] == 'hawq':
            sub_set = train_data
        else:
            sub_set = sample_target_samples(train_data, len(train_data), args['random'])
        target_model = PTQ_quantizer(target_model, sub_set, args)

    training_acc=target_model.test_model_acc(train_loader)


    print(f'train_acc:{training_acc}')

    return training_acc

def compute_expression(N, p):
    if p==100.0:
        p=1
    if p == 0 or p == 1:
        entropy = 0
    else:
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return N * (1 - entropy)
