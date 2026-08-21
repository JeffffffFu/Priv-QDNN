import numpy as np

from data.load_data import get_data
from data.prepare_data import split_dataset, split_dataset4
import torch

from model.DNN import DNN
from quantization import get_PTQ_quantization
from opacus import PrivacyEngine

import pandas as pd
import os

from training.utlis import lazy_split, save_output_AIA, save_output, sample_target_samples, save_output_RMIA, \
    save_output_MInv, visualize_tensor_as_image_and_save, test_model_acc, print_model_parameters_and_size, \
    compute_expression


def model_training_target(args):
    training_set_acc=[]
    test_set_acc=[]

    train_data, test_data = get_data(args['dataset_name'])
    C=[]
    for t in range(args['trials']):
        print("trials:",t,"dataset and net_name:",args['dataset_name'],args['net_name'],args['quantization'],args['w_bit'])


        target_model = DNN(args)

        #for AIA
        if args['dataset_name']=='UTKFace' or args['dataset_name']=='celebA' or args['dataset_name']=='fairface':
            target_m, shadow_m = lazy_split (train_data, args['random'])

            train_loader = torch.utils.data.DataLoader(
                target_m, batch_size=args['batch_size'], shuffle=True,num_workers=4,pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True,num_workers=4,pin_memory=True)
            label_key='gender'
            target_model.train_model(train_loader, test_loader,label_key)
            if args['quantization'] != 'Original':
                PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or  args['quantization'] == 'hawq' or  args['quantization'] == 'bsq':
                    sub_set=target_m
                else:
                    sub_set=sample_target_samples(target_m,4096,args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args,label_key)
            training_set_acc.append(target_model.test_model_acc(train_loader,label_key))
            test_set_acc.append(target_model.test_model_acc(test_loader,label_key))
            print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')

            # AIA doesn't have shadow model, gets attack features through target model. No need to train shadow model
            save_output_AIA('target', args, target_model, test_data, t)
            save_output_AIA('shadow', args, target_model, shadow_m, t)

        #For Minv
        elif args['dataset_name']=='mnist' or args['dataset_name']=='fmnist':
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True)

            target_model.train_model(train_loader, test_loader)
            if args['quantization'] != 'Original':
                PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                    'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                    sub_set = train_data
                else:
                    sub_set = sample_target_samples(train_data, 4096, args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args)
            training_set_acc.append(target_model.test_model_acc(train_loader, label_key=None))
            C.append(compute_expression(len(train_data), target_model.test_model_acc(train_loader, label_key=None)))
            test_set_acc.append(target_model.test_model_acc(test_loader, label_key=None))
            print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')
            save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/"
            os.makedirs(save_path, exist_ok=True)
            torch.save(target_model, f'{save_path}/target_model.pkl')

        #for MIA
        else:
            target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

            train_loader = torch.utils.data.DataLoader(
                target_m, batch_size=args['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True)

            target_model.train_model(train_loader, test_loader)
            if args['quantization'] != 'Original':
                PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or  args['quantization'] == 'hawq' or  args['quantization'] == 'bsq':
                    sub_set = target_m
                else:
                    sub_set = sample_target_samples(target_m, 4096, args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args)
            training_set_acc.append(target_model.test_model_acc(train_loader,label_key=None))
            test_set_acc.append(target_model.test_model_acc(test_loader,label_key=None))
            print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')
            save_output('target', args, target_model, target_m, test_data, t)


    metrics_dict = {
        "training_set_acc_mean": round(np.mean(training_set_acc), 4),
        "training_set_acc_std": round(np.std(training_set_acc), 4),
        "training_set_acc_var": round(np.var(training_set_acc), 4),
        "training_set_acc_min": round(np.min(training_set_acc), 4),
        "training_set_acc_max": round(np.max(training_set_acc), 4),
        "training_set_acc_range": round(np.max(training_set_acc) - np.min(training_set_acc), 4),
        "test_set_acc_mean": round(np.mean(test_set_acc), 4),
        "test_set_acc_std": round(np.std(test_set_acc), 4),
        "test_set_acc_var": round(np.var(test_set_acc), 4),
        "test_set_acc_min": round(np.min(test_set_acc), 4),
        "test_set_acc_max": round(np.max(test_set_acc), 4),
        "test_set_acc_range": round(np.max(test_set_acc) - np.min(test_set_acc), 4),
        'C':C
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/"

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save performance metrics
    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)

    # Save model parameters
  #  model_save_path = os.path.join(save_path, "model_parameters.pth")
  #  torch.save(target_model.model.state_dict(), model_save_path)


def model_training_shadow(args):
    training_set_acc=[]
    test_set_acc=[]
    for t in range(args['observations']):
        print("dataset and net_name:",args['dataset_name'],args['net_name'],args['quantization'],args['w_bit'])

        train_data, test_data = get_data(args['dataset_name'])
        target_model = DNN(args)


        target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

        train_loader = torch.utils.data.DataLoader(
            shadow_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            shadow_um, batch_size=args['batch_size'], shuffle=True)

        target_model.train_model(train_loader, test_loader)
        if args['quantization'] != 'Original':
            PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
            if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args['quantization'] == 'hawq' or  args['quantization'] == 'bsq':
                sub_set = shadow_m
            else:
                sub_set = sample_target_samples(shadow_m, 4096, args['random'])
            target_model = PTQ_quantizer(target_model, sub_set, args)

        training_set_acc.append(target_model.test_model_acc(train_loader))
        test_set_acc.append(target_model.test_model_acc(test_loader))
        save_output('shadow', args, target_model, shadow_m, shadow_um, t)

        print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')

    metrics_dict = {
        "training_set_acc_mean": round(np.mean(training_set_acc), 4),
        "training_set_acc_std": round(np.std(training_set_acc), 4),
        "training_set_acc_var": round(np.var(training_set_acc), 4),
        "training_set_acc_min": round(np.min(training_set_acc), 4),
        "training_set_acc_max": round(np.max(training_set_acc), 4),
        "training_set_acc_range": round(np.max(training_set_acc) - np.min(training_set_acc), 4),
        "test_set_acc_mean": round(np.mean(test_set_acc), 4),
        "test_set_acc_std": round(np.std(test_set_acc), 4),
        "test_set_acc_var": round(np.var(test_set_acc), 4),
        "test_set_acc_min": round(np.min(test_set_acc), 4),
        "test_set_acc_max": round(np.max(test_set_acc), 4),
        "test_set_acc_range": round(np.max(test_set_acc) - np.min(test_set_acc), 4),
        "C":round(np.mean(training_set_acc), 4),
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/"
    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)



def model_training_target_RMIA(args):
    training_set_acc=[]
    test_set_acc=[]
    for t in range(args['trials']):
        print("dataset and net_name:",args['dataset_name'],args['net_name'],args['quantization'],args['w_bit'])

        train_data, test_data = get_data(args['dataset_name'])
        target_model = DNN(args)


        target_m, shadow_m = split_dataset4(train_data, args['random'])

        train_loader = torch.utils.data.DataLoader(
            target_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args['batch_size'], shuffle=True)

        target_model.train_model(train_loader, test_loader)

        if args['quantization'] != 'Original':
            PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
            if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args['quantization'] == 'hawq' or  args['quantization'] == 'bsq':
                sub_set = target_m
            else:
                sub_set = sample_target_samples(target_m, 4096, args['random'])
            target_model = PTQ_quantizer(target_model, sub_set, args)

        training_set_acc.append(target_model.test_model_acc(train_loader))
        test_set_acc.append(target_model.test_model_acc(test_loader))
        save_output_RMIA('target', args, target_model, t)

        print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')

    metrics_dict = {
        "training_set_acc_mean": round(np.mean(training_set_acc), 4),
        "training_set_acc_std": round(np.std(training_set_acc), 4),
        "training_set_acc_var": round(np.var(training_set_acc), 4),
        "training_set_acc_min": round(np.min(training_set_acc), 4),
        "training_set_acc_max": round(np.max(training_set_acc), 4),
        "training_set_acc_range": round(np.max(training_set_acc) - np.min(training_set_acc), 4),
        "test_set_acc_mean": round(np.mean(test_set_acc), 4),
        "test_set_acc_std": round(np.std(test_set_acc), 4),
        "test_set_acc_var": round(np.var(test_set_acc), 4),
        "test_set_acc_min": round(np.min(test_set_acc), 4),
        "test_set_acc_max": round(np.max(test_set_acc), 4),
        "test_set_acc_range": round(np.max(test_set_acc) - np.min(test_set_acc), 4),
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])


    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/"
    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)

def model_training_shadow_RMIA(args):
    training_set_acc=[]
    test_set_acc=[]
    for t in range(args['observations']):
        print("dataset and net_name:",args['dataset_name'],args['net_name'],args['quantization'],args['w_bit'])

        train_data, test_data = get_data(args['dataset_name'])
        target_model = DNN(args)

        target_m, shadow_m = split_dataset4(train_data, args['random'])
        shadow_m = sample_target_samples(shadow_m, int(len(shadow_m)/2),'None')  #half samples to train

        train_loader = torch.utils.data.DataLoader(
            shadow_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args['batch_size'], shuffle=True)

        target_model.train_model(train_loader, test_loader)

        if args['quantization'] != 'Original':
            PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
            if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args['quantization'] == 'hawq':
                sub_set = shadow_m
            else:
                sub_set = sample_target_samples(shadow_m, 4096, args['random'])
            target_model = PTQ_quantizer(target_model, sub_set, args)

        training_set_acc.append(target_model.test_model_acc(train_loader))
        test_set_acc.append(target_model.test_model_acc(test_loader))
        save_output_RMIA('shadow', args, target_model, t)

        print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')

    metrics_dict = {
        "training_set_acc_mean": round(np.mean(training_set_acc), 4),
        "training_set_acc_std": round(np.std(training_set_acc), 4),
        "training_set_acc_var": round(np.var(training_set_acc), 4),
        "training_set_acc_min": round(np.min(training_set_acc), 4),
        "training_set_acc_max": round(np.max(training_set_acc), 4),
        "training_set_acc_range": round(np.max(training_set_acc) - np.min(training_set_acc), 4),
        "test_set_acc_mean": round(np.mean(test_set_acc), 4),
        "test_set_acc_std": round(np.std(test_set_acc), 4),
        "test_set_acc_var": round(np.var(test_set_acc), 4),
        "test_set_acc_min": round(np.min(test_set_acc), 4),
        "test_set_acc_max": round(np.max(test_set_acc), 4),
        "test_set_acc_range": round(np.max(test_set_acc) - np.min(test_set_acc), 4),
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])


    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/"
    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)




