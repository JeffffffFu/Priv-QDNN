import numpy as np
from tqdm import tqdm

from data.load_data import get_data
from data.prepare_data import split_dataset, split_dataset4
import torch

from dp.compute_dp_sgd import apply_dp_sgd_analysis
from dp.dp_optimizer import get_dp_optimizer
from dp.get_max import get_steps, get_noise_multiplier
from model.DNN import DNN
# Lazy import to avoid circular imports
# from quantization import get_PTQ_quantization
from opacus import PrivacyEngine
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch.nn.functional as F

import pandas as pd
import os

from training.utlis import lazy_split, save_output_AIA, save_output, sample_target_samples, save_output_RMIA, \
    save_output_MInv, visualize_tensor_as_image_and_save, test_model_acc, print_model_parameters_and_size


def model_training_target_dp(args):
    training_set_acc = []
    test_set_acc = []
    for t in range(args['trials']):
        print("DP:dataset and net_name:", args['dataset_name'], args['net_name'], args['quantization'], args['w_bit'])

        train_data, test_data = get_data(args['dataset_name'])

        target_model = DNN(args)
        target_model.to(args['device'])

        # for AIA
        if args['dataset_name'] == 'UTKFace' or args['dataset_name'] == 'celebA':
            target_m, shadow_m = lazy_split(train_data, args['random'])

            train_loader = torch.utils.data.DataLoader(
                target_m, batch_size=args['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True)
            label_key = 'gender'


            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
            delta = 1 / len(train_data)
            if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                steps = 2 * args['num_epochs'] * len(train_loader)
            else:
                steps = args['num_epochs']* len(train_loader)
            epsilon = args['eps']
            sigma = get_noise_multiplier(epsilon, 1 / len(target_m), args['batch_size'] / len(target_m),
                                         steps=steps, alphas=orders)
            print(f'eps:{epsilon} | sigma:{sigma}')
            args['sigma']=sigma

            optimizer = get_dp_optimizer(lr=args['lr'], C_t=1.0, sigma=sigma, batch_size=args['batch_size'],
                                         model=target_model)

            for epoch in range(args['num_epochs']):
                epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size']/ len(target_m), sigma,
                                                            (epoch + 1) * len(train_loader), orders,
                                                            delta)

                train_with_dp(target_model, train_loader, optimizer, args['device'],label_key)
                train_loss, train_acc = validation(target_model, train_loader, args['device'],label_key)

                test_loss, test_accuracy = validation(target_model, test_loader, args['device'],label_key)

                print(
                    f' DP-epoches:{epoch}| 'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')

            if args['quantization'] != 'Original':
                from quantization import get_PTQ_quantization
                PTQ_quantizer = get_PTQ_quantization(args['quantization'],net_name=args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                    'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                    sub_set = target_m
                else:
                    sub_set = sample_target_samples(target_m, 4096, args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args, label_key)
            training_set_acc.append(target_model.test_model_acc(train_loader, label_key))
            test_set_acc.append(target_model.test_model_acc(test_loader, label_key))
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

            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
            delta = 1 / len(train_data)
            if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                steps = 2 * args['num_epochs'] * len(train_loader)
            else:
                steps = args['num_epochs'] * len(train_loader)
            epsilon = args['eps']
            sigma = get_noise_multiplier(epsilon, 1 / len(train_data), args['batch_size'] / len(train_data),
                                         steps=steps, alphas=orders)
            print(f'eps:{epsilon} | sigma:{sigma}')
            args['sigma'] = sigma

            optimizer = get_dp_optimizer(lr=args['lr'], C_t=1.0, sigma=sigma, batch_size=args['batch_size'],
                                         model=target_model)

            for epoch in range(args['num_epochs']):
                epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size'] / len(train_data), sigma,
                                                            (epoch + 1) * len(train_loader), orders,
                                                            delta)

                train_with_dp(target_model, train_loader, optimizer, args['device'])
                train_loss, train_acc = validation(target_model, train_loader, args['device'])

                test_loss, test_accuracy = validation(target_model, test_loader, args['device'])

                print(
                    f'epoches:{epoch},'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')

            if args['quantization'] != 'Original':
                # Lazy import to avoid circular imports
                from quantization import get_PTQ_quantization
                PTQ_quantizer = get_PTQ_quantization(args['quantization'], net_name=args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                    'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                    sub_set = train_data
                else:
                    sub_set = sample_target_samples(train_data, 4096, args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args)

            training_set_acc.append(target_model.test_model_acc(train_loader, label_key=None))
            test_set_acc.append(target_model.test_model_acc(test_loader, label_key=None))
            print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')
            save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/"
            os.makedirs(save_path, exist_ok=True)
            torch.save(target_model, f'{save_path}/target_model.pkl')

        # for MIA
        else:
            target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

            train_loader = torch.utils.data.DataLoader(
                target_m, batch_size=args['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True)

            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
            delta = 1 / len(train_data)
            if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                steps = 2 * args['num_epochs'] * len(train_loader)
            else:
                steps = args['num_epochs'] * len(train_loader)
            epsilon = args['eps']
            sigma = get_noise_multiplier(epsilon, 1 / len(target_m), args['batch_size'] / len(target_m),
                                         steps=steps, alphas=orders)
            print(f'eps:{epsilon} | sigma:{sigma}')
            args['sigma']=sigma

            optimizer = get_dp_optimizer(lr=args['lr'], C_t=1.0, sigma=sigma, batch_size=args['batch_size'],
                                         model=target_model)

            for epoch in range(args['num_epochs']):
                epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size']/ len(target_m), sigma,
                                                            (epoch + 1) * len(train_loader), orders,
                                                            delta)


                train_with_dp(target_model, train_loader, optimizer, args['device'])
                train_loss, train_acc = validation(target_model, train_loader, args['device'])

                test_loss, test_accuracy = validation(target_model, test_loader, args['device'])

                print(
                    f'epoches:{epoch},'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')

            if args['quantization'] != 'Original':
                # Lazy import to avoid circular imports
                from quantization import get_PTQ_quantization
                PTQ_quantizer = get_PTQ_quantization(args['quantization'],net_name=args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                    'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                    sub_set = target_m
                else:
                    sub_set = sample_target_samples(target_m, 4096, args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args)

            training_set_acc.append(train_acc)
            test_set_acc.append(test_accuracy)
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
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/"

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save performance metrics
    data_to_save.to_csv(f"{save_path}/model_performance_dp.csv", index=False)

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