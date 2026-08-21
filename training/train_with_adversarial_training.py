from attack.utils import MLP2Layer, load_mia_attack_model
from data.load_data import get_data
from data.prepare_data import split_dataset, split_dataset4
import torch

from model.DNN import DNN
from quantization import get_PTQ_quantization
from opacus import PrivacyEngine
from tqdm import tqdm
import torch.optim as optim

from data.load_data import get_data
from data.prepare_data import split_dataset
from model.DNN import DNN
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import pandas as pd
import os

from training.utlis import lazy_split, save_output_AIA, save_output, sample_target_samples, save_output_RMIA, \
    save_output_MInv, visualize_tensor_as_image_and_save, test_model_acc, print_model_parameters_and_size


#FOR AIA
def model_training_with_adversarial(args):
    training_set_acc = []
    test_set_acc = []
    for t in range(args['trials']):
        print("dataset and net_name:", args['dataset_name'], args['net_name'], args['quantization'], args['w_bit'])

        train_data, test_data = get_data(args['dataset_name'])

        target_model = DNN(args).to(args['device'])

        # for AIA
        if args['dataset_name'] == 'UTKFace' or args['dataset_name'] == 'celebA':
            target_m, shadow_m = lazy_split(train_data, args['random'])

            train_loader = torch.utils.data.DataLoader(
                target_m, batch_size=args['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True)
            label_key = 'gender'

            train_with_adversarial_AIA(target_model,train_loader,test_loader,args,'black',label_key)

            if args['quantization'] != 'Original':
                PTQ_quantizer = get_PTQ_quantization(args['quantization'],args['net_name'])
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

        #Minv
        elif args['dataset_name']=='mnist' or args['dataset_name']=='fmnist':
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True)

            optimizer = optim.Adam(target_model.parameters(), lr=args['lr'], weight_decay=1e-4)
            for epoch in range(args['num_epochs']):

                if args['net_name']=='CNN_MNIST_adversarial':
                    train_with_IVB(target_model, train_loader, optimizer, args)
                elif  args['net_name']=='CNN_MNIST_HSIC':
                    train_with_HSIC(target_model, train_loader, optimizer, args)
                else:
                    print("can not support")
                training_acc = target_model.test_model_acc(train_loader)
                print(f'epoch: {epoch} | train_acc:{training_acc}')

            if args['quantization'] != 'Original':
                PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                    'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                    sub_set = train_data
                else:
                    sub_set = sample_target_samples(train_data, 4096, args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args)

            training_set_acc.append(target_model.test_model_acc(train_loader, label_key=None))
            test_set_acc.append(target_model.test_model_acc(test_loader, label_key=None))
            print(f'Round {t}, train_acc: {training_set_acc}, test_set_acc: {test_set_acc}')
            save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/"
            os.makedirs(save_path, exist_ok=True)
            torch.save(target_model, f'{save_path}/target_model.pkl')

        # for MIA
        else:
            target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

            train_loader = torch.utils.data.DataLoader(
                target_m, batch_size=args['batch_size'], shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args['batch_size'], shuffle=True)

            train_with_adversarial_MIA(target_model,target_m, shadow_m, test_data,args,label_key='None')
            if args['quantization'] != 'Original':
                PTQ_quantizer = get_PTQ_quantization(args['quantization'],args['net_name'])
                if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args[
                    'quantization'] == 'hawq' or args['quantization'] == 'bsq':
                    if args['net_name'] in ('resnet18_adversarial', 'mlp_5layer_adversarial'):
                        sub_set=[target_m,shadow_m]

                    else:
                        sub_set = target_m
                else:
                    sub_set = sample_target_samples(target_m, 4096, args['random'])
                target_model = PTQ_quantizer(target_model, sub_set, args)
            training_set_acc.append(target_model.test_model_acc(train_loader, label_key=None))
            test_set_acc.append(target_model.test_model_acc(test_loader, label_key=None))
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
    }
    print(metrics_dict)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/"

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save performance metrics
    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)

# just for celebA
def train_with_adversarial_AIA(model,train_loader,test_loader,args,method,label_key='None'):

    optimizer_remained = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    model.train()
    if method == 'black':
        clf = MLP2Layer(in_dim=2, out_dim=4, layer_list=[128, 64], device=args['device']).to(args['device'])

    else:
        clf = MLP2Layer(in_dim=64, out_dim=4, layer_list=[128, 64], device=args['device']).to(args['device'])
    optimizer_ascent = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)

    model_path = os.path.join(os.getcwd(), "pre_trained_model", f"AIA_attack_model_celebA_{method}.pth")
    clf.load_state_dict(torch.load(model_path, map_location=args['device']))

    for epoch in range(args['num_epochs']):
        refine_remained( model, train_loader, optimizer_remained, criterion, args,label_key)
        if epoch % 3==0:
            gradient_ascent(model,clf,train_loader, optimizer_ascent, criterion,args,method)
        acc_u,acc_a=test_model(clf, model, test_loader, args, method, label_key)
        print(f"epoch:{epoch}| target model acc: {acc_u:.4f} | attack acc: {acc_a:.4f}")



def gradient_ascent(target_model,clf, train_loader, optimizer, criterion, args,method):
    correct=0
    total = 0

    for data, target in train_loader:
        if isinstance(target, dict):
            target = target['sen_attr']
        data, target = data.to(args['device']), target.to(args['device'])
        optimizer.zero_grad()
        if method == 'black':
            output = target_model(data)

        else:
            output = target_model.get_last_hidden_layer(data)

        output=output.to(args['device'])
        pred_attack = clf(output)


        loss = -criterion(pred_attack, target)

        loss.backward()
        optimizer.step()

def refine_remained(model, train_loader, optimizer, criterion, args,label_key):
    correct=0
    total = 0
    for data, target in train_loader:
        if isinstance(target, dict):
            if label_key is None:
                raise ValueError("label_key must be specified for multi-label datasets.")
            target = target[label_key]

        data, target = data.to(args['device']), target.to(args['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test_model(clf,target_model,test_loader,args,method,label_key):
    correct_u=0
    correct_a=0
    total= 0
    for data, target in test_loader:
        if isinstance(target, dict):
            if label_key is None:
                raise ValueError("label_key must be specified for multi-label datasets.")
            target_u = target[label_key]
            target_a = target['sen_attr']
        data, target_u,target_a = data.to(args['device']), target_u.to(args['device']), target_a.to(args['device'])
        output_u = target_model(data)
        pred_u = output_u.argmax(dim=1)
        correct_u += (pred_u == target_u).sum().item()
        total += target_u.size(0)

        # for attack evalution
        if method == 'black':
            output_a = target_model(data)

        else:
            output_a = target_model.get_last_hidden_layer(data)
        pred_attack = clf(output_a)
        pred_a = pred_attack.argmax(dim=1)
        correct_a += (pred_a == target_a).sum().item()

    acc_u = correct_u / total if total > 0 else 0
    acc_a = correct_a / total if total > 0 else 0

    return  acc_u,acc_a




def train_with_adversarial_MIA(model,target_m, shadow_m, test_data,args,label_key='None'):
    lambda_param=6
    criterion = nn.CrossEntropyLoss()



    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    number = int(0.2 * len(target_m))

    member_set = sample_target_samples(target_m, number,args['random'])
    non_member_set = sample_target_samples(shadow_m, number,args['random'])


    optimizer_target= optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)


    attack_model = MLP2Layer(in_dim=1, out_dim=2, layer_list=[32, 16], device=torch.device(args['device']))
    load_mia_attack_model(attack_model, args['dataset_name'], args['device'])
    optimizer_attack = optim.Adam(attack_model.parameters(), lr=args['lr'], weight_decay=1e-4)

    for epoch in range(args['num_epochs']):
        train_mia_attack(attack_model, optimizer_attack,target_m,model,lambda_param,member_set,non_member_set,args)
        train_target_model(model, train_loader, optimizer_target, criterion, lambda_param,attack_model,args)
        train_acc=test_model_acc(model, train_loader, args['device'], label_key=None)
        test_acc=test_model_acc(model, test_loader, args['device'], label_key=None)

        print(f'epoch:{epoch} |  training acc: {train_acc} | testing acc:{test_acc}')


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

def train_with_IVB(target_model,train_loader,optimizer,args):

    criterion = nn.CrossEntropyLoss()
    beta=0.1
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



def train_with_HSIC(target_model,train_loader,optimizer,args):

    criterion = nn.CrossEntropyLoss()
    a1=2
    a2=20

    def _median_heuristic(x):
        with torch.no_grad():
            n = x.size(0)
            if n > 1024:
                idx = torch.randperm(n, device=x.device)[:1024]
                x = x[idx]
            d = torch.cdist(x, x, p=2)
            med = torch.median(d[d > 0])
            return (med.clamp(min=1e-6)) ** 2

    def _rbf(x, sigma2=None):
        if sigma2 is None:
            sigma2 = _median_heuristic(x)
        dist2 = torch.cdist(x, x, p=2) ** 2
        return torch.exp(-dist2 / (2.0 * sigma2))

    def _center(k):
        n = k.size(0)
        one = torch.full((n, n), 1.0 / n, device=k.device, dtype=k.dtype)
        return k - one @ k - k @ one + one @ k @ one

    def _hsic_unbiased(z, y):
        kz = _center(_rbf(z))
        ky = _center(_rbf(y))
        n = z.size(0)
        return torch.trace(kz @ ky) / ((n - 1) ** 2 + 1e-8)

    target_model.train()
    for data, target in tqdm(train_loader):
        data = data.to(args['device'])
        target = target.to(torch.int64).to(args['device'])

        optimizer.zero_grad()

        hiddens = None
        out = target_model(data)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            hiddens, logits = out
        else:
            logits = out

        cross_loss = criterion(logits, target)
        total_loss = cross_loss

        bs = data.size(0)
        h_data = data.view(bs, -1)
        num_classes = logits.size(1)
        h_target = torch.zeros(bs, num_classes, device=logits.device)
        h_target.scatter_(1, target.view(-1, 1), 1.0)

        if hiddens is None:
            if hasattr(target_model, 'get_mu_std'):
                mu, _ = target_model.get_mu_std(data)
                hiddens = [mu]
            elif hasattr(target_model, 'get_last_hidden_layer'):
                hiddens = [target_model.get_last_hidden_layer(data)]
            else:
                hiddens = [logits]

        for hidden in hiddens:
            z = hidden.view(bs, -1)
            z = (z - z.mean(dim=0, keepdim=True))
            z = z / (z.std(dim=0, keepdim=True) + 1e-6)

            hxz_l = _hsic_unbiased(z, h_data)
            hyz_l = _hsic_unbiased(z, h_target)

            temp_hsic = a1 * hxz_l - a2 * hyz_l
            total_loss = total_loss + temp_hsic

        total_loss.backward()
        optimizer.step()