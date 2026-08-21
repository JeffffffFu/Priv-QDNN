import os
import random

import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from sklearn import metrics

from model.DNN import DNN
from torch.utils.data import ConcatDataset, Subset
from sklearn.manifold import TSNE


def get_dataset_num_classes(dataset_name):
    """Return num_classes for attack feature construction (one-hot, SAMIA, etc.)."""
    return {
        'cifar10': 10,
        'cifar100': 100,
        'cinic10': 10,
        'mnist': 10,
        'fmnist': 10,
        'svhn': 10,
        'stl10': 10,
        'tinyimagenet': 200,
        'UTKFace': 2,
        'ATnT': 40,
        'texas': 100,
        'location': 8,
    }.get(dataset_name, 10)


def defense_shadow_backbone(net_name: str) -> str:
    """Map defense target net_name to the backbone shadow cache directory."""
    if net_name.startswith('resnet18_drop') or net_name in (
        'resnet18_dp',
        'resnet18_adversarial',
    ):
        return 'resnet18'
    if net_name == 'mlp_5layer_dp':
        return 'mlp_5layer'
    if net_name.startswith('mlp_5layer_drop'):
        return 'mlp_5layer'
    if net_name == 'mlp_5layer_adversarial':
        return 'mlp_5layer'
    return net_name


def get_mia_attack_model_path(dataset_name: str) -> str:
    """Return CT attack-model checkpoint path for adversarial MIA defense."""
    return os.path.join(
        os.getcwd(),
        "pre_trained_model",
        f"MIA_attack_model_{dataset_name}_CT.pth",
    )


def load_mia_attack_model(attack_model, dataset_name: str, device):
    """Load dataset-specific CT attack model, or random-init if missing."""
    model_path = get_mia_attack_model_path(dataset_name)
    if os.path.isfile(model_path):
        attack_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded MIA attack model from {model_path}")
    else:
        attack_model.apply(weights_init)
        print(f"MIA attack model not found at {model_path}, using random init")
    return attack_model.to(device)


def attack_feature_base(P, label_list,method):
    assert method in ["CDS", "CT",'DC','SC','DD','SD','ED',"CP", "DF", 'SM','test','T3','LO']

    attack_feature=[]
    if  method == "CT":
        for posterior,label in zip(P,label_list):

            attack_feature.append([posterior[label]])
    elif method == 'CP':
        for posterior in P:
            attack_feature.append(posterior)

    elif method == 'T3':
        for posterior in P:
            top_3 = sorted(posterior, reverse=True)[:3]
            attack_feature.append(top_3)
    elif method =='LO':
        for posterior,label in zip(P,label_list):
            # Normalize posterior_in
            max_index_in = np.argmax(posterior)
            posterior_new = np.zeros_like(posterior)
            posterior_new[max_index_in] = 1
            attack_feature.append(posterior_new)


    else:
        raise ValueError("this algorithm is not exist")

    ss = StandardScaler()
    attack_feature = ss.fit_transform(attack_feature)

    return attack_feature

# shadow just have member and non-member
def baseline_prep_single_trial(args, trial_idx, load_sensitivity=False):
    """trial.load_sensitivity=True  CompLeakNR SAMIA """
    P_model_target = []
    target_class = []
    P_model_shadow = []
    shadow_class = []
    shadow_sample_label=[]
    target_sample_label=[]
    P_target_sens = [] if load_sensitivity else None
    P_shadow_sens = [] if load_sensitivity else None

    # Target path:
    # - DP runs: include eps subfolder
    # - non-DP runs: no eps subfolder
    if args.get('eps', 0) != 0:
        target_base = os.getcwd() + (
            f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
            f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}"
        )
    else:
        target_base = os.getcwd() + (
            f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
            f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
        )

    # Shadow path is shared and does NOT include eps.
    shadow_net_name = args['net_name']
    shadow_base = os.getcwd() + (
        f"/save/{args['attack_method']}/{shadow_net_name}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
    )

    target_save_path = f"{target_base}/target/0"
    target_shadow_path = f"{shadow_base}/shadow/0"
    shadow_backbone = defense_shadow_backbone(shadow_net_name)
    if not os.path.isdir(target_shadow_path) and shadow_net_name != shadow_backbone:
        shadow_net_name = shadow_backbone
        shadow_base = os.getcwd() + (
            f"/save/{args['attack_method']}/{shadow_net_name}/{args['dataset_name']}/"
            f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
        )
        target_shadow_path = f"{shadow_base}/shadow/0"

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))

    # target data for current trial
    for i in range(target_num_subfolders):
        save_path = f"{target_base}/target/{trial_idx}/{i}"

        # non_member
        unseen_sample_original_model_posterior = torch.load(f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
        unseen_sample_label=torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

        P_model_target.append(unseen_sample_original_model_posterior)
        target_class.append(0)
        target_sample_label.append(unseen_sample_label)
        if load_sensitivity:
            sens_path = f"{save_path}/non_member_sample_sensitivity.pth"
            arr = np.array(unseen_sample_original_model_posterior)
            P_target_sens.append(torch.load(sens_path, weights_only=False) if os.path.exists(sens_path) else np.zeros_like(arr))

        # member
        member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
        member_sample_label=torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

        P_model_target.append(member_sample_original_model_posterior)
        target_class.append(1)
        target_sample_label.append(member_sample_label)
        if load_sensitivity:
            sens_path = f"{save_path}/member_sample_sensitivity.pth"
            arr = np.array(member_sample_original_model_posterior)
            P_target_sens.append(torch.load(sens_path, weights_only=False) if os.path.exists(sens_path) else np.zeros_like(arr))

    for t in range(args['observations']):
        for i in range(shadow_num_subfolders):
            save_path = f"{shadow_base}/shadow/{t}/{i}"

            # non_member
            unseen_sample_original_model_posterior = torch.load(
                f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
            unseen_sample_label = torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

            P_model_shadow.append(unseen_sample_original_model_posterior)
            shadow_class.append(0)
            shadow_sample_label.append(unseen_sample_label)
            if load_sensitivity:
                sens_path = f"{save_path}/non_member_sample_sensitivity.pth"
                arr = np.array(unseen_sample_original_model_posterior)
                P_shadow_sens.append(torch.load(sens_path, weights_only=False) if os.path.exists(sens_path) else np.zeros_like(arr))

            # member
            member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
            member_sample_label = torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

            P_model_shadow.append(member_sample_original_model_posterior)
            shadow_class.append(1)
            shadow_sample_label.append(member_sample_label)
            if load_sensitivity:
                sens_path = f"{save_path}/member_sample_sensitivity.pth"
                arr = np.array(member_sample_original_model_posterior)
                P_shadow_sens.append(torch.load(sens_path, weights_only=False) if os.path.exists(sens_path) else np.zeros_like(arr))

    P_model_shadow = np.array(P_model_shadow)
    P_model_target = np.array(P_model_target)
    if load_sensitivity and P_target_sens and P_shadow_sens:
        P_target_sens = np.array(P_target_sens)
        P_shadow_sens = np.array(P_shadow_sens)
        return P_model_shadow, P_model_target, shadow_class, target_class, shadow_sample_label, target_sample_label, P_shadow_sens, P_target_sens

    return P_model_shadow, P_model_target, shadow_class, target_class, shadow_sample_label, target_sample_label

def baseline_prep(args):

    P_model_target = []
    target_class = []
    P_model_shadow = []
    shadow_class = []
    shadow_sample_label=[]
    target_sample_label=[]

    target_save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/0"
    target_shadow_path = os.getcwd() + (
        f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/0"
    )

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))

    for t in tqdm(range(args['trials']), desc='preparing members'):

        for i in range(target_num_subfolders):
            save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/{t}/{i}"

            # non_member
            unseen_sample_original_model_posterior = torch.load(f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
            unseen_sample_label=torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

            P_model_target.append(unseen_sample_original_model_posterior)
            target_class.append(0)
            target_sample_label.append(unseen_sample_label)

            # member
            member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
            member_sample_label=torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

            P_model_target.append(member_sample_original_model_posterior)
            target_class.append(1)
            target_sample_label.append(member_sample_label)

    #shadow
    for t in tqdm(range(args['observations']), desc='preparing members'):
        for i in range(shadow_num_subfolders):
            save_path = os.getcwd() + (
                f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
                f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/{t}/{i}"
            )

            # non_member
            unseen_sample_original_model_posterior = torch.load(
                f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
            unseen_sample_label = torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

            P_model_shadow.append(unseen_sample_original_model_posterior)
            shadow_class.append(0)
            shadow_sample_label.append(unseen_sample_label)

            # member
            member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
            member_sample_label = torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

            P_model_shadow.append(member_sample_original_model_posterior)
            shadow_class.append(1)
            shadow_sample_label.append(member_sample_label)

    P_model_shadow = np.array(P_model_shadow)
    P_model_target = np.array(P_model_target)

    return P_model_shadow, P_model_target, shadow_class, target_class,shadow_sample_label,target_sample_label



def baseline_prep_single_trial_dp(args, trial_idx):
    """DPtrial"""
    P_model_target = []
    target_class = []
    P_model_shadow = []
    shadow_class = []
    shadow_sample_label=[]
    target_sample_label=[]

    target_save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/0"
    target_shadow_path = os.getcwd() + (
        f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/0"
    )

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))

    # target data for current trial
    for i in range(target_num_subfolders):
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/{trial_idx}/{i}"

        # non_member
        unseen_sample_original_model_posterior = torch.load(f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
        unseen_sample_label=torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

        P_model_target.append(unseen_sample_original_model_posterior)
        target_class.append(0)
        target_sample_label.append(unseen_sample_label)

        # member
        member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
        member_sample_label=torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

        P_model_target.append(member_sample_original_model_posterior)
        target_class.append(1)
        target_sample_label.append(member_sample_label)

    for t in range(args['observations']):
        for i in range(shadow_num_subfolders):
            save_path = os.getcwd() + (
                f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
                f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/{t}/{i}"
            )

            # non_member
            unseen_sample_original_model_posterior = torch.load(
                f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
            unseen_sample_label = torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

            P_model_shadow.append(unseen_sample_original_model_posterior)
            shadow_class.append(0)
            shadow_sample_label.append(unseen_sample_label)

            # member
            member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
            member_sample_label = torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

            P_model_shadow.append(member_sample_original_model_posterior)
            shadow_class.append(1)
            shadow_sample_label.append(member_sample_label)

    P_model_shadow = np.array(P_model_shadow)
    P_model_target = np.array(P_model_target)

    return P_model_shadow, P_model_target, shadow_class, target_class,shadow_sample_label,target_sample_label

def baseline_prep_dp(args):

    P_model_target = []
    target_class = []
    P_model_shadow = []
    shadow_class = []
    shadow_sample_label=[]
    target_sample_label=[]

    target_save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/0"
    target_shadow_path = os.getcwd() + (
        f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/0"
    )

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))

    for t in tqdm(range(args['trials']), desc='preparing members'):

        for i in range(target_num_subfolders):
            save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/{t}/{i}"

            # non_member
            unseen_sample_original_model_posterior = torch.load(f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
            unseen_sample_label=torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

            P_model_target.append(unseen_sample_original_model_posterior)
            target_class.append(0)
            target_sample_label.append(unseen_sample_label)

            # member
            member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
            member_sample_label=torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

            P_model_target.append(member_sample_original_model_posterior)
            target_class.append(1)
            target_sample_label.append(member_sample_label)

    #shadow
    for t in tqdm(range(args['observations']), desc='preparing members'):
        for i in range(shadow_num_subfolders):
            save_path = os.getcwd() + (
                f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/"
                f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/{t}/{i}"
            )

            # non_member
            unseen_sample_original_model_posterior = torch.load(
                f"{save_path}/non_member_sample_target_model_posterior.pth",weights_only=False)
            unseen_sample_label = torch.load(f"{save_path}/non_member_sample_label.pth",weights_only=False)

            P_model_shadow.append(unseen_sample_original_model_posterior)
            shadow_class.append(0)
            shadow_sample_label.append(unseen_sample_label)

            # member
            member_sample_original_model_posterior = torch.load(f"{save_path}/member_sample_target_model_posterior.pth",weights_only=False)
            member_sample_label = torch.load(f"{save_path}/member_sample_label.pth",weights_only=False)

            P_model_shadow.append(member_sample_original_model_posterior)
            shadow_class.append(1)
            shadow_sample_label.append(member_sample_label)

    P_model_shadow = np.array(P_model_shadow)
    P_model_target = np.array(P_model_target)

    return P_model_shadow, P_model_target, shadow_class, target_class,shadow_sample_label,target_sample_label


def baseline_prep_for_AIA(args,t):

    P_model_target = []
    class_target = []
    last_layer_target=[]
    P_model_shadow = []
    class_shadow = []
    last_layer_shadow=[]


    target_save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/0"
    target_shadow_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/0"

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))
  #  shadow_num_subfolders=int(shadow_num_subfolders*0.1)


    for i in range(target_num_subfolders):
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/{t}/{i}"
        model_posterior = torch.load(f"{save_path}/target_model_posterior.pth",weights_only=False)
        last_layer = torch.load(f"{save_path}/target_model_last_layer.pth",weights_only=False)
        target_class=torch.load(f"{save_path}/target_class.pth",weights_only=False)

        P_model_target.append(model_posterior)
        class_target.append(target_class.item())
        if isinstance(last_layer, torch.Tensor):
            last_layer_target.append(last_layer.detach().cpu().numpy())
        else:
            last_layer_target.append(last_layer)

    for i in range(shadow_num_subfolders):
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/{t}/{i}"

        model_posterior = torch.load(f"{save_path}/target_model_posterior.pth",weights_only=False)
        last_layer = torch.load(f"{save_path}/target_model_last_layer.pth",weights_only=False)
        target_class = torch.load(f"{save_path}/target_class.pth",weights_only=False)

        P_model_shadow.append(model_posterior)
        class_shadow.append(target_class.item())
        if isinstance(last_layer, torch.Tensor):
            last_layer_shadow.append(last_layer.detach().cpu().numpy())
        else:
            last_layer_shadow.append(last_layer)

    P_model_shadow = np.array(P_model_shadow)
    P_model_target = np.array(P_model_target)
    last_layer_target = np.array(last_layer_target)
    last_layer_shadow = np.array(last_layer_shadow)
    return P_model_shadow, P_model_target, last_layer_shadow, last_layer_target,class_shadow,class_target


def baseline_prep_for_AIA_dp(args,t):

    P_model_target = []
    class_target = []
    last_layer_target=[]
    P_model_shadow = []
    class_shadow = []
    last_layer_shadow=[]


    target_save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/0"
    target_shadow_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/shadow/0"

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))


    for i in range(target_num_subfolders):
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/target/{t}/{i}"
        model_posterior = torch.load(f"{save_path}/target_model_posterior.pth",weights_only=False)
        last_layer = torch.load(f"{save_path}/target_model_last_layer.pth",weights_only=False)
        target_class=torch.load(f"{save_path}/target_class.pth",weights_only=False)

        P_model_target.append(model_posterior)
        class_target.append(target_class.item())
        if isinstance(last_layer, torch.Tensor):
            last_layer_target.append(last_layer.detach().cpu().numpy())
        else:
            last_layer_target.append(last_layer)


    for i in range(shadow_num_subfolders):
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/shadow/{t}/{i}"

        model_posterior = torch.load(f"{save_path}/target_model_posterior.pth",weights_only=False)
        last_layer = torch.load(f"{save_path}/target_model_last_layer.pth",weights_only=False)
        target_class = torch.load(f"{save_path}/target_class.pth",weights_only=False)

        P_model_shadow.append(model_posterior)
        class_shadow.append(target_class.item())
        if isinstance(last_layer, torch.Tensor):
            last_layer_shadow.append(last_layer.detach().cpu().numpy())
        else:
            last_layer_shadow.append(last_layer)

    P_model_shadow = np.array(P_model_shadow)
    P_model_target = np.array(P_model_target)
    last_layer_target = np.array(last_layer_target)
    last_layer_shadow = np.array(last_layer_shadow)
    return P_model_shadow, P_model_target, last_layer_shadow, last_layer_target,class_shadow,class_target




# compute distance between two posterior
def baseline_prep_for_AIA_distance(args,t):

    P_model_target = []
    class_target = []
    last_layer_target=[]
    P_model_shadow = []
    class_shadow = []
    last_layer_shadow=[]


    target_save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/0"
    target_shadow_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/Original_32_32/target/0"

    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    shadow_num_subfolders=sum(os.path.isdir(os.path.join(target_shadow_path, name)) for name in os.listdir(target_shadow_path))
  #  shadow_num_subfolders=int(shadow_num_subfolders*0.1)


    for i in range(target_num_subfolders):
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/{t}/{i}"
        model_posterior = torch.load(f"{save_path}/target_model_posterior.pth",weights_only=False)
        last_layer = torch.load(f"{save_path}/target_model_last_layer.pth",weights_only=False)
        target_class=torch.load(f"{save_path}/target_class.pth",weights_only=False)

        P_model_target.append(model_posterior)
        class_target.append(target_class.item())
        if isinstance(last_layer, torch.Tensor):
            last_layer_target.append(last_layer.detach().cpu().numpy())
        else:
            last_layer_target.append(last_layer)


    for i in range(shadow_num_subfolders):
      #  save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/{t}/{i}"
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/Original_32_32/target/{t}/{i}"

        model_posterior = torch.load(f"{save_path}/target_model_posterior.pth",weights_only=False)
        last_layer = torch.load(f"{save_path}/target_model_last_layer.pth",weights_only=False)
        target_class = torch.load(f"{save_path}/target_class.pth",weights_only=False)

        P_model_shadow.append(model_posterior)
        class_shadow.append(target_class.item())
        if isinstance(last_layer, torch.Tensor):
            last_layer_shadow.append(last_layer.detach().cpu().numpy())
        else:
            last_layer_shadow.append(last_layer)

    P_model_shadow = np.array(P_model_shadow)
    P_model_target = np.array(P_model_target)
    last_layer_target = np.array(last_layer_target)
    last_layer_shadow = np.array(last_layer_shadow)
    return P_model_shadow, P_model_target, last_layer_shadow, last_layer_target,class_shadow,class_target



def baseline_prep_RMIA(args):

    target_model_list=[]
    shadow_model_list=[]


    for t in tqdm(range(args['trials']), desc='preparing members'):

        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/{t}"
        device=torch.load(f'{save_path}/device.pth')
        model = torch.load(f'{save_path}/model.pth',map_location=device, weights_only=False)
        target_model_list.append(model)
        torch.cuda.empty_cache()
    #shadow
    for t in tqdm(range(args['observations']), desc='preparing members'):

        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/{t}"
        device=torch.load(f'{save_path}/device.pth')
        model = torch.load(f'{save_path}/model.pth', map_location=device, weights_only=False)
        shadow_model_list.append(model)
        torch.cuda.empty_cache()

    return target_model_list, shadow_model_list



def adjust_ratio_samples(list_a, list_b, list_c, list_d,target_ratio=(2, 1, 2)):

    arr_a = np.array(list_a)
    arr_b = np.array(list_b)
    arr_c = np.array(list_c)
    arr_d = np.array(list_d)

    assert len(arr_a) == len(arr_b)==len(arr_c)==len(arr_d)

    unseen_idx = np.where(arr_a == 0)[0]
    forget_idx = np.where(arr_a == 1)[0]
    retain_idx = np.where(arr_a == 2)[0]

    ratio_unseen,ratio_forget,ratio_retain  = target_ratio

    min_count = min(
        len(forget_idx) // ratio_forget,
        len(unseen_idx) // ratio_unseen,
        len(retain_idx) // ratio_retain
    )

    target_forget = min_count * ratio_forget
    target_unseen = min_count * ratio_unseen
    target_retain = min_count * ratio_retain

    np.random.seed(0)

    selected_forget = np.random.choice(forget_idx, target_forget, replace=False)
    selected_unseen = np.random.choice(unseen_idx, target_unseen, replace=False)
    selected_retain = np.random.choice(retain_idx, target_retain, replace=False)


    selected_idx = np.concatenate([selected_forget, selected_unseen, selected_retain])
    selected_idx.sort()

    new_list_a = arr_a[selected_idx]
    new_list_b = arr_b[selected_idx]
    new_list_c = arr_c[selected_idx]
    new_list_d = arr_d[selected_idx]

    return new_list_a.tolist(), new_list_b, new_list_c,new_list_d


class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list, device):
        super(MLP2Layer, self).__init__()
        assert len(layer_list) == 2
        torch.manual_seed(3407)

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.criterion = None
        self.optimizer = None
        self.device = device
        
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def train_epochs(self, train_x, train_y, val_x, val_y, epochs, max_patient):
        pbar = tqdm(range(epochs), leave=True, desc=f"Attack Training")
        if val_x is None:
           val_x = train_x
           val_y = train_y
        opt_loss = 1e10
        patient = max_patient
        for i in pbar:
            self.train_one_epoch(train_x, train_y)
            train_loss, train_acc = self.loss_acc(train_x, train_y)
            val_loss, val_acc = self.loss_acc(val_x, val_y)

            pbar.set_postfix({'Loss': train_loss,
                              'Acc': train_acc,
                              'Val Loss': val_loss,
                              'Val Acc': val_acc})
            if opt_loss / 1.001 > val_loss:
                opt_loss = val_loss
                patient = max_patient
            else:
                patient = patient - 1

            if patient == 0:
                pbar.close()
                break

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)
        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def acc(self, X_target, y_target):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        acc = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        return acc

    def pred(self, X_target):
        '''Predict posteriors'''
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        return outputs_target.detach().numpy()




def tpr_at_fpr(y_true, y_score, fpr_th):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    ind = np.argmin(np.abs(fpr - fpr_th))
    tpr_res = max(tpr[fpr==fpr[ind]])
    return tpr_res


def tpr_at_fpr_roc(y_true, y_score, target_fpr):
    """TPR at a fixed FPR using sklearn ROC curve (legacy; prefer calculate_tpr_at_fpr)."""
    fpr_arr, tpr_arr, _ = metrics.roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr_arr, target_fpr, side='right') - 1
    idx = max(0, min(int(idx), len(tpr_arr) - 1))
    return float(tpr_arr[idx])


def calculate_tpr_at_fpr(y_true, y_score, target_fpr):
    """TPR @ target_fpr via non-member score percentile (all MIA attacks)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    members_scores = y_score[y_true == 1]
    non_members_scores = y_score[y_true == 0]
    percentile = 100 * (1 - target_fpr)
    threshold = np.percentile(non_members_scores, percentile)
    true_positives = np.sum(members_scores > threshold)
    return float(true_positives / len(members_scores))


def all_metrics(y_true,y_probs):
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.array(y_true)
    y_pred= np.array(y_pred)
    y_probs= np.array(y_probs[:,1])

    acc = metrics.accuracy_score(y_true,y_pred)

    auc_target = metrics.roc_auc_score(y_true, y_probs)
    tpr_001fpr = tpr_at_fpr(y_true, y_probs, 0.01)
    tpr_0005fpr = tpr_at_fpr(y_true, y_probs, 0.005)
    tpr_005fpr = tpr_at_fpr(y_true, y_probs, 0.05)
    all_metrics_value=[acc, auc_target,tpr_001fpr, tpr_0005fpr, tpr_005fpr]
    return all_metrics_value

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


def get_subset_for_attack(target_m, test_data, n, args):

    random.seed(args['random'])
    torch.manual_seed(args['random'])

    target_indices = random.sample(range(len(target_m)), n)
    target_samples = [target_m[i][0] for i in target_indices]
    target_subset_data = torch.stack(target_samples)

    test_indices = random.sample(range(len(test_data)), n)
    test_samples = [test_data[i][0] for i in test_indices]
    test_subset_data = torch.stack(test_samples)

    subset_data = torch.cat([target_subset_data, test_subset_data])

    labels_target = torch.ones(n, dtype=torch.long)
    labels_non_target = torch.zeros(n, dtype=torch.long)
    labels = torch.cat([labels_target, labels_non_target])

    return subset_data, labels


def find_best_threshold(predictions, true_labels):
    """
    Finds the best threshold to classify predictions (probabilities between 0 and 1)
    to match true labels (0 or 1), maximizing accuracy.

    Args:
        predictions (list or numpy.ndarray): List of predicted probabilities (0 to 1).
        true_labels (list or numpy.ndarray): List of true labels (0 or 1).

    Returns:
        float: The best threshold.
        float: The maximum accuracy achieved with that threshold.
    """

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    best_accuracy = 0.0
    best_threshold = 0.5  # Default threshold

    # Try a range of thresholds
    thresholds = np.arange(0.01, 1.00, 0.05)

    for threshold in thresholds:
        predicted_labels = (predictions >= threshold).astype(int)  # 1 if >= threshold, 0 otherwise
        accuracy = np.mean(predicted_labels == true_labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def t_SNE(A,random):
    A_array = np.array(A)

    # tsne = TSNE(n_components=1, perplexity=50,
    # early_exaggeration=12, random_state=random)
    tsne = TSNE(n_components=1, random_state=random)
    A_tsne = tsne.fit_transform(A_array)

    A_flat = A_tsne.flatten()
    return A_flat


def shuffle_labels(dataset):
    inputs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    random.shuffle(labels)

    shuffled_dataset = torch.utils.data.TensorDataset(
        torch.stack(inputs),
        torch.tensor(labels)
    )
    return shuffled_dataset