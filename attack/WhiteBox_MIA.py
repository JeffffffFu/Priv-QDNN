"""
WhiteBox MIA: white-box membership inference attack.

This implementation extends confidence-based MIA with white-box-only signals:
  - P(true_class): softmax confidence (strong black-box signal)
  - Gradient L2 Norm: L2 norm of the last-layer weight gradient (white-box signal)

The two scalar features are fed into a small MLP to classify member/non-member.

Supported modes:
  - WhiteBoxMIA (shadow mode):
      train attack model on shadow features, test on target features
  - WhiteBoxMIA_Partial (partial mode):
      split target features into two halves for train/test
      no shadow model required

Training artifacts are reused from MIA outputs under `save/MIA/.../model.pth`.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
import torchvision.models as tv_models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from model.WhiteBoxAttackModel import WhiteBoxAttackModel
from model.ResNet import resnet18 as custom_resnet18
from attack.utils import defense_shadow_backbone, get_dataset_num_classes, tpr_at_fpr


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_num_classes(dataset_name):
    """Get number of classes by dataset name."""
    return get_dataset_num_classes(dataset_name)


def _get_eval_data(dataset_name):
    """
    Load dataset with deterministic transform only (ToTensor + Normalize).

    Differences from `get_data(augment=True)`:
      - remove random augmentations such as RandomHorizontalFlip / RandomCrop
      - keep Normalize (same as model training pipeline)
      - apply the same deterministic transform to both train and test sets

    This ensures stable pixels across runs during MIA evaluation and avoids
    random-augmentation mismatch with memorized training samples.
    """
    import torchvision.transforms as transforms
    from torchvision import datasets

    load_path = os.getcwd() + '/data'

    # Normalize parameters (aligned with load_data.py augment=True branch).
    NORM_PARAMS = {
        'cifar10':       {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'cifar100':      {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
        'cinic10':       {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
        'tinyimagenet':  {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
        'svhn':          {'mean': [0.4377, 0.4438, 0.4728], 'std': [0.198, 0.201, 0.197]},
        'stl10':         {'mean': [0.4377, 0.4438, 0.4728], 'std': [0.198, 0.201, 0.197]},
    }

    if dataset_name in NORM_PARAMS:
        p = NORM_PARAMS[dataset_name]
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=p['mean'], std=p['std']),
        ])
    else:
        # Fallback for datasets not listed above.
        from data.load_data import get_data
        print(f"[WhiteBox] Warning: no eval transform defined for {dataset_name}, "
              "falling back to get_data(augment=True)")
        return get_data(dataset_name)

    # Build dataset by name.
    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(root=load_path, train=True,
                                     transform=eval_transform, download=True)
        test_set = datasets.CIFAR10(root=load_path, train=False,
                                    transform=eval_transform, download=True)
    elif dataset_name == 'cifar100':
        train_set = datasets.CIFAR100(root=load_path, train=True,
                                      transform=eval_transform, download=True)
        test_set = datasets.CIFAR100(root=load_path, train=False,
                                     transform=eval_transform, download=True)
    elif dataset_name == 'svhn':
        train_set = datasets.SVHN(root=load_path, split='train',
                                  transform=eval_transform, download=True)
        test_set = datasets.SVHN(root=load_path, split='test',
                                 transform=eval_transform, download=True)
    elif dataset_name == 'stl10':
        train_set = datasets.STL10(root=load_path, split='train',
                                   transform=eval_transform, download=True)
        test_set = datasets.STL10(root=load_path, split='test',
                                  transform=eval_transform, download=True)
    elif dataset_name == 'cinic10':
        # CINIC-10 uses ImageFolder with deterministic transform (no random aug).
        from data.cinic10 import fetch_cinic10
        fetch_cinic10(load_path)
        train_set = datasets.ImageFolder(
            root=f'{load_path}/cinic-10-batches-py/train',
            transform=eval_transform
        )
        test_set = datasets.ImageFolder(
            root=f'{load_path}/cinic-10-batches-py/test',
            transform=eval_transform
        )
    elif dataset_name in ('texas', 'location'):
        from data.load_data import get_data
        return get_data(dataset_name, augment=False)
    else:
        # Fallback for remaining datasets (may include random augmentation).
        from data.load_data import get_data
        print(f"[WhiteBox] Warning: eval loader not implemented for {dataset_name}, "
              "falling back to get_data(augment=True)")
        return get_data(dataset_name)

    return train_set, test_set


def _load_models(args, trial_idx=0):
    """Load target and shadow models."""
    target_base_path = os.getcwd() + (
        f"/save/MIA/{args['net_name']}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
    )
    # In DP setting, target model is stored under eps subdirectory.
    if float(args.get('eps', 0)) != 0.0:
        target_base_path = os.path.join(target_base_path, str(args['eps']))

    # Shadow cache is under net_name (no eps subfolder). Defense variants may
    # reuse a vanilla resnet18 shadow when their own shadow was never trained.
    shadow_net_name = args['net_name']
    shadow_base_path = os.getcwd() + (
        f"/save/MIA/{shadow_net_name}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
    )

    device = args['device']
    target_path = f"{target_base_path}/target/{trial_idx}/model.pth"
    shadow_path = f"{shadow_base_path}/shadow/0/model.pth"

    shadow_backbone = defense_shadow_backbone(shadow_net_name)
    if not os.path.exists(shadow_path) and shadow_net_name != shadow_backbone:
        shadow_net_name = shadow_backbone
        shadow_base_path = os.getcwd() + (
            f"/save/MIA/{shadow_net_name}/{args['dataset_name']}/"
            f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
        )
        shadow_path = f"{shadow_base_path}/shadow/0/model.pth"
        print(
            f"[WhiteBox] Shadow model not under net_name={args['net_name']}; "
            f"loading shadow from {shadow_net_name}: {shadow_path}"
        )

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target model not found: {target_path}")
    if not os.path.exists(shadow_path):
        raise FileNotFoundError(
            f"Shadow model not found: {shadow_path}. "
            f"(net_name={args['net_name']}, tried resnet18 fallback if applicable)"
        )

    target_model = torch.load(target_path, map_location=device, weights_only=False)
    shadow_model = torch.load(shadow_path, map_location=device, weights_only=False)
    return target_model, shadow_model


def _unwrap_model(model):
    """Extract raw PyTorch model from DNN wrapper."""
    net = model.model if hasattr(model, 'model') else model
    return net


def _get_in_channels(dataset_name):
    """Get input channels by dataset."""
    if dataset_name in ('mnist', 'fmnist', 'ATnT'):
        return 1
    return 3


def _create_base_model(args):
    """
    Create a clean base model (without quantization wrappers),
    consistent with DNN.determine_net behavior.
    Return None when reconstruction is unsupported (e.g., BSQ architecture).
    """
    net_name = args['net_name']
    quant = args['quantization']
    num_classes = _get_num_classes(args['dataset_name'])
    in_channels = _get_in_channels(args['dataset_name'])

    if net_name in ('resnet18', 'resnet18_adversarial'):
        if quant in ('qdrop', 'hawq'):
            model = tv_models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        elif quant == 'bsq':
            return None  # BSQ quantization is architecture-embedded.
        else:
            # Original, PYT_PTQ, adaround, pact
            return custom_resnet18(num_classes=num_classes, in_channels=in_channels)
    if net_name == 'mobilenetv2':
        from model.MobileNetV2 import mobilenetv2

        if quant == 'bsq':
            return None
        if quant in ('qdrop', 'hawq'):
            model = tv_models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        return mobilenetv2(num_classes=num_classes, in_channels=in_channels)
    if net_name == 'simple_cnn':
        if quant == 'bsq':
            return None
        from model.DNN import Simple_CNN_Tanh

        return Simple_CNN_Tanh(num_classes=num_classes)
    if net_name == 'simple_cnn_dropout':
        from model.DNN import Simple_CNN_Tanh_dropout

        return Simple_CNN_Tanh_dropout(num_classes=num_classes)
    if net_name in ('mlp_5layer', 'mlp_5layer_dp', 'mlp_5layer_adversarial') or net_name.startswith('mlp_5layer_drop'):
        if quant == 'bsq':
            return None
        from data.tabular_datasets import TABULAR_META

        meta = TABULAR_META[args['dataset_name']]
        if net_name.startswith('mlp_5layer_drop010'):
            from model.mlp_5layer_tanh_dropout import MLP_5Layer_Tanh_Dropout
            return MLP_5Layer_Tanh_Dropout(meta['input_dim'], meta['num_classes'], dropout_rate=0.1)
        if net_name.startswith('mlp_5layer_drop015'):
            from model.mlp_5layer_tanh_dropout import MLP_5Layer_Tanh_Dropout
            return MLP_5Layer_Tanh_Dropout(meta['input_dim'], meta['num_classes'], dropout_rate=0.15)
        if net_name.startswith('mlp_5layer_drop020'):
            from model.mlp_5layer_tanh_dropout import MLP_5Layer_Tanh_Dropout
            return MLP_5Layer_Tanh_Dropout(meta['input_dim'], meta['num_classes'], dropout_rate=0.2)
        from model.mlp_5layer_tanh import MLP_5Layer_Tanh
        return MLP_5Layer_Tanh(meta['input_dim'], meta['num_classes'])
    # Reconstruction is currently unsupported for other architectures.
    return None


def _reconstruct_clean_model(quantized_net, args, device):
    """
    Reconstruct a clean model from quantized weights (without fake-quant wrappers),
    following the ML-Doctor style: state_dict + clean architecture.
    Fallback to quantized model when reconstruction fails.
    """
    if args['quantization'] == 'Original':
        # Full-precision model is already clean.
        return quantized_net

    clean_model = _create_base_model(args)
    if clean_model is None:
        print("[WhiteBox] Cannot create clean model for this architecture. Using quantized model.")
        return quantized_net

    clean_model = clean_model.to(device)

    q_sd = quantized_net.state_dict()
    c_sd = clean_model.state_dict()
    c_keys = set(c_sd.keys())

    # Strategy 1: exact key and shape match.
    filtered_sd = {k: v for k, v in q_sd.items()
                   if k in c_keys and v.shape == c_sd[k].shape}

    # Strategy 2: if coverage is low, strip '.module.' prefix.
    if len(filtered_sd) < len(c_keys) * 0.5:
        mapped_sd = {}
        for k, v in q_sd.items():
            clean_key = k.replace('.module.', '.')
            if clean_key in c_keys and v.shape == c_sd[clean_key].shape:
                mapped_sd[clean_key] = v
        if len(mapped_sd) > len(filtered_sd):
            filtered_sd = mapped_sd

    missing, unexpected = clean_model.load_state_dict(filtered_sd, strict=False)
    loaded = len(c_keys) - len(missing)
    total = len(c_keys)

    if loaded < total * 0.5:
        print(f"[WhiteBox] Warning: Only {loaded}/{total} keys loaded. Using quantized model.")
        return quantized_net

    print(f"[WhiteBox] Reconstructed clean model: {loaded}/{total} keys loaded"
          + (f", {len(missing)} missing" if missing else ""))
    return clean_model


def weights_init(m):
    """Weight initialization used in ML-Doctor."""
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Load saved posteriors from save/MIA -> P(true_class)
# ---------------------------------------------------------------------------

def _load_posteriors_and_labels(args, trial_idx=0):
    """
    Load saved posteriors and labels from save/MIA/,
    fully consistent with the confidence attack pipeline.

    Returns:
        P_target:      [N_target, num_cls] softmax posteriors
        target_class:  [N_target]  membership (1=member, 0=non-member)
        target_labels: [N_target]  true class labels
        P_shadow:      [N_shadow, num_cls] softmax posteriors
        shadow_class:  [N_shadow]
        shadow_labels: [N_shadow]
    """
    from attack.utils import baseline_prep_single_trial

    load_args = dict(args)
    load_args['attack_method'] = 'MIA'

    P_shadow, P_target, shadow_class, target_class, shadow_labels, target_labels = \
        baseline_prep_single_trial(load_args, trial_idx)

    P_target = np.array(P_target)
    P_shadow = np.array(P_shadow)
    if P_target.ndim == 3:
        P_target = P_target[:, 0, :]
    if P_shadow.ndim == 3:
        P_shadow = P_shadow[:, 0, :]

    target_class = np.array(target_class)
    shadow_class = np.array(shadow_class)
    target_labels = np.array([int(x) if hasattr(x, 'item') else int(x) for x in target_labels])
    shadow_labels = np.array([int(x) if hasattr(x, 'item') else int(x) for x in shadow_labels])

    return P_target, target_class, target_labels, P_shadow, shadow_class, shadow_labels


def _posteriors_to_confidence(posteriors, labels):
    """Extract P(true_class) from posteriors."""
    return np.array([posteriors[i, labels[i]] for i in range(len(labels))])


# ---------------------------------------------------------------------------
# Compute Gradient L2 Norm (white-box-only signal)
# ---------------------------------------------------------------------------

def _compute_grad_norms_for_subset(model, subset, device):
    """
    Compute L2 norm of the last-layer weight gradient for each sample in a subset.

    Args:
        model: clean PyTorch model (already moved to device)
        subset: torch Subset (with .indices and .dataset)
        device: compute device

    Returns:
        np.array of shape [len(subset)] with per-sample gradient L2 norms
    """
    net = model
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()

    named_params = list(net.named_parameters())
    target_param_name = None
    for name, _ in reversed(named_params):
        if name.endswith('.weight'):
            target_param_name = name
            break
    if target_param_name is None and named_params:
        # For some quant wrappers (e.g., BSQ), parameter names may not end with ".weight".
        for name, param in reversed(named_params):
            if param.requires_grad:
                target_param_name = name
                break
    if target_param_name is None:
        raise RuntimeError("No trainable parameter found for gradient norm computation.")

    grad_norms = []
    for i in tqdm(range(len(subset)), desc="Computing gradient norms"):
        data, label = subset[i]
        data = data.unsqueeze(0).to(device)
        label_t = torch.tensor([label], dtype=torch.long).to(device)

        with torch.enable_grad():
            logits = net(data)
            loss = criterion(logits, label_t)
            net.zero_grad()
            loss.backward()

            appended = False
            for name, param in net.named_parameters():
                if name == target_param_name:
                    grad_norms.append(param.grad.norm(2).item() if param.grad is not None else 0.0)
                    appended = True
                    break
            if not appended:
                # Fallback guard to avoid empty outputs on rare name-mapping failures.
                grad_norms.append(0.0)

    if len(grad_norms) != len(subset):
        print(f"[WhiteBox] Warning: grad_norms size {len(grad_norms)} != subset size {len(subset)}")
    return np.array(grad_norms, dtype=np.float32)


def _compute_grad_norms_matched(model, member_subset, nonmember_subset, device):
    """
    Compute gradient L2 norms and return in baseline_prep_single_trial order:
    [nonmem_0, mem_0, nonmem_1, mem_1, ...]

    This keeps exact alignment with posterior order from _load_posteriors_and_labels.
    """
    print("Computing gradient norms for members...")
    mem_norms = _compute_grad_norms_for_subset(model, member_subset, device)
    print("Computing gradient norms for non-members...")
    nonmem_norms = _compute_grad_norms_for_subset(model, nonmember_subset, device)

    n = min(len(mem_norms), len(nonmem_norms))
    interleaved = np.empty(2 * n)
    interleaved[0::2] = nonmem_norms[:n]
    interleaved[1::2] = mem_norms[:n]
    return interleaved


# ---------------------------------------------------------------------------
# Evaluation metrics — TPR@FPR via tpr_at_fpr (nearest ROC point, consistent with RMIA)
# ---------------------------------------------------------------------------

def _evaluate_mlp(features_train, labels_train, features_test, labels_test, device,
                  num_epochs=50):
    """
    Train and evaluate a small MLP attack model.

    Args:
        features_train: [N_train, 2] numpy array
        labels_train:   [N_train] numpy array (0/1)
        features_test:  [N_test, 2] numpy array
        labels_test:    [N_test] numpy array (0/1)

    Returns:
        dict with acc, auc, f1, tpr, best_epoch
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train)
    X_test = scaler.transform(features_test)

    train_X = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(labels_train, dtype=torch.long)
    test_X = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(labels_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    attack_model = WhiteBoxAttackModel(input_dim=2).to(device)
    torch.manual_seed(0)
    attack_model.apply(weights_init)
    optimizer = optim.Adam(attack_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_auc, best_f1, best_tpr, best_tpr_01 = 0.5, 0.5, 0.0, 0.0, 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        attack_model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = attack_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_y)
            train_correct += (out.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            attack_model.eval()
            with torch.no_grad():
                logits = attack_model(test_X.to(device))
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()

            test_y_np = test_y.numpy()
            acc = (preds == test_y_np).mean()
            try:
                auc = roc_auc_score(test_y_np, probs)
            except ValueError:
                auc = 0.5
            try:
                f1 = f1_score(test_y_np, preds)
            except ValueError:
                f1 = 0.0
            tpr = tpr_at_fpr(test_y_np, probs, 0.001)
            tpr_01 = tpr_at_fpr(test_y_np, probs, 0.01)

            train_acc = train_correct / max(train_total, 1)
            train_loss = train_loss_sum / max(train_total, 1)
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Train acc={train_acc:.4f} loss={train_loss:.4f} | "
                  f"Test acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}")

            if auc > best_auc:
                best_acc, best_auc, best_f1, best_tpr, best_tpr_01 = acc, auc, f1, tpr, tpr_01
                best_epoch = epoch + 1

    return {
        'acc': best_acc,
        'auc': best_auc,
        'f1': best_f1,
        'tpr': best_tpr,
        'tpr_at_0.01': best_tpr_01,
        'best_epoch': best_epoch,
    }


# ---------------------------------------------------------------------------
# Main entry: WhiteBox MIA (shadow mode)
# ---------------------------------------------------------------------------

def WhiteBox_MIA(args):
    """
    White-box MIA in shadow mode.
    P(true_class) is loaded from save/MIA posteriors (same as confidence attack),
    while Gradient L2 Norm is computed online and fused by a small MLP.
    """
    print(f"\n{'='*60}")
    print(f"WhiteBox MIA: {args['net_name']} - {args['dataset_name']} - "
          f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}")
    print(f"{'='*60}")

    device = args['device']

    result_base = os.getcwd() + (
        f"/result/WhiteBoxMIA/{args['net_name']}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
    )
    if float(args.get('eps', 0)) != 0.0:
        result_base = os.path.join(result_base, str(args['eps']))
    os.makedirs(result_base, exist_ok=True)

    all_trials_results = []

    for trial in range(args['trials']):
        print(f"\n--- Trial {trial + 1}/{args['trials']} ---")

        # 1) Load saved posteriors -> P(true_class) (same as confidence attack).
        P_target, target_class, target_labels, \
            P_shadow, shadow_class, shadow_labels = _load_posteriors_and_labels(args, trial)

        s_conf = _posteriors_to_confidence(P_shadow, shadow_labels)
        t_conf = _posteriors_to_confidence(P_target, target_labels)
        print(f"[WhiteBox] Loaded posteriors: shadow={len(s_conf)}, target={len(t_conf)}")

        # 2) Load models and reconstruct clean models for gradient computation.
        target_model_wrapped, shadow_model_wrapped = _load_models(args, trial)
        target_net = _reconstruct_clean_model(_unwrap_model(target_model_wrapped), args, device)
        shadow_net = _reconstruct_clean_model(_unwrap_model(shadow_model_wrapped), args, device)
        target_net.to(device).eval()
        shadow_net.to(device).eval()

        # 3) Reproduce sampled subsets with the same random seed as save_output.
        from data.prepare_data import split_dataset
        from training.utlis import sample_target_samples

        train_data, test_data = _get_eval_data(args['dataset_name'])
        target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

        # target: 20% of target_m as members, 20% of test_data as non-members
        t_number = int(0.2 * len(target_m))
        t_member_set = sample_target_samples(target_m, t_number, args['random'])
        t_nonmember_set = sample_target_samples(test_data, t_number, args['random'])

        # shadow: 20% of shadow_m as members, 20% of shadow_um as non-members
        s_number = int(0.2 * len(shadow_m))
        s_member_set = sample_target_samples(shadow_m, s_number, args['random'])
        s_nonmember_set = sample_target_samples(shadow_um, s_number, args['random'])

        print(f"[WhiteBox] Reproduced samples: target {t_number}+{t_number}, "
              f"shadow {s_number}+{s_number}")

        # 4) Compute Gradient L2 Norms aligned with posterior order.
        t_grad_norms = _compute_grad_norms_matched(
            target_net, t_member_set, t_nonmember_set, device)
        s_grad_norms = _compute_grad_norms_matched(
            shadow_net, s_member_set, s_nonmember_set, device)

        # 5) Build features: [P(true_class), grad_L2_norm].
        if len(s_conf) != len(s_grad_norms):
            n = min(len(s_conf), len(s_grad_norms))
            print(f"[WhiteBox] Warning: shadow length mismatch conf={len(s_conf)} grad={len(s_grad_norms)}; truncate to {n}")
            s_conf = s_conf[:n]
            s_grad_norms = s_grad_norms[:n]
            shadow_class = shadow_class[:n]
        if len(t_conf) != len(t_grad_norms):
            n = min(len(t_conf), len(t_grad_norms))
            print(f"[WhiteBox] Warning: target length mismatch conf={len(t_conf)} grad={len(t_grad_norms)}; truncate to {n}")
            t_conf = t_conf[:n]
            t_grad_norms = t_grad_norms[:n]
            target_class = target_class[:n]

        s_features = np.column_stack([s_conf, s_grad_norms])
        t_features = np.column_stack([t_conf, t_grad_norms])

        print(f"[WhiteBox] Features: shadow {s_features.shape}, target {t_features.shape}")

        # 6) Train MLP and evaluate.
        result = _evaluate_mlp(
            s_features, np.array(shadow_class), t_features, np.array(target_class), device)

        print(f"  Best: epoch={result['best_epoch']} acc={result['acc']:.4f} "
              f"auc={result['auc']:.4f} f1={result['f1']:.4f} "
              f"tpr@0.1%={result['tpr']:.4f} tpr@1%={result['tpr_at_0.01']:.4f}")

        all_trials_results.append({
            'trial': trial, 'acc': result['acc'], 'auc': result['auc'],
            'f1': result['f1'], 'tpr_at_0.001': result['tpr'], 'tpr_at_0.01': result['tpr_at_0.01'],
        })

    # 7) Aggregate and save metrics.
    final_metrics = {}
    if all_trials_results:
        for key, metric_name in [
            ('acc', 'WhiteBoxMIA_acc_mean'),
            ('auc', 'WhiteBoxMIA_auc_mean'),
            ('f1', 'WhiteBoxMIA_f1_mean'),
            ('tpr_at_0.001', 'WhiteBoxMIA_tpr_mean'),
            ('tpr_at_0.01', 'WhiteBoxMIA_tpr_at_fpr_01_mean'),
        ]:
            vals = [r[key] for r in all_trials_results]
            final_metrics[metric_name] = round(np.mean(vals), 4)
        if len(all_trials_results) > 1:
            final_metrics['WhiteBoxMIA_acc_std'] = round(np.std([r['acc'] for r in all_trials_results]), 4)
            final_metrics['WhiteBoxMIA_auc_std'] = round(np.std([r['auc'] for r in all_trials_results]), 4)
            final_metrics['WhiteBoxMIA_tpr_at_fpr_01_std'] = round(
                np.std([r['tpr_at_0.01'] for r in all_trials_results]), 4
            )

    csv_path = f"{result_base}/attack_performance.csv"
    pd.DataFrame(list(final_metrics.items()), columns=["metric", "value"]).to_csv(csv_path, index=False)
    print(f"\nWhiteBox MIA Final metrics: {final_metrics}")
    print(f"Results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Main entry: WhiteBox MIA (partial mode)
# ---------------------------------------------------------------------------

def WhiteBox_MIA_Partial(args):
    """
    White-box MIA in partial mode.
    Uses only target-model posteriors and target data.
    Split data into two halves: one for training the attack model and one for testing.
    P(true_class) comes from save/MIA posteriors; Gradient L2 Norm is computed online.
    """
    print(f"\n{'='*60}")
    print(f"WhiteBox MIA (Partial): {args['net_name']} - {args['dataset_name']} - "
          f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}")
    print(f"{'='*60}")

    device = args['device']

    result_base = os.getcwd() + (
        f"/result/WhiteBoxMIA_Partial/{args['net_name']}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
    )
    if float(args.get('eps', 0)) != 0.0:
        result_base = os.path.join(result_base, str(args['eps']))
    os.makedirs(result_base, exist_ok=True)

    all_trials_results = []

    for trial in range(args['trials']):
        print(f"\n--- Trial {trial + 1}/{args['trials']} ---")

        # 1) Load target posteriors.
        P_target, target_class, target_labels, _, _, _ = \
            _load_posteriors_and_labels(args, trial)

        t_conf = _posteriors_to_confidence(P_target, target_labels)
        print(f"[WhiteBox Partial] Loaded target posteriors: {len(t_conf)} samples")

        # 2) Load target model.
        base_path = os.getcwd() + (
            f"/save/MIA/{args['net_name']}/{args['dataset_name']}/"
            f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
        )
        if float(args.get('eps', 0)) != 0.0:
            base_path = os.path.join(base_path, str(args['eps']))
        target_path = f"{base_path}/target/{trial}/model.pth"
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target model not found: {target_path}")

        target_model_wrapped = torch.load(target_path, map_location=device, weights_only=False)
        target_net = _reconstruct_clean_model(_unwrap_model(target_model_wrapped), args, device)
        target_net.to(device).eval()

        # 3) Reproduce sampled subsets and compute gradients.
        from data.prepare_data import split_dataset
        from training.utlis import sample_target_samples

        train_data, test_data = _get_eval_data(args['dataset_name'])
        target_m, _, _ = split_dataset(train_data, args['random'])

        t_number = int(0.2 * len(target_m))
        t_member_set = sample_target_samples(target_m, t_number, args['random'])
        t_nonmember_set = sample_target_samples(test_data, t_number, args['random'])

        t_grad_norms = _compute_grad_norms_matched(
            target_net, t_member_set, t_nonmember_set, device)

        # 4) Build features.
        t_features = np.column_stack([t_conf, t_grad_norms])
        t_class = np.array(target_class)

        # 5) Split into two halves: train first half, test second half.
        n = len(t_features)
        half = n // 2
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)

        train_idx, test_idx = idx[:half], idx[half:]
        features_train, labels_train = t_features[train_idx], t_class[train_idx]
        features_test, labels_test = t_features[test_idx], t_class[test_idx]

        print(f"[WhiteBox Partial] Train: {len(train_idx)} "
              f"(mem={np.sum(labels_train==1)}, nonmem={np.sum(labels_train==0)})")
        print(f"[WhiteBox Partial] Test:  {len(test_idx)} "
              f"(mem={np.sum(labels_test==1)}, nonmem={np.sum(labels_test==0)})")

        # 6) Train MLP and evaluate.
        result = _evaluate_mlp(features_train, labels_train, features_test, labels_test, device)

        print(f"  Best: epoch={result['best_epoch']} acc={result['acc']:.4f} "
              f"auc={result['auc']:.4f} f1={result['f1']:.4f} "
              f"tpr@0.1%={result['tpr']:.4f} tpr@1%={result['tpr_at_0.01']:.4f}")

        all_trials_results.append({
            'trial': trial, 'acc': result['acc'], 'auc': result['auc'],
            'f1': result['f1'], 'tpr_at_0.001': result['tpr'], 'tpr_at_0.01': result['tpr_at_0.01'],
        })

    # 7) Aggregate and save metrics.
    final_metrics = {}
    if all_trials_results:
        for key, metric_name in [
            ('acc', 'WhiteBoxMIA_Partial_acc_mean'),
            ('auc', 'WhiteBoxMIA_Partial_auc_mean'),
            ('f1', 'WhiteBoxMIA_Partial_f1_mean'),
            ('tpr_at_0.001', 'WhiteBoxMIA_Partial_tpr_mean'),
            ('tpr_at_0.01', 'WhiteBoxMIA_Partial_tpr_at_fpr_01_mean'),
        ]:
            vals = [r[key] for r in all_trials_results]
            final_metrics[metric_name] = round(np.mean(vals), 4)
        if len(all_trials_results) > 1:
            final_metrics['WhiteBoxMIA_Partial_acc_std'] = round(np.std([r['acc'] for r in all_trials_results]), 4)
            final_metrics['WhiteBoxMIA_Partial_auc_std'] = round(np.std([r['auc'] for r in all_trials_results]), 4)
            final_metrics['WhiteBoxMIA_Partial_tpr_at_fpr_01_std'] = round(
                np.std([r['tpr_at_0.01'] for r in all_trials_results]), 4
            )

    csv_path = f"{result_base}/attack_performance.csv"
    pd.DataFrame(list(final_metrics.items()), columns=["metric", "value"]).to_csv(csv_path, index=False)
    print(f"\nWhiteBox MIA (Partial) Final metrics: {final_metrics}")
    print(f"Results saved to: {csv_path}")
