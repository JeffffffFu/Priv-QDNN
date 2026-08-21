"""Pretrain CT membership attack model for adversarial MIA defense."""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from attack.utils import (
    MLP2Layer,
    baseline_prep_single_trial,
    get_mia_attack_model_path,
)


def _extract_ct_features(posteriors, sample_labels, membership_labels):
    features = []
    labels = []
    for posterior, label, membership in zip(posteriors, sample_labels, membership_labels):
        label_value = int(label.item()) if torch.is_tensor(label) else int(label)
        post = posterior
        if torch.is_tensor(post):
            post = post.detach().cpu().numpy()
        post = np.asarray(post)
        if post.ndim > 1:
            post = post[0]
        features.append([float(post[label_value])])
        labels.append(membership)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def pretrain_mia_attack_model(args):
    prep_args = dict(args)
    prep_args["net_name"] = args["shadow_net_name"]
    prep_args["eps"] = 0.0
    prep_args["observations"] = 1

    (
        p_shadow,
        _p_target,
        shadow_class,
        _target_class,
        shadow_sample_label,
        _target_sample_label,
    ) = baseline_prep_single_trial(prep_args, trial_idx=0)

    x_train, y_train = _extract_ct_features(p_shadow, shadow_sample_label, shadow_class)
    if len(x_train) == 0:
        raise RuntimeError(
            "No shadow CT features found. Train shadow model first under "
            f"save/MIA/{args['shadow_net_name']}/{args['dataset_name']}/"
            f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/"
        )

    device = torch.device(args["device"])
    attack_model = MLP2Layer(
        in_dim=1, out_dim=2, layer_list=[32, 16], device=device
    ).to(device)
    attack_model.criterion = nn.CrossEntropyLoss()
    attack_model.optimizer = optim.Adam(
        attack_model.parameters(), lr=args["lr"], weight_decay=1e-4
    )
    attack_model.train_epochs(
        x_train,
        y_train,
        x_train,
        y_train,
        epochs=args["num_epochs"],
        max_patient=10,
    )

    save_path = get_mia_attack_model_path(args["dataset_name"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(attack_model.state_dict(), save_path)
    print(f"Saved MIA attack model to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="location")
    parser.add_argument("--shadow_net_name", type=str, default="mlp_5layer")
    parser.add_argument("--attack_method", type=str, default="MIA")
    parser.add_argument("--quantization", type=str, default="PYT_PTQ")
    parser.add_argument("--a_bit", type=int, default=4)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda:0")
    return vars(parser.parse_args())


if __name__ == "__main__":
    pretrain_mia_attack_model(parse_args())
