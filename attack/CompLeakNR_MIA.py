"""
CompLeakNR: six black-box MIA methods adapted from CompLeak for Priv-QDNN.

Methods:
1. Confidence-based (Shokri) - confidence threshold.
2. Entropy-based (Salem) - entropy threshold.
3. Modified entropy (LiRA/Enhanced) - modified entropy threshold.
4. Train-based method 1 - LR/RF with posterior probabilities.
5. Train-based method 2 - LR/RF with posterior + label.
6. SAMIA - neural-network-based attack (posterior + sensitivity).

Before use, run training first: --training both --attack_method MIA
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from attack.utils import baseline_prep_single_trial, get_dataset_num_classes
from attack.attack_model import LR, RF
from attack.attack_utils import (
    ConstructFeature, Attack, ThresholdAttacker,
)
from attack.utils import calculate_tpr_at_fpr


def _get_num_classes(dataset_name):
    """Get number of classes by dataset name."""
    return get_dataset_num_classes(dataset_name)


def _predict_target_sensitivity(model, data_loader, device, num_cls, m=10, epsilon=1e-3):
    """
    Compute target-class sensitivity for model outputs (used by SAMIA).
    Compatible with Priv-QDNN style DNN wrappers.
    """
    net = model.model if hasattr(model, 'model') else model
    net = net.to(device)
    net.eval()
    predict_list, sensitivity_list, target_list = [], [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            outputs = net(inputs)
            predicts = F.softmax(outputs, dim=-1)
            predict_list.append(predicts.detach().cpu())
            target_list.append(targets)
            if len(inputs.size()) == 4:
                x = inputs.repeat((m, 1, 1, 1))
            elif len(inputs.size()) == 3:
                x = inputs.repeat((m, 1, 1))
            else:
                x = inputs.repeat((m, 1))
            u = torch.randn_like(x, device=device)
            evaluation_points = x + epsilon * u
            new_predicts = F.softmax(net(evaluation_points), dim=-1)
            diff = torch.abs(new_predicts - predicts.repeat((m, 1)))
            diff = diff.view(m, -1, num_cls)
            sensitivity = diff.mean(dim=0) / epsilon
            sensitivity_list.append(sensitivity.detach().cpu())
    targets = torch.cat(target_list, dim=0)
    predicts = torch.cat(predict_list, dim=0)
    sensitivities = torch.cat(sensitivity_list, dim=0)
    return predicts, targets, sensitivities


def _load_models_for_attack(args, trial_idx=0):
    """Load target and shadow models (used by SAMIA and RMIA)."""
    base_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}"
    device = args['device'] if isinstance(args['device'], str) else args['device']
    target_path = f"{base_path}/target/{trial_idx}/model.pth"
    shadow_path = f"{base_path}/shadow/0/model.pth"
    if not os.path.exists(target_path) or not os.path.exists(shadow_path):
        return None, None
    target_model = torch.load(target_path, map_location=device, weights_only=False)
    shadow_model = torch.load(shadow_path, map_location=device, weights_only=False)
    return target_model, shadow_model


def CompLeakNR_MIA(args):
    """Run six CompLeakNR black-box MIA attacks, or confidence-only mode."""
    confidence_only = args.get('compleaknr_attacks', 'all') == 'confidence'
    print(f"CompLeakNR-{args['net_name']}-{args['dataset_name']}-----")
    if confidence_only:
        print("CompLeakNR mode: confidence attack only (skip Entropy / RF / SAMIA / ...)")
    num_cls = _get_num_classes(args['dataset_name'])
    batch_size = args.get('batch_size', 256)
    device = args['device']

    all_trials_results = {
        'confidence': [], 'entropy': [], 'modified_entropy': [],
        'train_LR_m1': [], 'train_RF_m1': [], 'train_LR_m2': [], 'train_RF_m2': [],
        'SAMIA': []
    }

    load_args = dict(args)
    load_args['attack_method'] = 'MIA'

    for trial in range(args['trials']):
        print(f"\n=== Trial {trial + 1}/{args['trials']} ===")
        prep_result = baseline_prep_single_trial(load_args, trial, load_sensitivity=True)
        if len(prep_result) == 8:
            P_shadow, P_target, shadow_class, target_class, shadow_labels, target_labels, P_shadow_sens, P_target_sens = prep_result
            has_saved_sensitivity = True
        else:
            P_shadow, P_target, shadow_class, target_class, shadow_labels, target_labels = prep_result
            has_saved_sensitivity = False
            P_shadow_sens = P_target_sens = None

        P_shadow = np.array(P_shadow)
        P_target = np.array(P_target)
        if P_shadow.ndim == 3:
            P_shadow = P_shadow[:, 0, :]
        if P_target.ndim == 3:
            P_target = P_target[:, 0, :]

        shadow_class = np.array(shadow_class)
        target_class = np.array(target_class)
        shadow_labels = np.array([int(x) if hasattr(x, 'item') else int(x) for x in shadow_labels])
        target_labels = np.array([int(x) if hasattr(x, 'item') else int(x) for x in target_labels])

        s_tr_mask = shadow_class == 1
        s_te_mask = shadow_class == 0
        s_tr_outputs = P_shadow[s_tr_mask]
        s_tr_labels = shadow_labels[s_tr_mask]
        s_te_outputs = P_shadow[s_te_mask]
        s_te_labels = shadow_labels[s_te_mask]
        t_tr_outputs = P_target[target_class == 1]
        t_tr_labels = target_labels[target_class == 1]
        t_te_outputs = P_target[target_class == 0]
        t_te_labels = target_labels[target_class == 0]

        # ----- 1-3: Metric-based MIA (Confidence, Entropy, Modified Entropy) -----
        attacker = ThresholdAttacker(
            (s_tr_outputs, s_tr_labels), (s_te_outputs, s_te_labels),
            (t_tr_outputs, t_tr_labels), (t_te_outputs, t_te_labels),
            num_cls
        )
        conf_acc, conf_auc, conf_tpr, conf_tpr_01 = attacker._mem_inf_thre(
            'confidence', attacker.s_tr_conf, attacker.s_te_conf,
            attacker.t_tr_conf, attacker.t_te_conf
        )
        all_trials_results['confidence'].append((conf_acc, conf_auc, conf_tpr, conf_tpr_01))

        if confidence_only:
            print(f"Metric-based (confidence only): acc={conf_acc:.4f} auc={conf_auc:.4f}")
        else:
            ent_acc, ent_auc, ent_tpr, ent_tpr_01 = attacker._mem_inf_thre(
                'entropy', -attacker.s_tr_entr, -attacker.s_te_entr,
                -attacker.t_tr_entr, -attacker.t_te_entr
            )
            mod_acc, mod_auc, mod_tpr, mod_tpr_01 = attacker._mem_inf_thre(
                'mod_entropy', -attacker.s_tr_m_entr, -attacker.s_te_m_entr,
                -attacker.t_tr_m_entr, -attacker.t_te_m_entr
            )
            all_trials_results['entropy'].append((ent_acc, ent_auc, ent_tpr, ent_tpr_01))
            all_trials_results['modified_entropy'].append((mod_acc, mod_auc, mod_tpr, mod_tpr_01))
            print(
                f"Metric-based: Conf acc={conf_acc:.4f} auc={conf_auc:.4f} | "
                f"Ent acc={ent_acc:.4f} auc={ent_auc:.4f} | "
                f"ModEnt acc={mod_acc:.4f} auc={mod_auc:.4f}"
            )

        # ----- 4-5: Train-based MIA (LR, RF with method 1 and 2) -----
        if confidence_only:
            continue
        shadow_post_df = pd.DataFrame(columns=["compress", "targets", "label"])
        target_post_df = pd.DataFrame(columns=["compress", "targets", "label"])
        for i in range(len(P_shadow)):
            comp = torch.tensor(P_shadow[i:i+1] if P_shadow.ndim > 1 else P_shadow[i:i+1].reshape(1, -1),
                               dtype=torch.float32)
            lbl = int(shadow_labels[i])
            t_onehot = F.one_hot(torch.tensor(lbl), num_cls).float().reshape(1, -1)
            shadow_post_df.loc[len(shadow_post_df)] = [
                comp, t_onehot, int(shadow_class[i])
            ]
        for i in range(len(P_target)):
            comp = torch.tensor(P_target[i:i+1] if P_target.ndim > 1 else P_target[i:i+1].reshape(1, -1),
                               dtype=torch.float32)
            lbl = int(target_labels[i])
            t_onehot = F.one_hot(torch.tensor(lbl), num_cls).float().reshape(1, -1)
            target_post_df.loc[len(target_post_df)] = [
                comp, t_onehot, int(target_class[i])
            ]

        for method in [
            'CompLeakNR_train_based_method_1(target_compressed_model)',
            'CompLeakNR_train_based_method_2(target_compressed_model)'
        ]:
            ConstructFeature(shadow_post_df).obtain_feature(method, shadow_post_df)
            ConstructFeature(target_post_df).obtain_feature(method, target_post_df)
            for attack_name in ['LR', 'RF']:
                att = Attack(attack_name, shadow_post_df, target_post_df)
                att.train_attack_model(method, "")
                test_acc, test_auc, _, _, test_tpr, test_tpr_01 = att.test_attack_model(method)
                key = f"train_{attack_name}_m1" if "method_1" in method else f"train_{attack_name}_m2"
                all_trials_results[key].append((test_acc, test_auc, test_tpr, test_tpr_01))

        samia_done = False
        has_valid_sensitivity = (has_saved_sensitivity and P_shadow_sens is not None and P_target_sens is not None
                                 and np.any(np.abs(np.array(P_shadow_sens)) > 1e-8))
        if has_valid_sensitivity:
            try:
                P_s = np.array(P_shadow)
                P_t = np.array(P_target)
                S_s = np.array(P_shadow_sens)
                S_t = np.array(P_target_sens)
                if P_s.ndim == 3:
                    P_s = P_s[:, 0, :]
                if P_t.ndim == 3:
                    P_t = P_t[:, 0, :]
                if S_s.ndim == 3:
                    S_s = S_s[:, 0, :]
                if S_t.ndim == 3:
                    S_t = S_t[:, 0, :]
                sh_lbl = np.array([int(x) if hasattr(x, 'item') else int(x) for x in shadow_labels])
                tgt_lbl = np.array([int(x) if hasattr(x, 'item') else int(x) for x in target_labels])
                sh_onehot = F.one_hot(torch.tensor(sh_lbl).long(), num_cls).numpy().astype(np.float32)
                tgt_onehot = F.one_hot(torch.tensor(tgt_lbl).long(), num_cls).numpy().astype(np.float32)
                attack_train_x = np.concatenate([P_s, S_s, sh_onehot], axis=1).astype(np.float32)
                attack_train_y = np.array(shadow_class, dtype=np.int64)
                attack_test_x = np.concatenate([P_t, S_t, tgt_onehot], axis=1).astype(np.float32)
                attack_test_y = np.array(target_class, dtype=np.int64)

                in_dim = 3 * num_cls
                from model.Transformer import SAMIATransformer
                samia_net = SAMIATransformer(input_dim=in_dim, output_dim=2).to(device)
                samia_opt = torch.optim.SGD(samia_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                samia_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(samia_opt, T_max=100)
                samia_crit = torch.nn.CrossEntropyLoss()
                train_ds = TensorDataset(torch.tensor(attack_train_x).float(), torch.tensor(attack_train_y).long())
                train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                best_acc, best_auc, best_tpr, best_tpr_01 = 0.0, 0.0, 0.0, 0.0
                for epoch in range(100):
                    samia_net.train()
                    for bx, by in train_dl:
                        bx, by = bx.to(device), by.to(device)
                        samia_opt.zero_grad()
                        out = samia_net(bx)
                        loss = samia_crit(out, by)
                        l2_reg = sum(torch.norm(p, 2) ** 2 for p in samia_net.parameters()) / 2
                        loss = loss + 5e-5 * l2_reg
                        loss.backward()
                        samia_opt.step()
                    samia_scheduler.step()
                    samia_net.eval()
                    with torch.no_grad():
                        logits = samia_net(torch.tensor(attack_test_x).float().to(device))
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                    scores_pos = probs[:, 1]
                    auc_val = roc_auc_score(attack_test_y, scores_pos)
                    if auc_val < 0.5:
                        scores_pos = 1.0 - scores_pos
                        auc_val = 1.0 - auc_val
                    tpr_val = calculate_tpr_at_fpr(attack_test_y, scores_pos, 0.001)
                    tpr_val_01 = calculate_tpr_at_fpr(attack_test_y, scores_pos, 0.01)
                    pred = (scores_pos > 0.5).astype(np.int64)
                    acc_val = np.mean(pred == attack_test_y)
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_auc = auc_val
                        best_tpr = tpr_val
                        best_tpr_01 = tpr_val_01
                all_trials_results['SAMIA'].append((best_acc, best_auc, best_tpr, best_tpr_01))
                print(
                    f"SAMIA (saved): acc={best_acc:.4f} auc={best_auc:.4f} "
                    f"tpr@0.1%={best_tpr:.4f} tpr@1%={best_tpr_01:.4f}"
                )
                samia_done = True
            except Exception as e:
                import traceback
                print(f"SAMIA (saved) skip: {e}")
                traceback.print_exc()

        if not samia_done:
            target_model, shadow_model = _load_models_for_attack(load_args, trial)
            if target_model is not None and shadow_model is not None:
                try:
                    from data.load_data import get_data
                    from data.prepare_data import split_dataset
                    from model.Transformer import SAMIATransformer
                    train_data, test_data = get_data(args['dataset_name'])
                    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
                    shadow_train_loader = DataLoader(shadow_m, batch_size=batch_size, shuffle=False)
                    shadow_test_loader = DataLoader(shadow_um, batch_size=batch_size, shuffle=False)
                    victim_train_loader = DataLoader(target_m, batch_size=batch_size, shuffle=False)
                    victim_test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                    sh_in_pred, sh_in_tgt, sh_in_sens = _predict_target_sensitivity(
                        shadow_model, shadow_train_loader, device, num_cls)
                    sh_out_pred, sh_out_tgt, sh_out_sens = _predict_target_sensitivity(
                        shadow_model, shadow_test_loader, device, num_cls)
                    vic_in_pred, vic_in_tgt, vic_in_sens = _predict_target_sensitivity(
                        target_model, victim_train_loader, device, num_cls)
                    vic_out_pred, vic_out_tgt, vic_out_sens = _predict_target_sensitivity(
                        target_model, victim_test_loader, device, num_cls)

                    sh_in_onehot = F.one_hot(sh_in_tgt.long(), num_cls).float()
                    sh_out_onehot = F.one_hot(sh_out_tgt.long(), num_cls).float()
                    vic_in_onehot = F.one_hot(vic_in_tgt.long(), num_cls).float()
                    vic_out_onehot = F.one_hot(vic_out_tgt.long(), num_cls).float()

                    attack_train_x = torch.cat([
                        torch.cat([sh_in_pred, sh_in_sens, sh_in_onehot], dim=1),
                        torch.cat([sh_out_pred, sh_out_sens, sh_out_onehot], dim=1)
                    ], dim=0).numpy()
                    attack_train_y = np.concatenate([np.ones(len(sh_in_pred)), np.zeros(len(sh_out_pred))])
                    attack_test_x = torch.cat([
                        torch.cat([vic_in_pred, vic_in_sens, vic_in_onehot], dim=1),
                        torch.cat([vic_out_pred, vic_out_sens, vic_out_onehot], dim=1)
                    ], dim=0).numpy()
                    attack_test_y = np.concatenate([np.ones(len(vic_in_pred)), np.zeros(len(vic_out_pred))])

                    in_dim = 3 * num_cls
                    samia_net = SAMIATransformer(input_dim=in_dim, output_dim=2).to(device)
                    samia_opt = torch.optim.SGD(samia_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                    samia_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(samia_opt, T_max=100)
                    samia_crit = torch.nn.CrossEntropyLoss()
                    train_ds = TensorDataset(torch.tensor(attack_train_x).float(), torch.tensor(attack_train_y).long())
                    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                    best_acc, best_auc, best_tpr, best_tpr_01 = 0.0, 0.0, 0.0, 0.0
                    for epoch in range(100):
                        samia_net.train()
                        for bx, by in train_dl:
                            bx, by = bx.to(device), by.to(device)
                            samia_opt.zero_grad()
                            out = samia_net(bx)
                            loss = samia_crit(out, by)
                            l2_reg = sum(torch.norm(p, 2) ** 2 for p in samia_net.parameters()) / 2
                            loss = loss + 5e-5 * l2_reg
                            loss.backward()
                            samia_opt.step()
                        samia_scheduler.step()
                        samia_net.eval()
                        with torch.no_grad():
                            logits = samia_net(torch.tensor(attack_test_x).float().to(device))
                            probs = F.softmax(logits, dim=1).cpu().numpy()
                        scores_pos = probs[:, 1]
                        auc_val = roc_auc_score(attack_test_y, scores_pos)
                        if auc_val < 0.5:
                            scores_pos = 1.0 - scores_pos
                            auc_val = 1.0 - auc_val
                        tpr_val = calculate_tpr_at_fpr(attack_test_y, scores_pos, 0.001)
                        tpr_val_01 = calculate_tpr_at_fpr(attack_test_y, scores_pos, 0.01)
                        pred = (scores_pos > 0.5).astype(np.int64)
                        acc_val = np.mean(pred == attack_test_y)
                        if acc_val > best_acc:
                            best_acc = acc_val
                            best_auc = auc_val
                            best_tpr = tpr_val
                            best_tpr_01 = tpr_val_01
                    all_trials_results['SAMIA'].append((best_acc, best_auc, best_tpr, best_tpr_01))
                    print(
                        f"SAMIA (live): acc={best_acc:.4f} auc={best_auc:.4f} "
                        f"tpr@0.1%={best_tpr:.4f} tpr@1%={best_tpr_01:.4f}"
                    )
                    samia_done = True
                except Exception as e:
                    import traceback
                    print(f"SAMIA skip: {e}")
                    traceback.print_exc()

        if not samia_done:
            print("SAMIA skip: no saved sensitivity (run training) or models not found")
            all_trials_results['SAMIA'].append((0.5, 0.5, 0.0, 0.0))

    final_metrics = {}

    def _add_full_stats(prefix, arr):
        """Add mean/std/var/min/max/range statistics for a value list."""
        a = np.array(arr, dtype=float)
        final_metrics[f'{prefix}_mean'] = round(np.mean(a), 4)
        final_metrics[f'{prefix}_std'] = round(np.std(a), 4)
        final_metrics[f'{prefix}_var'] = round(np.var(a), 4)
        final_metrics[f'{prefix}_min'] = round(np.min(a), 4)
        final_metrics[f'{prefix}_max'] = round(np.max(a), 4)
        final_metrics[f'{prefix}_range'] = round(float(np.max(a) - np.min(a)), 4)

    for name, vals in [
        ('Confidence', all_trials_results['confidence']),
        ('Entropy', all_trials_results['entropy']),
        ('ModifiedEntropy', all_trials_results['modified_entropy']),
    ]:
        if vals:
            accs = [v[0] for v in vals]
            aucs = [v[1] for v in vals]
            tprs = [v[2] for v in vals]
            tprs_01 = [v[3] for v in vals]
            _add_full_stats(f'{name}_acc', accs)
            _add_full_stats(f'{name}_auc', aucs)
            _add_full_stats(f'{name}_tpr', tprs)
            _add_full_stats(f'{name}_tpr_at_fpr_01', tprs_01)

    for key in ['train_LR_m1', 'train_RF_m1', 'train_LR_m2', 'train_RF_m2']:
        if all_trials_results[key]:
            accs = [v[0] for v in all_trials_results[key]]
            aucs = [v[1] for v in all_trials_results[key]]
            tprs = [v[2] for v in all_trials_results[key]]
            tprs_01 = [v[3] for v in all_trials_results[key]]
            _add_full_stats(f'{key}_acc', accs)
            _add_full_stats(f'{key}_auc', aucs)
            _add_full_stats(f'{key}_tpr', tprs)
            _add_full_stats(f'{key}_tpr_at_fpr_01', tprs_01)

    if all_trials_results['SAMIA']:
        accs = [v[0] for v in all_trials_results['SAMIA']]
        aucs = [v[1] for v in all_trials_results['SAMIA']]
        tprs = [v[2] for v in all_trials_results['SAMIA']]
        tprs_01 = [v[3] for v in all_trials_results['SAMIA']]
        _add_full_stats('SAMIA_auc', aucs)
        _add_full_stats('SAMIA_tpr', tprs)
        _add_full_stats('SAMIA_tpr_at_fpr_01', tprs_01)
        _add_full_stats('SAMIA_acc', accs)

    # Unify output path under result/MIA for compatibility with MIA workflow.
    save_path = os.getcwd() + (
        f"/result/MIA/{args['net_name']}/{args['dataset_name']}/"
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}/"
    )
    if float(args.get('eps', 0)) != 0.0:
        save_path = os.path.join(save_path, str(args['eps']))
    os.makedirs(save_path, exist_ok=True)
    pd.DataFrame(list(final_metrics.items()), columns=["metric", "value"]).to_csv(
        f"{save_path}/attack_performance.csv", index=False
    )
    print("\nCompLeakNR Final metrics:", final_metrics)
