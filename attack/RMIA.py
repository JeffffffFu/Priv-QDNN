import numpy as np
from tqdm import tqdm

from attack.utils import find_best_threshold, tpr_at_fpr
from data.load_data import get_data
from data.prepare_data import split_dataset4
from model.DNN import DNN
from training.utlis import sample_target_samples
from sklearn import metrics
import torch
import pandas as pd
import os

def Score_RMIA(target_sample, target_model, shadow_model_list, Ratio_z_list, args,gamma=1.0, alpha=0.5):
    """
    RMIA (Relative Membership Inference Attack) Algorithm

    Parameters:
    - x: target sample (features)
    - theta: target model (should have a predict_prob method)
    - Theta: list of k shadow models (each should have a predict_prob method)
    - Z: population dataset (subset used for comparison)
    - gamma: threshold parameter (default=1.0)
    - alpha: scaling factor (default=0.5) ,a = 0.3 for CIFAR-10 and CINIC-10, a = 0.6 for CIFAR-100

    Returns:
    - Score_MIA: empirical MIA score for the target sample x
    """
    if args['dataset_name']=='cifar10' or args['dataset_name']=='cinic10':
        alpha=0.3
    elif args['dataset_name']=='cifar100' :
        alpha=0.6

    target_x=target_sample[0]
    target_y=target_sample[1]


    Pr_x_out = np.mean([model.predict_proba(target_x)[0][target_y] for model in shadow_model_list])

    Pr_x = 0.5 * ((1 + alpha) * Pr_x_out + (1 - alpha))

    Pr_x_theta = target_model.predict_proba(target_x)[0][target_y]
    Ratio_x = Pr_x_theta / Pr_x

    C = 0

    for Ratio_z in Ratio_z_list:
        if Ratio_z == 0:
            continue
        if Ratio_x / Ratio_z >= gamma:
            C += 1

    Score_MIA = C / len(Ratio_z_list)
    return Score_MIA


# train and attack together
def RMIA(args):
    print(f"{args['attack_method']}-{args['net_name']}-{args['dataset_name']}-----")
    
    # Train shadow models once and reuse across trials.
    shadow_model_list = model_training_shadow_RMIA(args)
    
    all_trials_metrics = []
    
    # Run attack for each trial.
    for trial in range(args['trials']):
        print(f"Processing trial {trial+1}/{args['trials']}")
        
        target_model_list = model_training_target_RMIA(args)

        train_data, test_data = get_data(args['dataset_name'])
        target_m, shadow_m = split_dataset4(train_data, args['random'])

        Z = sample_target_samples(shadow_m, 100, args['random'])
        number = int(0.2 * len(shadow_m))
        target_member = sample_target_samples(target_m, number, args['random'])
        target_nonmember = sample_target_samples(test_data, number, args['random'])
        target_classes = [1] * number + [0] * number
        
        trial_metrics = []
        
        for i, target_model in enumerate(target_model_list):
            score_list = []
            Ratio_z_list = []
            
            for z in Z:
                z_x = z[0]
                z_y = z[1]
                Pr_z = np.mean([model.predict_proba(z_x)[0][z_y] for model in shadow_model_list])

                Pr_z_theta = target_model.predict_proba(z_x)[0][z_y]
                Ratio_z = Pr_z_theta / Pr_z
                Ratio_z_list.append(Ratio_z)

            print('member--------')
            for target_sample in tqdm(target_member):
                score = Score_RMIA(target_sample, target_model, shadow_model_list, Ratio_z_list, args)
                score_list.append(score)

            print('nonmember--------')
            for target_sample in tqdm(target_nonmember):
                score = Score_RMIA(target_sample, target_model, shadow_model_list, Ratio_z_list, args)
                score_list.append(score)

            save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/{trial}"
            os.makedirs(save_path, exist_ok=True)
            torch.save(score_list, f'{save_path}/score_list.pth')
            torch.save(target_classes, f'{save_path}/target_classes.pth')

            auc_target = metrics.roc_auc_score(target_classes, score_list)
            best_threshold, best_accuracy = find_best_threshold(score_list, target_classes)

            tpr_at_fpr_value_0005 = tpr_at_fpr(target_classes, score_list, 0.005)
            tpr_at_fpr_value_0001 = tpr_at_fpr(target_classes, score_list, 0.01)
            tpr_at_fpr_value_005 = tpr_at_fpr(target_classes, score_list, 0.05)

            trial_result = {
                "accuracy": best_accuracy,
                "auc": auc_target,
                "tpr_at_fpr_0005": tpr_at_fpr_value_0005,
                "tpr_at_fpr_0001": tpr_at_fpr_value_0001,
                "tpr_at_fpr_005": tpr_at_fpr_value_005,
            }
            trial_metrics.append(trial_result)

            print(f'Trial {trial}, Model {i}: Accuracy={best_accuracy:.4f}, AUC={auc_target:.4f}')

        # Average metrics for the current trial
        trial_avg_metrics = {
            "accuracy": round(np.mean([m["accuracy"] for m in trial_metrics]), 4),
            "auc": round(np.mean([m["auc"] for m in trial_metrics]), 4),
            "tpr_at_fpr_0005": round(np.mean([m["tpr_at_fpr_0005"] for m in trial_metrics]), 4),
            "tpr_at_fpr_0001": round(np.mean([m["tpr_at_fpr_0001"] for m in trial_metrics]), 4),
            "tpr_at_fpr_005": round(np.mean([m["tpr_at_fpr_005"] for m in trial_metrics]), 4),
        }
        all_trials_metrics.append(trial_avg_metrics)

    final_metrics = {}
    
    accuracies = [result['accuracy'] for result in all_trials_metrics]
    aucs = [result['auc'] for result in all_trials_metrics]
    tpr_0005s = [result['tpr_at_fpr_0005'] for result in all_trials_metrics]
    tpr_0001s = [result['tpr_at_fpr_0001'] for result in all_trials_metrics]
    tpr_005s = [result['tpr_at_fpr_005'] for result in all_trials_metrics]
    
    # Compute summary statistics
    final_metrics["accuracy_mean"] = round(np.mean(accuracies), 4)
    final_metrics["accuracy_std"] = round(np.std(accuracies), 4)
    final_metrics["accuracy_var"] = round(np.var(accuracies), 4)
    final_metrics["accuracy_min"] = round(np.min(accuracies), 4)
    final_metrics["accuracy_max"] = round(np.max(accuracies), 4)
    final_metrics["accuracy_range"] = round(np.max(accuracies) - np.min(accuracies), 4)
    
    final_metrics["auc_mean"] = round(np.mean(aucs), 4)
    final_metrics["auc_std"] = round(np.std(aucs), 4)
    final_metrics["auc_var"] = round(np.var(aucs), 4)
    final_metrics["auc_min"] = round(np.min(aucs), 4)
    final_metrics["auc_max"] = round(np.max(aucs), 4)
    final_metrics["auc_range"] = round(np.max(aucs) - np.min(aucs), 4)
    
    final_metrics["tpr_at_fpr_0005_mean"] = round(np.mean(tpr_0005s), 4)
    final_metrics["tpr_at_fpr_0005_std"] = round(np.std(tpr_0005s), 4)
    final_metrics["tpr_at_fpr_0005_var"] = round(np.var(tpr_0005s), 4)
    final_metrics["tpr_at_fpr_0005_min"] = round(np.min(tpr_0005s), 4)
    final_metrics["tpr_at_fpr_0005_max"] = round(np.max(tpr_0005s), 4)
    final_metrics["tpr_at_fpr_0005_range"] = round(np.max(tpr_0005s) - np.min(tpr_0005s), 4)
    
    final_metrics["tpr_at_fpr_0001_mean"] = round(np.mean(tpr_0001s), 4)
    final_metrics["tpr_at_fpr_0001_std"] = round(np.std(tpr_0001s), 4)
    final_metrics["tpr_at_fpr_0001_var"] = round(np.var(tpr_0001s), 4)
    final_metrics["tpr_at_fpr_0001_min"] = round(np.min(tpr_0001s), 4)
    final_metrics["tpr_at_fpr_0001_max"] = round(np.max(tpr_0001s), 4)
    final_metrics["tpr_at_fpr_0001_range"] = round(np.max(tpr_0001s) - np.min(tpr_0001s), 4)
    
    final_metrics["tpr_at_fpr_005_mean"] = round(np.mean(tpr_005s), 4)
    final_metrics["tpr_at_fpr_005_std"] = round(np.std(tpr_005s), 4)
    final_metrics["tpr_at_fpr_005_var"] = round(np.var(tpr_005s), 4)
    final_metrics["tpr_at_fpr_005_min"] = round(np.min(tpr_005s), 4)
    final_metrics["tpr_at_fpr_005_max"] = round(np.max(tpr_005s), 4)
    final_metrics["tpr_at_fpr_005_range"] = round(np.max(tpr_005s) - np.min(tpr_005s), 4)

    # Build a DataFrame, stored transposed (metric name | value)
    data_to_save = pd.DataFrame(list(final_metrics.items()), columns=["metric", "value"])

    save_path = os.getcwd() + f"/result/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/"

    # Create the output directory if it does not exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the CSV in **long/vertical format**
    data_to_save.to_csv(f"{save_path}/attack_performance.csv", index=False)
    
    print("Final aggregated metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")



def model_training_target_RMIA(args):
    training_set_acc=[]
    test_set_acc=[]
    target_model_list=[]
    print("dataset and net_name:", args['dataset_name'], args['net_name'])

    train_data, test_data = get_data(args['dataset_name'])
    target_model = DNN(args)

    target_m, shadow_m = split_dataset4(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    target_model.train_model(train_loader, test_loader)

    if args['quantization'] != 'Original':
        from quantization import get_PTQ_quantization

        PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
        if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args['quantization'] == 'hawq' or \
                args['quantization'] == 'bsq':
            sub_set = target_m
        else:
            sub_set = sample_target_samples(target_m, 4096, args['random'])
        target_model = PTQ_quantizer(target_model, sub_set, args)

    training_set_acc.append(target_model.test_model_acc(train_loader))
    test_set_acc.append(target_model.test_model_acc(test_loader))

    print(f'train_acc:{training_set_acc},test_set_acc{test_set_acc}')
    target_model_list.append(target_model)

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
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/target/"
    os.makedirs(save_path, exist_ok=True)
    data_to_save.to_csv(f"{save_path}/model_performance2.csv", index=False)
    return target_model_list


def model_training_shadow_RMIA(args):
    training_set_acc=[]
    test_set_acc=[]
    shadow_model_list=[]
    for t in range(args['observations']):
        print("dataset and net_name:", args['dataset_name'], args['net_name'])

        train_data, test_data = get_data(args['dataset_name'])
        target_model = DNN(args)

        target_m, shadow_m = split_dataset4(train_data, args['random'])
        shadow_m = sample_target_samples(shadow_m, int(len(shadow_m)/2), 'None')  # Use half of shadow samples for training.

        train_loader = torch.utils.data.DataLoader(
            shadow_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args['batch_size'], shuffle=True)

        target_model.train_model(train_loader, test_loader)

        if args['quantization'] != 'Original':
            from quantization import get_PTQ_quantization

            PTQ_quantizer = get_PTQ_quantization(args['quantization'], args['net_name'])
            if args['quantization'] == 'pact' or args['quantization'] == 'PYT_QAT' or args['quantization'] == 'hawq' or args['quantization'] == 'bsq':
                sub_set = shadow_m
            else:
                sub_set = sample_target_samples(shadow_m, 4096, args['random'])
            target_model = PTQ_quantizer(target_model, sub_set, args)

        training_set_acc.append(target_model.test_model_acc(train_loader))
        test_set_acc.append(target_model.test_model_acc(test_loader))
        shadow_model_list.append(target_model)

        print(f'Round {t}, train_acc:{training_set_acc}, test_set_acc:{test_set_acc}')

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
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/shadow/"
    os.makedirs(save_path, exist_ok=True)
    data_to_save.to_csv(f"{save_path}/model_performance.csv", index=False)
    return shadow_model_list
