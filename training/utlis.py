import random

import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Subset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def _compute_sensitivity_single(model, data, device, num_cls, m=10, epsilon=1e-3):
    """CompLeakNR SAMIA:  ()"""
    import torch.nn.functional as F
    net = model.model if hasattr(model, 'model') else model
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        data = data.to(device)
        # Tabular samples (Texas/Location) are shape (feat_dim,); image samples are (C,H,W).
        if data.dim() == 1:
            data = data.unsqueeze(0)
        elif data.dim() == 3:
            data = data.unsqueeze(0)
        pred = F.softmax(net(data), dim=-1)
        # Use dynamic class dimension from model output to avoid
        # mismatches when num_cls fallback is incorrect.
        cls_dim = pred.shape[-1]
        if len(data.size()) == 4:
            x = data.repeat((m, 1, 1, 1))
        elif len(data.size()) == 3:
            x = data.repeat((m, 1, 1))
        else:
            x = data.repeat((m, 1))
        u = torch.randn_like(x, device=device)
        eval_pts = x + epsilon * u
        new_pred = F.softmax(net(eval_pts), dim=-1)
        diff = torch.abs(new_pred - pred.repeat((m, 1)))
        diff = diff.view(m, -1, cls_dim)
        sens = diff.mean(dim=0) / epsilon
    return sens.cpu().numpy()


def save_output(shadow_or_target,args,target_model,training_set,test_set,t):

    if args['eps']!=0:

        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/{shadow_or_target}/{t}/"
    else:
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{shadow_or_target}/{t}/"

    os.makedirs(save_path, exist_ok=True)

    if args['attack_method'] in ('CompLeakNR', 'MIA'):
        torch.save(target_model, f'{save_path}/model.pth')
        torch.save(args['device'], f'{save_path}/device.pth')

    save_sensitivity = (args['attack_method'] in ('CompLeakNR', 'MIA'))
    num_cls = getattr(target_model, 'num_classes', 100)
    device = args['device'] if isinstance(args['device'], str) else args['device']

    print("member sample:------------------")

    if shadow_or_target=='shadow':
        number=int(0.2*len(training_set))  #member is 20% to shadow set, all is 40% to shadow set
    else:
        number=int(0.2*len(training_set))  #member is 20% to training set, all is 40% to training set

    member_set = sample_target_samples(training_set, number,args['random'])

    for i, idx in enumerate(member_set.indices):
        data, label = member_set.dataset[idx]
        save_path_target_sample = f'{save_path}/{i}'
        os.makedirs(save_path_target_sample, exist_ok=True)
        member_sample_target_model_posterior = target_model.predict_proba(data)
        torch.save(member_sample_target_model_posterior,
                   f"{save_path_target_sample}/member_sample_target_model_posterior.pth")
        torch.save(label, f"{save_path_target_sample}/member_sample_label.pth")
        if save_sensitivity:
            sens = _compute_sensitivity_single(target_model, data, device, num_cls)
            torch.save(sens, f"{save_path_target_sample}/member_sample_sensitivity.pth")

    print("non_member sample:--------------------")

    non_member_set = sample_target_samples(test_set, number,args['random'])

    for i, idx in enumerate(non_member_set.indices):
        data, label = non_member_set.dataset[idx]
        save_path_test_sample = f'{save_path}/{i}'
        os.makedirs(save_path_test_sample, exist_ok=True)
        test_sample_target_model_posterior = target_model.predict_proba(data)
        torch.save(test_sample_target_model_posterior,
                   f"{save_path_test_sample}/non_member_sample_target_model_posterior.pth")
        torch.save(label, f"{save_path_test_sample}/non_member_sample_label.pth")
        if save_sensitivity:
            sens = _compute_sensitivity_single(target_model, data, device, num_cls)
            torch.save(sens, f"{save_path_test_sample}/non_member_sample_sensitivity.pth")


def save_output_RMIA(shadow_or_target,args,target_model,t):


    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{shadow_or_target}/{t}/"
    os.makedirs(save_path, exist_ok=True)
    print("save model:------------------")
    torch.save(target_model, f'{save_path}/model.pth')
    torch.save(args['device'], f'{save_path}/device.pth')




def save_output_AIA(shadow_or_target,args,target_model,target_set,t):

    if args['eps']!=0:
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{args['eps']}/{shadow_or_target}/{t}/"
    else:
        save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{shadow_or_target}/{t}/"

    os.makedirs(save_path, exist_ok=True)
    number=int(0.4*len(target_set))  #40% to shadow set, 40% to all test set
    print('sample_number:',number)
    target_set = sample_target_samples(target_set, number,args['random'])
    for i, idx in enumerate(target_set.indices):
        data, labels = target_set.dataset[idx]
        target_class=labels['sen_attr']

        save_path_test_sample = f'{save_path}/{i}'
        os.makedirs(save_path_test_sample, exist_ok=True)
        target_model_posterior = target_model.predict_proba(data)
        data = data.to(args['device'])
        target_model_last_layer = target_model.get_last_hidden_layer(data)
        torch.save(target_model_posterior,
                   f"{save_path_test_sample}/target_model_posterior.pth")
        torch.save(target_model_last_layer,
                   f"{save_path_test_sample}/target_model_last_layer.pth")
        torch.save(target_class,
                   f"{save_path_test_sample}/target_class.pth")

def save_output_MInv(shadow_or_target,args,target_model,training_set,test_set,t):


    save_path = os.getcwd() + f"/save/{args['attack_method']}/{args['net_name']}/{args['dataset_name']}/{args['quantization']}_{args['a_bit']}_{args['w_bit']}/{shadow_or_target}/{t}/"
    os.makedirs(save_path, exist_ok=True)
    torch.save(target_model,f'{save_path}/model.pth')
    torch.save(args['device'],f'{save_path}/device.pth')

def sample_target_samples(train_data, number,random_seed):

    if random_seed!='None':
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    all_indices = list(range(len(train_data)))
    
    # Ensure sampling number doesn't exceed available sample count
    actual_number = min(number, len(all_indices))
    if actual_number != number:
        print(f"Warning: Requested {number} samples but only {len(all_indices)} available. Using {actual_number} samples.")

    random_indices = random.sample(all_indices, actual_number)

    selected_samples = Subset(train_data, random_indices)


    return selected_samples


import numpy as np
def lazy_split(dataset, random_seed=3407, ratio=0.5):
    indices = list(range(len(dataset)))
    rng = np.random.RandomState(random_seed)
    rng.shuffle(indices)
    split = int(ratio * len(indices))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def generate_random_data(num_samples, img_h=112, img_w=92, output_dim=2, seed=None):
    """
    Generate AT&T style (112x92) random binary image data

    Parameters:
        num_samples (int): Number of samples
        img_h (int): Image height (default 112)
        img_w (int): Image width (default 92)
        output_dim (int): Output class number (default 2 classes)
        seed (int): Random seed

    Returns:
        X_images (torch.Tensor): Original image data (112x92)
        X_flattened (torch.Tensor): Flattened data (10304 dimensions)
        y (torch.Tensor): Random labels
        dataset (TensorDataset): Dataset ready for training
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Generate random binary images (num_samples, 112, 92)
    X_images = torch.FloatTensor(np.random.randint(2, size=(num_samples, img_h, img_w)))

    # Flatten processing (num_samples, 112*92=10304)
    X_flattened = X_images.view(num_samples, -1)

    # Generate random labels
    y = torch.LongTensor(np.random.randint(output_dim, size=num_samples))

    # Create dataset
    dataset = TensorDataset(X_flattened, y)

    return X_images, X_flattened, y, dataset


def visualize_tensor_as_image_and_save(tensor_data, output_filepath, title="Tensor Visualization"):
    """
    Visualize a PyTorch tensor as an image and save it to the specified path.

    Args:
        tensor_data (torch.Tensor): The tensor to visualize.
                                   Expected dimensions are (Channels, Height, Width) or (Height, Width).
        output_filepath (str): Complete path and filename for saving the image (e.g., "./output_images/my_image.png").
        title (str): Title of the image.
    """
    # Ensure tensor is on CPU and convert to NumPy array
    # If tensor has only 2 dimensions (Height, Width), use directly
    # If it has 3 dimensions (Channels, Height, Width), transpose
    if tensor_data.dim() == 3:
        # Convert tensor from (C, H, W) to (H, W, C) for matplotlib.imshow to correctly display color images
        image_np = tensor_data.cpu().numpy().transpose(1, 2, 0)
    elif tensor_data.dim() == 2:
        # For grayscale images (H, W)
        image_np = tensor_data.cpu().numpy()
    else:
        raise ValueError("Unsupported tensor dimensions. Expected 2D (H, W) or 3D (C, H, W).")

    # Normalize image data to [0, 1] range for correct display
    min_val = image_np.min()
    max_val = image_np.max()
    if max_val - min_val > 0:
        normalized_image = (image_np - min_val) / (max_val - min_val)
    else:
        # If all pixel values are the same (max_val == min_val), use directly
        normalized_image = image_np

    # Display image (don't pop up window, save directly)
    plt.figure(figsize=(8, 8)) # Can adjust image size as needed
    plt.imshow(normalized_image)
    plt.title(title)
    plt.axis('off') # Hide axes

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save image
    plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0.1) # bbox_inches='tight' reduces white space
    plt.close() # Close figure, release memory

    print(f"Image successfully saved to: {output_filepath}")

def test_model_acc(model, test_loader,args,label_key=None):
    model.eval()
    device=args['device']
    correct = 0

    with torch.no_grad():

        for data, target  in test_loader:
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

def print_model_parameters_and_size(model, model_name="Target Model"):
    """
    Calculate and print the total number of parameters and model size (MB) of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count parameters for.
        model_name (str): The name of the model, used for printing output.
    """
    print(f"--- {model_name} Statistics ---")

    # 1. Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}") # Use comma separator for better readability

    # 2. Calculate model size (MB)
    # Assume parameters are all floating point numbers, typically torch.float32 (4 bytes) or torch.float16 (2 bytes)
    # p.element_size() returns the size of each element (bytes)
    # p.numel() returns the total number of elements in the tensor
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_mb = total_size_bytes / (1024 * 1024) # Convert to MB

    print(f"Model Size (Bytes): {total_size_bytes:,}")
    print(f"Model Size (MB): {total_size_mb:.4f} MB")

def compute_expression(N, p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    # Compute entropy with numerical stability
    if p == 0 or p == 1:
        entropy = 0
    else:
        entropy = -p * np.log2(p + 1e-10) - (1 - p) * np.log2(1 - p + 1e-10)
    return N * (1 - entropy)
