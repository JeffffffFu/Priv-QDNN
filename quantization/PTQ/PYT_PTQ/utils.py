import torch
from . import quantizer


def enable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantizer.Quantizer):
            child.ptq = True
        else:
            enable_calibrate(child)
    return module

def disable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantizer.Quantizer):
            child.ptq = False
        else:
            disable_calibrate(child)
    return module

def calibrate(model, train_loader,label_key, device='cpu'):
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx == 10: break
            if isinstance(targets, dict):
                if label_key is None:
                    raise ValueError("label_key must be specified for multi-label datasets.")
                targets = targets[label_key]
            images, targets = images.to(device), targets.to(device)
            _ = model(images)