import copy
import torch
import torch.nn as nn
from typing import Optional
from . import quantizer
from .base_quant_modules import QuantConv2d, QuantLinear

def enable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantizer.Quantizer):
            child.ptq = True
        else:
            enable_calibrate(child)

def disable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantizer.Quantizer):
            child.ptq = False
        else:
            disable_calibrate(child)

def _extract_inputs_targets(batch, device: str = 'cpu', label_key: Optional[str] = None):
    """
    Support both tuple/list batches and dict-based batches.

    - Inputs are taken from common keys if dict: 'image', 'images', 'input', 'inputs', 'pixel_values', 'data', 'x'.
    - Targets are taken using provided label_key if given; otherwise common keys: 'label', 'labels', 'y', 'target', 'targets'.
    """
    # Tuple or list: expect (inputs, targets)
    if isinstance(batch, (tuple, list)):
        inputs, targets = batch[0], batch[1] if len(batch) > 1 else (batch[0], None)
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        if targets is not None and hasattr(targets, 'to'):
            targets = targets.to(device)
        return inputs, targets

    # Dict batch
    if isinstance(batch, dict):
        # Inputs
        input_keys = ['image', 'images', 'input', 'inputs', 'pixel_values', 'data', 'x']
        inputs = None
        for k in input_keys:
            if k in batch:
                inputs = batch[k]
                break
        if inputs is None:
            # Fallback: first tensor-like value
            for v in batch.values():
                if hasattr(v, 'to'):
                    inputs = v
                    break
        # Targets/labels
        targets = None
        if label_key is not None and label_key in batch:
            targets = batch[label_key]
        else:
            label_keys = ['label', 'labels', 'y', 'target', 'targets']
            for k in label_keys:
                if k in batch:
                    targets = batch[k]
                    break

        if inputs is not None and hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        if targets is not None and hasattr(targets, 'to'):
            targets = targets.to(device)
        return inputs, targets

    # Unknown batch type
    inputs = batch
    if hasattr(inputs, 'to'):
        inputs = inputs.to(device)
    return inputs, None


def calibrate(model, train_loader, device='cpu', label_key: Optional[str] = None):
    model.to(device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 10:
                break
            inputs, _ = _extract_inputs_targets(batch, device=device, label_key=label_key)
            if inputs is None:
                continue
            _ = model(inputs)


def prepare_model(module, w_bit, a_bit, weight_quantizer=None, input_quantizer=None):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_layer = QuantConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=True if child.bias is not None else False,
                w_bit=w_bit,
                a_bit=a_bit,
                weight_quantizer=copy.deepcopy(weight_quantizer),
                input_quantizer=copy.deepcopy(input_quantizer),
            )

            new_layer.weight = nn.Parameter(child.weight.detach().cpu())
            if child.bias is not None:
                new_layer.bias = nn.Parameter(child.bias.detach().cpu())
            setattr(module, name, new_layer)
        elif isinstance(child, nn.Linear):
            new_layer = QuantLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=True if child.bias is not None else False,
                w_bits=w_bit,
                a_bit=a_bit,
                weight_quantizer=copy.deepcopy(weight_quantizer),
                input_quantizer=copy.deepcopy(input_quantizer),
            )

            new_layer.weight = nn.Parameter(child.weight.detach().cpu())
            if child.bias is not None:
                new_layer.bias = nn.Parameter(child.bias.detach().cpu())
            setattr(module, name, new_layer)
        else:
            prepare_model(child, w_bit, a_bit, weight_quantizer, input_quantizer)