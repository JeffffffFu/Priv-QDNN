import math
import copy
from functools import reduce
from typing import Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from scipy.optimize import fmin_cg

from . import base_quant_modules as quant, utils


def to_vector(v):
    if isinstance(v, tuple) or isinstance(v, list):
        # return v.cpu().numpy().reshape(-1)
        return np.concatenate([vv.cpu().numpy().reshape(-1) for vv in v])
    else:
        return v.cpu().numpy().reshape(-1)

def to_list(v, sizes, device):
    _v = v
    result = []
    for size in sizes:
        total = reduce(lambda a, b: a * b, size)
        result.append(_v[:total].reshape(size).float().to(device))
        _v = _v[total:]
    return tuple(result)

def hessian_vector_product(model, train_loader, v, p, device):
    # parameters = [p for p in model.parameters() if p.requires_grad]
    # if p_idx is not None:
    #     parameters = parameters[p_idx:p_idx + 1]

    # hvp = [torch.zeros_like(p, device=device) for p in parameters]
    hvp = torch.zeros_like(p, device=device)
    n_samples = len(train_loader.dataset)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        train_loss = F.cross_entropy(outputs, labels) / n_samples

        grad_mean = grad(train_loss, [p], create_graph=True)
        v_dot_grad = [(v.data * g).sum() for v, g in zip(v, grad_mean)]
        _hvp = grad(v_dot_grad, [p], retain_graph=False)
        hvp = [h + h_tot for h, h_tot in zip(_hvp, hvp)]

    # _, train_loss = _as_tuple(train_loss, "outputs of the user-provided function", "hvp")

    # with torch.enable_grad():
    #     jac = _autograd_grad(train_loss, parameters, create_graph=True)
    #     grad_jac = tuple(
    #         torch.zeros_like(p, requires_grad=True, device=device) for p in parameters
    #     )
    #     double_back = _autograd_grad(jac, parameters, grad_jac, create_graph=True)

    # grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=False)
    # hvp = _fill_in_zeros(grad_res, parameters, False, False, "double_back_trick")
    # hvp = _grad_postprocess(hvp, False)
    return hvp

def _mini_batch_hvp(x, **kwargs):
    model = kwargs['model']
    train_loader = kwargs['train_loader']
    damping = kwargs['damping']
    device = kwargs['device']
    size = kwargs['size']
    p = kwargs['p']
    # p_idx = kwargs['p_idx']
    # use_torch = kwargs['use_torch']

    x = to_list(x, [size], device)
    _hvp = hessian_vector_product(model, train_loader, x, p, device)
    return [(a + damping * b).view(-1) for a, b in zip(_hvp, x)]

def _get_fmin_loss_fn(v, **kwargs):
    device = kwargs.get('device', 'cpu')

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
        return obj.detach().cpu().numpy()
    
    return get_fmin_loss

def _get_fmin_grad_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        # return to_vector(hvp - v.view(-1))
        return (torch.cat(hvp, dim=0) - v).cpu().numpy()

    return get_fmin_grad

def _inverse_hvp_cg(model, train_loader, v, p, device, damping_factor):
    """
    Compute the inverse Hessian-vector product using conjugate gradient method.
    """
    inverse_hvp = []

    size = p.size()
    v = v.view(-1)
    fmin_loss_fn = _get_fmin_loss_fn(
        v, model=model, train_loader=train_loader, p=p,
        damping=damping_factor, size=size, device=device,
    )
    fmin_grad_fn = _get_fmin_grad_fn(
        v, model=model, train_loader=train_loader, p=p,
        damping=damping_factor, size=size, device=device,
    )
    res = fmin_cg(
        f=fmin_loss_fn,
        x0=to_vector(v),
        fprime=fmin_grad_fn,
        disp=False,
        full_output=True,
        maxiter=50,
    )
    inverse_hvp = to_list(torch.from_numpy(res[0]), [size], device)[0]
    return inverse_hvp

def get_qconfig_for_bitwidth(w_bit, a_bit):
    qmin = -2 ** (w_bit - 1)
    qmax = 2 ** (w_bit - 1) - 1

def inplace_quantize_layers(module,  bit_widths, device):
    for name, child in module.named_children():
        print(name, type(child))

        if isinstance(child, torch.nn.Conv2d):
            new_layer = quant.QuantConv2d(
                child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding,
                child.dilation, child.groups, True if child.bias is not None else False, n_bits=bit_widths,
                device=device
            )
            new_layer.weight = child.weight
            if child.bias is not None:
                new_layer.bias = child.bias

            module._modules[name] = new_layer

        elif isinstance(child, torch.nn.Linear):
            new_layer = quant.QuantLinear(
                child.in_features, child.out_features, True if child.bias is not None else False, n_bits=bit_widths,
                device=device
            )
            new_layer.weight = child.weight

            if child.bias is not None:
                new_layer.bias = child.bias

            module._modules[name] = new_layer
        inplace_quantize_layers(child,  bit_widths, device)

def Asymmetric_quant(fp_model,test_set,args,label_key=None):

    quant_model=copy.deepcopy(fp_model)

    inplace_quantize_layers(
        quant_model, args['w_bit'], args['device']
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    utils.enable_calibrate(quant_model)
    utils.calibrate(quant_model, test_loader,label_key, device=args['device'])
    utils.disable_calibrate(quant_model)

    return quant_model