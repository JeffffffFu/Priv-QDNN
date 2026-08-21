import copy
import torch
import torch.nn as nn
from tqdm import tqdm

from . import utils
from . import quantizer
from . import observer
from .base_quant_modules import QuantConv2d, QuantLinear

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
def calibrate_adaround(module, adaround_iter, b_start, b_end, warmup, trainloader, device, label_key=None):
    torch.autograd.set_detect_anomaly(True)
    opt_params = []
    for name, child in module.named_modules():
        if isinstance(child, quantizer.AdaRoundQuantizer):
            # Only add alpha if it's not None (has been initialized)
            if child.alpha is not None:
                opt_params += [child.alpha]
            # print(opt_params)
    if len(opt_params) == 0:
        print("Warning: No optimizable parameters found for AdaRound calibration")
        return module
    optimizer = torch.optim.Adam(opt_params)
    scheduler = None

    temp_decay = LinearTempDecay(adaround_iter, rel_start_decay=warmup,
                                          start_b=b_start, end_b=b_end)
    for j in tqdm(range(adaround_iter)):
        b = temp_decay(j)
        for batch_idx, batch in enumerate(trainloader):
            if batch_idx == 10: break
            inputs, _ = utils._extract_inputs_targets(batch, device=device, label_key=label_key)
            if inputs is None:
                continue
            outputs = module(inputs)
            round_loss = 0
            for name, child in module.named_modules():
                if isinstance(child, quantizer.AdaRoundQuantizer):
                    # Only add to loss if alpha has been initialized
                    if child.alpha is not None:
                        round_vals = child.get_soft_targets()
                        round_loss += (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

            optimizer.zero_grad()
            round_loss.backward()
            optimizer.step()   
    for name, child in module.named_modules():
        if isinstance(child, quantizer.AdaRoundQuantizer):
            child.soft_targets = False

def adaround(model,test_set,args,label_key=None):
    quant_model = copy.deepcopy(model)
    device=args['device']
    quant_model.to(device)
    w_bit=args['w_bit']
    a_bit=args['a_bit']

    train_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)

    weight_quantizer = quantizer.AdaRoundQuantizer(
        bit=w_bit, observer=observer.MinMaxObserver(None, level='L'), ptq=True, sign=True
    )
    input_quantizer = quantizer.AsymmetricQuantizer(
        bit=a_bit, observer=observer.MinMaxObserver(None, level='L'), ptq=True, sign=False
    )

    utils.enable_calibrate(quant_model)
    utils.prepare_model(
        quant_model, w_bit, a_bit, weight_quantizer=weight_quantizer, input_quantizer=input_quantizer,
    )
    utils.calibrate(quant_model, train_loader, device=device, label_key=label_key)
    utils.disable_calibrate(quant_model)
    
    # Run one forward pass to initialize all alpha parameters
    quant_model.train()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 1:  # Just need one batch
                break
            inputs, _ = utils._extract_inputs_targets(batch, device=device, label_key=label_key)
            if inputs is not None:
                _ = quant_model(inputs)
            break
    
   # calibrate_adaround(quant_model, 10, 10, 4, 0.2, train_loader, device, label_key=label_key)
    calibrate_adaround(quant_model, 10, 10, 4, 0.2, train_loader, device, label_key=label_key)

    return quant_model


def dequantize(module):

    for name, child in module.named_children():
        if isinstance(child, QuantConv2d):
            new_conv = nn.Conv2d(
                child.in_channels, child.out_channels, child.kernel_size, child.stride,
                child.padding, child.dilation, child.groups, True if child.bias is not None else False,
            )
            new_conv.weight = nn.Parameter(child.weight_quantizer(child.weight).detach().cpu())
            if child.bias is not None:
                new_conv.bias = child.bias
            module._modules[name] = new_conv
        elif isinstance(child, QuantLinear):
            new_linear = nn.Linear(
                child.in_features, child.out_features, 
                True if child.bias is not None else False,
            )
            new_linear.weight = nn.Parameter(child.weight_quantizer(child.weight).detach().cpu())
            if child.bias is not None:
                new_linear.bias = child.bias
            module._modules[name] = new_linear
        else:
            dequantize(child)