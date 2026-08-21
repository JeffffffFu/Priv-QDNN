import torch
import torch.nn as nn
import torch.nn.functional as F

from . import quantizer
from . import observer


class QuantLinear(nn.Linear):
    """
    Adaptive rounding quantization for Linear
    """
    def __init__(self, in_features, out_features, bias=True, n_bits=8, sign=False, bias_correction=False, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias=bias)
        self.bias_correction = bias_correction

        device = kwargs.get('device', 'cpu')
        self.weight_quantizer = quantizer.AsymmetricQuantizer(
            bit=n_bits,
            observer=observer.EMAMinMaxObserver(out_features, level='L', device=device),
            ptq=True, sign=False, device=device
        )
        self.input_quantizer = quantizer.AsymmetricQuantizer(
            bit=n_bits,
            observer=observer.EMAMinMaxObserver(out_features, level='L', device=device), 
            ptq=True, sign=False, device=device
        )
    
    def forward(self, x):
        x = self.input_quantizer(x)
        weight_quant = self.weight_quantizer(self.weight)

        output = F.linear(x, weight_quant, self.bias)
        return output


class QuantConv2d(nn.Conv2d):
    """
    Adaptive rounding quantization for Conv2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', n_bits=8, sign=False, bias_correction=True, **kwargs):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups,
                                      bias=bias, padding_mode=padding_mode)
        self.n_bits = n_bits
        self.bias_correction = bias_correction

        device = kwargs.get('device', 'cpu')

        self.weight_quantizer = kwargs.get('weight_quantizer', quantizer.AsymmetricQuantizer(
            bit=n_bits,
            observer=observer.EMAMinMaxObserver(out_channels, level='L', device=device),
            ptq=True, sign=False, device=device
        ))

        self.input_quantizer = kwargs.get('input_quantizer', quantizer.AsymmetricQuantizer(
            bit=n_bits,
            observer=observer.EMAMinMaxObserver(out_channels, level='L', device=device), 
            ptq=True, sign=False, device=device
        ))

    def forward(self, x):
        x = self.input_quantizer(x)
        weight_quant = self.weight_quantizer(self.weight)
        output = F.conv2d(x, weight_quant, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
