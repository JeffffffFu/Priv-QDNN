import torch
import torch.nn as nn
import torch.nn.functional as F

from . import quantizer
from . import observer


class QuantLinear(nn.Linear):
    """
    Adaptive rounding quantization for Linear
    """
    def __init__(self, in_features, out_features, bias=True, w_bit=8, a_bit=8, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias=bias)

        weight_quantizer = kwargs.get('weight_quantizer', quantizer.SymmetricQuantizer(
            bit=w_bit,
            observer=observer.MinMaxObserver(out_features, level='L'),
            ptq=True, sign=True
        ))
        input_quantizer = kwargs.get('input_quantizer', quantizer.AsymmetricQuantizer(
            bit=a_bit,
            observer=observer.MinMaxObserver(out_features, level='L'), 
            ptq=True, sign=False
        ))

        self.weight_quantizer =  weight_quantizer
        self.input_quantizer = input_quantizer    
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
                 padding_mode='zeros', w_bit=8, a_bit=8, **kwargs):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups,
                                      bias=bias, padding_mode=padding_mode)

        weight_quantizer = kwargs.get('weight_quantizer', quantizer.SymmetricQuantizer(
            bit=w_bit,
            observer=observer.MinMaxObserver(out_channels, level='L'),
            ptq=True, sign=True
        ))
        input_quantizer = kwargs.get('input_quantizer', quantizer.AsymmetricQuantizer(
            bit=a_bit,
            observer=observer.MinMaxObserver(out_channels, level='L'), 
            ptq=True, sign=False
        ))        

        self.weight_quantizer = weight_quantizer
        self.input_quantizer = input_quantizer

    def forward(self, x):
        x = self.input_quantizer(x)
        weight_quant = self.weight_quantizer(self.weight)

        output = F.conv2d(x, weight_quant, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output