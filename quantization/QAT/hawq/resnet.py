import torch
import torch.nn as nn
import torch.ao.quantization as quantization


class QuantizedBasicBlock(nn.Module):

    def __init__(self, block):
        super(QuantizedBasicBlock, self).__init__()

        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample

        self.stride = block.stride

        self.dequant = quantization.DeQuantStub()
        self.quant = quantization.QuantStub()
        self.add_relu = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = out.dequantize()
        # identity = identity.dequantize()
        # out += identity
        # out = self.add_relu.add_relu(out, identity)
        out = torch.ops.quantized.add(out, identity, scale=out.q_scale(), zero_point=out.q_zero_point())

        # print('out', out.shape)
        # out = torch.quantize_per_tensor(out, scale=1.0/128, zero_point=0, dtype=torch.qint8)
        # print('out', out[0])
        out = self.relu(out)

        return out