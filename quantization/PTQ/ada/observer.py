import torch
import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()
        # self.level = level

    # def update_range(self, input):
    #     raise NotImplementedError
    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        # input = input.dequantize()
        if self.level == "L":  # layer level (activation/weight)
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.level == "C":  # channel level (conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.level == "FC":  # channel level (fc_weight)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]
        # self.update_range(input)
        self.update_range(min_val.cpu(), max_val.cpu())

        return input
    

class EMAMinMaxObserver(ObserverBase):
    def __init__(self, out_channels, level, momentum=0.1):
        super(EMAMinMaxObserver, self).__init__()
        self.momentum = momentum
        self.level = level
        self.num_flag = 0
        self.out_channels = out_channels
        if self.level == 'L':
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        # min_val_cur = torch.min(input)
        # max_val_cur = torch.max(input)

        # min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
        # max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur

        # self.min_val.copy_(min_val)
        # self.max_val.copy_(max_val)
        if self.level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val.dequantize())
        self.max_val.copy_(max_val.dequantize())
    

class MinMaxObserver(ObserverBase):
    def __init__(self, out_channels, level):
        super(MinMaxObserver, self).__init__()
        self.out_channels = out_channels
        self.level = level
        self.num_flag = 0

        if self.level == 'L':
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class OmseObserver(ObserverBase):

    def __init__(self, out_channels, level):
        super(OmseObserver, self).__init__()

        self.out_channels = out_channels
        self.level = level
        self.num_flag = 0

        if self.level == 'L':
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)