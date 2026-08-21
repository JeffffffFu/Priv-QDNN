import torch
import torch.nn as nn


class MLP_5Layer_Tanh(nn.Module):
    """5-layer FC MLP with Tanh activations: [2048, 1024, 512, 256, 100]."""

    HIDDEN = (2048, 1024, 512, 256, 100)

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, self.HIDDEN[0])
        self.hidden_fc1 = nn.Linear(self.HIDDEN[0], self.HIDDEN[1])
        self.hidden_fc2 = nn.Linear(self.HIDDEN[1], self.HIDDEN[2])
        self.hidden_fc3 = nn.Linear(self.HIDDEN[2], self.HIDDEN[3])
        self.hidden_fc4 = nn.Linear(self.HIDDEN[3], self.HIDDEN[4])
        if num_classes != self.HIDDEN[-1]:
            self.output_fc = nn.Linear(self.HIDDEN[-1], num_classes)
        else:
            self.output_fc = None

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        x = torch.tanh(self.input_fc(x))
        x = torch.tanh(self.hidden_fc1(x))
        x = torch.tanh(self.hidden_fc2(x))
        x = torch.tanh(self.hidden_fc3(x))
        x = self.hidden_fc4(x)
        if self.output_fc is not None:
            x = torch.tanh(x)
            x = self.output_fc(x)
        return x
