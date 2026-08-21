import torch
import torch.nn as nn


class MLP_5Layer_Tanh_Dropout(nn.Module):
    """5-layer FC MLP with Tanh activations and dropout (defense variant)."""

    HIDDEN = (2048, 1024, 512, 256, 100)

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.input_fc = nn.Linear(input_dim, self.HIDDEN[0])
        self.drop_input = nn.Dropout(dropout_rate)
        self.hidden_fc1 = nn.Linear(self.HIDDEN[0], self.HIDDEN[1])
        self.drop1 = nn.Dropout(dropout_rate)
        self.hidden_fc2 = nn.Linear(self.HIDDEN[1], self.HIDDEN[2])
        self.drop2 = nn.Dropout(dropout_rate)
        self.hidden_fc3 = nn.Linear(self.HIDDEN[2], self.HIDDEN[3])
        self.drop3 = nn.Dropout(dropout_rate)
        self.hidden_fc4 = nn.Linear(self.HIDDEN[3], self.HIDDEN[4])
        self.drop4 = nn.Dropout(dropout_rate)
        if num_classes != self.HIDDEN[-1]:
            self.output_fc = nn.Linear(self.HIDDEN[-1], num_classes)
            self.drop_out = nn.Dropout(dropout_rate)
        else:
            self.output_fc = None
            self.drop_out = None

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        x = torch.tanh(self.input_fc(x))
        x = self.drop_input(x)
        x = torch.tanh(self.hidden_fc1(x))
        x = self.drop1(x)
        x = torch.tanh(self.hidden_fc2(x))
        x = self.drop2(x)
        x = torch.tanh(self.hidden_fc3(x))
        x = self.drop3(x)
        x = self.hidden_fc4(x)
        x = self.drop4(x)
        if self.output_fc is not None:
            x = torch.tanh(x)
            x = self.drop_out(x)
            x = self.output_fc(x)
        return x
