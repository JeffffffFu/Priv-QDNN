"""
WhiteBox MIA .

 Confidence Attack :
  - P(true_class):  softmax  ( membership )
  - Gradient L2 Norm:  L2  ()

 MLP  2  (member / non-member).
"""

import torch.nn as nn


class WhiteBoxAttackModel(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 2),
        )

    def forward(self, features):
        return self.net(features)
