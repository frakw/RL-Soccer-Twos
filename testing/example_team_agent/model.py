import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPOCA(nn.Module):
    def __init__(self):
        super(MAPOCA, self).__init__()
        self.body_encoder = nn.Sequential(
            nn.Linear(336, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.branches = nn.ModuleList([
            nn.Linear(512, 3),
            nn.Linear(512, 3),
            nn.Linear(512, 3)
        ])

    def forward(self, x):
        x = self.body_encoder(x)
        outputs = [branch(x) for branch in self.branches]
        return outputs
