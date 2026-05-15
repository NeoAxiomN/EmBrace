import torch.nn as nn
from einops.layers.torch import Rearrange


class head_for_seedv(nn.Module):
    def __init__(self, num_classes, dropout):
        super(head_for_seedv, self).__init__()
        self.classifier = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(62 * 1 * 200, 4 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, num_classes),
        )

    def forward(self, feats):
        return self.classifier(feats)
