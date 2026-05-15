import torch.nn as nn
from einops.layers.torch import Rearrange


class head_for_faced(nn.Module):
    def __init__(self, num_classes, dropout):
        super(head_for_faced, self).__init__()
        self.classifier = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            # nn.Linear(32 * 10 * 200, 10 * 200),
            nn.LazyLinear(10 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, num_classes),
        )

    def forward(self, feats):
        return self.classifier(feats)
