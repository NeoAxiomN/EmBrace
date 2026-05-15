import torch.nn as nn
from einops.layers.torch import Rearrange


class head_for_attention(nn.Module):
    def __init__(self, num_classes, dropout):
        super(head_for_attention, self).__init__()
        self.classifier = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(28 * 4 * 200, 4 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 1),
            Rearrange('b 1 -> (b 1)'),
        )

    def forward(self, feats):
        return self.classifier(feats)
