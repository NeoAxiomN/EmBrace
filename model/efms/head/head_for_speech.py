import torch.nn as nn
from einops.layers.torch import Rearrange


class head_for_speech(nn.Module):
    def __init__(self, num_classes, dropout):
        super(head_for_speech, self).__init__()
        self.classifier = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(64 * 3 * 200, 3 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(3 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, num_classes),
        )

    def forward(self, feats):
        return self.classifier(feats)
