import torch.nn as nn
from abc import ABC, abstractmethod

class BaseEEGFoundationModel(ABC, nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, **kwargs):
        feats = self.backbone(x)
        return self.classifier(feats)