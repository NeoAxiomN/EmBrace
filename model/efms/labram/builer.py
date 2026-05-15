from ..efm import BaseEEGFoundationModel
from .core import get_labram as Backbone
from .. import head
from .utils import get_input_chans_for_labram


import torch
import torch.nn as nn


class LaBraM(BaseEEGFoundationModel):
    def __init__(
            self, 
            dataset, 
            num_classes, 
            dropout, 
            use_pretrained_weights, 
            foundation_dir
        ):
        self.dataset = dataset

        backbone = Backbone(
            dataset=dataset,
            num_classes=num_classes,
            use_pretrained_weights=use_pretrained_weights,
            foundation_dir=foundation_dir,
            drop_path_rate=dropout
        )
        
        if dataset in ["isruc-s1", "isruc-s3"]:
            classifier = head.head_for_isruc(num_classes, dropout)
        else:
            classifier = backbone.head

        super(LaBraM, self).__init__(
            backbone=backbone, classifier=classifier
        )

        self.input_chans = get_input_chans_for_labram(dataset)

    def forward(self, x, **kwargs):
        feats = self.backbone.forward_features(
            x, 
            input_chans=self.input_chans, 
            return_all_tokens=True if self.dataset in ["isruc-s1", "isruc-s3"] else False
        )
        return self.classifier(feats)

