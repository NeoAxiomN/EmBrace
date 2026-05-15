from ..efm import BaseEEGFoundationModel
from .core import CBraMod as Backbone
from .. import head

import torch
import torch.nn as nn


class CBraMod(BaseEEGFoundationModel):
    def __init__(
        self, 
        dataset, 
        num_classes, 
        dropout, 
        use_pretrained_weights, 
        foundation_dir
    ):
        backbone = Backbone(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800, 
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        if use_pretrained_weights:
            backbone.load_state_dict(
                torch.load(foundation_dir, map_location="cpu")
            )
        backbone.proj_out = nn.Identity()

        classifier = head.create_head(dataset, num_classes, dropout)

        super(CBraMod, self).__init__(
            backbone=backbone, classifier=classifier
        )
