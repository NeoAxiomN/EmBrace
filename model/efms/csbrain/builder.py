from ..efm import BaseEEGFoundationModel
from .core import CSBrain as Backbone
from .. import head
from .utils import load_dataset_configs, get_csbrain_config

import torch
import torch.nn as nn


class CSBrain(BaseEEGFoundationModel):
    def __init__(
            self, 
            dataset, 
            num_classes, 
            dropout, 
            use_pretrained_weights, 
            foundation_dir
        ):
        self.dataset = dataset

        configs = load_dataset_configs()
        regions, sorted_indices, keep_indices = get_csbrain_config(dataset, configs)
        self.keep_indices = keep_indices

        backbone = Backbone(
            in_dim=200, 
            out_dim=200, 
            d_model=200,
            dim_feedforward=800, 
            seq_len=30,
            n_layer=12, 
            nhead=8,
            brain_regions=regions,
            sorted_indices=sorted_indices
        )
        if use_pretrained_weights:
            state_dict = torch.load(foundation_dir, map_location="cpu")
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value

            model_state_dict = backbone.state_dict()

            # Filter matching weights by shape
            matching_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            model_state_dict.update(matching_dict)
            backbone.load_state_dict(model_state_dict)
        backbone.proj_out = nn.Identity()

        classifier = head.create_head(dataset, num_classes, dropout)

        super(CSBrain, self).__init__(
            backbone=backbone, classifier=classifier
        )


    def forward(self, x):
        if self.dataset not in ["isruc-s1", "isruc-s3"]:
            x = x[:, self.keep_indices, :, :]
        return super().forward(x)

