from ..efm import BaseEEGFoundationModel
from .core import CodeBrain as Backbone
from .. import head

import torch
import torch.nn as nn


class CodeBrain(BaseEEGFoundationModel):
    def __init__(
        self, 
        dataset, 
        num_classes, 
        dropout, 
        use_pretrained_weights, 
        foundation_dir
    ):
        backbone = Backbone(
            in_channels = 200, 
            res_channels = 200,
            skip_channels = 200, 
            out_channels = 200,
            num_res_layers = 8,
            diffusion_step_embed_dim_in = 200,
            diffusion_step_embed_dim_mid = 200,
            diffusion_step_embed_dim_out = 200,
            s4_lmax = 570,
            s4_d_state = 64,
            s4_dropout = dropout,
            s4_bidirectional = True,
            s4_layernorm = True,
            codebook_size_t = 4096,
            codebook_size_f = 4096,
            if_codebook = False
        )
        if use_pretrained_weights:
            state_dict = torch.load(foundation_dir, map_location="cpu")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            backbone.load_state_dict(new_state_dict)
        backbone.proj_out = nn.Identity()

        classifier = head.create_head(dataset, num_classes, dropout)

        super(CodeBrain, self).__init__(
            backbone=backbone, classifier=classifier
        )
