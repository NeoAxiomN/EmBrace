import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchinfo import summary
import torch.nn.functional as F
from einops import rearrange


def _reshape_etm_input(x):
    # ISRUC collate already flattens sequence windows into the batch dimension.
    return rearrange(x, 'b c p d -> b c (p d)')


from .EEGNet import EEGNet
class Model_EEGNet(nn.Module):
    def __init__(self, cfg):
        super(Model_EEGNet, self).__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = EEGNet(
            nChan=cfg.dataset.chans,
            nTime=cfg.dataset.patch_num*200,
            nClass=cfg.dataset.num_classes,
            dropoutP=cfg.model.params.dropout
        )
    def forward(self, x):
        x = _reshape_etm_input(x)
        return self.backbone(x)
    

from .EEGDeformer import Deformer
class Model_EEGDeformer(nn.Module):
    def __init__(self, cfg):
        super(Model_EEGDeformer, self).__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = Deformer(num_chan=cfg.dataset.chans, num_time=cfg.dataset.patch_num*200, 
                                 temporal_kernel=11, num_kernel=cfg.dataset.chans*cfg.dataset.patch_num,
                                 num_classes=cfg.dataset.num_classes, depth=4, heads=16,
                                 mlp_dim=16, dim_head=16, dropout=cfg.model.params.dropout)
    def forward(self, x):
        x = _reshape_etm_input(x)
        return self.backbone(x)
    

from .Sera import SERA
from scipy.signal import resample
import numpy as np
class Model_SERA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = SERA(num_classes=cfg.dataset.num_classes, 
                          input_size=(1, cfg.dataset.chans, cfg.dataset.patch_num*200), 
                          sampling_rate=200, num_T=8, patch_size=20,
                          patch_stride=10, dropout_rate=cfg.model.params.dropout,
                          num_head=8, dimz=32)
    def forward(self, x):
        x = _reshape_etm_input(x)
        # x: (bz, ch, T)
        x = x.unsqueeze(1) # (bz, 1, ch, T)
        return self.backbone(x)


from .SPaRCNet import SPaRCNet
class Model_SPaRCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = SPaRCNet(
            in_channels=cfg.dataset.chans,
            sample_length=cfg.dataset.patch_num*200,
            n_classes=cfg.dataset.num_classes
        )
    def forward(self, x):
        x = _reshape_etm_input(x)
        return self.backbone(x)


from .ContraWR import ContraWR
class Model_ContraWR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = ContraWR(
            in_channels=cfg.dataset.chans,
            n_classes=cfg.dataset.num_classes,
            fft=200
        )
    def forward(self, x):
        x = _reshape_etm_input(x)
        return self.backbone(x)
    
    
from .CNNTransformer import CNNTransformer
class Model_CNNTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = CNNTransformer(
            in_channels=cfg.dataset.chans,
            n_classes=cfg.dataset.num_classes,
            fft=200,
        )

    def forward(self, x):
        x = _reshape_etm_input(x)
        if x.shape[-1] < 512:
            x = F.pad(x, (0, 512 - x.shape[-1]), "constant", 0)
        return self.backbone(x)


from .FFCL import FFCL
class Model_FFCL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = FFCL(
            in_channels=cfg.dataset.chans,
            n_classes=cfg.dataset.num_classes,
            sample_length=cfg.dataset.patch_num*200,
            fft=200,
        )
    def forward(self, x):
        x = _reshape_etm_input(x)
        return self.backbone(x)
    
from .STTransformer import STTransformer
class Model_STTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_name = cfg.dataset.name
        self.backbone = STTransformer(
            n_channels=cfg.dataset.chans,
            n_classes=cfg.dataset.num_classes,
            channel_legnth=cfg.dataset.patch_num*200,
        )
    def forward(self, x):
        x = _reshape_etm_input(x)
        return self.backbone(x)

