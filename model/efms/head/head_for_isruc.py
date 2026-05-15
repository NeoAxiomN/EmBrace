import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class head_for_isruc(nn.Module):
    def __init__(self, num_classes, dropout):
        super(head_for_isruc, self).__init__()
        self.head = nn.Sequential(
            # nn.Linear(6*30*200, 512),
            nn.LazyLinear(512),
            # nn.Linear(6*30*200, 512),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=2048, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, feats):
        bz = feats.shape[0]//20
        epoch_features = feats.contiguous().view(bz, 20, -1)
        epoch_features = self.head(epoch_features)
        seq_features = self.sequence_encoder(epoch_features)
        return self.classifier(seq_features)
