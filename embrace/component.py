import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class Bridge(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        bridge_type="mlp", 
        norm=False, 
        do_normalize=False
    ):
        super().__init__()
        self.norm = norm
        self.do_normalize = do_normalize

        bridge = {
            "mlp": lambda: nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.LayerNorm(input_dim * 2),
                nn.GELU(),
                nn.Linear(input_dim * 2, output_dim),
            ),
            "linear": lambda: nn.Linear(input_dim, output_dim),
            "identity": lambda: nn.Identity(),
        }
        assert bridge_type in bridge, f"Invalid bridge_type: {bridge_type}"
        self.proj = bridge[bridge_type]()
        self.ln = nn.LayerNorm(output_dim) if norm else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(self.proj(x))
        return F.normalize(x, p=2, dim=-1) if self.do_normalize else x


class Fusor(nn.Module):
    def __init__(
        self, 
        chans,
        n_src,
        temp=1.0,
        topk=5,
        using_meta_features=True,
        learnable_weight=False,
        mode="mean_first",
        norm_meta=False,
        dropout=0.5,
    ):
        super().__init__()
        self.chans = chans
        self.topk = topk
        self.temp = temp
        self.mode = mode
        self.learnable_weight = learnable_weight
        self.norm_meta = norm_meta
        self.dropout = dropout

        if using_meta_features:
            self._setup_dynamic_mode(n_src)
            assert (mode in self.strategies), f"Unsupported mode: {mode}"
            self.executor = self.strategies[mode]

        else:
            initial_w = torch.full((n_src,), 1.0 / n_src)
            if learnable_weight:
                self.weight = nn.Parameter(initial_w)
            else:
                self.register_buffer("weight", initial_w)
            self.executor = self._static_forward

        self.apply(self._init_weights)

    def _setup_dynamic_mode(self, n_src, input_dim=10):
        if self.mode == "linear":
            self.fc = nn.Sequential(
                nn.Linear(self.chans * input_dim, input_dim),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(input_dim, n_src),
            )
        else:
            self.fc = nn.Linear(input_dim, n_src)


        self.strategies = {
            "mean_first": self._mean_first,
            "linear": self._linear,
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.constant_(m.weight, 0)
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0)
            
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # nn.init.kaiming_uniform_(m.weight, a=1.0)

    def forward(self, bz, meta):
        if meta.dim() == 4:  # for isruc
            meta = rearrange(meta, "b s c d -> (b s) c d")

        out = self.executor(meta)
        return out.expand(bz, -1) if out.dim() == 1 else out

    def _static_forward(self, _):
        return (
            F.softmax(self.weight / self.temp, dim=-1)
            if self.learnable_weight
            else self.weight
        )

    def _mean_first(self, x):
        p = x.mean(dim=1)
        if self.norm_meta:
            p = F.layer_norm(p, p.shape[1:])
        return F.softmax(self.fc(p) / self.temp, dim=-1)

    def _linear(self, x):
        return F.softmax(self.fc(rearrange(x, "b c d -> b (c d)")) / self.temp, dim=-1)

