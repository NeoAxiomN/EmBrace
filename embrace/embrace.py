from .ke import KnowledgeExtractor
from .component import Bridge, Fusor

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import re

def natural_sort_key(s):
    # 将字符串中的数字部分转为整数进行比较
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class ModelProtocol:
    @staticmethod
    def _base_attn(extractor, prefix, fused_layers, softmax=False):
        keys = sorted([k for k in extractor.knowledge_attn.keys() 
                       if k.startswith(f"{prefix}_attn")], key=natural_sort_key)

        # print(f"\n[Debug] Base Attention Keys ({prefix}):")
        # for i, k in enumerate(keys):
        #     print(f"  Layer {i}: {k}")
        if not keys: return None
        
        attn_list = [extractor.knowledge_attn[k] for k in keys]
        attn = torch.stack(attn_list, dim=1) 
        attn = torch.mean(attn, dim=2) 
        attn = attn[:, -fused_layers:]
        if softmax:
            attn = F.softmax(attn, dim=-1)
        return attn

    @staticmethod
    def _align_cbramod(extractor, prefix, fused_layers, ch_num, patch_num, bz, mode='dual', softmax=False):
        """mode: spatial, dual"""
        C, P = ch_num, patch_num
        s_keys = sorted([k for k in extractor.knowledge_attn.keys() 
                         if f"{prefix}_attn" in k and "self_attn_s" in k], 
                        key=natural_sort_key)
        t_keys = sorted([k for k in extractor.knowledge_attn.keys() 
                         if f"{prefix}_attn" in k and "self_attn_t" in k],
                         key=natural_sort_key)
        
        # print(f"\n[Debug] CBRAMod Keys ({prefix}):")
        # print(f"spatial Keys (S): {s_keys}")
        # print(f"\n[Debug] CBRAMod Keys ({prefix}):")
        # print(f"  Spatial Keys (S): {s_keys}")
        # if t_keys:
        #     print(f"  Temporal Keys (T): {t_keys}")
        if not s_keys: return None

        device = extractor.knowledge_attn[s_keys[0]].device
        L_total = C * P
        
        global_attn = torch.zeros((bz, fused_layers, L_total, L_total), device=device)

        s_attn_raw = torch.mean(torch.stack([extractor.knowledge_attn[k] for k in s_keys], dim=1), dim=2)
        s_attn = s_attn_raw[:, -fused_layers:]
        block_s = rearrange(s_attn, "(b p) l c1 c2 -> b l p c1 c2", b=bz, p=P)
        
        for p in range(P):
            indices = torch.arange(C, device=device) * P + p
            global_attn[:, :, indices.unsqueeze(1), indices] += block_s[:, :, p]

        if mode == 'dual' and t_keys:
            t_attn_raw = torch.mean(torch.stack([extractor.knowledge_attn[k] for k in t_keys], dim=1), dim=2)
            t_attn = t_attn_raw[:, -fused_layers:]
            block_t = rearrange(t_attn, "(b c) l p1 p2 -> b l c p1 p2", b=bz, c=C)
            
            for c in range(C):
                indices = torch.arange(P, device=device) + c * P
                global_attn[:, :, indices.unsqueeze(1), indices] += block_t[:, :, c]

        if softmax:
            global_attn = F.softmax(global_attn, dim=-1)
        return global_attn
    

    @staticmethod
    def _align_csbrain(extractor, prefix, fused_layers, ch_num, patch_num, bz, dataset, mode='dual', softmax=False):
        C, P = ch_num, patch_num
        win_keys = sorted([k for k in extractor.knowledge_attn.keys() 
                           if f"{prefix}_attn_window" in k], key=natural_sort_key)
        reg_keys = sorted([k for k in extractor.knowledge_attn.keys() 
                           if f"{prefix}_attn_region" in k], key=natural_sort_key)
        if not win_keys or not reg_keys: return None

        
        reg_attn_raw = torch.mean(torch.stack([extractor.knowledge_attn[k] for k in reg_keys], dim=1), dim=2)[:, -fused_layers:].contiguous()


        device = reg_attn_raw.device
        L_total = C * P

        from model.efms.csbrain.utils import load_dataset_configs, get_csbrain_config
        configs = load_dataset_configs()
        _, _, keep_indices = get_csbrain_config(dataset, configs)
        keep_indices = torch.tensor(keep_indices, device=device)

        
        
        reg_attn = rearrange(reg_attn_raw, '(b p) l c1 c2 -> b l p c1 c2', b=bz, p=P)
        aligned_attn = torch.zeros((bz, fused_layers, L_total, L_total), device=device, dtype=reg_attn_raw.dtype)
        
        for p in range(P):
            idx = p + keep_indices.to(device) * P
            aligned_attn[:, :, idx.unsqueeze(1), idx] = reg_attn[:, :, p]


        if mode == 'dual':
            win_attn_raw = torch.mean(torch.stack([extractor.knowledge_attn[k] for k in win_keys], dim=1), dim=2)[:, -fused_layers:].contiguous()
            window_size = min(P, 5)
            num_windows = P // window_size
            C_real = len(keep_indices)
            win_attn = rearrange(win_attn_raw, '(b ws c) l w1 w2 -> b ws c l w1 w2', b=bz, ws=window_size, c=C_real)
            
            delta_attn = torch.zeros_like(aligned_attn)
            for ws in range(window_size):
                for c_idx, real_ch_idx in enumerate(keep_indices.to(device)):
                    idx = (torch.arange(num_windows, device=device) * window_size + ws) + real_ch_idx * P
                    delta_attn[:, :, idx.unsqueeze(1), idx] = win_attn[:, ws, c_idx]
            attn = (aligned_attn + delta_attn).contiguous()
        else:
            attn = aligned_attn.contiguous()

        if softmax:
            attn = F.softmax(attn, dim=-1)
        return attn

    
    @staticmethod
    def _align_emb(emb, dataset, ch_num, patch_num, bz):
        target_len = ch_num * patch_num   
        bz_curr, current_len, D = emb.shape

        if current_len == target_len:
            return emb.contiguous()

        from model.efms.csbrain.utils import load_dataset_configs, get_csbrain_config
        configs = load_dataset_configs()
        _, _, keep_indices = get_csbrain_config(dataset, configs)
        
        device = emb.device
        dtype = emb.dtype
        aligned_emb = torch.zeros(bz_curr, target_len, D, device=device, dtype=dtype)

        full_indices = []
        for ch_idx in keep_indices:
            start = ch_idx * patch_num
            full_indices.extend(range(start, start + patch_num))
        
        full_indices = torch.tensor(full_indices, device=device)
        aligned_emb[:, full_indices, :] = emb
        
        return aligned_emb.contiguous()

    REGISTRY = {
        "labram": {
            "get_emb": lambda ex, p, b, cfg, **kw: b(ex.knowledge_emb[f"{p}_emb"].contiguous().view(kw['bz'], -1, 200)),
            "get_attn": lambda ex, p, cfg, **kw: ModelProtocol._base_attn(ex, p, cfg['fused_layers'], cfg['softmax'])[:, :, 1:, 1:],
        },
        "cbramod": {
            "get_emb": lambda ex, p, b, cfg, **kw: b(ex.knowledge_emb[f"{p}_emb"].contiguous().view(kw['bz'], -1, 200)),
            "get_attn": lambda ex, p, cfg, **kw: ModelProtocol._align_cbramod(
                ex, p, cfg['fused_layers'], cfg['ch_num'], cfg['patch_num'], kw['bz'], cfg['cbramod_mode'], cfg['softmax']
            ),
        },
        "codebrain": {
            "get_emb": lambda ex, p, b, cfg, **kw: b(ex.knowledge_emb[f"{p}_emb"].contiguous().view(kw['bz'], -1, 200)),
            "get_attn": lambda ex, p, cfg, **kw: ModelProtocol._base_attn(ex, p, cfg['fused_layers'], cfg['softmax']),
        },
        "csbrain": {
            "get_emb": lambda ex, p, b, cfg, **kw: b(ModelProtocol._align_emb(
                ex.knowledge_emb[f"{p}_emb"].contiguous().view(kw['bz'], -1, 200),
                cfg['dataset'], cfg['ch_num'], cfg['patch_num'], kw['bz']).contiguous()
            ),
            "get_attn": lambda ex, p, cfg, **kw: ModelProtocol._align_csbrain(
                ex, p, cfg['fused_layers'], cfg['ch_num'], cfg['patch_num'], kw['bz'], cfg['dataset'], cfg['csbrain_mode'], cfg['softmax']
            ),
        }
    }

class EmBrace(nn.Module):
    def __init__(
        self, 
        target_name,
        model_target,        # model_obj
        model_sources,       # Dict: {name: model_obj}
        dataset,
        chans,
        patch_num,
        fused_attn_layers=1,
        cbramod_mode="spatial",
        csbrain_mode='spatial',
        attn_softmax=False,
        # params for Bridge 
        input_dim=200,       
        proj_dim=256,        
        bridge_type="mlp",
        use_shared_bridge=True,
        use_unified_bridge=False,
        # params for Fusor 
        coupled_weight=True,
        fusor_temp=8.0,
        fusor_mode="mean_first",
        using_meta_features=True,
    ):
        super().__init__()
        self.model_target = model_target
        self.target_name = target_name.lower()
        self.model_sources = nn.ModuleDict(model_sources)
        self.coupled_weight = coupled_weight
        
        self.protocol_cfg = {
            'ch_num': chans,
            'patch_num': patch_num,
            'fused_layers': fused_attn_layers,
            'cbramod_mode': cbramod_mode,
            'csbrain_mode': csbrain_mode,
            'dataset': dataset,
            'softmax': attn_softmax,
        }
        
        self.ke = KnowledgeExtractor() 
        
        n_src = len(model_sources)
        bridge_kwargs = {
            "input_dim": input_dim, 
            "output_dim": proj_dim, 
            "bridge_type": bridge_type,
        }
        fusor_kwargs = {
            "chans": chans,
            "n_src": n_src,
            "temp": fusor_temp,
            "mode": fusor_mode,
            "using_meta_features": using_meta_features
        }
        
        shared = Bridge(**bridge_kwargs) if use_shared_bridge else None

        self.source_bridges = nn.ModuleDict()
        for name, m in self.model_sources.items():
            m.eval()
            for p in m.parameters(): p.requires_grad = False
            self.source_bridges[name] = shared or Bridge(**bridge_kwargs)
            self.ke.register_efm(m, model_prefix=name, detach=True) 

        self.target_bridge = (use_unified_bridge and shared) or Bridge(**bridge_kwargs)
        self.ke.register_efm(self.model_target.backbone, model_prefix=self.target_name, detach=False)

        if coupled_weight:
            self.fusor = Fusor(**fusor_kwargs)
        else:
            self.fusor_attn = Fusor(**fusor_kwargs)
            self.fusor_emb = Fusor(**fusor_kwargs)

    def forward(self, x, meta):
        bz = x.shape[0]
        self.ke.flush() 
        if meta is None:
            raise RuntimeError("EmBrace fusion requires meta features, but the current batch did not provide them.")

        k_sources = []
        with torch.no_grad():
            for name, m in self.model_sources.items():
                _ = m(x) 
                proto = ModelProtocol.REGISTRY[name.lower()]
                source_emb = proto["get_emb"](self.ke, name, self.source_bridges[name], self.protocol_cfg, bz=bz)
                source_attn = proto["get_attn"](self.ke, name, self.protocol_cfg, bz=bz)
                self._require_protocol_tensor(name, "emb", source_emb)
                self._require_protocol_tensor(name, "attn", source_attn)
                k_sources.append({
                    "emb": source_emb,
                    "attn": source_attn,
                })

        target_logits = self.model_target(x) 
        t_proto = ModelProtocol.REGISTRY[self.target_name]
        target_emb = t_proto["get_emb"](self.ke, self.target_name, self.target_bridge, self.protocol_cfg, bz=bz)
        target_attn = t_proto["get_attn"](self.ke, self.target_name, self.protocol_cfg, bz=bz)
        self._require_protocol_tensor(self.target_name, "emb", target_emb)
        self._require_protocol_tensor(self.target_name, "attn", target_attn)
        
        target_data = {
            "logits": target_logits,
            "emb": target_emb,
            "attn": target_attn,
        }
       

        if self.coupled_weight:
            w = self.fusor(bz, meta)
            w_emb, w_attn = w, w
        else:
            w_attn = self.fusor_attn(bz, meta)
            w_emb = self.fusor_emb(bz, meta)

        k_emb = sum(w.view(bz, 1, 1) * s["emb"] 
                    for w, s in zip(w_emb.unbind(1), k_sources))
        k_attn = sum(w.view(bz, 1, 1, 1) * s["attn"] 
                     for w, s in zip(w_attn.unbind(1), k_sources))
        
        assert k_emb.shape == target_data["emb"].shape, f"Shape mismatch: {k_emb.shape} vs {target_data['emb'].shape}"
        assert k_attn.shape == target_data["attn"].shape, f"Shape mismatch: {k_attn.shape} vs {target_data['attn'].shape}"


        return {    
            "logits": target_data["logits"], 
            "t_emb": target_data["emb"], 
            "t_attn": target_data["attn"],   
            "k_emb": k_emb, 
            "k_attn": k_attn, 
            "w_emb": w_emb,
            "w_attn": w_attn, 
        }

    def _require_protocol_tensor(self, model_name, field_name, value):
        if value is not None:
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        raise RuntimeError(
            f"[Rank {rank}] Missing extracted `{field_name}` for model `{model_name}`. "
            f"Available attention keys: {sorted(self.ke.knowledge_attn.keys())}. "
            f"Available embedding keys: {sorted(self.ke.knowledge_emb.keys())}."
        )


class EmBraceTask(nn.Module):
    MODEL_MAP = {
        "cbramod": "CBraMod",
        "codebrain": "CodeBrain",
        "labram": "LaBraM",
        "csbrain": "CSBrain"
    }
    def __init__(
        self, 
        target_name,   # e.g. "codebrain"
        source_names,  # list, e.g. ["labram", "cbramod"]
        dataset,
        chans,
        patch_num,
        dataset_name,
        num_classes,
        foundation_dir_dict,
        dropout,
        fusor_temp=8.0,
        cbramod_mode='spatial',   
        csbrain_mode='spatial', 
        attn_softmax=False,
        proj_dim=256,        
        bridge_type="mlp",
        use_shared_bridge=True,
        use_unified_bridge=False,
        coupled_weight=True,
        fusor_mode="mean_first",
        weight_ce=0.6,
        weight_emb=0.4,
        weight_attn=0.4,
    ):
        super().__init__()
        from model import efms 
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_emb = weight_emb
        self.weight_attn = weight_attn

        def get_model_class(name):
            cls_name = self.MODEL_MAP.get(name.lower(), name.capitalize())
            return getattr(efms, cls_name)

        TargetClass = get_model_class(target_name)
        model_target = TargetClass(
            dataset=dataset_name,
            num_classes=num_classes,
            dropout=dropout,
            foundation_dir=foundation_dir_dict[target_name],
            use_pretrained_weights=True
        )

        model_sources = {}
        for s_name in source_names:
            SourceClass = get_model_class(s_name)
            full_source = SourceClass(
                dataset=dataset_name,
                num_classes=num_classes,
                dropout=dropout,
                foundation_dir=foundation_dir_dict[s_name],
                use_pretrained_weights=True
            )
            model_sources[s_name] = full_source.backbone
            del full_source.classifier


        self.embrace = EmBrace(
            target_name=target_name,
            model_target=model_target,
            model_sources=model_sources,
            dataset=dataset,
            chans=chans,
            patch_num=patch_num,
            fusor_temp=fusor_temp,
            cbramod_mode=cbramod_mode,
            csbrain_mode=csbrain_mode,
            attn_softmax=attn_softmax,
            proj_dim=proj_dim,
            bridge_type=bridge_type,
            use_shared_bridge=use_shared_bridge,
            use_unified_bridge=use_unified_bridge,
            coupled_weight=coupled_weight,
            fusor_mode=fusor_mode,
        )

    def forward(self, x, meta=None):
        if self.training:
            return self.embrace(x, meta)
        
        logits = self.embrace.model_target(x)
        return {
            "logits": logits
        }


    
