from . import labram, utils

__all__ = [
    "get_labram"
]

from timm.models import create_model
from collections import OrderedDict
import torch.nn as nn
import torch
import warnings
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from einops import rearrange

def load_state_dict(
    model, 
    state_dict, 
    prefix="", 
    ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


def get_labram(
    dataset, 
    num_classes, 
    use_pretrained_weights, 
    foundation_dir,
    drop_rate=0.0,
    drop_path_rate=0.1,
    attn_drop_rate=0.0,
    use_mean_pooling=True,
    init_scale=0.001,
    use_rel_pos_bias=False,
    use_abs_pos_emb=True,
    layer_scale_init_value=0.1,
    qkv_bias=False,
    show_model=False,
):
    model = create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        attn_drop_rate=attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=use_mean_pooling,
        init_scale=init_scale,
        use_rel_pos_bias=use_rel_pos_bias,
        use_abs_pos_emb=use_abs_pos_emb,
        init_values=layer_scale_init_value,
        qkv_bias=qkv_bias,
        dataset=dataset,
    )

    if show_model:
        print(model)

    if not use_pretrained_weights:
        return model

    try:
        checkpoint = torch.load(
            foundation_dir, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(foundation_dir, map_location="cpu")

    checkpoint_model = None
    model_key = "model|module"
    for model_key in model_key.split("|"):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    if checkpoint_model is not None:
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith("student."):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                pass
        checkpoint_model = new_dict

    if show_model:
        for key in checkpoint_model.keys():
            print(f"\t{key}")

    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    load_state_dict(model, checkpoint_model, prefix="")
    return model


