from .config import ConfigComposeRequest
from .utils import (
    assign_nested,
    cfg_get,
    normalize_int_list,
    normalize_string_list,
)

from argparse import Namespace


class LaunchArgumentAdapter:
    def __init__(self, args: Namespace):
        self.args = args

    def build_compose_request(self):
        return ConfigComposeRequest(
            task=self.args.command,
            base_config_path=getattr(self.args, "config", None),
            dataset=getattr(self.args, "dataset", None),
            dataset_config_path=getattr(self.args, "dataset_config", None),
            model=getattr(self.args, "model", None),
            direct_overrides=self.build_direct_overrides(),
        )

    def build_direct_overrides(self, task_name=None):
        payload = {}
        self._apply_common_overrides(payload)
        self._apply_dataset_overrides(payload)
        self._apply_runtime_overrides(payload)
        self._apply_task_overrides(payload, task_name=task_name)
        return payload or None

    def resolve_seeds(self, cfg):
        if getattr(self.args, "seeds", None) is not None:
            return normalize_int_list(self.args.seeds)
        if getattr(self.args, "seed", None) is not None:
            return [int(self.args.seed)]

        cfg_seed = cfg_get(cfg, "seed", 42)
        if cfg_seed is None:
            raise ValueError("No seed configured. Use `--seed`, `--seeds`, or set `seed` in YAML.")
        return [int(cfg_seed)]

    def _apply_common_overrides(self, payload):
        assign_nested(payload, getattr(self.args, "device", None), "device")
        assign_nested(payload, getattr(self.args, "output_base_dir", None), "output", "base_dir")
        assign_nested(payload, getattr(self.args, "model_folder_id", None), "model", "model_folder_id")

    def _apply_dataset_overrides(self, payload):
        assign_nested(payload, getattr(self.args, "dataset_path", None), "dataset", "path")
        assign_nested(payload, getattr(self.args, "num_workers", None), "dataset", "num_workers")
        assign_nested(payload, getattr(self.args, "pin_memory", None), "dataset", "pin_memory")
        assign_nested(payload, getattr(self.args, "drop_last", None), "dataset", "drop_last")
        assign_nested(payload, getattr(self.args, "chans", None), "dataset", "chans")
        assign_nested(payload, getattr(self.args, "patch_num", None), "dataset", "patch_num")
        assign_nested(payload, getattr(self.args, "dataset_task", None), "dataset", "task")
        assign_nested(payload, getattr(self.args, "num_classes", None), "dataset", "num_classes")
        assign_nested(payload, getattr(self.args, "monitor_key", None), "dataset", "monitor_key")
        assign_nested(payload, getattr(self.args, "scale", None), "dataset", "scale")
        assign_nested(payload, getattr(self.args, "seq_len", None), "dataset", "seq_len")

    def _apply_runtime_overrides(self, payload):
        assign_nested(payload, getattr(self.args, "runtime_mode", None), "runtime", "mode")
        assign_nested(payload, getattr(self.args, "runtime_backend", None), "runtime", "backend")
        assign_nested(payload, getattr(self.args, "runtime_world_size", None), "runtime", "world_size")
        assign_nested(payload, getattr(self.args, "runtime_master_addr", None), "runtime", "master_addr")
        assign_nested(payload, getattr(self.args, "runtime_master_port", None), "runtime", "master_port")
        assign_nested(
            payload,
            getattr(self.args, "runtime_timeout_seconds", None),
            "runtime",
            "timeout_seconds",
        )
        assign_nested(payload, getattr(self.args, "runtime_debug", None), "runtime", "debug")
        assign_nested(
            payload,
            getattr(self.args, "runtime_find_unused_parameters", None),
            "runtime",
            "find_unused_parameters",
        )

    def _apply_task_overrides(self, payload, task_name=None):
        command = task_name or self.args.command
        if command == "finetune":
            self._apply_training_overrides(payload)
            assign_nested(
                payload,
                getattr(self.args, "use_pretrained_weights", None),
                "model",
                "params",
                "use_pretrained_weights",
            )
            assign_nested(payload, getattr(self.args, "dropout", None), "model", "params", "dropout")
            assign_nested(payload, getattr(self.args, "foundation_dir", None), "model", "params", "foundation_dir")
            return

        if command == "etms":
            self._apply_training_overrides(payload)
            assign_nested(payload, getattr(self.args, "dropout", None), "model", "params", "dropout")
            return

        if command == "fuse":
            self._apply_training_overrides(payload)
            self._apply_embrace_overrides(payload)
            return

        if command == "select":
            assign_nested(payload, getattr(self.args, "batch_size", None), "model", "trainer", "batch_size")
            assign_nested(payload, getattr(self.args, "sample_ratio", None), "sample_ratio")
            assign_nested(payload, getattr(self.args, "parc_n_dims", None), "parc", "n_dims")
            self._apply_embrace_overrides(payload)

    def _apply_training_overrides(self, payload):
        assign_nested(payload, getattr(self.args, "visualize", None), "visualize")
        assign_nested(payload, getattr(self.args, "batch_size", None), "model", "trainer", "batch_size")
        assign_nested(payload, getattr(self.args, "lr", None), "model", "trainer", "lr")
        assign_nested(payload, getattr(self.args, "lr_backbone", None), "model", "trainer", "lr_backbone")
        assign_nested(payload, getattr(self.args, "epochs", None), "model", "trainer", "epochs")
        assign_nested(payload, getattr(self.args, "patience", None), "model", "trainer", "patience")
        assign_nested(payload, getattr(self.args, "clip_grad_norm", None), "model", "trainer", "clip_grad_norm")
        assign_nested(payload, getattr(self.args, "weight_decay", None), "model", "trainer", "weight_decay")
        assign_nested(payload, getattr(self.args, "label_smoothing", None), "model", "trainer", "label_smoothing")
        assign_nested(payload, getattr(self.args, "betas", None), "model", "trainer", "betas")
        assign_nested(payload, getattr(self.args, "eps", None), "model", "trainer", "eps")
        assign_nested(payload, getattr(self.args, "save_every_epoch", None), "model", "trainer", "save_every_epoch")

    def _apply_embrace_overrides(self, payload):
        assign_nested(payload, getattr(self.args, "target_name", None), "model", "target_name")
        assign_nested(
            payload,
            normalize_string_list(getattr(self.args, "source_names", None)),
            "model",
            "source_names",
        )
        assign_nested(payload, getattr(self.args, "source_tag", None), "model", "source_tag")
        assign_nested(payload, getattr(self.args, "proj_dim", None), "model", "proj_dim")
        assign_nested(payload, getattr(self.args, "use_shared_bridge", None), "model", "use_shared_bridge")
        assign_nested(payload, getattr(self.args, "use_unified_bridge", None), "model", "use_unified_bridge")
        assign_nested(payload, getattr(self.args, "cbramod_mode", None), "model", "cbramod_mode")
        assign_nested(payload, getattr(self.args, "csbrain_mode", None), "model", "csbrain_mode")
        assign_nested(payload, getattr(self.args, "coupled_weight", None), "model", "coupled_weight")
        assign_nested(payload, getattr(self.args, "fusor_mode", None), "model", "fusor_mode")
        assign_nested(payload, getattr(self.args, "fusor_temp", None), "model", "fusor_temp")
        assign_nested(payload, getattr(self.args, "attn_softmax", None), "model", "attn_softmax")
        assign_nested(payload, getattr(self.args, "dropout", None), "model", "dropout")
        assign_nested(payload, getattr(self.args, "weight_ce", None), "model", "weight_ce")
        assign_nested(payload, getattr(self.args, "weight_emb", None), "model", "weight_emb")
        assign_nested(payload, getattr(self.args, "weight_attn", None), "model", "weight_attn")
        assign_nested(payload, getattr(self.args, "foundation_dir_cbramod", None), "model", "foundation_dir_dict", "cbramod")
        assign_nested(payload, getattr(self.args, "foundation_dir_labram", None), "model", "foundation_dir_dict", "labram")
        assign_nested(payload, getattr(self.args, "foundation_dir_codebrain", None), "model", "foundation_dir_dict", "codebrain")
        assign_nested(payload, getattr(self.args, "foundation_dir_csbrain", None), "model", "foundation_dir_dict", "csbrain")
