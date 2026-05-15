from dataset import create_datasets
import trainer as T
from .app.utils import cfg_get, config_to_dict
from .embrace import EmBraceTask

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

ISRUC_DATASETS = {"isruc-s1", "isruc-s3"}
ETM_MODEL_MAP = {
    "eegnet": "Model_EEGNet",
    "deformer": "Model_EEGDeformer",
    "sera": "Model_SERA",
    "sparc": "Model_SPaRCNet",
    "contra": "Model_ContraWR",
    "cnntrans": "Model_CNNTransformer",
    "ffcl": "Model_FFCL",
    "sttrans": "Model_STTransformer",
}


def _runtime_value(runtime_context, key, default):
    if runtime_context is None:
        return default
    return getattr(runtime_context, key, default)


def _resolve_device(device, cfg):
    if device is not None:
        return torch.device(device)
    return torch.device(str(cfg.device))


def _pin_memory(cfg, runtime_context):
    dataset_pref = bool(cfg_get(cfg, "dataset.pin_memory", True))
    if not dataset_pref:
        return False
    if runtime_context is not None:
        return _runtime_value(runtime_context, "device_type", "cpu") == "cuda"
    return str(cfg.device).startswith("cuda") and torch.cuda.is_available()


def _build_datasets(cfg):
    return create_datasets(
        dataset=cfg.dataset.name,
        data_dir=cfg.dataset.path,
        loading_metafeatures=(cfg.task == "fuse"),
        scale=cfg.dataset.scale,
    )


def _batch_size_for_split(cfg, split):
    if cfg.dataset.name in ISRUC_DATASETS and split != "train":
        return 1
    return cfg.model.trainer.batch_size


def create_dataloaders(cfg, runtime_context=None):
    datasets = _build_datasets(cfg)
    distributed = _runtime_value(runtime_context, "distributed", False)
    is_main = _runtime_value(runtime_context, "is_main", True)

    if is_main:
        print(
            f">> Train: {len(datasets['train'])} | Val: {len(datasets['val'])} | Test: {len(datasets['test'])}"
        )

    data_loaders = {}
    for split, dataset in datasets.items():
        sampler = None
        shuffle = split == "train"
        drop_last = bool(cfg_get(cfg, "dataset.drop_last", False)) if split == "train" else False
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=_runtime_value(runtime_context, "world_size", 1),
                rank=_runtime_value(runtime_context, "rank", 0),
                shuffle=shuffle,
                drop_last=shuffle,
            )
            shuffle = False

        data_loaders[split] = DataLoader(
            dataset,
            batch_size=_batch_size_for_split(cfg, split),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.dataset.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=_pin_memory(cfg, runtime_context),
            drop_last=drop_last,
        )

    return data_loaders


def _materialize_model(model, cfg, device):
    batch_size = cfg.model.trainer.batch_size * 20 if cfg.dataset.name in ISRUC_DATASETS else cfg.model.trainer.batch_size
    dummy_input = torch.randn(batch_size, cfg.dataset.chans, cfg.dataset.patch_num, 200, device=device)

    with torch.no_grad():
        model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f">> Model: {cfg.model.name.lower()}")
    print(f">> Total Params: {total_params:,} | Trainable Params: {trainable_params:,}")

    return model


def create_model(cfg, use_pretrained_weights=True, device=None):
    name = cfg.model.name.lower()
    target_device = _resolve_device(device, cfg)

    if name == "cbramod":
        from model.efms.cbramod import CBraMod as model_cls
    elif name == "codebrain":
        from model.efms.codebrain import CodeBrain as model_cls
    elif name == "csbrain":
        from model.efms.csbrain import CSBrain as model_cls
    elif name == "labram":
        from model.efms.labram import LaBraM as model_cls
    else:
        raise NotImplementedError(f"Model {name} is not implemented")

    model = model_cls(
        dataset=cfg.dataset.name,
        num_classes=cfg.dataset.num_classes,
        dropout=cfg.model.params.dropout,
        use_pretrained_weights=use_pretrained_weights,
        foundation_dir=cfg.model.params.foundation_dir,
    ).to(target_device)

    return _materialize_model(model, cfg, target_device)


def create_fuse_model(cfg, device=None):
    target_device = _resolve_device(device, cfg)
    model = EmBraceTask(
        target_name=cfg.model.target_name,
        source_names=cfg.model.source_names,
        dataset=cfg.dataset.name,
        chans=cfg.dataset.chans,
        patch_num=cfg.dataset.patch_num,
        dataset_name=cfg.dataset.name,
        num_classes=cfg.dataset.num_classes,
        foundation_dir_dict=cfg.model.foundation_dir_dict,
        dropout=cfg.model.dropout,
        fusor_temp=cfg.model.fusor_temp,
        cbramod_mode=cfg.model.cbramod_mode,
        csbrain_mode=cfg.model.csbrain_mode,
        attn_softmax=cfg.model.attn_softmax,
        proj_dim=cfg.model.proj_dim,
        bridge_type="mlp",
        use_shared_bridge=cfg.model.use_shared_bridge,
        use_unified_bridge=cfg.model.use_unified_bridge,
        coupled_weight=cfg.model.coupled_weight,
        fusor_mode=cfg.model.fusor_mode,
        weight_ce=cfg.model.weight_ce,
        weight_emb=cfg.model.weight_emb,
        weight_attn=cfg.model.weight_attn,
    ).to(target_device)
    return model


def create_etm_model(cfg, device=None):
    name = cfg.model.name.lower()
    target_device = _resolve_device(device, cfg)

    try:
        model_name = ETM_MODEL_MAP[name]
    except KeyError as exc:
        raise NotImplementedError(f"Model {name} is not implemented") from exc

    from model import etms

    model_cls = getattr(etms, model_name)
    model = model_cls(cfg).to(target_device)
    return _materialize_model(model, cfg, target_device)


def create_task_model(cfg, device=None):
    if cfg.task == "finetune":
        return create_model(
            cfg,
            use_pretrained_weights=cfg.model.params.use_pretrained_weights,
            device=device,
        )
    if cfg.task == "fuse":
        return create_fuse_model(cfg, device=device)
    if cfg.task == "etms":
        return create_etm_model(cfg, device=device)
    raise ValueError(f"Task {cfg.task} does not build a training model.")


def _trainer_class_for_cfg(cfg):
    if cfg.task in {"finetune", "etms"}:
        trainer_map = {
            "multi": T.commonMultiClassTrainer,
            "binary": T.commonBinaryClassTrainer,
        }
    elif cfg.task == "fuse":
        trainer_map = {
            "multi": T.embraceFuseMultiClassTrainer,
            "binary": T.embraceFuseBinaryClassTrainer,
        }
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    try:
        return trainer_map[cfg.dataset.task]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset task: {cfg.dataset.task}") from exc


def _build_trainer_hparams(cfg, checkpoints_path, runtime_context=None):
    hparams = config_to_dict(cfg.model.trainer)
    if hparams.get("lr_backbone") is None:
        hparams["lr_backbone"] = hparams["lr"]

    hparams.update(
        {
            "device": str(_runtime_value(runtime_context, "device", cfg.device)),
            "monitor_key": cfg.dataset.monitor_key,
            "checkpoints_path": checkpoints_path,
            "is_higher_better": False if cfg.dataset.monitor_key in ["loss"] else True,
            "visualize": cfg.visualize,
            "distributed": _runtime_value(runtime_context, "distributed", False),
            "rank": _runtime_value(runtime_context, "rank", 0),
            "world_size": _runtime_value(runtime_context, "world_size", 1),
            "ddp_device_ids": _runtime_value(runtime_context, "ddp_device_ids", None),
            "find_unused_parameters": bool(cfg_get(cfg, "runtime.find_unused_parameters", True)),
            "runtime_debug": bool(cfg_get(cfg, "runtime.debug", False)),
        }
    )
    return hparams


def create_trainer(model, data_loaders, checkpoints_path, cfg, runtime_context=None):
    trainer_cls = _trainer_class_for_cfg(cfg)
    trainer_kwargs = {
        "model": model,
        "data_loaders": data_loaders,
        "hparams": _build_trainer_hparams(
            cfg,
            checkpoints_path=checkpoints_path,
            runtime_context=runtime_context,
        ),
    }
    return trainer_cls(**trainer_kwargs)
