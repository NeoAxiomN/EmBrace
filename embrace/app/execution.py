from .artifacts import RunArtifacts
from .utils import config_to_dict, cfg_get, dict_to_config, load_yaml_file
from ..factory import (
    create_dataloaders,
    create_fuse_model,
    create_task_model,
    create_trainer,
)
from ..runtime import (
    RuntimeSpec,
    build_runtime_context,
    cleanup_distributed_runtime,
    prepare_distributed_runtime,
    resolve_runtime_spec,
)
from ..selector.model_selector import Selector

import os
import time
import traceback
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import setup_seed

TRAINING_TASKS = {"finetune", "fuse", "etms"}


def _runtime_mode_requested(cfg):
    return str(cfg_get(cfg, "runtime.mode", "auto")).lower()


def _runtime_debug_enabled(cfg):
    return bool(cfg_get(cfg, "runtime.debug", False))


def _log_rank_event(cfg, runtime_context, message):
    if not _runtime_debug_enabled(cfg):
        return
    print(
        f"[Rank {runtime_context.rank}/{runtime_context.world_size} | {runtime_context.mode}] {message}",
        flush=True,
    )


def _resolve_non_distributed_runtime_spec(cfg, task_name):
    runtime_spec = resolve_runtime_spec(cfg)
    if runtime_spec.mode != "ddp":
        return runtime_spec

    if task_name == "evaluate":
        if runtime_spec.device_type == "cuda":
            return RuntimeSpec(mode="single", device_type="cuda", backend=None, world_size=1)
        return RuntimeSpec(mode="cpu", device_type="cpu", backend=None, world_size=1)

    if _runtime_mode_requested(cfg) != "auto":
        raise ValueError(f"Task `{task_name}` does not support runtime.mode=ddp.")

    if runtime_spec.device_type == "cuda":
        return RuntimeSpec(mode="single", device_type="cuda", backend=None, world_size=1)
    return RuntimeSpec(mode="cpu", device_type="cpu", backend=None, world_size=1)


@dataclass
class TrainingRun:
    cfg: object
    output_dir: str

    def __post_init__(self):
        self.artifacts = RunArtifacts(
            cfg=self.cfg,
            output_dir=self.output_dir,
        )

    def run(self):
        runtime_spec = resolve_runtime_spec(self.cfg)
        self.artifacts.write_resolved_config()

        if runtime_spec.mode == "ddp":
            self._run_distributed(runtime_spec)
        else:
            runtime_context = build_runtime_context(self.cfg, runtime_spec, rank=0)
            self._run_task(runtime_context)

        self.artifacts.finalize_training()

    def _run_distributed(self, runtime_spec):
        cfg_payload = config_to_dict(self.cfg)
        mp.spawn(
            _distributed_training_worker,
            args=(runtime_spec.world_size, cfg_payload, self.output_dir),
            nprocs=runtime_spec.world_size,
            join=True,
        )

    def _run_task(self, runtime_context):
        setup_seed(self.cfg.seed + runtime_context.rank)
        _log_rank_event(
            self.cfg,
            runtime_context,
            f"seed initialized to {self.cfg.seed + runtime_context.rank}",
        )

        checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        if runtime_context.is_main:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if runtime_context.distributed:
            _log_rank_event(self.cfg, runtime_context, "waiting at pre-dataloader barrier")
            dist.barrier()
            _log_rank_event(self.cfg, runtime_context, "passed pre-dataloader barrier")

        data_loaders = create_dataloaders(self.cfg, runtime_context=runtime_context)
        _log_rank_event(self.cfg, runtime_context, "dataloaders constructed")
        model = create_task_model(self.cfg, device=runtime_context.device)
        _log_rank_event(self.cfg, runtime_context, "model constructed")
        trainer = create_trainer(
            model=model,
            data_loaders=data_loaders,
            checkpoints_path=checkpoint_dir,
            cfg=self.cfg,
            runtime_context=runtime_context,
        )
        _log_rank_event(self.cfg, runtime_context, "trainer constructed; starting fit")
        results = trainer.fit()
        _log_rank_event(self.cfg, runtime_context, "fit completed")

        if runtime_context.is_main:
            self.artifacts.write_json("results.json", results)
            self.artifacts.write_test_results(results.get("test", {}))

        if runtime_context.distributed:
            _log_rank_event(self.cfg, runtime_context, "waiting at post-training barrier")
            dist.barrier()
            _log_rank_event(self.cfg, runtime_context, "passed post-training barrier")

        del model, trainer, data_loaders
        self._clear_device_cache(runtime_context)

    @staticmethod
    def _clear_device_cache(runtime_context):
        if runtime_context.device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class SelectionRun:
    cfg: object
    output_dir: str

    def __post_init__(self):
        self.artifacts = RunArtifacts(
            cfg=self.cfg,
            output_dir=self.output_dir,
        )

    def run(self):
        runtime_spec = _resolve_non_distributed_runtime_spec(self.cfg, "select")
        runtime_context = build_runtime_context(self.cfg, runtime_spec, rank=0)

        self.artifacts.write_resolved_config()

        setup_seed(self.cfg.seed)
        data_loaders = create_dataloaders(self.cfg, runtime_context=runtime_context)
        selector = Selector(
            num_classes=self.cfg.dataset.num_classes,
            model_sources=self.cfg.model.source_names,
            train_dataloader=data_loaders["train"],
            val_dataloader=data_loaders["val"],
            batch_size=self.cfg.model.trainer.batch_size,
            device=str(runtime_context.device),
            sample_ratio=self.cfg.sample_ratio,
            cfg=self.cfg,
        )

        start_time = time.time()
        results = selector.select_best_model()
        elapsed_time = time.time() - start_time

        final_report = {
            "task_info": {
                "dataset": self.cfg.dataset.name if hasattr(self.cfg.dataset, "name") else "unknown",
                "sample_ratio": self.cfg.sample_ratio,
                "num_classes": self.cfg.dataset.num_classes,
                "total_execution_time_sec": round(elapsed_time, 2),
            },
            "selection_results": results,
        }

        self.artifacts.write_json("selection_results.json", final_report, ensure_ascii=True)
        self.artifacts.write_json("scores_only.json", results.get("parc_scores", {}))

        print("\n" + "=" * 30)
        print("MODEL SELECTION COMPLETED")
        print(f"Best Model: {results['best_model']}")
        print(f"Reason: {results['reason']}")
        print(f"Time Taken: {elapsed_time:.2f}s")
        print("=" * 30 + "\n")
        print(f">> Experiment finished. All artifacts saved to: {self.output_dir}")

        del selector, data_loaders
        TrainingRun._clear_device_cache(runtime_context)


@dataclass
class CheckpointEvaluator:
    exp_dir: str
    ckpt_name: str = "best_checkpoint.pth"

    def run(self):
        cfg = dict_to_config(load_yaml_file(os.path.join(self.exp_dir, "config_resolved.yaml")))
        runtime_spec = _resolve_non_distributed_runtime_spec(cfg, "evaluate")
        runtime_context = build_runtime_context(cfg, runtime_spec, rank=0)

        setup_seed(cfg.seed)
        data_loaders = create_dataloaders(cfg, runtime_context=runtime_context)
        model = self._load_model(cfg, runtime_context)
        trainer = create_trainer(
            model=model,
            data_loaders=data_loaders,
            checkpoints_path=os.path.join(self.exp_dir, "checkpoint"),
            cfg=cfg,
            runtime_context=runtime_context,
        )

        results = trainer._evaluate(mode="test")
        trainer.report({"Test": results}, title="[Test]")

        del model, trainer, data_loaders
        TrainingRun._clear_device_cache(runtime_context)

    def _load_model(self, cfg, runtime_context):
        if cfg.task == "fuse":
            model = create_fuse_model(cfg, device=runtime_context.device)
        elif cfg.task in {"finetune", "etms"}:
            model = create_task_model(cfg, device=runtime_context.device)
        else:
            raise ValueError(f"Unsupported evaluate task: {cfg.task}")

        checkpoint_path = os.path.join(self.exp_dir, "checkpoint", self.ckpt_name)
        print(f">> Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        message = model.load_state_dict(state_dict, strict=False)
        print(f">> Task Load Result: {message}")
        model.eval()
        return model


@dataclass
class TaskApplication:
    cfg: object
    output_dir: str

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if self.cfg.task in TRAINING_TASKS:
            TrainingRun(
                cfg=self.cfg,
                output_dir=self.output_dir,
            ).run()
            return

        if self.cfg.task == "select":
            SelectionRun(
                cfg=self.cfg,
                output_dir=self.output_dir,
            ).run()
            return

        raise ValueError(f"Unsupported task: {self.cfg.task}")


def _distributed_training_worker(rank, world_size, cfg_payload, output_dir):
    del world_size
    cfg = dict_to_config(cfg_payload)
    runtime_spec = resolve_runtime_spec(cfg)
    runtime_context = build_runtime_context(cfg, runtime_spec, rank=rank)
    prepare_distributed_runtime(cfg, runtime_context)
    try:
        TrainingRun(cfg, output_dir)._run_task(runtime_context)
    except Exception as exc:
        print(f"[Rank {rank}] Unhandled exception: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed_runtime()
