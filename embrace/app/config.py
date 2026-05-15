from .utils import (
    ConfigNode,
    cfg_get,
    clone_config,
    config_to_dict,
    dict_to_config,
    load_yaml_file,
    merge_dicts,
)

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path



DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "config"


@dataclass(frozen=True)
class ConfigComposeRequest:
    task: str
    base_config_path: str | None = None
    dataset: str | None = None
    dataset_config_path: str | None = None
    model: str | None = None
    direct_overrides: dict | None = None


@dataclass(frozen=True)
class LaunchPlan:
    task: str
    seeds: list[int]
    timestamp: str
    category: str
    model_key: str
    run_root_dir: str
    multirun_root_dir: str
    active_root_dir: str
    is_multirun: bool

    def output_dir_for_seed(self, seed):
        return os.path.join(self.active_root_dir, f"seed_{seed}")


def sanitize_path_token(value):
    if value is None:
        return "none"

    sanitized = str(value).strip()
    if not sanitized:
        return "none"

    sanitized = sanitized.replace(os.sep, "-")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "-")
    return sanitized.replace(" ", "-")


def build_model_key(cfg):
    task = str(cfg_get(cfg, "task", ""))
    model_name = sanitize_path_token(cfg_get(cfg, "model.name", "unknown"))
    if task not in {"fuse", "select"}:
        return model_name

    target_name = sanitize_path_token(cfg_get(cfg, "model.target_name", "unknown"))
    source_tag = sanitize_path_token(cfg_get(cfg, "model.source_tag", "none"))
    return f"{model_name}__tgt-{target_name}__src-{source_tag}"


class ConfigRepository:
    def __init__(self, config_root: str | os.PathLike | None = None):
        self.config_root = Path(config_root) if config_root is not None else DEFAULT_CONFIG_ROOT
        self.task_root = self.config_root / "tasks"
        self.dataset_root = self.config_root / "dataset"

    def load_task_payload(self, task, config_path=None):
        resolved_path = Path(config_path) if config_path is not None else self.task_root / f"{task}.yaml"
        payload = load_yaml_file(resolved_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Task config must be a mapping: {resolved_path}")
        return payload

    def load_dataset_payload(self, dataset_name=None, config_path=None):
        if config_path is None and dataset_name is None:
            return None

        resolved_path = Path(config_path) if config_path is not None else self.dataset_root / f"{dataset_name}.yaml"
        payload = load_yaml_file(resolved_path)
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Dataset config is empty or invalid: {resolved_path}")
        if "name" not in payload:
            payload["name"] = resolved_path.stem
        return payload

    def list_dataset_payloads(self):
        payloads = []
        for path in sorted(self.dataset_root.glob("*.yaml")):
            try:
                payload = load_yaml_file(path)
            except Exception:
                continue
            if isinstance(payload, dict) and payload:
                if "name" not in payload:
                    payload["name"] = path.stem
                payloads.append(payload)
        return payloads

    @staticmethod
    def saved_seed_config_path(exp_dir):
        return Path(exp_dir) / "config_resolved.yaml"

    def load_saved_seed_payload(self, exp_dir):
        resolved_path = self.saved_seed_config_path(exp_dir)
        payload = load_yaml_file(resolved_path)
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Saved config is empty or invalid: {resolved_path}")
        return payload


class ConfigComposer:
    def __init__(self, config_root: str | os.PathLike | None = None):
        self.repository = ConfigRepository(config_root=config_root)

    def compose(self, request: ConfigComposeRequest):
        task_payload = self.repository.load_task_payload(request.task, config_path=request.base_config_path)
        base_payload = {
            key: value
            for key, value in task_payload.items()
            if key not in {"default_model", "models"}
        }

        model_payload = self._resolve_model_payload(task_payload, request.task, request.model)
        dataset_payload = self.repository.load_dataset_payload(
            dataset_name=request.dataset,
            config_path=request.dataset_config_path,
        )

        payload = merge_dicts(
            base_payload,
            {"task": request.task},
            {"dataset": dataset_payload} if dataset_payload is not None else None,
            {"model": model_payload} if model_payload is not None else None,
            request.direct_overrides,
        )
        self._apply_computed_defaults(payload)
        self._validate(payload, request.task)
        return dict_to_config(payload)

    def compose_saved_seed_cfg(self, exp_dir, direct_overrides=None):
        payload = self.repository.load_saved_seed_payload(exp_dir)
        task = str(payload.get("task", "")).strip().lower()
        if not task:
            raise ValueError(f"Saved config has no task: {self.repository.saved_seed_config_path(exp_dir)}")

        payload = merge_dicts(payload, direct_overrides)
        self._apply_computed_defaults(payload)
        self._validate(payload, task)
        return dict_to_config(payload)

    def materialize_seed_cfg(self, cfg, launch_plan: LaunchPlan, seed):
        seed_cfg = clone_config(cfg)
        seed_cfg.seed = int(seed)
        seed_cfg.output.timestamp = launch_plan.timestamp
        seed_cfg.output.category = launch_plan.category
        seed_cfg.output.model_key = launch_plan.model_key
        seed_cfg.output.run_root_dir = launch_plan.run_root_dir
        seed_cfg.output.multirun_root_dir = launch_plan.multirun_root_dir
        seed_cfg.output.current_root_dir = launch_plan.active_root_dir
        seed_cfg.output.is_multirun = launch_plan.is_multirun
        seed_cfg.output.expected_seeds = list(launch_plan.seeds)
        output_dir = launch_plan.output_dir_for_seed(seed)
        seed_cfg.output.output_dir = output_dir
        return seed_cfg, output_dir

    def _resolve_model_payload(self, task_payload, task, requested_model):
        models = task_payload.get("models") or {}
        model_name = requested_model or task_payload.get("default_model")
        if model_name is None:
            raise ValueError("Model is required. Use `--model`.")

        try:
            model_payload = config_to_dict(models[model_name])
        except KeyError as exc:
            available = ", ".join(sorted(models))
            raise ValueError(f"Unknown model `{model_name}` for task `{task}`. Available: {available}") from exc

        model_payload.setdefault("name", model_name)
        return model_payload

    @staticmethod
    def _apply_computed_defaults(payload):
        payload.setdefault("output", {})
        payload["output"].setdefault("base_dir", "outputs")

        dataset_payload = payload.get("dataset")
        if isinstance(dataset_payload, dict):
            dataset_payload.setdefault("drop_last", False)

        model_payload = payload.get("model")
        if not isinstance(model_payload, dict):
            return

        if not model_payload.get("model_folder_id"):
            model_payload["model_folder_id"] = ConfigComposer._default_model_folder_id(payload)

    @staticmethod
    def _default_model_folder_id(payload):
        task = str(payload.get("task", ""))
        model_payload = payload.get("model") or {}
        model_name = str(model_payload.get("name", "unknown"))

        if task == "finetune":
            return f"finetune_{model_name}"
        if task == "fuse":
            return (
                f"{model_name}_tgt-{model_payload.get('target_name', 'unknown')}"
                f"_src-{model_payload.get('source_tag', 'none')}"
            )
        if task == "select":
            return (
                f"{model_name}_select_tgt-{model_payload.get('target_name', 'unknown')}"
                f"_src-{model_payload.get('source_tag', 'none')}"
            )
        if task == "etms":
            return f"task-specific_{model_name}"
        return model_name

    @staticmethod
    def _validate(payload, task):
        if str(payload.get("task", "")).lower() != task:
            raise ValueError(f"Configuration task mismatch: expected `{task}`, got `{payload.get('task', '')}`.")

        dataset_payload = payload.get("dataset")
        if not isinstance(dataset_payload, dict) or not dataset_payload.get("name"):
            raise ValueError("Dataset config is required. Use `--dataset` or `--dataset-config`.")

        model_payload = payload.get("model")
        if not isinstance(model_payload, dict) or not model_payload.get("name"):
            raise ValueError("Model config is required. Use `--model`.")


class OutputLayoutPlanner:
    def build(self, cfg, seeds):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        task = str(cfg_get(cfg, "task", ""))
        dataset_name = sanitize_path_token(cfg_get(cfg, "dataset.name", "unknown"))
        model_name = sanitize_path_token(cfg_get(cfg, "model.name", "unknown"))
        base_dir = os.path.normpath(str(cfg_get(cfg, "output.base_dir", "outputs")))
        category = task
        model_key = build_model_key(cfg)
        path_model_segment = model_key if task in {"fuse", "select"} else model_name

        run_root_dir = os.path.normpath(os.path.join(base_dir, task, dataset_name, path_model_segment, timestamp))
        normalized_seeds = [int(seed) for seed in seeds]
        multirun_root_dir = run_root_dir
        active_root_dir = run_root_dir
        is_multirun = True

        return LaunchPlan(
            task=task,
            seeds=normalized_seeds,
            timestamp=timestamp,
            category=category,
            model_key=model_key,
            run_root_dir=run_root_dir,
            multirun_root_dir=multirun_root_dir,
            active_root_dir=active_root_dir,
            is_multirun=is_multirun,
        )
