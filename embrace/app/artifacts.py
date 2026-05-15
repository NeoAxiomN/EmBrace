from .config import build_model_key
from .utils import cfg_get, dump_yaml_text, is_numeric_scalar, locked_file, seed_dir_sort_key

import json
import os
from dataclasses import dataclass


SUMMARY_YAML_FILENAME = "summary.yaml"
RESULT_MARKER_EXTENSION = ".done"
RESULT_MARKER_SPECS = {
    "multi": (("b_acc", "bacc"), ("f1", "f1"), ("kappa", "kappa")),
    "binary": (("b_acc", "bacc"), ("auc_pr", "aucpr"), ("auroc", "auroc")),
}
SUMMARY_METRIC_SPECS = {
    "multi": ("acc", "b_acc", "f1", "kappa"),
    "binary": ("acc", "b_acc", "auc_pr", "auroc"),
}
FALLBACK_PRIMARY_METRIC_GROUP = ("acc", "f1", "kappa")
FALLBACK_PRIMARY_METRIC_CANDIDATES = (
    "acc",
    "f1",
    "kappa",
    "b_acc",
    "auroc",
    "auc_pr",
    "precision",
    "recall",
    "mse",
)


@dataclass
class RunArtifacts:
    cfg: object
    output_dir: str

    def write_json(self, filename, payload, ensure_ascii=False):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=ensure_ascii, default=str)

    def write_test_results(self, payload):
        test_results = payload if isinstance(payload, dict) else {}
        self.write_json("test_results.json", test_results)
        self.write_result_marker(test_results)

    def write_resolved_config(self):
        path = os.path.join(self.output_dir, "config_resolved.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(dump_yaml_text(self.cfg))

    def write_result_marker(self, test_results):
        metric_parts = []
        for source_key, label in self._result_marker_spec():
            try:
                metric_value = float(test_results[source_key])
            except KeyError as exc:
                raise ValueError(
                    f"Missing test metric `{source_key}` for dataset.task={cfg_get(self.cfg, 'dataset.task', '')!r}."
                ) from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid test metric `{source_key}` for dataset.task={cfg_get(self.cfg, 'dataset.task', '')!r}: "
                    f"{test_results.get(source_key)!r}"
                ) from exc
            metric_parts.append(f"{label}-{metric_value:.6f}")

        self._remove_existing_result_markers()
        marker_name = "_".join(metric_parts) + RESULT_MARKER_EXTENSION
        marker_path = os.path.join(self.output_dir, marker_name)
        with open(marker_path, "w", encoding="utf-8"):
            pass

    def _result_marker_spec(self):
        dataset_task = str(cfg_get(self.cfg, "dataset.task", "")).strip().lower()
        try:
            return RESULT_MARKER_SPECS[dataset_task]
        except KeyError as exc:
            raise ValueError(f"Unsupported dataset.task for result marker: {dataset_task!r}") from exc

    def _remove_existing_result_markers(self):
        if not os.path.isdir(self.output_dir):
            return

        for entry in os.listdir(self.output_dir):
            if not self._is_result_marker_filename(entry):
                continue
            try:
                os.remove(os.path.join(self.output_dir, entry))
            except OSError:
                continue

    @staticmethod
    def _is_result_marker_filename(filename):
        if not filename.endswith(RESULT_MARKER_EXTENSION):
            return False

        stem = filename[: -len(RESULT_MARKER_EXTENSION)]
        parts = stem.split("_")
        for spec in RESULT_MARKER_SPECS.values():
            if len(parts) != len(spec):
                continue
            if RunArtifacts._matches_result_marker_parts(parts, spec):
                return True
        return False

    @staticmethod
    def _matches_result_marker_parts(parts, spec):
        for part, (_, label) in zip(parts, spec):
            prefix = f"{label}-"
            if not part.startswith(prefix):
                return False
            try:
                float(part[len(prefix) :])
            except ValueError:
                return False
        return True

    def finalize_training(self):
        summary_yaml_path = maybe_write_multirun_summary(self.cfg, self.output_dir)
        print(">> Experiment finished.")
        print(f">> Results saved to: {self.output_dir}")
        if summary_yaml_path is not None:
            print(f">> Multirun summary saved to: {summary_yaml_path}")


@dataclass
class MultirunSummaryBuilder:
    cfg: object
    multirun_dir: str
    result_filename: str = "test_results.json"

    def build(self):
        if not os.path.isdir(self.multirun_dir):
            return None

        per_seed_results = self._load_per_seed_results()
        if not per_seed_results:
            return None

        expected_seeds = self._expected_seeds()
        completed_seed_names = sorted(per_seed_results.keys(), key=seed_dir_sort_key)
        completed_seed_values = [seed_name.split("seed_", 1)[-1] for seed_name in completed_seed_names]
        missing_seed_values = [seed for seed in expected_seeds if seed not in completed_seed_values]
        first_seed_result = per_seed_results[completed_seed_names[0]]
        monitor_key = cfg_get(self.cfg, "dataset.monitor_key", "")
        primary_metrics = self._select_primary_metrics(first_seed_result, monitor_key)
        metrics = self._build_metric_summary(primary_metrics, per_seed_results)
        metric_name = monitor_key if monitor_key in metrics else (primary_metrics[0] if primary_metrics else "")
        metric_summary = metrics.get(metric_name, {})

        return {
            "task": cfg_get(self.cfg, "task", ""),
            "dataset": cfg_get(self.cfg, "dataset.name", ""),
            "model": cfg_get(self.cfg, "model.name", ""),
            "model_key": build_model_key(self.cfg),
            "target_name": cfg_get(self.cfg, "model.target_name", ""),
            "source_tag": cfg_get(self.cfg, "model.source_tag", ""),
            "source_names": list(cfg_get(self.cfg, "model.source_names", []) or []),
            "metric_name": metric_name,
            "monitor_key": monitor_key,
            "best_metric": metric_summary.get("best_value"),
            "best_seed": metric_summary.get("best_seed"),
            "created_at": cfg_get(self.cfg, "output.timestamp", os.path.basename(self.multirun_dir)),
            "summary_file": SUMMARY_YAML_FILENAME,
            "multirun_dir": os.path.abspath(self.multirun_dir),
            "is_complete": bool(expected_seeds) and len(missing_seed_values) == 0,
            "expected_seeds": expected_seeds,
            "completed_seeds": completed_seed_values,
            "missing_seeds": missing_seed_values,
            "completed_seed_count": len(completed_seed_values),
            "primary_metrics": primary_metrics,
            "metrics": metrics,
            "per_seed": {
                seed_name: {
                    "output_dir": os.path.join(self.multirun_dir, seed_name),
                    "test_results": per_seed_results[seed_name],
                }
                for seed_name in completed_seed_names
            },
        }

    def _expected_seeds(self):
        expected_seeds = cfg_get(self.cfg, "output.expected_seeds", []) or []
        return [str(seed).strip() for seed in expected_seeds if str(seed).strip()]

    def _load_per_seed_results(self):
        seed_dirs = []
        for entry in os.listdir(self.multirun_dir):
            entry_path = os.path.join(self.multirun_dir, entry)
            if entry.startswith("seed_") and os.path.isdir(entry_path):
                seed_dirs.append(entry)
        seed_dirs.sort(key=seed_dir_sort_key)

        per_seed_results = {}
        for seed_dir in seed_dirs:
            result_path = os.path.join(self.multirun_dir, seed_dir, self.result_filename)
            if not os.path.exists(result_path):
                continue
            try:
                with open(result_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            if isinstance(payload, dict):
                per_seed_results[seed_dir] = payload

        return per_seed_results

    def _build_metric_summary(self, primary_metrics, per_seed_results):
        metrics = {}
        for metric_name in primary_metrics:
            metric_summary = self._summarize_metric(metric_name, per_seed_results)
            if metric_summary is not None:
                metrics[metric_name] = metric_summary
        return metrics

    def _select_primary_metrics(self, test_metrics, monitor_key):
        numeric_keys = [key for key, value in test_metrics.items() if is_numeric_scalar(value)]
        dataset_task = str(cfg_get(self.cfg, "dataset.task", "")).strip().lower()

        task_metrics = SUMMARY_METRIC_SPECS.get(dataset_task)
        if task_metrics is not None:
            selected = [metric for metric in task_metrics if metric in numeric_keys]
            if selected:
                return selected

        if all(metric in numeric_keys for metric in FALLBACK_PRIMARY_METRIC_GROUP):
            return list(FALLBACK_PRIMARY_METRIC_GROUP)

        selected = []
        if monitor_key in numeric_keys:
            selected.append(monitor_key)

        for metric in FALLBACK_PRIMARY_METRIC_CANDIDATES:
            if metric in numeric_keys and metric not in selected:
                selected.append(metric)
            if len(selected) == 3:
                break

        return selected[:3]

    @staticmethod
    def _summarize_metric(metric_name, per_seed_results):
        values_by_seed = {}
        for seed_name, metrics in per_seed_results.items():
            value = metrics.get(metric_name)
            if is_numeric_scalar(value):
                values_by_seed[seed_name] = float(value)

        if not values_by_seed:
            return None

        values = list(values_by_seed.values())
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = variance ** 0.5
        best_seed_name, best_value = max(values_by_seed.items(), key=lambda item: item[1])
        worst_seed_name, worst_value = min(values_by_seed.items(), key=lambda item: item[1])

        return {
            "mean": mean_value,
            "std": std_value,
            "best_seed": best_seed_name.split("seed_", 1)[-1],
            "best_seed_dir": best_seed_name,
            "best_value": best_value,
            "worst_seed": worst_seed_name.split("seed_", 1)[-1],
            "worst_seed_dir": worst_seed_name,
            "worst_value": worst_value,
            "values": values_by_seed,
        }


def maybe_write_multirun_summary(cfg, output_dir, result_filename="test_results.json"):
    seed_dir_name = os.path.basename(output_dir)
    if not seed_dir_name.startswith("seed_"):
        return None

    multirun_dir = os.path.dirname(output_dir)
    summary_path = os.path.join(multirun_dir, SUMMARY_YAML_FILENAME)
    lock_path = f"{summary_path}.lock"
    with locked_file(lock_path):
        summary = MultirunSummaryBuilder(
            cfg=cfg,
            multirun_dir=multirun_dir,
            result_filename=result_filename,
        ).build()
        if summary is None:
            return None

        tmp_path = f"{summary_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(dump_yaml_text(summary))
        os.replace(tmp_path, summary_path)
    return summary_path
