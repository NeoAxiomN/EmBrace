import os
import time
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


def _coerce_hparams(hparams):
    if isinstance(hparams, SimpleNamespace):
        return hparams
    if isinstance(hparams, dict):
        return SimpleNamespace(**hparams)
    raise TypeError(f"Unsupported hparams type: {type(hparams)!r}")


class baseTrainer(ABC):
    def __init__(
        self,
        model,
        data_loaders,
        hparams,
        *args,
        **kwargs,
    ):
        self.hparams = _coerce_hparams(hparams)
        self.device = torch.device(self.hparams.device)
        self.model = model.to(self.device)
        self.data_loaders = data_loaders
        self.monitor_key = self.hparams.monitor_key
        self.checkpoint_path = self.hparams.checkpoints_path
        self.is_higher_better = self.hparams.is_higher_better

        # Distributed training attributes
        self.distributed = getattr(self.hparams, "distributed", False)
        self.rank = getattr(self.hparams, "rank", 0)
        self.world_size = getattr(self.hparams, "world_size", 1)
        self.is_main = self.rank == 0
        self.ddp_device_ids = getattr(self.hparams, "ddp_device_ids", None)
        self.find_unused_parameters = getattr(self.hparams, "find_unused_parameters", True)
        self.runtime_debug = getattr(self.hparams, "runtime_debug", False)

        self.current_epoch = 0
        self.criterion = self._setup_criterion()

        if self.distributed:
            self._initialize_lazy_modules()
            self._wrap_distributed_model()

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

    @abstractmethod
    def _setup_optimizer(self):
        pass

    @abstractmethod
    def _setup_criterion(self):
        pass

    def _setup_scheduler(self):
        return None

    @abstractmethod
    def step(self, batch):
        pass

    @abstractmethod
    def summarize(self, tracker):
        pass

    def _wrap_distributed_model(self):
        if self.device.type == "cuda":
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=self.ddp_device_ids,
                find_unused_parameters=self.find_unused_parameters,
            )
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=self.find_unused_parameters,
            )

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _initialize_lazy_modules(self):
        if "train" not in self.data_loaders:
            return

        try:
            dummy_batch = next(iter(self.data_loaders["train"]))
        except StopIteration:
            return

        previous_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.step(dummy_batch)
        self.model.train(previous_mode)

        if self.is_main:
            print(">> Lazy modules initialized.")

    def _is_cuda_runtime(self):
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _reduce_scalar(self, value, op):
        tensor = torch.tensor(float(value), device=self.device if self._is_cuda_runtime() else "cpu")
        if self.distributed:
            dist.all_reduce(tensor, op=op)
        return tensor

    def _merge_trackers(self, gathered_trackers):
        merged = {}
        for tracker in gathered_trackers:
            for key, values in tracker.items():
                merged.setdefault(key, []).extend(values)
        return merged

    def _gather_tracker(self, tracker):
        if not self.distributed:
            return tracker

        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, tracker)
        return self._merge_trackers(gathered)

    def _debug_log(self, message):
        if not self.runtime_debug:
            return
        print(f"[Rank {self.rank}/{self.world_size}] {message}", flush=True)

    def _ensure_finite_loss(self, loss, mode):
        if not torch.is_tensor(loss):
            raise TypeError(f"Expected tensor loss in mode `{mode}`, got {type(loss)!r}.")

        local_ok = bool(torch.isfinite(loss.detach()).all().item())
        if self.distributed:
            flag = torch.tensor(
                1 if local_ok else 0,
                device=self.device if self._is_cuda_runtime() else "cpu",
                dtype=torch.int32,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) != 1:
                gathered = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered, local_ok)
                bad_ranks = [idx for idx, ok in enumerate(gathered) if not ok]
                raise RuntimeError(
                    f"Non-finite loss detected during `{mode}` on rank(s) {bad_ranks} "
                    f"at epoch {self.current_epoch}."
                )
        elif not local_ok:
            raise RuntimeError(f"Non-finite loss detected during `{mode}` at epoch {self.current_epoch}.")

    def collect(self, tracker, step_output):
        if step_output is None:
            return

        for key, value in step_output.items():
            if key not in tracker:
                tracker[key] = []

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.ndim == 0:
                    tracker[key].append(value.item())
                else:
                    tracker[key].append(value.numpy())
            elif isinstance(value, (float, int, np.number)):
                tracker[key].append(value)
            else:
                tracker[key].append(value)

    def _run_epoch(self, mode):
        is_train = mode == "train"
        tracker = {}
        total_samples = 0
        start_time = time.time()

        sampler = getattr(self.data_loaders[mode], "sampler", None)
        if is_train and self.distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

        pbar = tqdm(
            self.data_loaders[mode],
            desc=f"[Epoch {self.current_epoch} {mode}]",
            leave=False,
            disable=not self.is_main,
        )

        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 0:
                self._debug_log(f"epoch={self.current_epoch} mode={mode} batch=0 start")

            if isinstance(batch, (list, tuple)):
                total_samples += batch[0].size(0)
            elif isinstance(batch, dict):
                total_samples += next(iter(batch.values())).size(0)

            step_output = self.step(batch)
            if not isinstance(step_output, dict) or "loss" not in step_output:
                raise RuntimeError(
                    f"`step()` must return a dict containing `loss`; got {type(step_output)!r} in mode `{mode}`."
                )
            self._ensure_finite_loss(step_output["loss"], mode)

            if is_train:
                if batch_idx == 0:
                    self._debug_log(f"epoch={self.current_epoch} mode={mode} batch=0 backward")
                self.optimizer.zero_grad()
                step_output["loss"].backward()
                if self.hparams.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.clip_grad_norm)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                if batch_idx == 0:
                    self._debug_log(f"epoch={self.current_epoch} mode={mode} batch=0 optimizer step complete")

            self.collect(tracker, step_output)

            if self.is_main:
                pbar.set_postfix(
                    {
                        "batch_loss": f"{step_output['loss'].item():.4f}",
                        "avg_loss": f"{np.mean(tracker['loss']):.4f}",
                        **(
                            {"lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"}
                            if is_train
                            else {}
                        ),
                    }
                )

        if self._is_cuda_runtime():
            torch.cuda.synchronize(self.device)

        duration = time.time() - start_time
        if self.distributed:
            total_samples = int(self._reduce_scalar(total_samples, dist.ReduceOp.SUM).item())
            duration = float(self._reduce_scalar(duration, dist.ReduceOp.MAX).item())

        throughput = total_samples / max(duration, 1e-8)

        vram_peak = 0.0
        if self._is_cuda_runtime():
            vram_peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            if self.distributed:
                vram_peak = float(self._reduce_scalar(vram_peak, dist.ReduceOp.MAX).item())

        tracker = self._gather_tracker(tracker)
        epoch_results = self.summarize(tracker)
        epoch_results["throughput"] = throughput
        epoch_results["vram_mb"] = vram_peak
        epoch_results["duration"] = duration
        return epoch_results

    def _train_one_epoch(self):
        self.model.train()
        self.current_epoch += 1
        return self._run_epoch(mode="train")

    @torch.no_grad()
    def _evaluate(self, mode="val"):
        self.model.eval()
        return self._run_epoch(mode=mode)

    def fit(self, skip_val=False, skip_test=False):
        results = {
            "history": [],
            "best_epoch": 0,
            "test": {},
            "epochs": 0,
            "total_train_duration": 0.0,
            "max_vram_mb": 0.0,
        }
        if self._is_cuda_runtime():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        fit_start_time = time.time()
        best_score = float("-inf")
        patience_counter = 0

        for epoch in range(1, self.hparams.epochs + 1):
            epoch_start = time.time()
            train_start = time.time()
            results_train = self._train_one_epoch()
            train_duration = time.time() - train_start

            if results_train["vram_mb"] > results["max_vram_mb"]:
                results["max_vram_mb"] = results_train["vram_mb"]
            results["total_train_duration"] += train_duration

            if skip_val:
                self.report({"Train": results_train}, title=f"[Epoch {epoch}/{self.hparams.epochs}]")
                continue

            results_val = self._evaluate(mode="val")
            self.report(
                {"Train": results_train, "Val": results_val},
                title=f"[Epoch {epoch}/{self.hparams.epochs}]",
            )

            optimized_score = results_val[self.monitor_key]
            is_better = (
                self.is_higher_better and optimized_score > best_score
            ) or (
                not self.is_higher_better and optimized_score < best_score
            )
            if is_better:
                best_score = optimized_score
                results["best_epoch"] = epoch
                patience_counter = 0
                if self.is_main:
                    print(f"  [!] New Best {self.monitor_key}: {best_score:.6f} (Epoch {epoch}).")
                self.save_checkpoint(path=self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.hparams.patience:
                    if self.is_main:
                        print(f"Early stopping at epoch {epoch}.")
                    results["history"].append(
                        {
                            "epoch": epoch,
                            "train": results_train,
                            "val": results_val,
                            "epoch_duration": time.time() - epoch_start,
                        }
                    )
                    results["epochs"] = epoch
                    break

            if self.hparams.save_every_epoch:
                self.save_checkpoint(path=self.checkpoint_path, name=f"checkpoint_epoch_{epoch}.pth")

            results["history"].append(
                {
                    "epoch": epoch,
                    "train": results_train,
                    "val": results_val,
                    "epoch_duration": time.time() - epoch_start,
                }
            )
            results["epochs"] = epoch
            if self.is_main:
                elapsed = time.time() - epoch_start
                print(f"  [Time] Epoch {epoch} took {elapsed:.2f}s (Pure Train: {train_duration:.2f}s)")
                print()

        results["total_training_time"] = time.time() - fit_start_time

        if skip_test:
            return results

        if self.distributed:
            self._debug_log("waiting at pre-test barrier")
            dist.barrier()
            self._debug_log("passed pre-test barrier")

        self.load_checkpoint(path=self.checkpoint_path)
        test_start = time.time()
        results["test"] = self._evaluate(mode="test")
        results["test"]["test_time"] = time.time() - test_start
        results["test"]["training_duration_seconds"] = results["total_train_duration"]
        num_epochs = len(results["history"])
        if num_epochs > 0:
            results["test"]["avg_time_per_epoch"] = results["total_train_duration"] / num_epochs
        self.report({"Test": results["test"]}, title="[Test]")
        return results

    def save_checkpoint(self, path, extras=None, name="best_checkpoint.pth"):
        if not self.is_main:
            return

        path = os.path.normpath(path)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name)

        state = {
            "epoch": self.current_epoch,
            "state_dict": self._unwrap_model().state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }
        if extras:
            state.update(extras)
        torch.save(state, path)
        print(f">> Saved [{name}] checkpoint at {path}")

    def load_checkpoint(self, path, name="best_checkpoint.pth"):
        path = os.path.join(path, name)
        path = os.path.normpath(path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            if self.is_main and self.data_loaders["val"] is not None:
                print(f">> Failed to load checkpoint from {path}: {exc}")
            return None

        self._unwrap_model().load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.current_epoch = checkpoint["epoch"]

        if self.optimizer is not None and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.is_main:
            print(f">> Loaded checkpoint from: {path}")
        return checkpoint

    def report(self, results_dict, title):
        if not self.is_main:
            return

        print(f"\n{title}")
        keys = sorted({key for values in results_dict.values() for key in values if key != "cm"})

        for mode, metrics in results_dict.items():
            formatted_parts = []
            for key in keys:
                if key not in metrics:
                    continue

                value = metrics[key]
                if isinstance(value, (float, np.floating)):
                    formatted_parts.append(f"{key}={value:.6f}")
                elif isinstance(value, (list, np.ndarray)):
                    list_str = "[" + ", ".join(f"{v:.4f}" for v in value) + "]"
                    formatted_parts.append(f"{key}={list_str}")
                else:
                    formatted_parts.append(f"{key}={value}")

            content = ", ".join(formatted_parts)
            print(f"  {mode:<5}: {content}")
            if "cm" in metrics:
                print(f"  Confusion Matrix:\n{np.array(metrics['cm'])}")
