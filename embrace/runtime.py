from .app.utils import cfg_get

from datetime import timedelta
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist




RUNTIME_MODES = {"auto", "cpu", "single", "ddp"}
AUTO_BACKEND = "auto"


@dataclass(frozen=True)
class RuntimeSpec:
    mode: str
    device_type: str
    backend: str | None
    world_size: int


@dataclass(frozen=True)
class RuntimeContext:
    mode: str
    device: torch.device
    device_type: str
    distributed: bool
    rank: int
    world_size: int
    is_main: bool
    backend: str | None
    ddp_device_ids: list[int] | None


def _normalize_mode(value):
    mode = str(value or "auto").lower()
    if mode not in RUNTIME_MODES:
        raise ValueError(f"Unsupported runtime.mode: {value}")
    return mode


def _requested_device_type(cfg):
    device = str(cfg_get(cfg, "device", "cuda")).lower()
    return "cuda" if device.startswith("cuda") else "cpu"


def _cuda_is_usable():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _resolve_backend(device_type, configured_backend):
    backend = str(configured_backend or AUTO_BACKEND).lower()
    if backend == AUTO_BACKEND:
        return "nccl" if device_type == "cuda" else "gloo"
    return backend


def resolve_runtime_spec(cfg):
    requested_mode = _normalize_mode(cfg_get(cfg, "runtime.mode", "auto"))
    requested_device_type = _requested_device_type(cfg)
    cuda_usable = _cuda_is_usable()
    available_cuda = torch.cuda.device_count() if cuda_usable else 0
    configured_world_size = cfg_get(cfg, "runtime.world_size", None)
    configured_world_size = None if configured_world_size in (None, "null") else int(configured_world_size)

    if requested_mode == "auto":
        if requested_device_type != "cuda" or not cuda_usable:
            return RuntimeSpec(mode="cpu", device_type="cpu", backend=None, world_size=1)
        if available_cuda > 1:
            return RuntimeSpec(
                mode="ddp",
                device_type="cuda",
                backend=_resolve_backend("cuda", cfg_get(cfg, "runtime.backend", AUTO_BACKEND)),
                world_size=configured_world_size or available_cuda,
            )
        return RuntimeSpec(mode="single", device_type="cuda", backend=None, world_size=1)

    if requested_mode == "cpu":
        return RuntimeSpec(mode="cpu", device_type="cpu", backend=None, world_size=1)

    if requested_mode == "single":
        if requested_device_type != "cuda":
            raise ValueError("runtime.mode=single requires a CUDA device configuration.")
        if not cuda_usable:
            raise ValueError("runtime.mode=single requires CUDA, but no visible GPU is available.")
        return RuntimeSpec(mode="single", device_type="cuda", backend=None, world_size=1)

    device_type = "cuda" if requested_device_type == "cuda" and cuda_usable else "cpu"
    if device_type == "cuda":
        world_size = configured_world_size or available_cuda
        if world_size > available_cuda:
            raise ValueError(
                f"runtime.world_size={world_size} exceeds visible CUDA devices ({available_cuda})."
            )
    else:
        world_size = configured_world_size or 2

    if world_size < 2:
        raise ValueError("runtime.mode=ddp requires world_size >= 2.")

    return RuntimeSpec(
        mode="ddp",
        device_type=device_type,
        backend=_resolve_backend(device_type, cfg_get(cfg, "runtime.backend", AUTO_BACKEND)),
        world_size=world_size,
    )


def build_runtime_context(cfg, runtime_spec, rank=0):
    if runtime_spec.mode == "cpu":
        device = torch.device("cpu")
        return RuntimeContext(
            mode="cpu",
            device=device,
            device_type="cpu",
            distributed=False,
            rank=0,
            world_size=1,
            is_main=True,
            backend=None,
            ddp_device_ids=None,
        )

    if runtime_spec.mode == "single":
        device = torch.device(str(cfg_get(cfg, "device", "cuda")))
        return RuntimeContext(
            mode="single",
            device=device,
            device_type=device.type,
            distributed=False,
            rank=0,
            world_size=1,
            is_main=True,
            backend=None,
            ddp_device_ids=None,
        )

    if runtime_spec.device_type == "cuda":
        device = torch.device(f"cuda:{rank}")
        ddp_device_ids = [rank]
    else:
        device = torch.device("cpu")
        ddp_device_ids = None

    return RuntimeContext(
        mode="ddp",
        device=device,
        device_type=runtime_spec.device_type,
        distributed=True,
        rank=rank,
        world_size=runtime_spec.world_size,
        is_main=(rank == 0),
        backend=runtime_spec.backend,
        ddp_device_ids=ddp_device_ids,
    )


def prepare_distributed_runtime(cfg, runtime_context):
    if not runtime_context.distributed:
        return

    os.environ["MASTER_ADDR"] = str(cfg_get(cfg, "runtime.master_addr", "127.0.0.1"))
    os.environ["MASTER_PORT"] = str(cfg_get(cfg, "runtime.master_port", "29500"))
    debug_enabled = bool(cfg_get(cfg, "runtime.debug", False))
    if debug_enabled:
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
        if runtime_context.backend == "nccl":
            os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
            os.environ.setdefault("NCCL_DEBUG", "INFO")

    if runtime_context.device_type == "cuda":
        torch.cuda.set_device(runtime_context.rank)

    timeout_seconds = cfg_get(cfg, "runtime.timeout_seconds", None)
    init_kwargs = {}
    if timeout_seconds not in (None, "null"):
        timeout_seconds = int(timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("runtime.timeout_seconds must be a positive integer.")
        init_kwargs["timeout"] = timedelta(seconds=timeout_seconds)

    if debug_enabled:
        print(
            f"[Rank {runtime_context.rank}/{runtime_context.world_size}] init_process_group "
            f"backend={runtime_context.backend} device={runtime_context.device} "
            f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} "
            f"timeout={init_kwargs.get('timeout', 'default')}",
            flush=True,
        )

    dist.init_process_group(
        backend=runtime_context.backend,
        init_method="env://",
        world_size=runtime_context.world_size,
        rank=runtime_context.rank,
        **init_kwargs,
    )


def cleanup_distributed_runtime():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
