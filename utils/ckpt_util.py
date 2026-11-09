from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from utils.logging import log_for_0


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove the _orig_mod. prefix that torch.compile adds to parameter names."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def _is_rank_zero() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _to_python_int(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().reshape(-1)[0].item())
    return int(x)


def _output_root(workdir: Optional[str] = None) -> Path:
    if workdir:
        return Path(workdir).resolve()
    return Path("runs").resolve()


def _job_ckpt_dir(workdir: Optional[str] = None) -> Path:
    return _output_root(workdir) / "checkpoints"


def _ckpt_name(step: int) -> str:
    return f"step_{int(step):09d}.pt"


def _checkpoints_desc(ckpt_dir: Path) -> list[Path]:
    """All step_*.pt, newest first."""
    return sorted(ckpt_dir.glob("step_*.pt"), reverse=True)


def _latest_checkpoint_path(ckpt_dir: Path) -> Path | None:
    ckpts = _checkpoints_desc(ckpt_dir)
    if not ckpts:
        return None
    return ckpts[0]


def _load_payload(path: Path) -> dict | None:
    """Load a checkpoint, returning None if the file is unreadable/truncated.

    A preempted job can leave a partially-written file behind; because saves are
    atomic (tmp + rename) the final name should never be truncated, but we still
    tolerate it so one bad file cannot kill a long run.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - corrupt/truncated pickle
        log_for_0("WARNING: cannot read checkpoint %s (%s); skipping", str(path), type(exc).__name__)
        return None
    if not isinstance(payload, dict) or "model" not in payload:
        log_for_0("WARNING: checkpoint %s has no 'model' entry; skipping", str(path))
        return None
    return payload


def _restore_rng(payload: dict) -> None:
    rng = payload.get("rng")
    if not isinstance(rng, dict):
        return
    try:
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            import numpy as np

            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            import random

            random.setstate(rng["python"])
    except Exception as exc:  # noqa: BLE001 - best-effort fidelity only
        log_for_0("WARNING: could not restore RNG state (%s); continuing", type(exc).__name__)


def _capture_rng() -> dict:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001
        pass
    try:
        import random

        state["python"] = random.getstate()
    except Exception:  # noqa: BLE001
        pass
    return state


def restore_checkpoint(step=None, state=None, workdir: Optional[str] = None):
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    if not ckpt_dir.exists():
        log_for_0("No local checkpoint dir at %s", str(ckpt_dir))
        return state

    if step is not None:
        candidates = [ckpt_dir / _ckpt_name(int(step))]
    else:
        candidates = _checkpoints_desc(ckpt_dir)
    if not candidates:
        return state

    payload = None
    ckpt_path = None
    for path in candidates:
        if not path.exists():
            continue
        payload = _load_payload(path)
        if payload is not None:
            ckpt_path = path
            break
    if payload is None:
        log_for_0("No readable checkpoint in %s", str(ckpt_dir))
        return state

    if state is None:
        return payload

    log_for_0("Resuming from %s (step=%s)", str(ckpt_path), payload.get("step"))
    state.model.load_state_dict(_strip_compiled_prefix(payload["model"]))
    ema_payload = payload.get("ema_model", payload.get("ema_params"))
    if hasattr(state, "ema_model") and state.ema_model is not None and ema_payload is not None:
        state.ema_model.load_state_dict(_strip_compiled_prefix(ema_payload))
    if hasattr(state, "ema_params") and ema_payload is not None:
        state.ema_params = {k: v.clone() for k, v in ema_payload.items()}
    if state.optimizer is not None and "optimizer" in payload:
        state.optimizer.load_state_dict(payload["optimizer"])
    state.step = int(payload.get("step", 0))
    state.ema_decay = float(payload.get("ema_decay", state.ema_decay))
    _restore_rng(payload)
    return state


def save_checkpoint(state, keep=2, keep_every=None, workdir: Optional[str] = None):
    if not _is_rank_zero():
        return
    ckpt_dir = _job_ckpt_dir(workdir=workdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(state, "ema_model") and state.ema_model is not None:
        ema_payload = state.ema_model.state_dict()
    elif hasattr(state, "ema_params"):
        ema_payload = state.ema_params
    else:
        ema_payload = None

    payload = {
        "step": _to_python_int(state.step),
        "model": _strip_compiled_prefix(state.model.state_dict()),
        "ema_model": _strip_compiled_prefix(ema_payload) if ema_payload is not None else None,
        "ema_params": _strip_compiled_prefix(ema_payload) if ema_payload is not None else None,
        "optimizer": state.optimizer.state_dict() if state.optimizer is not None else None,
        "ema_decay": float(state.ema_decay),
        "rng": _capture_rng(),
    }
    out = ckpt_dir / _ckpt_name(_to_python_int(state.step))
    # Write to a temp file in the same directory and rename, so a preemption or
    # crash mid-write cannot leave a truncated file under the final name.
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    log_for_0("Saving checkpoint step %d to %s", _to_python_int(state.step), str(out))

    keep_every = int(keep_every) if keep_every else None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if keep is not None and keep > 0 and len(ckpts) > keep:
        for p in ckpts[: len(ckpts) - keep]:
            try:
                step = int(p.stem.removeprefix("step_"))
            except ValueError:
                step = None
            if keep_every and step is not None and step % keep_every == 0:
                continue
            p.unlink(missing_ok=True)


def save_params_ema_artifact(
    state: Any,
    *,
    workdir: Optional[str] = None,
    kind: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> Path:
    if not _is_rank_zero():
        return _output_root(workdir) / "params_ema"
    step = _to_python_int(state.step)
    ema_decay = float(getattr(state, "ema_decay"))

    out_dir = _output_root(workdir) / "params_ema"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "ema_params.pt"
    if hasattr(state, "ema_model") and state.ema_model is not None:
        ema_payload = state.ema_model.state_dict()
    elif hasattr(state, "ema_params"):
        ema_payload = state.ema_params
    else:
        ema_payload = {}
    tmp_params = params_path.with_suffix(params_path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(_strip_compiled_prefix(ema_payload), tmp_params)
        os.replace(tmp_params, params_path)
    finally:
        if tmp_params.exists():
            tmp_params.unlink(missing_ok=True)

    metadata = {
        "format": "torch.state_dict",
        "kind": kind,
        "backend": "torch",
        "ema_decay": ema_decay,
        "step": step,
        "path": "params_ema/ema_params.pt",
        "model_config": dict(model_config or {}),
    }
    # metadata last: its presence marks the artifact as complete and readable.
    tmp_meta = out_dir / f"metadata.json.tmp.{os.getpid()}"
    tmp_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_meta, out_dir / "metadata.json")
    log_for_0("Saved EMA params artifact step %d to %s", step, str(out_dir))
    return out_dir
