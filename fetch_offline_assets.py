#!/usr/bin/env python
"""Download every external resource the ImageNet runs need, into one tree.

The training cluster is offline, so gather all weights/stats here (where there is
network) and copy the resulting directory across. Everything is laid out under
--dest (default: ./offline_bundle) with a manifest.json recording what was fetched.

What is fetched, and why (from the Drift / CrossFlow papers):
  - sdvae/                 SD-VAE f8 encoder+decoder. Needed to build cached
                           latents and to decode latent-space samples.
  - mae/<name>/            MAE feature extractors (the drift loss operates on
                           their multi-scale activations). MAE is trained on
                           latents (mae_latent_*) or pixels (mae_pixel_*).
  - convnext/              ConvNeXt-V2 (facebook/convnextv2-base-22k-224) used
                           as the extra pixel-space feature encoder for the
                           pixel / CrossFlow configs.
  - fid/pt_inception-*.pth Inception-V3 weights for FID/IS (pytorch-fid).
  - fid/imagenet_256_fid_stats.npz   Reference FID statistics for ImageNet-256.
  - imnet_stats/imagenet_val_prc_arr0.npz  Precision/recall stats (optional).

Usage:
    # on a networked machine
    python scripts/fetch_offline_assets.py --dest /path/to/offline_bundle
    # copy the whole directory to the cluster, then point env vars at it
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# The fetch script imports project modules (models.hf). Make sure the repo root
# is importable even when the script is invoked by absolute path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import shutil
import urllib.request
from pathlib import Path

FID_WEIGHTS_URL = (
    "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
)
# torch-fidelity (used by torchmetrics InceptionScore) ships its own copy of the
# same Inception-V3 weights under a different filename.
IS_WEIGHTS_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"
)
CONVNEXT_REPO = "facebook/convnextv2-base-22k-224"

# MAE artifacts published with the Drift release (HF repo Goodeat/drifting).
MAE_NAMES = ["mae_latent_256", "mae_latent_640", "mae_pixel_640"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _set_mirror(hf_endpoint: str | None) -> None:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        _log(f"[hf] using endpoint {hf_endpoint}")


def fetch_sdvae(dest: Path, repo_id: str) -> dict:
    from huggingface_hub import snapshot_download

    out = dest / "sdvae"
    out.mkdir(parents=True, exist_ok=True)
    _log(f"[sdvae] {repo_id} -> {out}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out),
        allow_patterns=["config.json", "configuration.json", "diffusion_pytorch_model.safetensors"],
    )
    ok = (out / "config.json").is_file() and (out / "diffusion_pytorch_model.safetensors").is_file()
    return {"sdvae": {"path": "sdvae", "repo": repo_id, "ok": ok}}


def fetch_mae(dest: Path, repo_id: str, names: list[str]) -> dict:
    """Download + convert the MAE artifacts into the loader's expected layout.

    The offline loader resolves ``hf://<name>`` through HF_ROOT as
    ``<HF_ROOT>/models/mae/pytorch/<name>/{metadata.json,ema_params.pt}``
    (see models/hf.py:_download_artifact / load_mae_torch), so the bundle must
    place them exactly there for offline runs to work without hub access.
    """
    import torch as _torch

    from models.hf import _convert_mae_jax_params_to_torch, _load_flax_msgpack, _download_artifact, read_metadata

    hf_root = dest / "hf_root"
    results = {}
    for name in names:
        out_dir = hf_root / "models" / "mae" / "pytorch" / name
        try:
            if (out_dir / "metadata.json").is_file() and (out_dir / "ema_params.pt").is_file():
                _log(f"[mae] {name} already present: {out_dir}")
                results[name] = {"path": str(out_dir.relative_to(dest)), "ok": True}
                continue

            artifact_dir = _download_artifact(
                repo_id=repo_id,
                kind="mae",
                backend="jax",
                model_id=name,
                output_root=str(dest / "_raw"),
                prefix=None,
            )
            metadata = read_metadata(artifact_dir)
            params = _load_flax_msgpack(artifact_dir / "ema_params.msgpack")
            torch_params = _convert_mae_jax_params_to_torch(params)

            out_dir.mkdir(parents=True, exist_ok=True)
            metadata = dict(metadata)
            metadata["backend"] = "pytorch"
            metadata["converted_from_backend"] = "jax"
            # Write params first, then metadata (whose presence marks completeness).
            tmp = out_dir / "ema_params.pt.tmp"
            _torch.save(torch_params, tmp)
            tmp.replace(out_dir / "ema_params.pt")
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            results[name] = {"path": str(out_dir.relative_to(dest)), "ok": True}
            _log(f"[mae] {name} -> {out_dir}")
        except Exception as exc:  # noqa: BLE001
            results[name] = {"path": str(out_dir.relative_to(dest)), "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"}
            _log(f"[mae] {name} FAILED: {type(exc).__name__}: {exc}")
    return results


def fetch_convnext(dest: Path, repo_id: str = CONVNEXT_REPO) -> dict:
    from huggingface_hub import snapshot_download

    out = dest / "convnext" / "convnextv2-base-22k-224"
    out.mkdir(parents=True, exist_ok=True)
    _log(f"[convnext] {repo_id} -> {out}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(out))
        return {"convnext": {"path": "convnext/convnextv2-base-22k-224", "repo": repo_id, "ok": True}}
    except Exception as exc:  # noqa: BLE001
        return {"convnext": {"path": "convnext/convnextv2-base-22k-224", "repo": repo_id,
                             "ok": False, "error": f"{type(exc).__name__}: {exc}"}}


def fetch_fid_weights(dest: Path) -> dict:
    """Download both Inception-V3 weight files into the torch hub cache layout.

    pytorch-fid and torch-fidelity both resolve their weights through
    torch.hub.load_state_dict_from_url, which reads $TORCH_HOME/hub/checkpoints
    before touching the network. Placing both files there makes FID and IS fully
    offline as long as TORCH_HOME points at this bundle.
    """
    out_dir = dest / "torch" / "hub" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for key, url in (("fid_weights", FID_WEIGHTS_URL), ("is_weights", IS_WEIGHTS_URL)):
        target = out_dir / Path(url).name
        rel = str(target.relative_to(dest))
        if target.is_file() and target.stat().st_size > 50_000_000:
            _log(f"[{key}] already present: {target}")
            results[key] = {"path": rel, "ok": True}
            continue
        _log(f"[{key}] {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(target)
            results[key] = {"path": rel, "ok": True}
        except Exception as exc:  # noqa: BLE001
            results[key] = {"path": rel, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


def _copy_stats(src: str | None, dest: Path, dest_path: Path, label: str) -> dict:
    rel = str(dest_path.relative_to(dest))
    if not src:
        return {label: {"path": rel, "ok": False, "error": "not provided"}}
    src_p = Path(src).expanduser()
    if not src_p.is_file():
        return {label: {"path": rel, "ok": False, "error": f"missing {src_p}"}}
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if src_p.resolve() != dest_path.resolve():
        shutil.copy2(src_p, dest_path)
    return {label: {"path": rel, "ok": True}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="offline_bundle", help="Output directory for the bundle.")
    ap.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID", "Goodeat/drifting"))
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                    help="HF mirror to use ('' to use the default endpoint).")
    ap.add_argument("--sdvae-repo", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--mae", nargs="*", default=MAE_NAMES)
    ap.add_argument("--fid-stats", default=os.environ.get("IMAGENET_FID_NPZ", ""),
                    help="Existing imagenet_256_fid_stats.npz to bundle.")
    ap.add_argument("--pr-stats", default=os.environ.get("IMAGENET_PR_NPZ", ""),
                    help="Existing imagenet_val_prc_arr0.npz to bundle (optional).")
    ap.add_argument("--skip", nargs="*", default=[], help="Subsets to skip: sdvae mae convnext fid_weights stats.")
    args = ap.parse_args()

    _set_mirror(args.hf_endpoint or None)
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)
    manifest: dict = {"dest": str(dest), "items": {}}

    if "sdvae" not in skip:
        manifest["items"].update(fetch_sdvae(dest, args.sdvae_repo))
    if "mae" not in skip:
        manifest["items"].update(fetch_mae(dest, args.hf_repo_id, args.mae))
    if "convnext" not in skip:
        manifest["items"].update(fetch_convnext(dest))
    if "fid_weights" not in skip:
        manifest["items"].update(fetch_fid_weights(dest))
    if "stats" not in skip:
        manifest["items"].update(
            _copy_stats(args.fid_stats, dest, dest / "fid" / "imagenet_256_fid_stats.npz", "fid_stats")
        )
        manifest["items"].update(
            _copy_stats(args.pr_stats, dest, dest / "imnet_stats" / "imagenet_val_prc_arr0.npz", "pr_stats")
        )

    # Drop the raw JAX downloads now that the torch conversion exists.
    raw = dest / "_raw"
    if raw.exists() and "mae" not in skip:
        shutil.rmtree(raw, ignore_errors=True)

    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _log("\n=== summary ===")
    for k, v in manifest["items"].items():
        _log(f"  {k:14s} {'OK ' if v.get('ok') else 'FAIL'} {v.get('path')}"
             + (f"  ({v['error']})" if not v.get("ok") and v.get("error") else ""))
    _log(f"\nmanifest: {dest / 'manifest.json'}")
    failed = [k for k, v in manifest["items"].items() if not v.get("ok")]
    if failed:
        _log(f"WARNING: {len(failed)} item(s) failed: {failed}")
        return 1
    _log("all assets fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
