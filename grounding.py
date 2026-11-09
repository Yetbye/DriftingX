"""Frozen-encoder representation grounding for X-Drifting.

This module supplies the drift loss's feature space phi(.) from a *frozen*
pretrained vision encoder rather than a pixel-MAE. It implements the grounding
mechanism that resolves the "latent representation" gap in cross-space drifting:

  * **Grounding (A)** — the drift attraction-repulsion is evaluated in the frozen
    encoder's representation space. Because the encoder is a representation prior
    (DINO-family, or an autoencoder's encoder), the *quality of the latent
    representation* directly shapes generation. Swapping the encoder (SD-VAE ->
    RAE -> PAE) therefore produces a measurable difference: this is the latent-
    representation axis of the paper.

  * **Predict-then-condition (B)** — the generator first predicts the frozen
    representation and then conditions on its own prediction (DINO-Forcing
    style). We expose `predict_condition_features` so the training loop can add a
    look-ahead representation head and self-condition, instead of co-denoising a
    representation target.

Encoders
--------
Each encoder is frozen and maps a clean (or noisy) image to per-patch tokens on a
shared grid. The drift loss consumes a dict of feature tensors, so we return a
multi-scale dict compatible with `models/mae_model.build_activation_function`:

    {
      "global":          (B, 1, D)          flattened image summary
      "g/<name>_mean":   (B, 1, D)          global mean over patches
      "g/<name>_std":    (B, 1, D)          global std over patches
      "g/<name>":        (B, L, D)          per-patch tokens
      "g/<name>_mean_2": (B, L/4, D)        2x2 patch means  (optional)
      ...
    }

This keeps X-Drifting's multi-temperature drift machinery unchanged while
swapping *where* phi lives: pixel-MAE (baseline) vs frozen foundation features
(grounded).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# feature dict helpers (mirror models/mae_model.MAEResNet.get_activations)
# --------------------------------------------------------------------------- #
def _safe_std(x: torch.Tensor, axis, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=axis, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=axis, keepdim=False)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)


def tokens_to_feature_dict(
    tokens: torch.Tensor,
    *,
    name: str = "enc",
    patch_mean_size=(2, 4),
    patch_std_size=(2, 4),
    use_mean: bool = True,
    use_std: bool = True,
) -> Dict[str, torch.Tensor]:
    """Turn (B, L, D) per-patch tokens into the drift feature dict.

    L is assumed to be a square grid (h = w = sqrt(L)).
    """
    B, L, D = tokens.shape
    h = w = int(math.sqrt(L))
    if h * w != L:
        # non-square token grid: keep only per-patch + global aggregates
        out = {f"g/{name}": tokens}
        if use_mean:
            out[f"g/{name}_mean"] = tokens.mean(dim=1, keepdim=True)
        if use_std:
            out[f"g/{name}_std"] = _safe_std(tokens, axis=1)[:, None, :]
        return out

    feat = tokens.reshape(B, h, w, D)
    out: Dict[str, torch.Tensor] = {}
    out[f"g/{name}"] = tokens  # (B, L, D)
    if use_mean:
        out[f"g/{name}_mean"] = feat.mean(dim=(1, 2), keepdim=False)[:, None, :]
    if use_std:
        out[f"g/{name}_std"] = _safe_std(feat, axis=(1, 2))[:, None, :]

    for size in patch_mean_size or ():
        if h % size == 0 and w % size == 0:
            resh = tokens.reshape(B, h // size, size, w // size, size, D)
            resh = resh.permute(0, 1, 3, 2, 4, 5).reshape(B, (h // size) * (w // size), size * size, D)
            out[f"g/{name}_mean_{size}"] = resh.mean(dim=2)
    for size in patch_std_size or ():
        if h % size == 0 and w % size == 0:
            resh = tokens.reshape(B, h // size, size, w // size, size, D)
            resh = resh.permute(0, 1, 3, 2, 4, 5).reshape(B, (h // size) * (w // size), size * size, D)
            out[f"g/{name}_std_{size}"] = _safe_std(resh, axis=2)
    return out


# --------------------------------------------------------------------------- #
# frozen encoders
# --------------------------------------------------------------------------- #
class FrozenEncoder(nn.Module):
    """Uniform interface: image (B,3,H,W) in [-1,1] -> per-patch tokens (B,L,D)."""

    latent_dim: int
    patch_grid: int  # sqrt(L)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DinoEncoder(FrozenEncoder):
    """Frozen DINOv2/DINOv3 backbone (PixelDiT2 / DinoForcing / RAE's encoder).

    `source` is a HF hub id or a local path to a torch HF-weights directory.
    """

    def __init__(self, source: str = "facebook/dinov2-with-registers-base", patch: int = 14, device=None):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(source)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.patch = patch
        self.latent_dim = int(self.model.config.hidden_size)
        self.patch_grid = 0  # depends on input size
        self._device = device

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1,1] NCHW -> [0,1] -> ImageNet-normalized
        x = (x + 1) / 2
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state  # (B, 1+L, D) or (B, L, D)
        return tokens[:, -self.model.config.num_patches :] if tokens.shape[1] > self._num_patches(x) else tokens


class AutoencoderEncoder(FrozenEncoder):
    """Frozen autoencoder *encoder* (SD-VAE / RAE / PAE).

    All three are autoencoders whose encoder maps image -> latent; we use only the
    encoder (never the decoder) so generation stays free of reconstruction loss.
    `encode_fn` maps an NCHW image in [-1,1] to (B, C, H, W) or (B, L, D) latent.
    """

    def __init__(self, encode_fn, latent_dim: int, patch_grid: int):
        super().__init__()
        self._encode = encode_fn
        self.latent_dim = latent_dim
        self.patch_grid = patch_grid

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # accept NHWC or NCHW; the underlying VAE encoder wants NCHW
        if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        z = self._encode(x)
        if z.dim() == 4:  # (B, C, H, W) -> (B, H*W, C)
            if z.shape[1] not in (1, 3, 4) and z.shape[-1] in (1, 3, 4):
                z = z.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
            B, C, H, W = z.shape
            z = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
            self.patch_grid = H
            self.latent_dim = C
        return z.to(x.device)


def _vae_encode_factory(device):
    from dataset.vae import vae_enc_decode

    enc, _ = vae_enc_decode()
    return enc


def build_frozen_encoder(kind: str, device=None, **kw) -> FrozenEncoder:
    """Build a frozen encoder for representation grounding.

    kind:
      "sdvae" | "vae"   -> SD-VAE encoder (4ch f8 latent)
      "rae"             -> DINO-based RAE encoder (768ch)
      "pae"             -> PAE (DINOv3 + prior-aligned compressor)
      "dino"            -> raw DINOv2/v3 backbone tokens
    """
    kind = kind.lower()
    if kind in ("sdvae", "vae"):
        enc = _vae_encode_factory(device)
        return AutoencoderEncoder(enc, latent_dim=4, patch_grid=0)
    if kind == "dino":
        return DinoEncoder(source=kw.get("source", "facebook/dinov2-with-registers-base"))
    if kind == "rae":
        # RAE = DINO encoder + normalization stats (models/stats/...). Its encoder is
        # a frozen DINO; the decoder is never used. Prefer a local RAE encoder if the
        # RAE repo is on disk, else fall back to the DINO backbone.
        rae_encoder = kw.get("rae_encode_fn")
        if rae_encoder is not None:
            return AutoencoderEncoder(rae_encoder, latent_dim=kw.get("latent_dim", 768), patch_grid=0)
        return DinoEncoder(source=kw.get("source", "facebook/dinov2-with-registers-base"))
    if kind == "pae":
        pae_encode_fn = kw.get("pae_encode_fn")
        if pae_encode_fn is not None:
            return AutoencoderEncoder(pae_encode_fn, latent_dim=kw.get("latent_dim", 32), patch_grid=0)
        return DinoEncoder(source=kw.get("source", "facebook/dinov3-large"))
    raise ValueError(f"Unknown frozen encoder kind: {kind!r}")


class GroundingFeatureExtractor(nn.Module):
    """Drift feature extractor phi(.) backed by a frozen encoder (grounding).

    Drop-in replacement for the pixel-MAE activation: call it on generated images
    (NHWC in [-1,1]) to get the feature dict consumed by drift_loss.
    """

    def __init__(self, encoder: FrozenEncoder, *, patch_mean_size=(2, 4),
                 patch_std_size=(2, 4), use_mean=True, use_std=True):
        super().__init__()
        self.encoder = encoder
        self.patch_mean_size = tuple(patch_mean_size or ())
        self.patch_std_size = tuple(patch_std_size or ())
        self.use_mean = use_mean
        self.use_std = use_std

    def get_activations(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # x: NHWC [-1,1] -> NCHW for the encoder
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x_nchw = x.permute(0, 3, 1, 2).contiguous()
        else:
            x_nchw = x
        tokens = self.encoder.encode(x_nchw.float())  # (B, L, D)
        feats = tokens_to_feature_dict(
            tokens,
            name="g",
            patch_mean_size=self.patch_mean_size,
            patch_std_size=self.patch_std_size,
            use_mean=self.use_mean,
            use_std=self.use_std,
        )
        # "global" first entry keeps drift_loss's shape contract
        return {"global": x.reshape(x.shape[0], 1, -1), **feats}

    def forward(self, x, **kwargs):
        return self.get_activations(x, **kwargs)


# --------------------------------------------------------------------------- #
# (B) predict-then-condition: representation head + self-conditioning
# --------------------------------------------------------------------------- #
class RepresentationHead(nn.Module):
    """Look-ahead head that predicts the frozen representation from a hidden state.

    DINO-Forcing style: the generator predicts the pretrained representation it
    needs, then conditions on its own prediction. The head maps a (B, L, D_model)
    hidden state to (B, L, D_enc) representation tokens, supervised against the
    frozen encoder's features with an alignment (REPA-style) loss.
    """

    def __init__(self, model_dim: int, enc_dim: int, num_tokens: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, enc_dim),
        )
        self.num_tokens = num_tokens

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)

    @staticmethod
    def align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """REPA-style cosine alignment between predicted and frozen tokens."""
        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).mean()


def predict_condition_features(
    hidden: torch.Tensor,
    rep_head: RepresentationHead,
    target_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (alignment loss, predicted representation) for (B).

    `hidden` is the generator's intermediate state; `target_tokens` is the frozen
    encoder's representation of the (clean) target image. The predicted rep is fed
    back as conditioning in the same step (self-conditioning), matching DINO-Forcing.
    """
    pred = rep_head(hidden)
    loss = RepresentationHead.align_loss(pred, target_tokens)
    return loss, pred
