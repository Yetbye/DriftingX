"""Focal Log-Frequency Loss (f-loss) for pixel-space supervision.

From "Balancing Frequencies and Pixels in Flow Matching" (arXiv 2609.02748).
Natural images follow a 1/f^2 spectral distribution: pixel-space reconstruction
losses let low frequencies dominate the optimization signal and delay learning
of fine-scale detail. f-loss balances the learning signal across frequencies by
working in the 2D Fourier domain, emphasizing high-frequency components.

Definition (per the paper):
  1. error spectrum  e_{u,v} = |F(pred)_{u,v} - F(target)_{u,v}|   (2D FFT)
  2. focal normalization  e_hat = e / max_{u,v} e
  3. log-focal weighting over the radial frequency radius rho_{u,v}
     w = log(1 + rho / rho_max)  (emphasizes high frequency)
  4. L_f = mean( w * e_hat )

The paper's training strategy: emphasize frequency supervision early, then
transition to pixel (v-) loss for spatial refinement. We expose `focal_alpha`
to blend f-loss into a pixel-supervision objective:

    L = (1 - a) * L_pixel + a * L_freq

where `a` is the frequency weight (high early, ->0 late).

This is a drop-in spectral regularizer on the *final pixel output* — used here
to reinforce X-Drifting's pixel-space supervision with frequency awareness.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _radial_frequency(h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalized radial frequency radius rho in [0, 1] for each (u, v) bin."""
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)  # cycles/pixel in [-0.5, 0.5)
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype)
    rho = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    rho_max = rho.max().clamp(min=1e-8)
    return rho / rho_max  # (h, w) in [0, 1]


def focal_log_frequency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weight_high: float = 1.0,
) -> torch.Tensor:
    """Focal Log-Frequency Loss between pred and target.

    Args:
        pred, target: images in NCHW (or NHWC auto-detected), same shape.
        weight_high: strength of the log-focal high-frequency emphasis.

    Returns:
        Scalar f-loss.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    # to NCHW
    if pred.shape[-1] in (1, 3) and pred.shape[1] not in (1, 3):
        pred = pred.permute(0, 3, 1, 2).contiguous()
        target = target.permute(0, 3, 1, 2).contiguous()

    pred = pred.float()
    target = target.float()
    _, _, h, w = pred.shape

    # 2D FFT per-sample over spatial dims
    F_pred = torch.fft.rfft2(pred, norm="ortho")
    F_target = torch.fft.rfft2(target, norm="ortho")
    e = torch.abs(F_pred - F_target)  # (B, C, h, w//2+1) complex->real

    # focal normalization by the max error spectrum per sample/channel
    e_max = e.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    e_hat = e / e_max

    # log-focal high-frequency weight on the radial frequency radius
    rho = _radial_frequency(h, w // 2 + 1, e.device, e.dtype)  # (h, w//2+1)
    w_focal = torch.log1p(weight_high * rho)  # 0 at DC, grows with frequency

    return (w_focal[None, None] * e_hat).mean()


def pixel_frequency_blend(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    freq_weight: float = 0.5,
    pixel_loss: str = "l2",
    weight_high: float = 1.0,
) -> dict:
    """Blend pixel reconstruction loss with f-loss for pixel-space supervision.

    `freq_weight` (a) is the frequency emphasis: the paper uses high a early and
    decays it to 0 for final spatial refinement. Returns dict with the total loss
    and its components for logging.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    if pred.shape[-1] in (1, 3) and pred.shape[1] not in (1, 3):
        pred_n = pred.permute(0, 3, 1, 2).contiguous()
        target_n = target.permute(0, 3, 1, 2).contiguous()
    else:
        pred_n, target_n = pred, target

    if pixel_loss == "l1":
        pix = F.l1_loss(pred_n, target_n)
    else:
        pix = F.mse_loss(pred_n, target_n)

    freq = focal_log_frequency_loss(pred_n, target_n, weight_high=weight_high)
    a = float(freq_weight)
    total = (1.0 - a) * pix + a * freq
    return {"loss": total, "pixel_loss": pix.detach(), "freq_loss": freq.detach(), "freq_weight": a}


def freq_weight_schedule(step: int, total_steps: int, *, start: float = 0.8, end: float = 0.0,
                         transition_frac: float = 0.4) -> float:
    """Linear frequency-weight schedule: high early, decay to `end` for refinement.

    Matches the paper's "frequency early, pixel later" strategy: hold `start` for
    the first `transition_frac` of training, then linearly decay to `end`.
    """
    hold = int(total_steps * transition_frac)
    if step <= hold:
        return float(start)
    t = (step - hold) / max(1, total_steps - hold)
    return float(start + (end - start) * min(1.0, t))
