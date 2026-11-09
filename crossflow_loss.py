"""CrossFlow L_CF objective (the paper's eq. 11'), for the cross-space comparison.

CrossFlow trains F_theta to map a *noised latent* z_t to a *pixel* image x0 under
a differential constraint along the latent trajectory. The concrete objective
(paper eq. 11') is

    L_CF = E_{x0, eps, t, r} || (r/t^2) * ( F(z_t,t,r) + (t(t-r)/r) * dF/dt - x0 ) ||^2

where dF/dt is the *total* derivative of F along the latent noising path
z_t = (1-t) z + t eps, z = E(x0), estimated with a forward-mode Jacobian-vector
product (torch.func.jvp) and a stop-gradient on the JVP term.

This module implements the drift-framework-compatible form. Because the Drift
framework generates from a single latent prior sample (t=1, r=0, one step), we
support the general (t, r) sampling used for training so the objective is faithful,
while one-step inference uses t=1, r=0.

We keep this *separate* from the drift loss (`drift_loss.py`) so the two
objectives can be compared cleanly under the same backbone (see
docs/EXPERIMENT-DESIGN.md, E5/E6 vs E3).
"""

from __future__ import annotations

from typing import Optional

import torch


def _linear_schedule_alpha(t):
    return 1.0 - t


def _linear_schedule_beta(t):
    return t


def sample_tr(
    batch_size: int,
    device: torch.device,
    *,
    time_shift: Optional[float] = None,
    p_recon: float = 0.5,
    generator: Optional[torch.Generator] = None,
):
    """Sample (t, r) with 0 <= r <= t <= 1, per CrossFlow's time sampling.

    With probability ``p_recon`` we set r = t (reconstruction endpoint). Otherwise
    we sample two shifted times independently and order them (t >= r). The
    optional ``time_shift`` applies the RAE shifted-time transform
    t -> a t / (1 + (a-1) t), a = sqrt(latent_dim / 4096).
    """
    def _shift(u):
        if time_shift is None:
            return u
        a = float(time_shift)
        return a * u / (1.0 + (a - 1.0) * u)

    t = torch.rand(batch_size, device=device, generator=generator)
    t = _shift(t)
    if torch.rand(batch_size, device=device, generator=generator).max() < p_recon:
        r = t.clone()
    else:
        u2 = _shift(torch.rand(batch_size, device=device, generator=generator))
        r = torch.minimum(t, u2)
        t = torch.maximum(t, u2)
    return t, r


def crossflow_loss(
    model,
    encoder_fn,
    x0_pixel,
    labels,
    *,
    t: torch.Tensor,
    r: torch.Tensor,
    pixel_loss: str = "l2",
    eps: float = 1e-5,
):
    """Compute the CrossFlow L_CF residual for a batch.

    Args:
        model: cross-space generator F(z_t, t, r, y) -> pixel image.
        encoder_fn: frozen VAE encoder E(x0_pixel) -> clean latent z (NHWC->NCHW ok).
        x0_pixel: clean pixel images (B, H, W, C) in [-1, 1].
        labels: class labels (B,).
        t, r: (B,) timesteps with 0 <= r <= t <= 1.
        pixel_loss: 'l1' or 'l2' on the residual.

    Returns:
        (loss, info) where loss is a scalar tensor.
    """
    if pixel_loss not in ("l1", "l2"):
        raise ValueError(f"pixel_loss must be 'l1' or 'l2', got {pixel_loss!r}")

    bsz = x0_pixel.shape[0]
    device = x0_pixel.device

    # Clean latent z = E(x0); noise eps ~ N(0, I) in latent space.
    with torch.no_grad():
        x0_nchw = x0_pixel.permute(0, 3, 1, 2).contiguous() if x0_pixel.shape[-1] in (1, 3) else x0_pixel
        z = encoder_fn(x0_nchw)  # (B, C_lat, H_lat, W_lat)

    eps = torch.randn_like(z)
    t_ = t.to(device=device, dtype=z.dtype).view(-1, 1, 1, 1)
    r_ = r.to(device=device, dtype=z.dtype).view(-1, 1, 1, 1)

    a_t = _linear_schedule_alpha(t_)
    b_t = _linear_schedule_beta(t_)
    z_t = a_t * z + b_t * eps  # (B, C_lat, H_lat, W_lat)

    # dF/dt along the path: total derivative via JVP with tangent = dz/dt = eps - z.
    # model_fn(z) returns pixel output; we need d(model_fn)/dt at z_t in direction (eps - z).
    def model_fn(z_in, tt, rr):
        return model(z_in, tt, rr)

    # Forward-mode JVP: directional derivative of F along the latent trajectory.
    # Use torch.func.jvp so it composes cleanly (no DDP wrapper inside).
    tangent_t = torch.ones_like(t)  # d/dt of the time argument
    tangent_r = torch.zeros_like(r)  # r held fixed
    tangent_z = (eps - z)  # dz/dt = eps - z  (for linear schedule)

    # Model signature varies; call via a closure that folds (t, r) as explicit inputs.
    def f(z_in, t_in, r_in):
        return model.forward_pixel(z_in, t_in, r_in, labels)

    if hasattr(model, "forward_pixel"):
        # JVP over (z, t) with (tangent_z, tangent_t); r constant.
        _, dF = torch.func.jvp(
            lambda zz, tt: f(zz, tt, r),
            (z_t, t),
            (tangent_z, tangent_t),
        )
    else:
        # Fallback: finite-difference directional derivative (slower, used only if
        # the model lacks a JVP-friendly forward).
        with torch.no_grad():
            f0 = model_fn(z_t, t, r)
        dt = eps.to(dtype=torch.float32)
        z2 = z_t + dt.view(-1, 1, 1, 1) * 1e-3
        f1 = model_fn(z2, t + 1e-3, r)
        dF = (f1 - f0) / 1e-3

    dF = dF.detach()  # stop-grad on the JVP term (CrossFlow design)

    F = model_fn(z_t, t, r)  # pixel output
    # Reconstruct-compatible prediction: F + (t(t-r)/r) * dF  (eq. 11' bracket)
    # Guard r -> 0 (reconstruction endpoint) where t(t-r)/r is singular.
    r_safe = torch.clamp(r_, min=eps)
    coeff = (t_ * (t_ - r_safe) / r_safe)
    pred = F + coeff * dF

    target = x0_pixel.permute(0, 3, 1, 2).contiguous() if x0_pixel.shape[-1] in (1, 3) else x0_pixel
    pred_nchw = pred.permute(0, 3, 1, 2).contiguous() if pred.shape[-1] in (1, 3) else pred

    residual = (r_safe / (t_ ** 2)) * (pred_nchw - target)
    if pixel_loss == "l2":
        loss = (residual ** 2).mean()
    else:
        loss = residual.abs().mean()

    info = {
        "cf_loss": loss.detach(),
        "cf_residual_norm": residual.norm().detach() / max(1, residual.numel()),
    }
    return loss, info
