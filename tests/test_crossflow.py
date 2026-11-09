"""CrossFlow backbone: official size presets and Drift-framework interface.

The backbone is vendored from the official C-Fun/cross-flow repo
(`models/crossflow_official.py`) and must match its parameter counts exactly at
`latent_channels=4` — this is the fairness guarantee for comparing against
CrossFlow. See docs/BACKBONE-COMPARISON.md.
"""

import torch

from CrossFlow import CrossFlow, _OFFICIAL_SIZE, resolve_size
from models.crossflow_official import CROSSFLOWDIT_PRESETS
from models.generator import build_generator_from_config


# Official reference param counts (from C-Fun/cross-flow, latent_channels=4).
OFFICIAL_PARAMS = {
    "B": 87_437_440,
    "L": 305_589_120,
    "XL": 450_327_104,
}


def _small_crossflow():
    return CrossFlow(
        num_classes=10,
        img_size=64,
        latent_channels=4,
        latent_size=8,
        latent_patch_size=1,
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_class_tokens=2,
        num_time_tokens=1,
    )


def test_params_match_official_exactly():
    """The fairness guarantee: byte-identical param counts to the official repo."""
    for size, expected in OFFICIAL_PARAMS.items():
        m = CrossFlow(size=size, num_classes=1000, latent_channels=4)
        assert m.num_parameters() == expected, f"{size}: {m.num_parameters()} != {expected}"


def test_vendored_backbone_matches_wrapper():
    for size in OFFICIAL_PARAMS:
        wrapped = CrossFlow(size=size, num_classes=1000, latent_channels=4).num_parameters()
        direct = sum(p.numel() for p in CROSSFLOWDIT_PRESETS[size]().parameters())
        assert wrapped == direct, f"{size}: wrapper {wrapped} != vendored {direct}"


def test_resolve_size_matches_official_depth_and_hidden():
    for name, preset in _OFFICIAL_SIZE.items():
        hidden, depth, heads = resolve_size(name)
        assert (hidden, depth, heads) == (preset["hidden_size"], preset["depth"], preset["num_heads"])


def test_resolve_size_prefers_explicit_overrides():
    hidden, depth, heads = resolve_size("B", hidden_size=256, depth=3, num_heads=8)
    assert (hidden, depth, heads) == (256, 3, 8)


def test_explicit_dims_without_size_still_work():
    model = _small_crossflow()
    assert model.size is None
    out = model(c=torch.arange(2))["samples"]
    assert out.shape == (2, 64, 64, 3)


def test_forward_shape_and_range():
    model = _small_crossflow()
    out = model(c=torch.tensor([1, 2, 3]), cfg_scale=torch.tensor([1.0, 1.5, 2.0]))["samples"]
    assert out.shape == (3, 64, 64, 3)
    assert torch.isfinite(out).all()


def test_backward_flows_to_all_parameters():
    model = _small_crossflow()
    model.train()
    loss = model(c=torch.arange(2))["samples"].pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_official_modules_present():
    """Sanity: the vendored backbone has the official module stack."""
    from models.crossflow_official import (
        BottleneckPatchEmbedder,
        RMSNorm,
        RoPEAttention,
        SwiGLUMlp,
        TransformerBlock,
    )

    m = _small_crossflow().backbone
    block = m.blocks[0]
    assert isinstance(block, TransformerBlock)
    assert isinstance(block.attn, RoPEAttention)
    assert isinstance(block.mlp, SwiGLUMlp)
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(m.x_embedder, BottleneckPatchEmbedder)
    assert hasattr(block, "attn_scale") and hasattr(block, "mlp_scale")


def test_factory_dispatches_crossflow_config():
    model = _small_crossflow()
    rebuilt = build_generator_from_config(model.model_config, num_classes=10)
    assert isinstance(rebuilt, CrossFlow)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.num_parameters() == model.num_parameters()
