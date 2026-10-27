"""
Tests for geoformer.quant subsystem:
BitNet b1.58 STE, TernaryLinear, and Variance-Preserving Initialization.
"""

import torch
import pytest
from geoformer.quant import (
    RoundSTE,
    SignSTE,
    AbsMaxQuantizer,
    variance_preserving_init_,
    QuantizedLinear
)

def test_round_and_sign_ste():
    x = torch.tensor([1.2, -0.7, 0.4, -1.8], requires_grad=True)
    r = RoundSTE.apply(x)
    assert torch.equal(r, torch.tensor([1.0, -1.0, 0.0, -2.0]))
    loss = r.sum()
    loss.backward()
    assert torch.equal(x.grad, torch.ones_like(x)), "STE must pass gradients through unmodified"

    x2 = torch.tensor([2.0, -3.0, 0.0], requires_grad=True)
    s = SignSTE.apply(x2)
    s.sum().backward()
    assert torch.equal(x2.grad, torch.ones_like(x2))

def test_absmax_quantizer_ternary():
    w = torch.randn(16, 32)
    w_q, gamma = AbsMaxQuantizer.quantize_weight_ternary(w)
    unique_vals = torch.unique(w_q).tolist()
    for val in unique_vals:
        assert val in (-1.0, 0.0, 1.0), f"Ternary weight must be in {{-1, 0, 1}}, got {val}"
    assert gamma > 0.0

def test_quantized_linear_forward_backward():
    layer = QuantizedLinear(in_features=32, out_features=16, mode="ternary")
    x = torch.randn(4, 32, requires_grad=True)
    out = layer(x)
    assert out.shape == (4, 16)

    loss = out.pow(2).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0, "Gradients must flow to continuous weights via STE"
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

def test_variance_preserving_init():
    w = torch.empty(128, 256)
    variance_preserving_init_(w, mode="ternary")
    std = w.std().item()
    # For fan_in=256, expected std = sqrt(1.5 / 256) = 0.0765
    assert 0.05 < std < 0.11, f"Expected std ~0.0765, got {std}"
