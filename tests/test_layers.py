"""
Tests for geoformer.layers subsystem:
Attention (Q-GHA), Continuous Relaxation (Q-TAC), S5 Projector, and SwiGLU.
"""

import torch
import pytest
from geoformer.layers import QuantizedGHA, QuantizedTAC, S5CayleyProjector, TernarySwiGLU

def test_ternary_swiglu():
    swiglu = TernarySwiGLU(d_model=32, mode="ternary")
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = swiglu(x)
    assert out.shape == (2, 8, 32)
    out.sum().backward()
    assert x.grad is not None

def test_s5_cayley_projector_bound():
    projector = S5CayleyProjector(d_model=32)
    h = torch.randn(2, 16, 32)
    h_sym, K_S = projector(h)
    assert h_sym.shape == (2, 16, 32)
    # Check Theorem Q-4 bound: K(S) <= 54.68 bits
    assert K_S.item() <= 54.68, f"Theorem Q-4 bound violated: K(S) = {K_S.item()} > 54.68"

def test_quantized_gha_attention():
    gha = QuantizedGHA(d_model=32, n_heads=4, mode="ternary", lambda_geo=0.1)
    h = torch.randn(2, 8, 32, requires_grad=True)
    d_mat = torch.ones(8, 8) * 2.0
    out, attn = gha(h, d_matrix=d_mat)
    assert out.shape == (2, 8, 32)
    assert attn.shape == (2, 4, 8, 8)
    # Check attention rows sum to 1.0
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-5)

def test_quantized_tac_relaxation():
    tac = QuantizedTAC(d_model=32, tau_max=4)
    h_init = torch.randn(2, 8, 32)
    L_matrix = torch.eye(8) * 2.0 - torch.ones(8, 8) * 0.25
    h_star, actual_tau, K_S = tac(h_init, L_matrix)
    assert h_star.shape == (2, 8, 32)
    assert 1 <= actual_tau <= 4
    assert K_S.item() <= 54.68
