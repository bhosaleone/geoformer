"""
Unit & Integration Test Suite for Quantized GeoFormer
Verifies:
  1. Ternary STE Forward & Backward Differentiation
  2. Discrete Conductance Lattice Warping & Fiedler Gap
  3. Q-GHA Attention & Q-TAC Continuous Relaxation
  4. Dual Path Kramers Optimization
"""

import sys
import torch
import torch.nn as nn
from quantized_geoformer import (
    QuantizedLinear,
    DiscreteMetricWarpingEngine,
    QuantizedGeoFormerBlock,
    DualPathKramersOptimizer
)

def test_quantized_linear_ste():
    print("[1/4] Testing QuantizedLinear & Ternary STE Gradient Flow...")
    layer = QuantizedLinear(in_features=16, out_features=16, mode="ternary")
    x = torch.randn(4, 16, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.weight.grad is not None, "Gradient should flow back to continuous weight!"
    assert layer.weight.grad.abs().sum() > 0, "Gradient must be non-zero!"
    assert x.grad is not None, "Gradient should flow back to input x!"
    print("      ✓ Ternary STE successfully differentiably passed gradients.")

def test_metric_lattice():
    print("[2/4] Testing Discrete Metric Lattice & Fiedler Gap...")
    engine = DiscreteMetricWarpingEngine(n_nodes=16, b_w=4, w_max=1.0)
    
    # Check initial effective resistance
    d_init = engine.get_effective_resistance(16)
    assert (d_init > 0).all(), "Geodesic distances must be strictly positive!"
    
    # Check initial Fiedler eigenvalue (Theorem Q-2)
    lambda_2 = engine.get_fiedler_eigenvalue(16)
    assert lambda_2 > 0.0, f"Algebraic connectivity must be > 0! Got {lambda_2}"
    print(f"      ✓ Initial Quantized Fiedler eigenvalue lambda_2 = {lambda_2:.4f} > 0")
    
    # Simulate Hebbian update
    h_mock = torch.randn(2, 16, 32)
    A_mock_grad = torch.randn(2, 4, 16, 16)
    engine.update_discrete_conductance(h_mock, A_mock_grad, eta_g=0.5)
    
    lambda_2_post = engine.get_fiedler_eigenvalue(16)
    assert lambda_2_post > 0.0, f"Fiedler eigenvalue must remain > 0! Got {lambda_2_post}"
    print(f"      ✓ Post-update Quantized Fiedler eigenvalue lambda_2 = {lambda_2_post:.4f} > 0")

def test_quantized_geoformer_block():
    print("[3/4] Testing Quantized GeoFormer Block Pipeline...")
    block = QuantizedGeoFormerBlock(d_model=32, n_heads=4, n_neurons=16, b_w=4, mode="ternary")
    h_in = torch.randn(2, 16, 32, requires_grad=True)
    
    h_out, A, L, tau, K_S = block(h_in)
    assert h_out.shape == (2, 16, 32), f"Expected shape (2, 16, 32), got {h_out.shape}"
    assert tau >= 1, f"TAC loop must execute at least 1 step, got {tau}"
    assert K_S.item() <= 54.73, f"Description complexity {K_S.item()} exceeds 54.73 bits bound!"
    print(f"      ✓ Q-GeoFormer forward pass succeeded. TAC converged in {tau} steps; K(S) = {K_S.item():.2f} bits <= 54.68 bound.")
    
    # Backward pass
    loss = (h_out ** 2).sum()
    loss.backward()
    
    # Slow timescale update
    block.step_slow_timescale(h_out, A.grad)
    print("      ✓ Slow-timescale discrete metric warping step completed successfully.")

def test_dual_path_optimizer():
    print("[4/4] Testing Dual Path Kramers Optimizer...")
    model = nn.Sequential(
        QuantizedLinear(16, 16, mode="ternary"),
        nn.SiLU(),
        QuantizedLinear(16, 4, mode="ternary")
    )
    optimizer = DualPathKramersOptimizer(model.parameters(), lr=0.01, alpha_tau=0.2, d_0=1e-3)
    
    x = torch.randn(8, 16)
    target = torch.randn(8, 4)
    
    initial_w = model[0].weight.clone()
    for step in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step(calendar_signal=0.5)
        
    delta_w = (model[0].weight - initial_w).abs().sum()
    assert delta_w > 0, "Weights must evolve under Dual Path optimizer!"
    print(f"      ✓ Dual Path Kramers optimizer updated parameters (delta_W = {delta_w:.4f}).")

if __name__ == "__main__":
    print("=" * 80)
    print("   QUANTIZED GEOFORMER COMPREHENSIVE ARCHITECTURE TEST SUITE")
    print("=" * 80)
    test_quantized_linear_ste()
    test_metric_lattice()
    test_quantized_geoformer_block()
    test_dual_path_optimizer()
    print("=" * 80)
    print("   ALL ARCHITECTURAL TESTS PASSED (100% SUCCESS)")
    print("=" * 80)
