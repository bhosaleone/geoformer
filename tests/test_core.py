"""
Tests for geoformer.core subsystem:
Metric Warping, Laplacian Dynamics, and Theorem Q-2 Invariant Preservation.
"""

import torch
import pytest
from geoformer.core import DiscreteMetricWarpingEngine, GraphLaplacian

def test_graph_laplacian_properties():
    # Symmetric 4-node ring graph
    W = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0]
    ])
    L = GraphLaplacian.compute_laplacian(W)
    # Check row sums are zero
    assert torch.allclose(L.sum(dim=-1), torch.zeros(4), atol=1e-5)

    # Check Fiedler eigenvalue lambda_2 > 0
    lambda_2 = GraphLaplacian.get_fiedler_eigenvalue(L)
    assert lambda_2 > 0.0, f"Connected graph must have positive algebraic connectivity, got {lambda_2}"

def test_discrete_metric_warping_engine():
    engine = DiscreteMetricWarpingEngine(n_nodes=16, b_w=4, w_max=1.0)
    
    # Theorem Q-2 Check: initial Fiedler eigenvalue lambda_2 > 0
    lambda_2_init = engine.get_fiedler_eigenvalue(16)
    assert lambda_2_init > 0.0, f"Theorem Q-2 violated: lambda_2 = {lambda_2_init}"

    # Effective resistance check
    d = engine.get_effective_resistance(16)
    assert (d > 0).all(), "All effective resistances must be strictly positive"

    # Simulate Hebbian update
    h_mock = torch.randn(2, 16, 32)
    A_mock_grad = torch.randn(2, 4, 16, 16)
    engine.update_discrete_conductance(h_mock, A_mock_grad, eta_g=0.5)

    # Conductance bounds check (off-diagonal elements must be >= delta_w, diagonal is 0)
    W = engine.W_discrete
    mask = ~torch.eye(16, dtype=torch.bool)
    assert (W[mask] >= engine.delta_w - 1e-6).all(), "Off-diagonal W must be >= delta_w"
    assert (W <= engine.w_max + 1e-6).all(), "W must be <= w_max"
    assert torch.allclose(W.diagonal(), torch.zeros(16)), "Diagonal must remain zero"

    # Theorem Q-2 Check post-update
    lambda_2_post = engine.get_fiedler_eigenvalue(16)
    assert lambda_2_post > 0.0, f"Theorem Q-2 post-update violated: lambda_2 = {lambda_2_post}"
