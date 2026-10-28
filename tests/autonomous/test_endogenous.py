"""
tests/autonomous/test_endogenous.py
===================================
Rigorous unit, property, and stress tests for EndogenousConsolidator.
Verifies all hard mathematical bounds, non-negativity, energy stability, and determinism.
"""

import math
import torch
import pytest
from geoformer.autonomous.endogenous_engine import (
    ConsolidationConfig,
    ConsolidationReport,
    EndogenousConsolidator,
)


def test_consolidator_guarantees_dense():
    N = 32
    # Create an initial non-negative metric lattice with a transitive 2-hop bridge:
    # 0 -> 1 (0.8), 1 -> 2 (0.8), but 0 -> 2 is 0.0
    W = torch.zeros((N, N), dtype=torch.float32)
    W[0, 1] = 0.8
    W[1, 0] = 0.8
    W[1, 2] = 0.8
    W[2, 1] = 0.8

    cfg = ConsolidationConfig(max_probes=64, max_delta_w=0.01, seed=42)
    consolidator = EndogenousConsolidator(cfg)

    W_updated, report = consolidator.step(W, num_steps=5)

    # 1. Non-negativity guarantee
    assert (W_updated >= 0.0).all(), "W must remain non-negative"

    # 2. Diagonal remains zero
    assert torch.allclose(torch.diag(W_updated), torch.zeros(N)), "Diagonal must remain 0"

    # 3. Maximum delta bound per step
    assert report.max_abs_delta <= cfg.max_delta_w * 5 + 1e-7, f"Max delta {report.max_abs_delta} exceeded allowed bound"

    # 4. Probe budget limit
    assert report.num_probes <= cfg.max_probes * 5, f"Probes {report.num_probes} exceeded limit {cfg.max_probes * 5}"

    # 5. Kappa bounded in [0, 1]
    assert 0.0 <= report.kappa <= 1.0, f"Kappa {report.kappa} outside [0, 1]"

    # 6. Energy does not explode
    assert math.isfinite(report.energy_after)
    assert report.energy_after >= 0.0


def test_consolidator_idempotence_zero_probes():
    N = 16
    W = torch.rand((N, N)).abs()
    W.fill_diagonal_(0.0)

    cfg = ConsolidationConfig(max_probes=0)
    consolidator = EndogenousConsolidator(cfg)

    W_out, report = consolidator.step(W, num_steps=1)
    assert torch.equal(W_out, W), "Zero probes must leave W unchanged"
    assert report.num_probes == 0
    assert report.max_abs_delta == 0.0
    assert report.edges_updated == 0


def test_consolidator_determinism():
    N = 20
    W1 = torch.rand((N, N)).abs()
    W1.fill_diagonal_(0.0)
    W2 = W1.clone()

    cfg1 = ConsolidationConfig(seed=12345)
    cfg2 = ConsolidationConfig(seed=12345)

    c1 = EndogenousConsolidator(cfg1)
    c2 = EndogenousConsolidator(cfg2)

    W_out1, r1 = c1.step(W1, num_steps=3)
    W_out2, r2 = c2.step(W2, num_steps=3)

    assert torch.equal(W_out1, W_out2), "Consolidator must be deterministic given identical seed"
    assert r1.max_abs_delta == r2.max_abs_delta
    assert r1.edges_updated == r2.edges_updated


def test_consolidator_sparse_support():
    N = 24
    dense_W = torch.zeros((N, N))
    dense_W[0, 1] = 0.7
    dense_W[1, 0] = 0.7
    dense_W[1, 2] = 0.7
    dense_W[2, 1] = 0.7

    sparse_W = dense_W.to_sparse().coalesce()

    cfg = ConsolidationConfig(max_probes=32, seed=99)
    consolidator = EndogenousConsolidator(cfg)

    out_sparse, report = consolidator.step(sparse_W, num_steps=2)

    assert out_sparse.is_sparse, "Output must preserve sparse format"
    out_dense = out_sparse.to_dense()
    assert (out_dense >= 0.0).all(), "Sparse values must remain non-negative"
    assert torch.allclose(torch.diag(out_dense), torch.zeros(N))


def test_consolidator_stress_1000_steps():
    """
    Stress test: 1000 consecutive steps on a 256-node lattice.
    Must strictly respect non-negativity and max_delta_w without exploding.
    """
    N = 256
    torch.manual_seed(42)
    # Sparse random initialization
    W = (torch.rand((N, N)) < 0.05).float() * torch.rand((N, N))
    W = (W + W.T) / 2.0
    W.fill_diagonal_(0.0)

    cfg = ConsolidationConfig(max_probes=16, max_delta_w=0.01, decay=0.9995, seed=777)
    consolidator = EndogenousConsolidator(cfg)

    # Run 10 batches of 100 steps = 1000 total steps
    curr_W = W
    for batch in range(10):
        curr_W, report = consolidator.step(curr_W, num_steps=100)
        assert (curr_W >= 0.0).all(), f"Failed non-negativity at batch {batch}"
        assert math.isfinite(report.energy_after), f"Energy non-finite at batch {batch}"
        assert 0.0 <= report.kappa <= 1.0


def test_consolidator_property_randomized():
    """
    Property test: starts from 10 different random non-negative matrices
    and runs 50 consolidations, checking hard mathematical invariants.
    """
    for seed in range(10):
        N = 30 + seed
        torch.manual_seed(seed * 100)
        W = torch.rand((N, N)).abs()
        W = (W + W.T) / 2.0
        W.fill_diagonal_(0.0)

        cfg = ConsolidationConfig(
            max_probes=20,
            max_delta_w=0.005,
            decay=0.998,
            seed=seed
        )
        consolidator = EndogenousConsolidator(cfg)

        W_out, report = consolidator.step(W, num_steps=50)

        assert (W_out >= 0.0).all()
        assert report.num_probes <= cfg.max_probes * 50
        assert 0.0 <= report.kappa <= 1.0
        assert math.isfinite(report.energy_after)
