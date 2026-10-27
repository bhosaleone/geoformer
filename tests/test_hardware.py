"""
Tests for geoformer.hardware subsystem:
PhysicalMemristorCrossbar, NeuromorphicQGeoFormer, and NanojouleEnergyAuditor.
"""

import numpy as np
import pytest
from geoformer.hardware import (
    PhysicalMemristorCrossbar,
    NeuromorphicQGeoFormer,
    NanojouleEnergyAuditor
)

def test_physical_memristor_crossbar():
    crossbar = PhysicalMemristorCrossbar(n_inputs=16, n_outputs=16, b_levels=16)
    W_mock = np.random.uniform(0.0, 1.0, (16, 16))
    crossbar.program_conductance_matrix(W_mock, noise_std=0.05)

    assert crossbar.G_actual.shape == (16, 16)
    assert np.all(crossbar.G_actual >= crossbar.g_min * 0.4)

    # Compute analog VMM
    v_in = np.zeros(16)
    v_in[0] = 0.2
    i_out = crossbar.compute_analog_vmm(v_in)
    assert i_out.shape == (16,)
    assert np.all(i_out >= 0.0)

def test_energy_auditor():
    crossbar = PhysicalMemristorCrossbar(n_inputs=64, n_outputs=64)
    W_mock = np.ones((64, 64)) * 0.1
    crossbar.program_conductance_matrix(W_mock, noise_std=0.0)

    comparison = NanojouleEnergyAuditor.audit(crossbar)
    assert "nvidia_gpu_fp32" in comparison
    ratio = comparison["nvidia_gpu_fp32"]["reduction_ratio"]
    assert ratio > 5000.0, f"Expected >5000x GPU energy reduction, got {ratio:.1f}x"

def test_neuromorphic_harness():
    harness = NeuromorphicQGeoFormer(num_entities=16, noise_std=0.01)
    W_mock = np.zeros((16, 16))
    # Create chain 0 -> 1 -> 2 (row=target, col=source)
    W_mock[1, 0] = 1.0  # edge 0 -> 1
    W_mock[2, 1] = 1.0  # edge 1 -> 2
    harness.program_learned_highways(W_mock)

    pred, currents = harness.route_2hop_query(src_entity=0)
    assert pred == 2, f"Expected 2-hop route to predict node 2, got {pred}"
