"""
tests/autonomous/test_encodings.py
==================================
Unit tests for quasi-crystalline cut-and-project encodings.
"""

import math
import torch
import pytest
from geoformer.autonomous.encodings import (
    quasi_crystalline_encoding,
    encoding_collision_rate,
)


def test_quasi_encoding_shape_and_finite():
    N, D = 128, 64
    enc = quasi_crystalline_encoding(N, D, basis="golden")
    assert enc.shape == (N, D)
    assert not torch.isnan(enc).any()
    assert not torch.isinf(enc).any()
    
    # Check normalized
    norms = torch.norm(enc, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_quasi_encoding_deterministic():
    N, D = 64, 32
    enc1 = quasi_crystalline_encoding(N, D, basis="golden")
    enc2 = quasi_crystalline_encoding(N, D, basis="golden")
    assert torch.equal(enc1, enc2)


def test_quasi_encoding_collision_rate():
    # Test for N = 1000, D = 32 with golden basis
    N, D = 1000, 32
    enc = quasi_crystalline_encoding(N, D, basis="golden")
    rate = encoding_collision_rate(enc, rtol=1e-5)
    # Collision rate must be near-zero (< 1e-3)
    assert rate < 1e-3, f"Collision rate too high: {rate}"


def test_quasi_encoding_bases():
    N, D = 32, 16
    enc_gold = quasi_crystalline_encoding(N, D, basis="golden")
    enc_sqrt2 = quasi_crystalline_encoding(N, D, basis="sqrt2")
    # Different bases must produce distinct geometric representations
    assert not torch.allclose(enc_gold, enc_sqrt2, atol=1e-2)
