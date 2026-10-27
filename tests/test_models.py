"""
Tests for geoformer.models subsystem:
QGeoFormerBlock, RelationalQGeoFormer, and DeepTransitiveModel.
"""

import torch
import pytest
from geoformer.models import QGeoFormerBlock, RelationalQGeoFormer, DeepTransitiveModel

def test_qgeoformer_block():
    block = QGeoFormerBlock(d_model=32, n_heads=4, n_neurons=16, b_w=4, mode="ternary")
    h_in = torch.randn(2, 16, 32, requires_grad=True)
    h_out, A, L, tau, K_S = block(h_in)

    assert h_out.shape == (2, 16, 32)
    assert tau >= 1
    assert K_S.item() <= 54.68

    loss = h_out.sum()
    loss.backward()
    assert h_in.grad is not None

    # Step slow timescale
    block.step_slow_timescale(h_out, A.grad)

def test_relational_qgeoformer():
    model = RelationalQGeoFormer(num_entities=32, num_relations=2, d_model=64, n_neurons=32, mode="ternary")
    entity_idx = torch.tensor([0, 5, 12])
    relation_idx = torch.tensor([0, 1, 0])
    logits = model(entity_idx, relation_idx)
    assert logits.shape == (3, 32)

def test_deep_transitive_model_powers():
    model = DeepTransitiveModel(num_entities=32, num_relations=4, d_model=64, n_neurons=32, mode="ternary")
    # Test 2-hop (rel=1), 3-hop (rel=2), and 4-hop (rel=3) queries
    entity_idx = torch.tensor([1, 2, 3])
    relation_idx = torch.tensor([1, 2, 3])
    logits = model(entity_idx, relation_idx)
    assert logits.shape == (3, 32)
