"""
Unit Tests for Hippocampal Relational Working Memory (RML)
"""

import pytest
import torch
from geoformer.layers.memory import HippocampalRelationalMemory
from geoformer.models.transitive import DeepTransitiveModel


def test_hippocampal_memory_forward():
    B = 4
    num_entities = 32
    d_model = 64
    memory = HippocampalRelationalMemory(d_model=d_model, num_entities=num_entities, num_slots=5)

    h_query = torch.randn(B, d_model)
    entity_embs = torch.randn(num_entities, d_model)
    W_conductance = torch.rand(num_entities, num_entities)

    # Test 2-hop routing
    logits, trace = memory(h_query, entity_embs, W_conductance, hops=2)
    assert logits.shape == (B, num_entities)
    assert trace.shape == (B, 5, d_model)
    assert not torch.isnan(logits).any()


def test_deep_transitive_model_with_memory():
    B = 8
    num_entities = 32
    model = DeepTransitiveModel(
        num_entities=num_entities,
        num_relations=4,
        d_model=64,
        adaptive=True,
        use_memory=True
    )

    entity_idx = torch.randint(0, num_entities, (B,))
    relation_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    logits = model(entity_idx, relation_idx)
    assert logits.shape == (B, num_entities)
    assert not torch.isnan(logits).any()
