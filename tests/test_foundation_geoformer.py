"""
Unit tests for the official GeoFormer Foundation Neural Network Model.
Verifies forward pass, backward pass, gradient flow through ternary STE,
stacked blocks, and checkpoint loading.
"""

import os
import pytest
import torch
import torch.nn as nn

from geoformer import GeoFormer, GeoFormerBlock


def test_geoformer_forward_and_backward_classification():
    """Tests GeoFormer on continuous vector classification with backprop."""
    batch_size = 4
    in_features = 32
    num_classes = 5
    d_model = 16
    n_layers = 2
    n_heads = 2
    
    model = GeoFormer(
        in_features=in_features,
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mode="ternary"
    )
    
    x = torch.randn(batch_size, in_features, requires_grad=True)
    logits = model(x)
    assert logits.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"
    
    # Verify backward pass gradients flow into parameters and input
    target = torch.randint(0, num_classes, (batch_size,))
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    
    # Check that in_proj and head received non-zero gradients
    assert model.in_proj.weight.grad is not None
    assert torch.norm(model.in_proj.weight.grad) > 0
    assert model.head.weight.grad is not None
    assert torch.norm(model.head.weight.grad) > 0


def test_geoformer_sequence_modeling():
    """Tests GeoFormer on discrete token sequence modeling."""
    vocab_size = 100
    seq_len = 16
    batch_size = 2
    d_model = 32
    n_layers = 2
    
    model = GeoFormer(
        vocab_size=vocab_size,
        num_classes=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        mode="ternary"
    )
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(token_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"


def test_geoformer_checkpoint_serialization(tmp_path):
    """Tests serializing and restoring GeoFormer state_dict."""
    model = GeoFormer(in_features=16, num_classes=3, d_model=16, n_layers=1, n_heads=2)
    ckpt_file = tmp_path / "test_geoformer.pt"
    
    torch.save(model.state_dict(), ckpt_file)
    assert os.path.exists(ckpt_file)
    
    model.eval()
    model_restored = GeoFormer(in_features=16, num_classes=3, d_model=16, n_layers=1, n_heads=2)
    model_restored.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
    model_restored.eval()
    
    x = torch.randn(2, 16)
    with torch.no_grad():
        out1 = model(x)
        out2 = model_restored(x)
    assert torch.allclose(out1, out2), "Restored model output does not match original"
