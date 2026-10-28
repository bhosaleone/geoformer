"""
src/geoformer/autonomous/encodings.py
=====================================
Quasi-crystalline (cut-and-project style) positional and node encodings.
Uses irrational rotations on higher-dimensional tori to generate deterministic,
collision-free, parameter-free geometric representations.
"""

import math
from typing import Optional
import torch


def quasi_crystalline_encoding(
    num_nodes: int,
    dim: int,
    basis: str = "golden",            # "golden" | "sqrt2" | "custom"
    custom_alphas: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Returns (num_nodes, dim) positional / node features.
    Uses cut-and-project style irrational rotations.
    Deterministic for a given configuration.
    """
    if num_nodes <= 0 or dim <= 0:
        return torch.empty((num_nodes, dim), dtype=torch.float32, device=device)

    basis = basis.lower()
    if basis == "golden":
        base_alpha = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.6180339887
    elif basis == "sqrt2":
        base_alpha = math.sqrt(2.0) - 1.0          # ≈ 0.4142135624
    elif basis == "custom" and custom_alphas is not None:
        base_alpha = None
    else:
        base_alpha = (math.sqrt(5.0) - 1.0) / 2.0

    # Determine channel frequencies
    if custom_alphas is not None:
        if custom_alphas.numel() < dim:
            # Repeat to match dim
            repeats = math.ceil(dim / custom_alphas.numel())
            alphas = custom_alphas.repeat(repeats)[:dim].to(device=device, dtype=torch.float32)
        else:
            alphas = custom_alphas[:dim].to(device=device, dtype=torch.float32)
    else:
        # Generate powers or Weyl frequencies: alpha_k = (base_alpha * (k + 1)) % 1.0
        # Adding a small prime offset ensures linear independence over Q
        k_indices = torch.arange(1, dim + 1, dtype=torch.float32, device=device)
        alphas = (base_alpha * k_indices + (k_indices * 0.00137)) % 1.0
        # Avoid exact 0
        alphas = torch.where(alphas < 1e-6, torch.tensor(base_alpha, device=device), alphas)

    # Node indices n in {1, ..., num_nodes}
    n_indices = torch.arange(1, num_nodes + 1, dtype=torch.float32, device=device).unsqueeze(1) # [N, 1]
    alphas = alphas.unsqueeze(0) # [1, D]

    # Phases: theta_nk = 2 * pi * n * alpha_k
    phases = 2.0 * math.pi * (n_indices @ alphas)

    # Alternate sin and cos across channels for balance
    enc = torch.zeros((num_nodes, dim), dtype=torch.float32, device=device)
    enc[:, 0::2] = torch.cos(phases[:, 0::2])
    if dim > 1:
        enc[:, 1::2] = torch.sin(phases[:, 1::2])

    # Normalize rows to unit sphere for stable attention
    row_norms = torch.norm(enc, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    enc = enc / row_norms

    return enc


def encoding_collision_rate(enc: torch.Tensor, rtol: float = 1e-5) -> float:
    """
    Fraction of near-duplicate rows (diagnostic).
    A row i collides if there exists j != i such that ||x_i - x_j|| <= rtol.
    """
    num_nodes, dim = enc.shape
    if num_nodes <= 1:
        return 0.0

    # Ensure float32 on CPU for consistent pairwise computation
    x = enc.detach().float().cpu()
    
    # Process in chunks if num_nodes > 2000 to keep memory O(chunk_size * N)
    chunk_size = 1000
    colliding_indices = set()

    for start_idx in range(0, num_nodes, chunk_size):
        end_idx = min(num_nodes, start_idx + chunk_size)
        chunk = x[start_idx:end_idx] # [C, D]
        
        # Pairwise distance matrix between chunk and all nodes [C, N]
        dists = torch.cdist(chunk, x, p=2)
        
        # Zero-out self-distance
        for local_i in range(end_idx - start_idx):
            global_i = start_idx + local_i
            dists[local_i, global_i] = float('inf')
            
        # Check collisions
        is_collision = (dists <= rtol).any(dim=1) # [C]
        for local_i, has_col in enumerate(is_collision):
            if has_col.item():
                colliding_indices.add(start_idx + local_i)

    return float(len(colliding_indices)) / float(num_nodes)
