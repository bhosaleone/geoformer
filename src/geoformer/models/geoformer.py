"""
GeoFormer Foundation Neural Network Architecture
================================================
The official, general-purpose GeoFormer deep neural network (PyTorch nn.Module).

Integrates:
  1. 1.58-Bit Ternary Synaptic Weights (BitNet b1.58 STE)
  2. Multi-Head Geodesic Highway Attention (Q-GHA)
  3. Continuous Thermodynamic Attractor Convergence (Q-TAC) Latent Relaxation
  4. Ternary SwiGLU Gated Feed-Forward Networks
  5. Pre-RMSNorm & Residual Highway Skip Connections

Trained 100% end-to-end via gradient backpropagation (no hardcoded lookups).

Author: Shrikant Bhosale (Atmabhan Pandit), MBBS GMC Sambhainagar
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant.ternary_linear import QuantizedLinear
from ..layers.attention import QuantizedGHA
from ..layers.ffn import TernarySwiGLU
from ..layers.tac import QuantizedTAC
from .block import RMSNorm


class GeoFormerBlock(nn.Module):
    """
    A single deep GeoFormer Transformer block.
    Sequential execution:
      1. h1 = h + Q_GHA(RMSNorm(h), d_matrix)
      2. h2 = h1 + SwiGLU(RMSNorm(h1))
      3. h* = Q_TAC(h2, L_matrix)
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        mode: str = "ternary",
        tau_max: int = 3,
        lambda_geo: float = 0.1,
        lambda_dir: float = 0.05,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.attn = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode=mode, lambda_geo=lambda_geo)
        
        self.norm2 = RMSNorm(d_model)
        self.ffn = TernarySwiGLU(d_model=d_model, mode=mode)
        
        self.tac = QuantizedTAC(d_model=d_model, tau_max=tau_max, lambda_dir=lambda_dir)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        d_matrix: Optional[torch.Tensor] = None,
        L_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Geodesic Highway Attention Sub-layer
        h_norm1 = self.norm1(h)
        attn_out, _ = self.attn(h_norm1, d_matrix=d_matrix)
        h = h + self.dropout(attn_out)

        # 2. Feedforward Sub-layer
        h_norm2 = self.norm2(h)
        ffn_out = self.ffn(h_norm2)
        h = h + self.dropout(ffn_out)

        # 3. Continuous Q-TAC Relaxation Sub-layer (if Laplacian provided)
        if L_matrix is not None:
            h, _, _ = self.tac(h, L_matrix)

        return h


class GeoFormer(nn.Module):
    """
    Full GeoFormer Neural Network Model.
    Supports sequence modeling, classification, and graph representation learning.
    """
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        in_features: Optional[int] = None,
        num_classes: int = 10,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 1024,
        mode: str = "ternary",
        tau_max: int = 3,
        lambda_geo: float = 0.1,
        lambda_dir: float = 0.05,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.in_features = in_features
        self.num_classes = num_classes
        self.mode = mode

        # Input Embedding: Token embedding or Linear feature projection
        if vocab_size is not None:
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        elif in_features is not None:
            self.in_proj = QuantizedLinear(in_features, d_model, mode=mode)
        else:
            raise ValueError("Must provide either vocab_size (for discrete tokens) or in_features (for vectors).")

        # Stacked Deep GeoFormer Blocks
        self.blocks = nn.ModuleList([
            GeoFormerBlock(
                d_model=d_model,
                n_heads=n_heads,
                mode=mode,
                tau_max=tau_max,
                lambda_geo=lambda_geo,
                lambda_dir=lambda_dir,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final Layer Norm and Projection Head
        self.final_norm = RMSNorm(d_model)
        self.head = QuantizedLinear(d_model, num_classes if num_classes is not None else vocab_size, mode=mode)

        # Endogenous Metric Lattice conductances: learned parameter matrix W [N, N]
        self.register_parameter("conductance_bias", nn.Parameter(torch.zeros(1)))

    def forward(
        self,
        x: torch.Tensor,
        L_matrix: Optional[torch.Tensor] = None,
        d_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass:
          x: [Batch, Seq_Len] (integer token IDs) OR [Batch, Seq_Len, in_features] / [Batch, in_features]
        """
        # 1. Project input to latent d_model dimension
        if self.vocab_size is not None and x.dtype in (torch.long, torch.int64):
            B, S = x.shape
            positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
            h = self.tok_emb(x) + self.pos_emb(positions)
        else:
            if x.dim() == 2:
                # [B, in_features] -> [B, 1, d_model]
                h = self.in_proj(x).unsqueeze(1)
            elif x.dim() == 3:
                # [B, S, in_features] -> [B, S, d_model]
                h = self.in_proj(x)
            else:
                h = x

        # 2. Build default 1D sequential Laplacian and metric distance if not passed
        B, N, D = h.shape
        if L_matrix is None and N > 1:
            # Standard tridiagonal 1D sequence Laplacian
            diag = 2.0 * torch.ones(N, device=h.device)
            off = -1.0 * torch.ones(N - 1, device=h.device)
            L_matrix = torch.diag(diag) + torch.diag(off, 1) + torch.diag(off, -1)
            L_matrix[0, 0] = 1.0
            L_matrix[-1, -1] = 1.0

        if d_matrix is None and N > 1:
            idx = torch.arange(N, device=h.device, dtype=torch.float32)
            d_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

        # 3. Pass through stacked GeoFormer blocks
        for block in self.blocks:
            h = block(h, d_matrix=d_matrix, L_matrix=L_matrix)

        # 4. Final normalization & projection
        h = self.final_norm(h)
        logits = self.head(h)

        # If sequence classification, squeeze sequence dim if N == 1
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)

        return logits

    def count_parameters(self) -> Tuple[int, float]:
        """Returns total parameter count and ternary packed memory footprint (KB)."""
        total_params = sum(p.numel() for p in self.parameters())
        # BitNet b1.58: 1.58 bits/param = 0.20 bytes/param (packed 5 trits/byte = 0.20 bytes)
        packed_kb = (total_params * 0.20) / 1024.0
        return total_params, packed_kb
