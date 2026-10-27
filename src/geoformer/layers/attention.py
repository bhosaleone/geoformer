"""
Quantized Geodesic Highway Attention (Q-GHA).
Applies discrete effective resistance distance penalization to multi-head softmax attention:
  S_ij = (Q_i K_j^T / sqrt(d_k)) - lambda_geo * d_ij^(b)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant.ternary_linear import QuantizedLinear

class QuantizedGHA(nn.Module):
    """
    Quantized Geodesic Highway Attention with BitNet b1.58 ternary projections
    and effective resistance distance bias d_ij.
    """
    def __init__(self, d_model=32, n_heads=4, mode="ternary", lambda_geo=0.1, act_bits=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.lambda_geo = lambda_geo

        self.q_proj = QuantizedLinear(d_model, d_model, bias=False, mode=mode, act_bits=act_bits)
        self.k_proj = QuantizedLinear(d_model, d_model, bias=False, mode=mode, act_bits=act_bits)
        self.v_proj = QuantizedLinear(d_model, d_model, bias=False, mode=mode, act_bits=act_bits)
        self.out_proj = QuantizedLinear(d_model, d_model, bias=False, mode=mode, act_bits=act_bits)

    def forward(self, h, d_matrix=None, directed_finsler=None):
        B, N, D = h.shape

        q = self.q_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Standard scaled dot-product logits
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Subtract effective resistance metric penalty: lambda_geo * d_ij
        if d_matrix is not None:
            # d_matrix: [N, N]
            scores = scores - self.lambda_geo * d_matrix.unsqueeze(0).unsqueeze(0)

        # Optional directed Finsler asymmetry
        if directed_finsler is not None:
            scores = scores + directed_finsler.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        # Store for gradient tracking in slow-timescale Hebbian updates
        if attn_weights.requires_grad:
            attn_weights.retain_grad()

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(context), attn_weights
