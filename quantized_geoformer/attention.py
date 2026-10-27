"""
Quantized Geodesic Highway Attention (Q-GHA)
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Computes attention affinities deformed by discrete Riemannian-Finsler
effective resistance distances d_ij^(b).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantized_ops import QuantizedLinear

class QuantizedGHA(nn.Module):
    def __init__(self, d_model=64, n_heads=4, mode="ternary", lambda_geo=0.15):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.lambda_geo = lambda_geo
        self.mode = mode
        
        # Quantized Projection Layers
        self.w_q = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_k = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_v = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_o = QuantizedLinear(d_model, d_model, mode=mode)

    def forward(self, h, D_metric):
        """
        h: [B, N, D]
        D_metric: [N, N] effective resistance distance matrix
        """
        B, N, D = h.shape
        
        Q = self.w_q(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # Raw projection affinities
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Deform routing by discrete geodesic distance penalty
        # High conductance (W -> W_max) => d -> 0 (Ballistic Geodesic Highway)
        # Low conductance (W -> 0) => d -> 2/delta_w (High Energy Barrier)
        geo_penalty = self.lambda_geo * D_metric.unsqueeze(0).unsqueeze(0)
        scores = scores - geo_penalty
        
        A = F.softmax(scores, dim=-1)
        
        # Enable gradient retention on attention map for Hebbian-gradient coupling
        if self.training and A.requires_grad:
            A.retain_grad()
            
        attn_out = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.w_o(attn_out)
        
        return out, A
