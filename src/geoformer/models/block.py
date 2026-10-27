"""
Quantized GeoFormer Block (QGeoFormerBlock).
Unified block integrating Pre-RMSNorm, Quantized Geodesic Highway Attention,
Ternary SwiGLU, and Quantized TAC Continuous Relaxation.
"""

import torch
import torch.nn as nn
from ..core.metric_warping import DiscreteMetricWarpingEngine
from ..layers.attention import QuantizedGHA
from ..layers.ffn import TernarySwiGLU
from ..layers.tac import QuantizedTAC

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class QGeoFormerBlock(nn.Module):
    """
    Standard Quantized GeoFormer Block.
    Executes:
      1. x1 = x + Q-GHA(RMSNorm(x), d_matrix)
      2. x2 = x1 + SwiGLU(RMSNorm(x1))
      3. h* = Q-TAC(x2, L_matrix)
    """
    def __init__(
        self,
        d_model=32,
        n_heads=4,
        n_neurons=32,
        b_w=4,
        w_max=1.0,
        mode="ternary",
        tau_max=4,
        lambda_geo=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_neurons = n_neurons

        # Metric Warping Engine
        self.metric_engine = DiscreteMetricWarpingEngine(n_nodes=n_neurons, b_w=b_w, w_max=w_max)

        # Pre-Norms
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Layers
        self.attn = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode=mode, lambda_geo=lambda_geo)
        self.ffn = TernarySwiGLU(d_model=d_model, mode=mode)
        self.tac = QuantizedTAC(d_model=d_model, tau_max=tau_max)

    def forward(self, h, directed_finsler=None):
        B, N, D = h.shape
        
        # 1. Fetch current effective resistance and Laplacian from metric lattice
        d_matrix = self.metric_engine.get_effective_resistance(N)
        L_matrix = self.metric_engine.get_laplacian(N)

        # 2. Attention Sub-layer
        h_norm1 = self.norm1(h)
        attn_out, attn_weights = self.attn(h_norm1, d_matrix=d_matrix, directed_finsler=directed_finsler)
        h_res1 = h + attn_out

        # 3. Feedforward Sub-layer
        h_norm2 = self.norm2(h_res1)
        ffn_out = self.ffn(h_norm2)
        h_res2 = h_res1 + ffn_out

        # 4. Thermodynamic Attractor Convergence (Q-TAC) Relaxation
        h_star, tau, K_S = self.tac(h_res2, L_matrix)

        return h_star, attn_weights, L_matrix, tau, K_S

    def step_slow_timescale(self, h_states, A_grads=None, eta_g=0.05):
        """Updates slow-timescale discrete conductance lattice."""
        self.metric_engine.update_discrete_conductance(h_states, A_grads=A_grads, eta_g=eta_g)
