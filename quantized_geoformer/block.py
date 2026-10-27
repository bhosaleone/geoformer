"""
The Complete Quantized GeoFormer Block (Q-GeoFormer)
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Architecture Pipeline:
    Input Hidden State h
           │
           ▼
     [ RMSNorm 1 ]
           │
           ▼
    [ Quantized GHA ] ◄── Uses discrete d_ij^(b)
           │
           ▼
     ( + Residual )
           │
           ▼
     [ RMSNorm 2 ]
           │
           ▼
  [ Ternary SwiGLU FFN ]
           │
           ▼
     ( + Residual )
           │
           ▼
   [ Quantized TAC ] ◄── Physical relaxation along L^(b)
           │
           ▼
   Output Hidden State h*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantized_ops import QuantizedLinear
from .metric_lattice import DiscreteMetricWarpingEngine
from .attention import QuantizedGHA
from .tac_relaxation import QuantizedTAC

class QuantizedGeoFormerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_neurons=32, b_w=4, mode="ternary", lambda_geo=0.15, use_s5=True, lambda_s5=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_neurons = n_neurons
        
        # 1. Attention Engine
        self.gha = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode=mode, lambda_geo=lambda_geo)
        
        # 2. Ternary SwiGLU FFN Projections
        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode=mode)
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode=mode)
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode=mode)
        
        # 3. Discrete Metric Conductance Lattice Engine
        self.metric_engine = DiscreteMetricWarpingEngine(n_nodes=n_neurons, b_w=b_w)
        
        # 4. Continuous TAC Relaxation Loop with S_5 Cayley Projector
        self.tac = QuantizedTAC(tau_max=3, eta_tac=0.08, lambda_dir=0.05, lambda_s5=lambda_s5, use_s5=use_s5, d_model=d_model)
        
        # 5. RMSNorm Scaling Parameters
        self.gamma_1 = nn.Parameter(torch.ones(d_model))
        self.gamma_2 = nn.Parameter(torch.ones(d_model))

    def rms_norm(self, x, gamma, eps=1e-5):
        return (x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)) * gamma

    def forward(self, h):
        B, N, D = h.shape
        
        # Step 1: Effective Resistance from discrete lattice
        D_metric = self.metric_engine.get_effective_resistance(N)
        
        # Step 2: Quantized Geodesic Highway Attention
        h_norm1 = self.rms_norm(h, self.gamma_1)
        attn_out, A = self.gha(h_norm1, D_metric)
        h_mid = h + attn_out
        
        # Step 3: SwiGLU FFN
        h_norm2 = self.rms_norm(h_mid, self.gamma_2)
        ffn_out = self.w_down(F.silu(self.w_gate(h_norm2)) * self.w_up(h_norm2))
        h_ffn = h_mid + ffn_out
        
        # Step 4: Quantized TAC Continuous Relaxation Loop with S_5 Filtering
        L = self.metric_engine.get_laplacian(N)
        h_star, tau_used, K_S = self.tac(h_ffn, L)
        
        return h_star, A, L, tau_used, K_S

    def step_slow_timescale(self, h_star, A_grad, eta_g=0.01, mu=0.5, gamma=0.1):
        """Advances slow timescale metric warping."""
        self.metric_engine.update_discrete_conductance(
            h_states=h_star,
            A_grad=A_grad,
            eta_g=eta_g,
            mu=mu,
            gamma=gamma
        )
