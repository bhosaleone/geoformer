"""
Quantized Thermodynamic Automaton Computer (Q-TAC) Continuous Relaxation Loop
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Relaxes latent states dynamically along physical potential gradients:
    dh/dtau = -k * delta F_latent / delta h
    F_latent = 1/2 ||h - h_ffn||^2 + lambda_Dir/2 Tr(h^T L^(b) h)
Solves fixed-point attractor convergence in O(1) physical time (ThermP complexity).
"""

import torch
import torch.nn as nn
from .s5_projector import S5CayleyProjector

class QuantizedTAC(nn.Module):
    def __init__(self, tau_max=3, eta_tac=0.08, lambda_dir=0.05, lambda_s5=0.1, eps_tac=1e-3, use_s5=True, d_model=32):
        super().__init__()
        self.tau_max = tau_max
        self.eta_tac = eta_tac
        self.lambda_dir = lambda_dir
        self.lambda_s5 = lambda_s5
        self.eps_tac = eps_tac
        self.use_s5 = use_s5
        self.s5_projector = S5CayleyProjector(d_model=d_model) if use_s5 else None

    def forward(self, h_ffn, L_laplacian):
        """
        h_ffn: [B, N, D] target state post-FFN
        L_laplacian: [N, N] quantized graph Laplacian
        """
        h_curr = h_ffn
        actual_tau = 0
        current_K_S = torch.tensor(0.0, device=h_ffn.device)
        
        for tau in range(self.tau_max):
            # 1. Base gradient: input fidelity + Dirichlet Laplacian smoothness
            laplacian_smoothness = torch.matmul(L_laplacian, h_curr)
            grad_F = (h_curr - h_ffn) + self.lambda_dir * laplacian_smoothness
            
            # 2. S_5 Algebraic Invariant Projection restoring force
            if self.use_s5 and self.s5_projector is not None:
                h_proj, K_S, _ = self.s5_projector(h_curr)
                current_K_S = K_S
                grad_s5 = self.lambda_s5 * (h_curr - h_proj)
                grad_F = grad_F + grad_s5
            
            h_next = h_curr - self.eta_tac * grad_F
            delta_norm = torch.norm(h_next - h_curr)
            actual_tau += 1
            h_curr = h_next
            
            if delta_norm < self.eps_tac:
                break
                
        return h_curr, actual_tau, current_K_S
