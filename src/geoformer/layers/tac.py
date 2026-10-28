"""
Quantized Thermodynamic Attractor Convergence (Q-TAC) Relaxation Loop.
Solves the continuous gradient flow:
  dh / dtau = -k * grad_{h} F_{latent}
where F_{latent} = 1/2 ||h - h_ffn||^2 + (lambda_Dir / 2) Tr(h^T L h) + (lambda_S5 / 2) ||h - P_S5 h||^2.
"""

import torch
import torch.nn as nn
from .cayley import S5CayleyProjector

class QuantizedTAC(nn.Module):
    """
    Thermodynamic Attractor Convergence loop operating over discrete graph Laplacians
    and S5 symmetric invariant projections.
    """
    def __init__(self, d_model=32, tau_max=4, delta_tau=0.1, lambda_dir=0.05, lambda_s5=0.05, schedule=None):
        super().__init__()
        self.d_model = d_model
        self.tau_max = tau_max
        self.delta_tau = delta_tau
        self.lambda_dir = lambda_dir
        self.lambda_s5 = lambda_s5
        self.schedule = schedule
        
        self.s5_projector = S5CayleyProjector(d_model=d_model)

    def forward(self, h_init, L_matrix, schedule=None):
        """
        Relaxes h_init to attractor h* via Euler steps:
          h_{tau+1} = h_tau - delta_tau * [ (h_tau - h_init) + lambda_dir * (L h_tau) + lambda_s5 * (h_tau - P_S5(h_tau)) ]
        Supports optional multi-scale schedule (e.g. Fibonacci [1, 2, 3, 5, 8, 13]).
        """
        B, N, D = h_init.shape
        h = h_init.clone()
        
        # Determine relaxation budget
        active_schedule = schedule if schedule is not None else self.schedule
        max_steps = active_schedule[-1] if active_schedule is not None and len(active_schedule) > 0 else self.tau_max
        
        # Ensure L is [N, N]
        L = L_matrix[:N, :N]
        
        final_K_S = torch.tensor(0.0, device=h.device)
        actual_tau = 0
        
        for step in range(max_steps):
            actual_tau += 1
            # 1. Metric Dirichlet energy gradient: L @ h
            grad_dirichlet = torch.matmul(L.unsqueeze(0), h)
            
            # 2. S_5 Cayley invariant projection gradient
            h_sym, K_S = self.s5_projector(h)
            grad_s5 = h - h_sym
            final_K_S = K_S
            
            # 3. Target restoration gradient
            grad_restoration = h - h_init
            
            # Combined continuous relaxation
            dh = -(grad_restoration + self.lambda_dir * grad_dirichlet + self.lambda_s5 * grad_s5)
            h_next = h + self.delta_tau * dh
            
            # Check convergence
            diff = torch.norm(h_next - h, dim=-1).mean()
            h = h_next
            if diff.item() < 1e-4:
                break
                
        return h, actual_tau, final_K_S
