"""
Discrete Metric Warping Engine.
Maintains the quantized conductance lattice W^(b) and updates it via Hebbian co-activations
and cross-entropy gradients, implementing Theorems Q-1 (Geodesic Contraction) and Q-2 (Fiedler Preservation).
"""

import torch
import torch.nn as nn
from .laplacian import GraphLaplacian

class DiscreteMetricWarpingEngine(nn.Module):
    """
    Maintains a discrete conductance matrix W in {0, Delta_w, 2*Delta_w, ..., W_max}.
    Computes effective geodesic distances and controls spectral relaxation.
    """
    def __init__(self, n_nodes=32, b_w=4, w_max=1.0, eps_dist=1e-4):
        super().__init__()
        self.n_nodes = n_nodes
        self.b_w = b_w
        self.w_max = w_max
        self.eps_dist = eps_dist
        
        # Quantization step size Delta_w
        self.num_levels = (1 << b_w) - 1
        self.delta_w = w_max / self.num_levels
        
        # Initialize with baseline ground conductance Delta_w (ensures lambda_2 > 0)
        init_W = torch.full((n_nodes, n_nodes), self.delta_w)
        init_W.fill_diagonal_(0.0)
        self.register_buffer("W_discrete", init_W)

    def get_effective_resistance(self, seq_len=None):
        """Returns the effective geodesic distance matrix d_ij = 1 / (W_ij + eps)."""
        N = seq_len if seq_len is not None else self.n_nodes
        W_sub = self.W_discrete[:N, :N]
        return GraphLaplacian.compute_effective_resistance(W_sub, self.eps_dist)

    def get_laplacian(self, seq_len=None, normalized=False):
        """Returns the discrete Graph Laplacian L = D - W."""
        N = seq_len if seq_len is not None else self.n_nodes
        W_sub = self.W_discrete[:N, :N]
        return GraphLaplacian.compute_laplacian(W_sub, normalized=normalized)

    def get_fiedler_eigenvalue(self, seq_len=None):
        """Calculates lambda_2(L) to certify Theorem Q-2 preservation."""
        L = self.get_laplacian(seq_len=seq_len)
        return GraphLaplacian.get_fiedler_eigenvalue(L)

    def update_discrete_conductance(self, h_states, A_grads=None, eta_g=0.05, gamma_leak=0.001):
        """
        Master jump update step:
            Delta W_ij = round( (eta_g * C_ij - gamma_leak * W_ij) / Delta_w ) * Delta_w
        where C_ij is the pairwise Hebbian correlation combined with attention gradients.
        """
        with torch.no_grad():
            N = min(h_states.size(1), self.n_nodes)
            h_norm = torch.nn.functional.normalize(h_states[:, :N, :], dim=-1)
            # Pairwise co-activation matrix
            C_hebb = torch.bmm(h_norm, h_norm.transpose(1, 2)).mean(dim=0)
            
            # Incorporate attention gradients if available
            if A_grads is not None:
                if A_grads.dim() == 4: # [B, H, N, N]
                    grad_metric = A_grads[:, :, :N, :N].mean(dim=(0, 1))
                else:
                    grad_metric = A_grads[:N, :N]
                C_drive = C_hebb + 0.5 * grad_metric.abs()
            else:
                C_drive = C_hebb
                
            W_sub = self.W_discrete[:N, :N]
            continuous_drift = eta_g * C_drive - gamma_leak * W_sub
            
            # Discretize jump increments to integer multiples of Delta_w
            jump_steps = torch.round(continuous_drift / self.delta_w)
            W_new = W_sub + jump_steps * self.delta_w
            
            # Enforce bounds: W_min = Delta_w, W_max, and zero diagonal
            W_new = torch.clamp(W_new, min=self.delta_w, max=self.w_max)
            W_new.fill_diagonal_(0.0)
            
            # Symmetrize to maintain undirected Riemannian/metric consistency
            W_sym = 0.5 * (W_new + W_new.transpose(0, 1))
            self.W_discrete[:N, :N] = W_sym
