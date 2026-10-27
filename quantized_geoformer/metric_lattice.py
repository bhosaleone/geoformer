"""
Discrete Metric Conductance Lattice Engine
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Manages:
  - b_W-bit discrete conductance lattice W^(b)
  - Klein-Randic effective resistance geodesic metric d_ij = 1 / (W_ij + eps)
  - Discrete Graph Laplacian L^(b) = D - W
  - Stochastic Markov jump updates over discrete levels
  - Spectral gap (Fiedler eigenvalue) auditing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteMetricWarpingEngine(nn.Module):
    def __init__(self, n_nodes=32, b_w=4, w_max=1.0, eps_dist=0.05):
        super().__init__()
        self.n_nodes = n_nodes
        self.b_w = b_w
        self.levels = 2 ** b_w - 1
        self.w_max = w_max
        self.delta_w = w_max / float(self.levels)
        self.eps_dist = eps_dist
        
        # Initialize uniform baseline conductance grid (guaranteeing lambda_2 > 0)
        init_W = torch.ones(n_nodes, n_nodes) * self.delta_w
        # Symmetrize and zero-diagonal
        init_W.fill_diagonal_(0.0)
        self.register_buffer('W', init_W)

    def get_effective_resistance(self, n=None):
        """
        Geodesic distance deformed by discrete conductance:
        d_ij = 1 / (W_ij + eps_dist)
        """
        n = n or self.n_nodes
        W_sub = self.W[:n, :n]
        return 1.0 / (W_sub + self.eps_dist)

    def get_laplacian(self, n=None):
        """
        Discrete Graph Laplacian:
        L = D - W
        where D_ii = sum_j W_ij
        """
        n = n or self.n_nodes
        W_sub = self.W[:n, :n]
        deg = torch.diag(W_sub.sum(dim=-1))
        return deg - W_sub

    def get_fiedler_eigenvalue(self, n=None):
        """Calculates algebraic connectivity lambda_2(L)."""
        L = self.get_laplacian(n)
        evals = torch.linalg.eigvalsh(L)
        if len(evals) >= 2:
            return evals[1].item()
        return 0.0

    def update_discrete_conductance(self, h_states, A_grad, eta_g=0.01, mu=0.5, gamma=0.1, alpha_a=0.5):
        """
        Stochastic Markov jump update on the conductance lattice:
        Jump rates: T+ = mu * [a - (gamma/mu)W]_+, T- = gamma * [W - (mu/gamma)a]_+
        """
        N = h_states.shape[1]
        with torch.no_grad():
            # 1. Gradient sensitivity across tokens: g_ij = ||dL / dA_ij||
            if A_grad is not None:
                g = A_grad.mean(dim=(0, 1)).abs()
                g_bar = g.mean() + 1e-6
                g_norm = g / g_bar
            else:
                g_norm = torch.zeros(N, N, device=h_states.device)

            # 2. Hebbian co-activation: c_ij = max(0, <h_i, h_j> / ||h_i|| ||h_j||)
            h_mean = h_states.mean(dim=0) # [N, D]
            h_normed = F.normalize(h_mean, p=2, dim=-1)
            c = F.relu(torch.matmul(h_normed, h_normed.T))
            c.fill_diagonal_(0.0)
            c_bar = c.mean() + 1e-6
            c_norm = c / c_bar

            # 3. Composite interaction field
            if A_grad is not None:
                a = alpha_a * g_norm + (1.0 - alpha_a) * c_norm
            else:
                a = c_norm

            # 4. Tendency flux
            W_sub = self.W[:N, :N]
            flux = mu * a - gamma * W_sub

            # 5. Stochastic lattice jump
            jump_prob = torch.clamp((flux.abs() / (self.w_max + 1e-6)) * eta_g, 0.0, 1.0)
            jump_mask = torch.bernoulli(jump_prob)
            jump_direction = torch.sign(flux)

            W_new = W_sub + jump_mask * jump_direction * self.delta_w
            # Enforce symmetry, minimum floor delta_w (Theorem Q-2), and maximum ceiling
            W_new = 0.5 * (W_new + W_new.T)
            W_new.fill_diagonal_(0.0)
            self.W[:N, :N] = torch.clamp(W_new, min=self.delta_w, max=self.w_max)
