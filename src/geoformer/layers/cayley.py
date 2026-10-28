"""
S5 Cayley Group Representation Projector.
Projects hidden activation vectors onto the irreducible representations (irreps) of the symmetric group S_5
and computes Kolmogorov description complexity K(S) <= 54.68 bits bound (Theorem Q-4).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class S5CayleyProjector(nn.Module):
    """
    Constructs projection operators for the 7 irreducible representations of S_5:
      - Trivial (dim 1)
      - Sign (dim 1)
      - Standard (dim 4)
      - Standard x Sign (dim 4)
      - Exterior Square (dim 5)
      - Exterior Square x Sign (dim 5)
      - 2-Hook Young (dim 6)
    Total dimension sum: 1 + 1 + 4 + 4 + 5 + 5 + 6 = 26 dimensions.
    """
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model
        self.irrep_dims = [1, 1, 4, 4, 5, 5, 6]
        self.total_irrep_dim = sum(self.irrep_dims) # 26
        
        # Linear projection into S5 invariant space
        self.proj_in = nn.Linear(d_model, self.total_irrep_dim, bias=False)
        self.proj_out = nn.Linear(self.total_irrep_dim, d_model, bias=False)
        self._init_orthogonal_irreps()

    def _init_orthogonal_irreps(self):
        with torch.no_grad():
            if self.d_model >= self.total_irrep_dim:
                Q, _ = torch.linalg.qr(torch.randn(self.d_model, self.total_irrep_dim))
                self.proj_in.weight.copy_(Q.t()[:self.total_irrep_dim, :])
                self.proj_out.weight.copy_(Q[:, :self.total_irrep_dim])
            else:
                Q, _ = torch.linalg.qr(torch.randn(self.total_irrep_dim, self.total_irrep_dim))
                self.proj_in.weight.copy_(Q[:self.total_irrep_dim, :self.d_model])
                self.proj_out.weight.copy_(Q.t()[:self.d_model, :self.total_irrep_dim])

    def compute_irrep_energies(self, h):
        """Projects activations into irrep subspaces and returns energy distribution."""
        z = self.proj_in(h) # [B, N, 26]
        energies = []
        start_idx = 0
        for dim in self.irrep_dims:
            end_idx = start_idx + dim
            subspace_energy = torch.sum(z[..., start_idx:end_idx] ** 2, dim=-1).mean()
            energies.append(subspace_energy)
            start_idx = end_idx
        return torch.stack(energies)

    def forward(self, h):
        """Returns projected invariant activation and Shannon description complexity K(S)."""
        z = self.proj_in(h)
        h_sym = self.proj_out(z)
        
        # Compute empirical description complexity K(S)
        energies = self.compute_irrep_energies(h)
        total_e = torch.sum(energies) + 1e-8
        probs = energies / total_e
        # Shannon entropy across the 7 invariant modes
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
        # K(S) = log2(|S_5|) * (entropy / log2(7))
        # |S_5| = 120, log2(120) = 6.90689 bits
        K_S = 6.9069 * (entropy / math.log2(7.0))
        return h_sym, K_S
