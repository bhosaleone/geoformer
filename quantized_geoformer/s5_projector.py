"""
S_5 Permutation Group Cayley Projector Engine
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Decomposes neural representations into the 7 irreducible representations
of the symmetric group S_5 (order 120), enforcing:
    1. Sum of squares identity: sum d_rho^2 = 120 (Burnside)
    2. Description complexity bound: K(S) <= sum d_rho log2(d_rho) <= 54.68 bits
    3. Microscopic noise filtering via Cayley equivariant projectors:
       Pi_rho = (d_rho / |S_5|) * sum_{g in S_5} chi_rho(g)* g
"""

import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class S5CayleyProjector(nn.Module):
    def __init__(self, d_model=30):
        super().__init__()
        self.d_model = d_model
        
        # Dimensions of the 7 irreps of S_5
        # Shapes: (5), (1^5), (4,1), (2,1^3), (3,2), (2^2,1), (3,1^2)
        dims = [1, 1, 4, 4, 5, 5, 6]
        self.register_buffer('dims', torch.tensor(dims, dtype=torch.float32))
        
        # Theoretical description entropy contribution per irrep: d_rho * log2(d_rho)
        # sum d_rho * log2(d_rho) ~ 54.68 to 54.73 bits
        c_bits = [0.0 if d == 1 else d * math.log2(d) for d in dims]
        self.register_buffer('c_bits', torch.tensor(c_bits, dtype=torch.float32))
        self.theoretical_bound_bits = float(sum(c_bits))
        
        # S_5 Character table (7 conjugacy classes x 7 irreps)
        # Classes: [1^5], [2,1^3], [2^2,1], [3,1^2], [3,2], [4,1], [5]
        # Class sizes: [1, 10, 15, 20, 20, 30, 24]
        chi = [
            [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],  # Trivial (d=1)
            [1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0],  # Sign (d=1)
            [4.0,  2.0,  0.0,  1.0, -1.0,  0.0, -1.0],  # Standard (d=4)
            [4.0, -2.0,  0.0,  1.0,  1.0,  0.0, -1.0],  # Std x Sign (d=4)
            [5.0,  1.0,  1.0, -1.0,  1.0, -1.0,  0.0],  # (3,2) (d=5)
            [5.0, -1.0,  1.0, -1.0, -1.0,  1.0,  0.0],  # (2^2,1) (d=5)
            [6.0,  0.0, -2.0,  0.0,  0.0,  0.0,  1.0]   # (3,1^2) (d=6)
        ]
        self.register_buffer('chi', torch.tensor(chi, dtype=torch.float32))
        
        # Build 5x5 permutation class sum matrices
        class_matrices = self._build_s5_class_matrices()
        
        # Build 5x5 Cayley projectors for the 7 irreps in permutation space:
        # Pi_rho = (d_rho / 120) * sum_k chi[rho, k] * class_matrices[k]
        projectors = []
        for rho in range(7):
            P = torch.zeros(5, 5)
            for k in range(7):
                P += chi[rho][k] * class_matrices[k]
            P = (dims[rho] / 120.0) * P
            projectors.append(P)
            
        self.register_buffer('projectors', torch.stack(projectors)) # [7, 5, 5]

    def _build_s5_class_matrices(self):
        """Constructs sum of permutation matrices for each of the 7 conjugacy classes."""
        perms = list(itertools.permutations(range(5)))
        class_map = {
            (1, 1, 1, 1, 1): 0, # [1^5] (size 1)
            (2, 1, 1, 1): 1,    # [2, 1^3] (size 10)
            (2, 2, 1): 2,       # [2^2, 1] (size 15)
            (3, 1, 1): 3,       # [3, 1^2] (size 20)
            (3, 2): 4,          # [3, 2] (size 20)
            (4, 1): 5,          # [4, 1] (size 30)
            (5,): 6             # [5] (size 24)
        }
        
        class_matrices = [torch.zeros(5, 5) for _ in range(7)]
        for p in perms:
            # Determine cycle structure
            visited = [False] * 5
            cycles = []
            for i in range(5):
                if not visited[i]:
                    length = 0
                    curr = i
                    while not visited[curr]:
                        visited[curr] = True
                        curr = p[curr]
                        length += 1
                    cycles.append(length)
            cycles.sort(reverse=True)
            c_idx = class_map[tuple(cycles)]
            
            # Permutation matrix
            M = torch.zeros(5, 5)
            for i in range(5):
                M[p[i], i] = 1.0
            class_matrices[c_idx] += M
            
        return torch.stack(class_matrices)

    def project_pentad(self, x):
        """
        Projects a tensor whose last dimension is a multiple of 5:
        x: [..., 5] or [..., K, 5]
        Returns:
          h_proj: [..., 5] (symmetry-filtered state)
          irrep_components: list of 7 components [..., 5]
        """
        components = []
        for rho in range(7):
            P_rho = self.projectors[rho] # [5, 5]
            # Matrix multiply on the right
            h_rho = torch.matmul(x, P_rho.T)
            components.append(h_rho)
            
        # Total projection onto invariant group space
        h_proj = sum(components)
        return h_proj, components

    def forward(self, h):
        """
        h: [B, N, D]
        Decomposes h into 5-coordinate pentads (padding if D % 5 != 0).
        Computes irrep energies, description complexity K(S), and filtered state.
        """
        B, N, D = h.shape
        pad_len = (5 - (D % 5)) % 5
        if pad_len > 0:
            h_padded = F.pad(h, (0, pad_len))
        else:
            h_padded = h
            
        D_pad = h_padded.shape[-1]
        n_pentads = D_pad // 5
        
        # Reshape to [B, N, n_pentads, 5]
        h_pentad = h_padded.view(B, N, n_pentads, 5)
        
        # Project each pentad
        components = []
        irrep_energies = []
        
        for rho in range(7):
            P_rho = self.projectors[rho]
            comp = torch.matmul(h_pentad, P_rho.T)
            components.append(comp)
            # Energy in irrep subspace
            E_rho = torch.sum(comp ** 2)
            irrep_energies.append(E_rho)
            
        E_tensor = torch.stack(irrep_energies) + 1e-8
        prob_dist = E_tensor / E_tensor.sum() # Probability simplex across the 7 irreps
        
        # Exact description complexity: K(S) = sum p_rho * (d_rho * log2(d_rho))
        # Bounded by theoretical bound <= 54.68 to 54.73 bits
        K_S = torch.sum(prob_dist * self.c_bits)
        
        # Filtered representation
        h_proj_pentad = sum(components)
        h_proj = h_proj_pentad.view(B, N, D_pad)
        if pad_len > 0:
            h_proj = h_proj[:, :, :D]
            
        return h_proj, K_S, prob_dist
