"""
Graph Laplacian & Spectral Dynamics Engine for Quantized Propagation Geometry.
Implements Discrete Laplacian computation, algebraic connectivity (Fiedler eigenvalue),
and effective resistance metrics.
"""

import torch

class GraphLaplacian:
    """
    Computes graph Laplacian matrices and spectral invariants from conductance matrices.
    """
    @staticmethod
    def compute_laplacian(W, normalized=False):
        """
        Computes L = D - W (or symmetric normalized L_sym = I - D^(-1/2) W D^(-1/2)).
        """
        deg = torch.sum(W, dim=-1)
        D = torch.diag_embed(deg)
        L = D - W

        if not normalized:
            return L

        # Symmetric normalized Laplacian
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-8), -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        L_norm = torch.eye(W.size(-1), device=W.device, dtype=W.dtype) - torch.matmul(torch.matmul(D_inv_sqrt, W), D_inv_sqrt)
        return L_norm

    @staticmethod
    def get_fiedler_eigenvalue(L):
        """
        Computes algebraic connectivity lambda_2(L) (the Fiedler eigenvalue).
        Under Theorem Q-2, lambda_2 >= N * Delta_w > 0 ensures graph connectedness.
        """
        if L.dim() == 2:
            L_eval = L
        else:
            L_eval = L[0]
            
        eigs = torch.linalg.eigvalsh(L_eval)
        # Sort ascending
        eigs_sorted, _ = torch.sort(eigs)
        if len(eigs_sorted) > 1:
            return eigs_sorted[1].item()
        return eigs_sorted[0].item()

    @staticmethod
    def compute_effective_resistance(W, eps_dist=1e-4):
        """
        Computes effective resistance metric d_ij = 1 / (W_ij + eps_dist).
        """
        return 1.0 / (W + eps_dist)
