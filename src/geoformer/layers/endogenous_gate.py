"""
Endogenous Dirichlet Gating Layer (Self-Purification via Propagation Geometry).
==============================================================================
Enables the GeoFormer model to autonomously evaluate candidate semantic associations
using continuous Riemannian metric energy, Dirichlet harmonic smoothness,
and Kolmogorov information entropy, completely eliminating external heuristic regexes.

Mathematical Formulation:
  1. Pairwise Geodesic Dirichlet Energy:
       E_dir(a, b) = 0.5 * ||h_a - h_b||^2
  2. Information Density (Shannon-Kolmogorov Entropy):
       H(x) = - sum(p * log2(p))
  3. Endogenous Decision Gate:
       G(a, b) = sigmoid( W_g @ [h_a, h_b] - lambda_dir * E_dir - lambda_ent * |H - H_0| )

If G >= 0.50, the association is admitted as harmonically coherent;
otherwise, it is discarded as metric dissonance (noise).

Author: Shrikant Bhosale (Atmabhan Pandit), TWIST POOL Labs
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoformer.quant.ternary_linear import QuantizedLinear


class EndogenousDirichletGate(nn.Module):
    """
    Self-Purification Gating Layer based on Dirichlet Energy Minimization.
    Allows the model's own metric manifold to autonomously accept or reject information.
    """
    def __init__(self, d_model=4096, lambda_dir=0.35, lambda_ent=0.15, target_entropy=4.2):
        super().__init__()
        self.d_model = d_model
        self.lambda_dir = lambda_dir
        self.lambda_ent = lambda_ent
        self.target_entropy = target_entropy

        # Fast ternary projection for semantic affinity
        self.w_affinity = QuantizedLinear(d_model * 2, 1, bias=True, mode="ternary")
        
        # Dirichlet reference energy calibration
        self.register_buffer("running_mean_energy", torch.tensor(1.0))
        self.register_buffer("admitted_count", torch.tensor(0))
        self.register_buffer("rejected_count", torch.tensor(0))

    def compute_dirichlet_energy(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise Dirichlet energy: E_dir = 0.5 * ||h_a - h_b||^2.
        Lower energy indicates smooth geodesic connection on the manifold.
        """
        return 0.5 * torch.sum((h_a - h_b) ** 2, dim=-1, keepdim=True)

    def estimate_token_entropy(self, text_snippet: str) -> float:
        """Estimates character/token entropy H(X)."""
        if not text_snippet:
            return 0.0
        n = len(text_snippet)
        from collections import Counter
        counts = Counter(text_snippet.lower())
        return -sum((cnt / n) * math.log2(cnt / n) for cnt in counts.values())

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor, text_snippet: str = "") -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Evaluates candidate association between entity representations h_a and h_b.
        Returns:
          - is_admitted: bool tensor indicating whether the model accepted the fact
          - gate_score: continuous confidence in [0, 1]
          - dirichlet_energy: computed metric Dirichlet energy
        """
        # 1. Pairwise Dirichlet Energy on representation manifold
        e_dir = self.compute_dirichlet_energy(h_a, h_b)
        
        # Update running mean energy online
        with torch.no_grad():
            self.running_mean_energy = 0.95 * self.running_mean_energy + 0.05 * e_dir.mean().detach()

        # Normalized energy disparity
        normalized_e_dir = e_dir / (self.running_mean_energy + 1e-6)

        # 2. Entropy distance penalty
        ent = self.estimate_token_entropy(text_snippet)
        ent_penalty = abs(ent - self.target_entropy)

        # 3. Fast Synaptic Affinity Score
        x_pair = torch.cat([h_a, h_b], dim=-1)
        raw_affinity = self.w_affinity(x_pair)

        # 4. Endogenous Decision Gate:
        # G = sigmoid( Affinity - lambda_dir * E_dir - lambda_ent * |H - H_0| )
        gate_logit = raw_affinity - (self.lambda_dir * normalized_e_dir) - (self.lambda_ent * ent_penalty)
        gate_score = torch.sigmoid(gate_logit)

        is_admitted = (gate_score >= 0.50)

        # Update running statistics
        with torch.no_grad():
            if is_admitted.any():
                self.admitted_count += is_admitted.sum().item()
            if (~is_admitted).any():
                self.rejected_count += (~is_admitted).sum().item()

        return is_admitted, gate_score, e_dir.mean().item()
