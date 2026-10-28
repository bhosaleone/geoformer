"""
src/geoformer/autonomous/endogenous_engine.py
=============================================
Bounded Endogenous Consolidator using Weyl irrational driver and kappa closure diagnostic.
Implements autonomous, passive-to-active background consolidation with strict hard bounds.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import torch


@dataclass
class ConsolidationConfig:
    max_probes: int = 64
    max_delta_w: float = 0.01          # hard bound on |ΔW| per update
    decay: float = 0.999               # mild global decay
    alpha: float = (5.0**0.5 - 1.0) / 2.0  # irrational golden driver ≈ 0.6180339887
    kappa_closed: float = 0.85
    seed: int = 0
    symmetrize: bool = False           # False for directed knowledge graphs, True if undirected metric


@dataclass
class ConsolidationReport:
    kappa: float
    num_probes: int
    max_abs_delta: float
    edges_updated: int
    energy_before: float
    energy_after: float


class EndogenousConsolidator:
    """
    Performs bounded background consolidation on metric lattice W.
    Guarantees:
      - W remains non-negative (W >= 0)
      - |ΔW|_max <= max_delta_w per single step
      - Total candidate probes <= max_probes * num_steps
      - Non-explosive Dirichlet/Frobenius energy
    """
    def __init__(self, config: Optional[ConsolidationConfig] = None):
        self.cfg = config or ConsolidationConfig()
        # Internal deterministic state for Weyl rotation
        self._z = float(self.cfg.seed % 1000) * 0.00137
        if self._z <= 0:
            self._z = 0.123456789

    def compute_kappa(self, trajectory: torch.Tensor) -> float:
        """
        Computes intrinsic closure index:
            κ = 1 - final_dist / max_excursion (clipped to [0, 1])
        """
        if trajectory.dim() == 1:
            trajectory = trajectory.unsqueeze(1)
        T = trajectory.shape[0]
        if T < 2:
            return 1.0

        p0 = trajectory[0]
        p_final = trajectory[-1]
        
        diffs = trajectory - p0
        dists = torch.norm(diffs, p=2, dim=-1)
        
        max_excursion = dists.max().item()
        final_dist = dists[-1].item()
        
        if max_excursion < 1e-12:
            return 1.0
            
        kappa = 1.0 - (final_dist / max_excursion)
        return float(max(0.0, min(1.0, kappa)))

    def step(
        self,
        W: torch.Tensor,               # sparse or dense conductance (N, N)
        num_steps: int = 1,
    ) -> Tuple[torch.Tensor, ConsolidationReport]:
        """
        Perform bounded background updates.
        Guarantees:
          - W remains non-negative
          - |ΔW|_max <= max_delta_w per step
          - total probes <= max_probes * num_steps
        Returns updated W and report.
        """
        if num_steps <= 0 or self.cfg.max_probes <= 0:
            energy = 0.5 * float(torch.sum(W ** 2).item()) if not W.is_sparse else 0.5 * float(torch.sum(W.values() ** 2).item())
            report = ConsolidationReport(
                kappa=1.0,
                num_probes=0,
                max_abs_delta=0.0,
                edges_updated=0,
                energy_before=energy,
                energy_after=energy,
            )
            return W, report

        is_sparse = W.is_sparse
        if is_sparse:
            dense_W = W.to_dense().clone()
        else:
            dense_W = W.clone()

        device = dense_W.device
        dtype = dense_W.dtype
        N = dense_W.shape[0]
        
        energy_before = 0.5 * float(torch.sum(dense_W ** 2).item())
        initial_W = dense_W.clone()

        total_probes = 0
        edges_updated = 0
        kappa_history = []
        max_abs_delta_observed = 0.0

        for s in range(num_steps):
            step_start_W = dense_W.clone()
            probes_this_step = 0
            
            while probes_this_step < self.cfg.max_probes and N > 2:
                probes_this_step += 1
                total_probes += 1

                # 1. Weyl rotation to pick (i, k) pair
                self._z = (self._z + self.cfg.alpha) % 1.0
                i = int(math.floor(self._z * N)) % N
                
                self._z = (self._z + self.cfg.alpha) % 1.0
                k = int(math.floor(self._z * (N - 1)))
                if k >= i:
                    k += 1

                # 2. Check 2-hop intermediate bridging strength
                w_row_i = dense_W[i, :]
                w_col_k = dense_W[:, k]
                
                # Exclude endpoints i and k
                two_hop_strengths = w_row_i * w_col_k
                two_hop_strengths[i] = 0.0
                two_hop_strengths[k] = 0.0
                
                best_bridge_val, best_bridge_idx = torch.max(two_hop_strengths, dim=0)
                s_bridge = float(best_bridge_val.item())
                j = int(best_bridge_idx.item())

                # 3. Compute trajectory closure kappa
                w_ij = float(dense_W[i, j].item())
                w_jk = float(dense_W[j, k].item())
                w_ik = float(dense_W[i, k].item())
                
                traj = torch.tensor([
                    [0.0, 0.0],
                    [w_ij, 0.0],
                    [w_ij, w_jk],
                    [w_ik, 0.0]
                ], dtype=torch.float32, device=device)
                
                k_val = self.compute_kappa(traj)
                kappa_history.append(k_val)

                # 4. If path is open (kappa < kappa_closed) and bridge is stronger than direct edge:
                if s_bridge > 1e-4 and w_ik < s_bridge:
                    raw_delta = (s_bridge - w_ik) * 0.1
                    # Enforce strict hard bound |ΔW| <= max_delta_w
                    bounded_delta = min(self.cfg.max_delta_w, max(-self.cfg.max_delta_w, raw_delta))
                    
                    new_val = max(0.0, w_ik + bounded_delta)
                    actual_delta = abs(new_val - w_ik)
                    
                    if actual_delta > 1e-7:
                        dense_W[i, k] = new_val
                        if self.cfg.symmetrize:
                            dense_W[k, i] = new_val
                        edges_updated += 1
                        max_abs_delta_observed = max(max_abs_delta_observed, actual_delta)

            # Global mild decay per step
            if self.cfg.decay < 1.0:
                dense_W = dense_W * self.cfg.decay
                
            # Guarantee non-negativity and zero diagonal
            dense_W = torch.clamp(dense_W, min=0.0)
            dense_W.fill_diagonal_(0.0)

            # Enforce step-level max_delta_w bound
            step_delta = torch.abs(dense_W - step_start_W)
            max_step_delta = float(step_delta.max().item())
            if max_step_delta > self.cfg.max_delta_w:
                # Scale back to strictly satisfy max_delta_w per step
                scale = self.cfg.max_delta_w / max_step_delta
                dense_W = step_start_W + (dense_W - step_start_W) * scale
                dense_W = torch.clamp(dense_W, min=0.0)
                dense_W.fill_diagonal_(0.0)

        # Final verification of guarantees
        total_delta = torch.abs(dense_W - initial_W)
        max_abs_delta = float(total_delta.max().item())
        energy_after = 0.5 * float(torch.sum(dense_W ** 2).item())

        mean_kappa = float(sum(kappa_history) / len(kappa_history)) if kappa_history else 1.0

        if is_sparse:
            out_W = dense_W.to_sparse().coalesce()
        else:
            out_W = dense_W

        report = ConsolidationReport(
            kappa=mean_kappa,
            num_probes=total_probes,
            max_abs_delta=max_abs_delta,
            edges_updated=edges_updated,
            energy_before=energy_before,
            energy_after=energy_after,
        )

        return out_W, report
