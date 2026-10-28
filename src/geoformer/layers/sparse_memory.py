"""
Sparse Hippocampal Relational Working Memory (Sparse-RML)
=========================================================
Memory-efficient relational working memory engine scaling to 100,000+ entities
by representing relational conductance lattices as sparse CSR / COO tensors.

Reduces spatial complexity from O(R * N^2) to O(R * |E|), enabling large-scale
knowledge graph reasoning (OGB / Wikidata) without DRAM explosion.

Author: Shrikant Bhosale (Atmabhan Pandit)
"""

import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant.ternary_linear import QuantizedLinear


class SparseHippocampalRelationalMemory(nn.Module):
    """
    Sparse relational memory buffer for large-scale knowledge graph reasoning.
    
    Args:
        d_model: Hidden feature dimension (e.g. 64 or 256).
        num_entities: Vocabulary size (scales effortlessly to 100,000+).
        num_slots: Number of episodic working memory registers.
        mode: Quantization mode ("ternary" or "fp32").
    """
    def __init__(self, d_model: int = 64, num_entities: int = 10000, num_slots: int = 4, mode: str = "ternary"):
        super().__init__()
        self.d_model = d_model
        self.num_entities = num_entities
        self.num_slots = num_slots
        self.mode = mode

        # 1.58-Bit Quantized Projections
        self.w_read_q = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_read_k = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_read_v = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_gate = QuantizedLinear(d_model * 2, d_model, mode=mode)
        self.layer_norm = nn.LayerNorm(d_model)

        # Dynamic sparse relation edges storage: dict mapping rel_id -> (indices, values)
        self.sparse_edges = {}

    def register_sparse_relation(self, r_id: int, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        """
        Registers a sparse relation slice r as an edge list [2, num_edges].
        Memory complexity: O(|E|).
        """
        num_edges = edge_index.shape[1]
        if edge_weight is None:
            edge_weight = torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)
            
        # Store as normalized sparse COO matrix
        # Column normalize: sum of weights for each target entity
        target_nodes = edge_index[0]
        deg = torch.zeros(self.num_entities, dtype=torch.float32, device=edge_index.device)
        deg.scatter_add_(0, target_nodes, edge_weight)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        norm_weights = edge_weight * deg_inv[target_nodes]

        sparse_tensor = torch.sparse_coo_tensor(
            edge_index, norm_weights, size=(self.num_entities, self.num_entities)
        ).coalesce()
        self.sparse_edges[r_id] = sparse_tensor

    def propagate_sparse_relation(self, r_id: int, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Propagates state vector [num_entities] or [num_entities, d] across sparse relation r:
            s_next = W_r_sparse @ s_curr
        Execution is O(|E|), strictly linear in the number of edges.
        """
        if r_id not in self.sparse_edges:
            return state_vector
            
        W_sparse = self.sparse_edges[r_id]
        if state_vector.dim() == 1:
            # [num_entities, 1]
            s_in = state_vector.unsqueeze(1)
            s_out = torch.sparse.mm(W_sparse, s_in)
            return s_out.squeeze(1)
        else:
            return torch.sparse.mm(W_sparse, state_vector)

    def forward_multi_hop_chain(self, start_entity: int, relation_sequence: List[int]) -> torch.Tensor:
        """
        Executes zero-shot multi-hop deductive chaining across sparse relations.
        """
        device = next(self.parameters()).device
        h = torch.zeros(self.num_entities, dtype=torch.float32, device=device)
        h[start_entity] = 1.0

        for r_id in relation_sequence:
            h = self.propagate_sparse_relation(r_id, h)

        return h

    def memory_footprint_bytes(self) -> int:
        """Computes exact RAM usage of the sparse relation graph."""
        total_bytes = 0
        for W_sp in self.sparse_edges.values():
            # indices: 2 * num_edges * 8 bytes (int64)
            # values: num_edges * 4 bytes (float32)
            nnz = W_sp._nnz()
            total_bytes += nnz * (2 * 8 + 4)
        return total_bytes
