"""
Hippocampal Relational Working Memory Layer (RML)
=================================================
A neurobiologically inspired working memory buffer modeled on the Hippocampal CA3
recurrent attractor network. Overcomes higher-order path attenuation during deep multi-hop
transitive inference by latching intermediate frontier entities into active memory slots.

Biological Analogy:
  • Neocortex: Slow metric conductance plasticity (Geodesic Highways W).
  • Hippocampus: Fast episodic working memory buffer (M) executing sharp-wave ripple
    replay to sustain transitive path coherence without exponential signal dilution.

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant.ternary_linear import QuantizedLinear


class HippocampalRelationalMemory(nn.Module):
    """
    Slot-based Hippocampal Working Memory Buffer for Higher-Order Path Reasoning.

    Parameters:
        d_model (int): Hidden representation dimension (e.g. 512).
        num_entities (int): Vocabulary size of entities (e.g. 128).
        num_slots (int): Number of episodic memory registers (default: 6, supports up to 5-hop chains).
        mode (str): Quantization mode for projections ("ternary" or "fp32").
    """
    def __init__(self, d_model=512, num_entities=128, num_slots=6, mode="ternary"):
        super().__init__()
        self.d_model = d_model
        self.num_entities = num_entities
        self.num_slots = num_slots
        self.mode = mode

        # Quantized Read and Write Projections
        self.w_read_q = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_read_k = QuantizedLinear(d_model, d_model, mode=mode)
        self.w_read_v = QuantizedLinear(d_model, d_model, mode=mode)

        # Episodic Latching Gate: decides how much intermediate frontier state to consolidate
        self.w_gate = QuantizedLinear(d_model * 2, d_model, mode=mode)

        # Output projection for memory-augmented trajectory scoring
        self.classifier = QuantizedLinear(d_model, num_entities, mode=mode)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, h_query, entity_embs, W_conductance, hops=1):
        """
        Executes multi-step hippocampal working memory routing.

        Args:
            h_query: [Batch, d_model] semantic representation of query (head + relation).
            entity_embs: [num_entities, d_model] entity embedding matrix.
            W_conductance: [num_entities, num_entities] learned metric conductance lattice.
            hops: int or Tensor indicating reasoning depth (1 for direct, 2, 3, 4 for multi-hop).

        Returns:
            trajectory_logits: [Batch, num_entities] memory-sharpened target prediction scores.
            memory_trace: [Batch, num_slots, d_model] state of hippocampal working memory registers.
        """
        B = h_query.shape[0]
        device = h_query.device

        # Initialize Working Memory Slots: Slot 0 holds initial query context
        M = torch.zeros(B, self.num_slots, self.d_model, device=device)
        M[:, 0, :] = h_query

        # Normalize conductance matrix for directed transition probabilities
        W_pos = torch.relu(W_conductance)
        row_sums = torch.clamp(W_pos.sum(dim=-1, keepdim=True), min=1e-6)
        P_trans = W_pos / row_sums  # Transition kernel: P[j, i] = prob of moving from i to j

        # Current active frontier representation
        h_curr = h_query.clone()

        # Sequential hippocampal memory stepping
        max_hops = int(hops.max().item()) if isinstance(hops, torch.Tensor) else int(hops)
        max_hops = max(1, min(max_hops, self.num_slots - 1))

        for step in range(1, max_hops + 1):
            # 1. Metric Frontier Diffusion: estimate next node distribution in embedding space
            # Soft projection of current state onto entity dictionary
            sims = torch.matmul(h_curr, entity_embs.T) / math.sqrt(self.d_model)
            frontier_dist = F.softmax(sims, dim=-1)  # [B, num_entities]

            # Propagate through plastic metric lattice
            next_frontier_dist = torch.matmul(P_trans, frontier_dist.T).T  # [B, num_entities]
            h_diffused = torch.matmul(next_frontier_dist, entity_embs)      # [B, d_model]

            # 2. Episodic Memory Retrieval (Read from previous slots)
            active_slots = M[:, :step, :]  # [B, step, d_model]
            q_mem = self.w_read_q(h_diffused).unsqueeze(1)               # [B, 1, d_model]
            k_mem = self.w_read_k(active_slots)                          # [B, step, d_model]
            v_mem = self.w_read_v(active_slots)                          # [B, step, d_model]

            attn_weights = F.softmax(torch.bmm(q_mem, k_mem.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
            context = torch.bmm(attn_weights, v_mem).squeeze(1)          # [B, d_model]

            # 3. Gated Latching: hippocampal consolidation of intermediate frontier
            gate_in = torch.cat([h_diffused, context], dim=-1)
            gate = torch.sigmoid(self.w_gate(gate_in))
            h_latched = gate * h_diffused + (1.0 - gate) * context
            h_curr = self.layer_norm(h_curr + h_latched)

            # 4. Write into working memory slot for this hop
            M[:, step, :] = h_curr

        # Memory-Augmented Output Classification:
        # Uses the final latched frontier state to classify into the entity dictionary
        out_logits = self.classifier(h_curr)
        return out_logits, M
