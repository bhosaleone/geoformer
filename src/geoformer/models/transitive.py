"""
Deep Transitive Reasoning Model for Higher-Order Path Scaling.
Supports 2-hop, 3-hop, and 4-hop zero-shot path discovery via matrix power routing.
"""

import torch
import torch.nn as nn
from ..quant.ternary_linear import QuantizedLinear

class DeepTransitiveModel(nn.Module):
    """
    Tier 5 Architecture configured for deep composition over 128+ entities.
    Employs higher-order conductance powers W^2, W^3, W^4 for multi-hop queries.
    """
    def __init__(
        self,
        num_entities=128,
        num_relations=4,
        d_model=512,
        n_neurons=256,
        mode="ternary",
        adaptive=True
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_model = d_model
        self.adaptive = adaptive

        self.entity_emb = nn.Embedding(num_entities, d_model)
        self.relation_emb = nn.Embedding(num_relations, d_model)

        self.fc1 = QuantizedLinear(d_model * 2, d_model, mode=mode)
        self.ln1 = nn.LayerNorm(d_model)
        self.fc2 = QuantizedLinear(d_model, d_model, mode=mode)
        self.ln2 = nn.LayerNorm(d_model)
        self.classifier = QuantizedLinear(d_model, num_entities, mode=mode)

        if adaptive:
            self.W = nn.Parameter(torch.full((num_entities, num_entities), 0.02))
        else:
            self.register_buffer("W", torch.full((num_entities, num_entities), 0.02))

    def forward(self, entity_idx, relation_idx):
        e_emb = self.entity_emb(entity_idx)
        r_emb = self.relation_emb(relation_idx)
        x = torch.cat([e_emb, r_emb], dim=-1)

        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))

        logits = self.classifier(x)

        if self.adaptive:
            W_pos = torch.relu(self.W)
            # Normalize conductance matrix for stable matrix powers
            row_sums = torch.clamp(W_pos.sum(dim=-1, keepdim=True), min=1e-6)
            W_norm = W_pos / row_sums

            # Matrix powers for higher-order paths
            W2 = torch.matmul(W_norm, W_norm)
            W3 = torch.matmul(W2, W_norm)
            W4 = torch.matmul(W3, W_norm)

            # Route by relation index:
            # relation 0: 1-hop (W_pos)
            # relation 1: 2-hop (W2)
            # relation 2: 3-hop (W3)
            # relation 3: 4-hop (W4)
            powers = [W_pos, W2, W3, W4]
            boost_list = []
            for i in range(len(entity_idx)):
                rel = relation_idx[i].item()
                rel_idx = min(rel, len(powers) - 1)
                boost_list.append(powers[rel_idx][entity_idx[i]])

            metric_boost = torch.stack(boost_list, dim=0)
            logits = logits + 3.0 * metric_boost

        return logits
