"""
Relational Q-GeoFormer Model.
End-to-end model for knowledge graph path finding and transitive relational inference.
"""

import torch
import torch.nn as nn
from ..quant.ternary_linear import QuantizedLinear

class RelationalQGeoFormer(nn.Module):
    """
    Model for relational reasoning over knowledge graphs.
    Uses discrete conductance lattice W and Finsler directed routing.
    """
    def __init__(
        self,
        num_entities=64,
        num_relations=4,
        d_model=256,
        n_neurons=128,
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

        # Conductance matrix for metric routing
        if adaptive:
            self.W = nn.Parameter(torch.full((num_entities, num_entities), 0.05))
        else:
            self.register_buffer("W", torch.full((num_entities, num_entities), 0.05))

    def forward(self, entity_idx, relation_idx):
        e_emb = self.entity_emb(entity_idx)
        r_emb = self.relation_emb(relation_idx)
        x = torch.cat([e_emb, r_emb], dim=-1)

        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))

        # Direct classifier logits
        logits = self.classifier(x)

        if self.adaptive:
            # Directed Finsler metric boost: Softplus conductance from source entity
            W_pos = torch.relu(self.W)
            metric_boost = W_pos[entity_idx]
            logits = logits + 2.0 * metric_boost

        return logits
