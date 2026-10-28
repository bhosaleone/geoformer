"""
GeoFormer-MCU: Micro-Scale Foundation Architecture for 8-Bit Microcontrollers
==============================================================================
Specifically constrained to execute inside the 2 KB SRAM / 32 KB Flash envelope
of the Atmel ATmega328P (Arduino Uno) @ 16 MHz.

Architecture:
- Entities (Nodes): N = 32
- Hidden Dimension: d = 16
- Synaptic Weights: 1.58-bit ternary {-1, 0, +1} (BitNet b1.58 STE)
- Metric Conductance: 4-bit integers W_ij in {0, ..., 15}
- Continuous Relaxation: Directed Q-TAC Graph Laplacian Diffusion

Author: Shrikant Bhosale (Atmabhan Pandit), GMC Sambhajinagar
Zenodo DOI: 10.5281/zenodo.23730305
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_NODES = 32
D_MODEL = 16


class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        scale = w.abs().mean().clamp(min=1e-5)
        w_scaled = w / scale
        w_quant = torch.round(w_scaled).clamp(-1.0, 1.0)
        return w_quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_ternary(w):
    return TernaryQuantizeSTE.apply(w)


class GeoFormerMCU(nn.Module):
    def __init__(self, num_nodes=NUM_NODES, d_model=D_MODEL):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # 1.58-bit Input & Output Projection Weights
        self.w_in = nn.Parameter(torch.randn(d_model, num_nodes) * 0.5)
        self.w_out = nn.Parameter(torch.randn(num_nodes, d_model) * 0.5)

        # 4-bit Metric Conductance Lattice (32x32)
        # Initialized positive
        self.w_lattice = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def get_conductance_matrix(self):
        # 4-bit integer conductances: [0, 15]
        w_int = torch.clamp(torch.round(F.relu(self.w_lattice)), 0.0, 15.0)
        return w_int

    def get_transition_matrix(self):
        W = self.get_conductance_matrix()
        row_sum = W.sum(dim=-1, keepdim=True)
        P = torch.where(row_sum > 0, W / (row_sum + 1e-6), torch.zeros_like(W))
        return P

    def forward(self, x, tau=2):
        """
        x: [batch_size, num_nodes] one-hot or activation pulse
        tau: number of Q-TAC diffusion steps
        """
        # Ternary Synaptic Weights
        w_in_q = quantize_ternary(self.w_in)
        w_out_q = quantize_ternary(self.w_out)

        # Input feature mapping: h_latent = x @ w_in_q^T
        # Shape: [B, D]
        h_latent = F.linear(x, w_in_q)

        # Project to node space for topological diffusion: h_nodes = h_latent @ w_out_q^T
        # Shape: [B, N]
        h_nodes = F.linear(h_latent, w_out_q)
        h_nodes = F.softmax(h_nodes, dim=-1)

        # Directed Q-TAC Heat Diffusion: h_{t+1} = h_t @ P
        P = self.get_transition_matrix()
        h_t = h_nodes
        for _ in range(tau):
            h_t = torch.matmul(h_t, P)

        return h_t


def generate_mcu_dataset():
    """
    Generates a structured 32-entity relational knowledge graph with:
    - 6 distinct multi-hop chains (2-hop, 3-hop, and 4-hop)
    - Branching crossroads
    - Clinical triage casualty deduction rules
    - Isolated distractor nodes
    """
    train_1hop = []
    test_2hop = []
    test_3hop = []
    adversarial_crossroads = []

    # Ground truth edges (src, dst)
    edges = [
        # Chain 1: 0 -> 1 -> 2 -> 3
        (0, 1), (1, 2), (2, 3),
        # Crossroad branch: 1 -> 4 -> 5
        (1, 4), (4, 5),
        # Chain 2: 6 -> 7 -> 8 -> 9
        (6, 7), (7, 8), (8, 9),
        # Crossroad branch: 7 -> 10 -> 11
        (7, 10), (10, 11),
        # Chain 3: 12 -> 13 -> 14 -> 15
        (12, 13), (13, 14), (14, 15),
        # Chain 4: 16 -> 17 -> 18 -> 19
        (16, 17), (17, 18), (18, 19),
        # Chain 5 (Medical Casualty Triage):
        # 20 (Severe Hemorrhagic Shock) -> 21 (Profound Hypotension) -> 22 (Vasopressor Infusion) -> 23 (ICU Resuscitation)
        (20, 21), (21, 22), (22, 23),
        # 20 -> 24 (Acute Respiratory Distress) -> 25 (Endotracheal Intubation)
        (20, 24), (24, 25),
        # Chain 6: 26 -> 27 -> 28 -> 29
        (26, 27), (27, 28), (28, 29),
    ]

    # Training set consists of all direct 1-hop facts
    for src, dst in edges:
        train_1hop.append((src, dst))

    # Test 2-hop compositions (Unseen during training)
    test_2hop = [
        (0, 2),  # via 1
        (1, 3),  # via 2
        (6, 8),  # via 7
        (7, 9),  # via 8
        (12, 14), # via 13
        (16, 18), # via 17
        (20, 22), # via 21 (Shock -> Vasopressors)
        (20, 25), # via 24 (Shock -> Intubation)
        (26, 28), # via 27
    ]

    # Test 3-hop compositions (Deep reasoning)
    test_3hop = [
        (0, 3),   # 0 -> 1 -> 2 -> 3
        (6, 9),   # 6 -> 7 -> 8 -> 9
        (12, 15), # 12 -> 13 -> 14 -> 15
        (16, 19), # 16 -> 17 -> 18 -> 19
        (20, 23), # 20 -> 21 -> 22 -> 23 (Shock -> ICU)
        (26, 29), # 26 -> 27 -> 28 -> 29
    ]

    # Adversarial Distractor Queries
    # Nodes that should NEVER be reached from Node 0 or Node 20
    adversarial_crossroads = [
        (0, 9, False),   # Node 0 cannot reach Node 9 (disconnected component)
        (0, 22, False),  # Node 0 cannot reach Medical ICU
        (20, 3, False),  # Medical shock cannot reach abstract node 3
        (26, 15, False), # Disconnected components
    ]

    return edges, train_1hop, test_2hop, test_3hop, adversarial_crossroads


def train_mcu_model():
    print("================================================================")
    print("      TRAINING GEOFORMER-MCU (OPTIMIZED FOR ATmega328P)         ")
    print("================================================================")
    edges, train_1hop, test_2hop, test_3hop, adversarial_crossroads = generate_mcu_dataset()
    print(f"Entities: {NUM_NODES} | Latent Dim: {D_MODEL}")
    print(f"1-Hop Train Facts: {len(train_1hop)}")
    print(f"2-Hop Test Queries: {len(test_2hop)}")
    print(f"3-Hop Test Queries: {len(test_3hop)}")
    print(f"Adversarial Probes: {len(adversarial_crossroads)}")

    model = GeoFormerMCU(num_nodes=NUM_NODES, d_model=D_MODEL)

    # Initialize Conductance Lattice with 1-hop facts
    with torch.no_grad():
        for src, dst in edges:
            model.w_lattice[src, dst] = 10.0  # High conductance (low resistance)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train for 150 epochs
    epochs = 150
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Shuffle training pairs
        random.shuffle(train_1hop)
        for src, dst in train_1hop:
            x = torch.zeros(1, NUM_NODES)
            x[0, src] = 1.0

            target = torch.tensor([dst], dtype=torch.long)

            # 1-hop forward
            out = model(x, tau=1)
            loss = F.nll_loss(torch.log(out + 1e-8), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp lattice to valid 4-bit non-negative values
            with torch.no_grad():
                model.w_lattice.clamp_(min=0.0, max=15.0)

            total_loss += loss.item()

        if epoch % 30 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(train_1hop):.4f}")

    # Evaluate on Unseen 2-Hop Test Set
    model.eval()
    print("\n---------------- EVALUATION: UNSEEN 2-HOP REASONING ----------------")
    correct_2hop = 0
    with torch.no_grad():
        for src, tgt in test_2hop:
            x = torch.zeros(1, NUM_NODES)
            x[0, src] = 1.0
            out = model(x, tau=2)
            pred = out.argmax(dim=-1).item()
            conf = out[0, pred].item() * 100
            is_match = (pred == tgt)
            if is_match:
                correct_2hop += 1
            print(f"Query {src:02d} -> ? | True: {tgt:02d} | Pred: {pred:02d} ({conf:.1f}%) | {'PASS' if is_match else 'FAIL'}")

    acc_2hop = (correct_2hop / len(test_2hop)) * 100
    print(f"2-Hop Zero-Shot Accuracy: {correct_2hop}/{len(test_2hop)} ({acc_2hop:.2f}%)")

    # Evaluate on Deep 3-Hop Test Set
    print("\n---------------- EVALUATION: DEEP 3-HOP REASONING ----------------")
    correct_3hop = 0
    with torch.no_grad():
        for src, tgt in test_3hop:
            x = torch.zeros(1, NUM_NODES)
            x[0, src] = 1.0
            out = model(x, tau=3)
            pred = out.argmax(dim=-1).item()
            conf = out[0, pred].item() * 100
            is_match = (pred == tgt)
            if is_match:
                correct_3hop += 1
            print(f"Query {src:02d} -> ? | True: {tgt:02d} | Pred: {pred:02d} ({conf:.1f}%) | {'PASS' if is_match else 'FAIL'}")

    acc_3hop = (correct_3hop / len(test_3hop)) * 100
    print(f"3-Hop Zero-Shot Accuracy: {correct_3hop}/{len(test_3hop)} ({acc_3hop:.2f}%)")

    # Save Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/geoformer_mcu_uno32.pt"
    
    # Extract quantized integer tensors
    w_in_ternary = quantize_ternary(model.w_in).detach().to(torch.int8)
    w_out_ternary = quantize_ternary(model.w_out).detach().to(torch.int8)
    w_lattice_int = model.get_conductance_matrix().detach().to(torch.uint8)

    torch.save({
        "model_name": "GeoFormer-MCU-Uno32",
        "num_nodes": NUM_NODES,
        "d_model": D_MODEL,
        "w_in": w_in_ternary,
        "w_out": w_out_ternary,
        "w_lattice": w_lattice_int,
        "test_2hop": test_2hop,
        "test_3hop": test_3hop,
        "adversarial_crossroads": adversarial_crossroads,
        "acc_2hop": acc_2hop,
        "acc_3hop": acc_3hop
    }, ckpt_path)

    print(f"\n[OK] Trained Checkpoint Saved: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    train_mcu_model()
