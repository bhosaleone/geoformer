"""
benchmarks/diagnose_multihop_failure.py
=======================================
Root-Cause Diagnosis of the 2-Hop / Multi-Hop Reasoning Failure.

Investigates why the neural model scored 0% on 2-hop while pure graph diffusion
achieves 100%, isolating the mathematical bottlenecks:
  1. Undirected Symmetrization (A -> B -> A bounce-back).
  2. Untrained Query Relation Embeddings.
  3. Residual Drowning in Continuous Diffusion.
  4. Output Head Misalignment.

Usage:
  python3 benchmarks/diagnose_multihop_failure.py
"""

import torch
import numpy as np


def main():
    print("=" * 80)
    print("🔬 ROOT-CAUSE DIAGNOSIS: MULTI-HOP TRANSITIVE REASONING FAILURE")
    print("=" * 80)

    # Synthetic 4-hop chains: A -> B -> C -> D -> E
    chains = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]
    ]
    N = 24

    # --------------------------------------------------------------------------
    # Test 1: Pure Directed Graph Adjacency Power (A^k)
    # --------------------------------------------------------------------------
    print("\n[Test 1] Pure Directed Graph Adjacency Powers (A^k @ h_0):")
    W_dir = torch.zeros(N, N)
    for c in chains:
        for k in range(len(c) - 1):
            W_dir[c[k], c[k+1]] = 1.0

    for hop in [1, 2, 3, 4]:
        hits = 0
        for c in chains:
            h = torch.zeros(N)
            h[c[0]] = 1.0
            # Flow forward along directed edges
            for _ in range(hop):
                h = torch.matmul(W_dir.T, h)
            pred = h.argmax().item()
            if pred == c[hop]:
                hits += 1
        acc = hits / len(chains) * 100.0
        print(f"  • {hop}-Hop Directed Diffusion Accuracy: {acc:.1f}%")

    # --------------------------------------------------------------------------
    # Test 2: Why Undirected Symmetrization Fails (A -> B -> A bounce-back)
    # --------------------------------------------------------------------------
    print("\n[Test 2] Impact of Undirected Symmetrization (W = W + W.T):")
    W_undir = W_dir + W_dir.T
    for hop in [2]:
        for c in chains[:2]:
            h = torch.zeros(N)
            h[c[0]] = 1.0
            for _ in range(hop):
                h = torch.matmul(W_undir, h)
            top3 = torch.topk(h, k=3).indices.tolist()
            print(f"  • Chain starting at {c[0]} (target 2-hop is {c[2]}):")
            print(f"    Predictions: {top3}")
            print(f"    Diagnostic: Node {c[0]} receives {h[c[0]].item():.1f} echo units (bounce-back), while target {c[2]} receives {h[c[2]].item():.1f} units.")

    # --------------------------------------------------------------------------
    # Test 3: Residual Drowning Analysis
    # --------------------------------------------------------------------------
    print("\n[Test 3] Residual Drowning in Continuous Relaxation:")
    # If h_out = h_attn (weight 1.0) + 0.1 * (W @ h_attn)
    # Target signal is diluted by 10x relative to 1-hop attended state
    h_attn = torch.zeros(N)
    h_attn[1] = 1.0 # Attended to node B
    diff_step = torch.matmul(W_dir.T, h_attn) # Pushes to node C (2)
    h_out_diluted = h_attn + 0.1 * diff_step
    pred_diluted = h_out_diluted.argmax().item()
    print(f"  • Argmax with 0.1 residual scaling: Node {pred_diluted} (1-hop neighbor B wins!)")

    # Un-drowned propagation:
    h_out_pure = diff_step
    pred_pure = h_out_pure.argmax().item()
    print(f"  • Argmax with pure diffusion step: Node {pred_pure} (2-hop target C correctly retrieved!)")

    print("\n" + "=" * 80)
    print("CONCLUSION & ACTIONABLE FIX:")
    print("1. Directed vs Undirected: Multi-hop reasoning requires directed propagation.")
    print("   Symmetrizing W causes 2-hop bounce-back to self.")
    print("2. State Transition: Multi-hop queries require stepping state h_{t+1} = W^T h_t,")
    print("   rather than adding an attenuated 0.1 residual onto the 1-hop attended state.")
    print("3. Query Relational Grounding: If relation index r=1 is tested, its embedding")
    print("   must be grounded during training or cast as composed relation (r_1 o r_2).")
    print("=" * 80)


if __name__ == "__main__":
    main()
