"""
Comprehensive 5-Step Diagnostic Protocol for Propagation Geometry
Author: Atmabhan Pandit (Shrikant Bhosale) Theoretical Research Program

Implements the user's decisive 5-step diagnostic sequence:
  Test A — Validate the task (p = 5, all 25 pairs verified, chance = 20.0%)
  Test B — Overfit a tiny dataset (test memorization capacity on 25 pairs)
  Test C — Establish the floating-point baseline (FP32 vs. Ternary comparison)
  Test D — Test rule generalization (18 train pairs / 7 unseen held-out test pairs across 5 seeds)
  Test E — Inspect the geometry itself (measure Frobenius norm ||Delta W|| and Delta lambda_2)
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantized_geoformer.quantized_ops import QuantizedLinear
from quantized_geoformer.metric_lattice import DiscreteMetricWarpingEngine
from quantized_geoformer.tac_relaxation import QuantizedTAC

# ==============================================================================
# TEST A: TASK VALIDATION & DATASET GENERATION
# ==============================================================================

def validate_and_generate_mod5_task():
    p = 5
    pairs = []
    labels = []
    for a in range(p):
        for b in range(p):
            pairs.append([a, b])
            labels.append((a + b) % p)
            
    pairs_tensor = torch.tensor(pairs, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Mathematical sanity verification
    class_counts = {c: labels.count(c) for c in range(p)}
    assert len(pairs) == 25, f"Expected 25 pairs, got {len(pairs)}"
    for c in range(p):
        assert class_counts[c] == 5, f"Class {c} has {class_counts[c]} examples, expected 5"
        
    return pairs_tensor, labels_tensor, class_counts

# ==============================================================================
# ARCHITECTURES FOR FP32 AND TERNARY COMPARISONS
# ==============================================================================

class DiagnosticModel(nn.Module):
    def __init__(self, p=5, d_model=64, mode="ternary", geometry="fixed", use_tac=False):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.mode = mode          # "fp32" or "ternary"
        self.geometry = geometry  # "fixed", "static", "adaptive"
        self.use_tac = use_tac
        
        self.embedding = nn.Embedding(p, d_model)
        
        # Attention Projections
        if mode == "ternary":
            self.w_q = QuantizedLinear(d_model, d_model, mode="ternary")
            self.w_k = QuantizedLinear(d_model, d_model, mode="ternary")
            self.w_v = QuantizedLinear(d_model, d_model, mode="ternary")
            self.w_o = QuantizedLinear(d_model, d_model, mode="ternary")
            self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
            self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
            self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")
        else:
            self.w_q = nn.Linear(d_model, d_model, bias=False)
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)
            self.w_o = nn.Linear(d_model, d_model, bias=False)
            self.w_gate = nn.Linear(d_model, d_model * 2, bias=False)
            self.w_up   = nn.Linear(d_model, d_model * 2, bias=False)
            self.w_down = nn.Linear(d_model * 2, d_model, bias=False)
            
        if geometry in ["static", "adaptive"]:
            self.metric_engine = DiscreteMetricWarpingEngine(n_nodes=2, b_w=4)
        else:
            self.metric_engine = None
            
        if use_tac:
            self.tac = QuantizedTAC(tau_max=2, eta_tac=0.08, lambda_dir=0.05, lambda_s5=0.1, use_s5=True, d_model=d_model)
        else:
            self.tac = None
            
        self.head = nn.Linear(2 * d_model, p, bias=False)

    def forward(self, x):
        B, N = x.shape
        h = self.embedding(x) # [B, 2, D]
        
        Q = self.w_q(h)
        K = self.w_k(h)
        V = self.w_v(h)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply metric distance penalty if geometry is not fixed
        if self.metric_engine is not None:
            D_metric = self.metric_engine.get_effective_resistance(N)
            scores = scores - 0.20 * D_metric.unsqueeze(0)
            
        A = F.softmax(scores, dim=-1)
        if self.training and A.requires_grad:
            A.retain_grad()
            
        attn_out = self.w_o(torch.matmul(A, V))
        h_mid = h + attn_out
        
        # SwiGLU FFN
        ffn_out = self.w_down(F.silu(self.w_gate(h_mid)) * self.w_up(h_mid))
        h_ffn = h_mid + ffn_out
        
        if self.tac is not None and self.metric_engine is not None:
            L = self.metric_engine.get_laplacian(N)
            h_final, _, _ = self.tac(h_ffn, L)
        else:
            h_final = h_ffn
            
        logits = self.head(h_final.view(B, -1))
        return logits, A, h_final

    def step_slow_timescale(self, h_final, A_grad, eta_g=0.02):
        if self.metric_engine is not None and self.geometry == "adaptive":
            self.metric_engine.update_discrete_conductance(
                h_states=h_final,
                A_grad=A_grad,
                eta_g=eta_g,
                mu=0.5,
                gamma=0.1
            )

# ==============================================================================
# RUN SUITE
# ==============================================================================

def run_5step_diagnostics():
    print("=" * 86)
    print("   QUANTIZED PROPAGATION GEOMETRY: 5-STEP RIGOROUS DIAGNOSTIC PROTOCOL")
    print("=" * 86)
    
    # --------------------------------------------------------------------------
    # TEST A: Validate the Task
    # --------------------------------------------------------------------------
    print("\n[TEST A] VALIDATE THE TASK (p = 5)")
    X_all, y_all, counts = validate_and_generate_mod5_task()
    print(f"  • Total input pairs generated: {len(X_all)}")
    print(f"  • Class distribution: {counts}")
    print(f"  • Random uniform chance level: 1/5 = {100.0/5:.1f}%")
    print(f"  • Sample pairs verified:")
    for i in [0, 1, 6, 12, 18, 24]:
        a, b = X_all[i].tolist()
        lbl = y_all[i].item()
        assert (a + b) % 5 == lbl
        print(f"      ({a} + {b}) mod 5 = {lbl} ✓")
    print("  ✓ Task A Validation: 100% Mathematically Certified & Balanced.")

    # --------------------------------------------------------------------------
    # TEST B & C: Overfit a Tiny Dataset & Floating-Point Baseline
    # --------------------------------------------------------------------------
    print("\n" + "-" * 86)
    print("[TEST B & C] OVERFIT CAPACITY: FP32 VS. TERNARY ON 25 PAIRS (600 Epochs)")
    print("-" * 86)
    
    configs = [
        ("FP32_Fixed_Euclidean",   "fp32",    "fixed",    False),
        ("FP32_Adaptive_Metric",   "fp32",    "adaptive", False),
        ("Ternary_Fixed_Euclidean","ternary", "fixed",    False),
        ("Ternary_Adaptive_Metric","ternary", "adaptive", False),
        ("Ternary_Full_GeoFormer", "ternary", "adaptive", True),
    ]
    
    overfit_results = {}
    
    for name, mode, geom, tac in configs:
        torch.manual_seed(42)
        model = DiagnosticModel(p=5, d_model=64, mode=mode, geometry=geom, use_tac=tac)
        lr = 0.015 if mode == "fp32" else 0.02
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        
        initial_loss = None
        initial_acc = None
        
        for ep in range(1, 601):
            model.train()
            opt.zero_grad()
            logits, A, h_out = model(X_all)
            loss = crit(logits, y_all)
            loss.backward()
            opt.step()
            
            if geom == "adaptive" and A.grad is not None:
                model.step_slow_timescale(h_out.detach(), A.grad, eta_g=0.02)
                
            if ep == 1:
                initial_loss = loss.item()
                initial_acc = (logits.argmax(dim=-1) == y_all).float().mean().item() * 100.0
                
        final_loss = loss.item()
        final_acc = (logits.argmax(dim=-1) == y_all).float().mean().item() * 100.0
        overfit_results[name] = {"init_acc": initial_acc, "final_acc": final_acc, "final_loss": final_loss}
        
        status = "PASSED (OVERFIT ACHIEVED)" if final_acc >= 90.0 else "PARTIAL/SUB-SATURATED"
        print(f"  • {name:<26} | Init: {initial_acc:>5.1f}% (L={initial_loss:.3f}) │ Final: {final_acc:>5.1f}% (L={final_loss:.4f}) │ [{status}]")

    # --------------------------------------------------------------------------
    # TEST D: Rule Generalization on Held-Out Unseen Pairs (5 Seeds)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 86)
    print("[TEST D] RULE GENERALIZATION: 18 TRAIN PAIRS / 7 UNSEEN TEST PAIRS (5 Seeds)")
    print("-" * 86)
    
    seeds = [42, 101, 2024, 777, 999]
    gen_results = {name: {"train_accs": [], "test_accs": []} for name, _, _, _ in configs}
    
    for s_idx, seed in enumerate(seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        perm = torch.randperm(25)
        train_idx = perm[:18]
        test_idx = perm[18:]
        
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]
        
        for name, mode, geom, tac in configs:
            torch.manual_seed(seed)
            model = DiagnosticModel(p=5, d_model=64, mode=mode, geometry=geom, use_tac=tac)
            lr = 0.015 if mode == "fp32" else 0.02
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()
            
            for ep in range(1, 501):
                model.train()
                opt.zero_grad()
                logits, A, h_out = model(X_tr)
                loss = crit(logits, y_tr)
                loss.backward()
                opt.step()
                
                if geom == "adaptive" and A.grad is not None:
                    model.step_slow_timescale(h_out.detach(), A.grad, eta_g=0.02)
                    
            model.eval()
            with torch.no_grad():
                tr_pred = logits.argmax(dim=-1)
                tr_acc = (tr_pred == y_tr).float().mean().item() * 100.0
                
                te_logits, _, _ = model(X_te)
                te_pred = te_logits.argmax(dim=-1)
                te_acc = (te_pred == y_te).float().mean().item() * 100.0
                
            gen_results[name]["train_accs"].append(tr_acc)
            gen_results[name]["test_accs"].append(te_acc)
            
    print(f"{'Model Architecture':<26} | {'Train Acc (18 pairs)':<22} | {'Unseen Test Acc (7 pairs)':<24}")
    print("-" * 86)
    for name, _, _, _ in configs:
        tr_mean = np.mean(gen_results[name]["train_accs"])
        tr_std  = np.std(gen_results[name]["train_accs"])
        te_mean = np.mean(gen_results[name]["test_accs"])
        te_std  = np.std(gen_results[name]["test_accs"])
        print(f"{name:<26} | {tr_mean:>5.1f}% ± {tr_std:>4.1f}%             | {te_mean:>5.1f}% ± {te_std:>4.1f}%")

    # --------------------------------------------------------------------------
    # TEST E: Inspect the Geometry Itself (Metric Plasticity Audit)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 86)
    print("[TEST E] INSPECT THE GEOMETRY ITSELF: METRIC CONDUCTANCE PLASTICITY AUDIT")
    print("-" * 86)
    
    torch.manual_seed(42)
    model = DiagnosticModel(p=5, d_model=64, mode="ternary", geometry="adaptive", use_tac=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    crit = nn.CrossEntropyLoss()
    
    W_init = model.metric_engine.W.clone()
    fiedler_init = model.metric_engine.get_fiedler_eigenvalue(2)
    
    for ep in range(1, 401):
        model.train()
        opt.zero_grad()
        logits, A, h_out = model(X_all)
        loss = crit(logits, y_all)
        loss.backward()
        opt.step()
        
        if A.grad is not None:
            model.step_slow_timescale(h_out.detach(), A.grad, eta_g=0.02)
            
    W_final = model.metric_engine.W.clone()
    fiedler_final = model.metric_engine.get_fiedler_eigenvalue(2)
    
    delta_W = torch.norm(W_final - W_init, p='fro').item()
    print(f"  • Initial Metric Conductance Lattice W(0):\n{W_init.numpy()}")
    print(f"  • Final Metric Conductance Lattice W(T):\n{W_final.numpy()}")
    print(f"  • Frobenius norm change ||W(T) - W(0)||_F: {delta_W:.4f}")
    print(f"  • Initial Fiedler eigenvalue lambda_2: {fiedler_init:.4f}")
    print(f"  • Final Fiedler eigenvalue lambda_2:   {fiedler_final:.4f}")
    
    if delta_W > 1e-4:
        print("  ✓ Plasticity Status: ACTIVE (Metric geometry physically warped during training).")
    else:
        print("  ✗ Plasticity Status: INACTIVE (Metric geometry remained frozen).")
        
    print("=" * 86)

if __name__ == "__main__":
    run_5step_diagnostics()
