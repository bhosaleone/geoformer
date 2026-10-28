"""
Decisive Scientific Benchmark: Adaptive Propagation Geometry vs. Fixed-Geometry Baselines
Author: Atmabhan Pandit (Shrikant Bhosale) Theoretical Research Program

Null Hypothesis (H0):
"A ternary-weight network with adaptive propagation geometry achieves no statistically
significant improvement in unseen test generalization over an identical ternary-weight
network with fixed Euclidean geometry on a symbolic rule learning task."

Controlled Design:
- Task: Modular Addition (a + b) mod p, p = 23 (529 total pairs).
- Split: Strict 50% Train (264 pairs), 50% Held-Out Unseen Test (265 pairs).
- Precision: Strictly Ternary {-1, 0, +1} weights across all projection layers in ALL models.
- Parameter Budget: Identical latent dimension (d=32), identical node count (N=32), identical heads (2).
- Optimizer: Identical DualPathKramersOptimizer (or AdamW / SGD) with identical lr and schedule.
- Multi-Seed Replication: 5 independent random seeds (42, 101, 2024, 777, 999).

Models Tested:
1. Model A [Control]: Fixed-Geometry Euclidean Transformer (Ternary weights, standard dot-product attention, no metric lattice).
2. Model B [Ablation]: Static-Metric Transformer (Ternary weights, static uniform metric distance penalty, zero plastic updates).
3. Model C [Treatment 1]: Adaptive-Metric Transformer (Ternary weights, plastic conductance lattice W^(b) updating via jump dynamics, no TAC).
4. Model D [Treatment 2]: Full Q-GeoFormer (Ternary weights, plastic conductance lattice W^(b), Q-GHA, and continuous Q-TAC with S_5).
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_paired_ttest(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_err = math.sqrt(var_diff / n) if var_diff > 1e-12 else 1e-6
    t_stat = mean_diff / std_err
    abs_t = abs(t_stat)
    # Closed form CDF for Student-t with df=4:
    f_t = 0.5 + abs_t / (4.0 * math.sqrt(4.0 + abs_t**2)) * (1.0 + 2.0 / (4.0 + abs_t**2))
    p_val = max(0.0, min(1.0, 2.0 * (1.0 - f_t)))
    return t_stat, p_val

from quantized_geoformer.quantized_ops import QuantizedLinear
from quantized_geoformer.metric_lattice import DiscreteMetricWarpingEngine
from quantized_geoformer.attention import QuantizedGHA
from quantized_geoformer.tac_relaxation import QuantizedTAC
from quantized_geoformer.optimizer import DualPathKramersOptimizer

# ==============================================================================
# 1. FAIR CONTROLLED ARCHITECTURES (IDENTICAL TERNARY BUDGETS)
# ==============================================================================

class FixedEuclideanTransformer(nn.Module):
    """Control: Standard Transformer block with Ternary {-1, 0, +1} weights, flat Euclidean space."""
    def __init__(self, p=23, d_model=32, n_heads=2):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.embedding = nn.Embedding(p + 1, d_model)
        
        # Ternary Attention Projections
        self.w_q = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_k = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_v = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_o = QuantizedLinear(d_model, d_model, mode="ternary")
        
        # Ternary SwiGLU FFN
        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")
        
        self.gamma_1 = nn.Parameter(torch.ones(d_model))
        self.gamma_2 = nn.Parameter(torch.ones(d_model))
        self.head = nn.Linear(d_model, p, bias=False)

    def rms_norm(self, x, gamma, eps=1e-5):
        return (x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)) * gamma

    def forward(self, x):
        h = self.embedding(x)
        B, N, D = h.shape
        
        # Standard Flat Dot-Product Attention
        h_norm1 = self.rms_norm(h, self.gamma_1)
        Q = self.w_q(h_norm1).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(h_norm1).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(h_norm1).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, N, D)
        h_mid = h + self.w_o(attn_out)
        
        # SwiGLU FFN
        h_norm2 = self.rms_norm(h_mid, self.gamma_2)
        ffn_out = self.w_down(F.silu(self.w_gate(h_norm2)) * self.w_up(h_norm2))
        h_final = h_mid + ffn_out
        
        logits = self.head(h_final[:, -1, :])
        return logits, None, None


class StaticMetricTransformer(nn.Module):
    """Ablation: Ternary weights with a static, non-adaptive metric distance penalty."""
    def __init__(self, p=23, d_model=32, n_heads=2, n_neurons=32, b_w=4, lambda_geo=0.2):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.lambda_geo = lambda_geo
        
        self.embedding = nn.Embedding(p + 1, d_model)
        self.gha = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode="ternary", lambda_geo=lambda_geo)
        
        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")
        
        # Metric engine is initialized but NEVER updated (static prior)
        self.metric_engine = DiscreteMetricWarpingEngine(n_nodes=n_neurons, b_w=b_w)
        
        self.gamma_1 = nn.Parameter(torch.ones(d_model))
        self.gamma_2 = nn.Parameter(torch.ones(d_model))
        self.head = nn.Linear(d_model, p, bias=False)

    def rms_norm(self, x, gamma, eps=1e-5):
        return (x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)) * gamma

    def forward(self, x):
        h = self.embedding(x)
        B, N, D = h.shape
        D_metric = self.metric_engine.get_effective_resistance(N)
        
        h_norm1 = self.rms_norm(h, self.gamma_1)
        attn_out, A = self.gha(h_norm1, D_metric)
        h_mid = h + attn_out
        
        h_norm2 = self.rms_norm(h_mid, self.gamma_2)
        ffn_out = self.w_down(F.silu(self.w_gate(h_norm2)) * self.w_up(h_norm2))
        h_final = h_mid + ffn_out
        
        logits = self.head(h_final[:, -1, :])
        return logits, A, None


class AdaptiveMetricTransformer(nn.Module):
    """Treatment 1: Ternary weights with dynamic, plastic 4-bit metric conductance lattice."""
    def __init__(self, p=23, d_model=32, n_heads=2, n_neurons=32, b_w=4, lambda_geo=0.2):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.lambda_geo = lambda_geo
        
        self.embedding = nn.Embedding(p + 1, d_model)
        self.gha = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode="ternary", lambda_geo=lambda_geo)
        
        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")
        
        self.metric_engine = DiscreteMetricWarpingEngine(n_nodes=n_neurons, b_w=b_w)
        
        self.gamma_1 = nn.Parameter(torch.ones(d_model))
        self.gamma_2 = nn.Parameter(torch.ones(d_model))
        self.head = nn.Linear(d_model, p, bias=False)

    def rms_norm(self, x, gamma, eps=1e-5):
        return (x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)) * gamma

    def forward(self, x):
        h = self.embedding(x)
        B, N, D = h.shape
        D_metric = self.metric_engine.get_effective_resistance(N)
        
        h_norm1 = self.rms_norm(h, self.gamma_1)
        attn_out, A = self.gha(h_norm1, D_metric)
        h_mid = h + attn_out
        
        h_norm2 = self.rms_norm(h_mid, self.gamma_2)
        ffn_out = self.w_down(F.silu(self.w_gate(h_norm2)) * self.w_up(h_norm2))
        h_final = h_mid + ffn_out
        
        logits = self.head(h_final[:, -1, :])
        return logits, A, h_final

    def step_slow_timescale(self, h_final, A_grad, eta_g=0.015):
        self.metric_engine.update_discrete_conductance(
            h_states=h_final,
            A_grad=A_grad,
            eta_g=eta_g,
            mu=0.5,
            gamma=0.1
        )


class FullAdaptiveGeoFormer(nn.Module):
    """Treatment 2: Complete Q-GeoFormer (Ternary weights, plastic metric lattice, and Q-TAC with S_5)."""
    def __init__(self, p=23, d_model=32, n_heads=2, n_neurons=32, b_w=4, lambda_geo=0.2):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.embedding = nn.Embedding(p + 1, d_model)
        self.gha = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode="ternary", lambda_geo=lambda_geo)
        
        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")
        
        self.metric_engine = DiscreteMetricWarpingEngine(n_nodes=n_neurons, b_w=b_w)
        self.tac = QuantizedTAC(tau_max=3, eta_tac=0.08, lambda_dir=0.05, lambda_s5=0.1, use_s5=True, d_model=d_model)
        
        self.gamma_1 = nn.Parameter(torch.ones(d_model))
        self.gamma_2 = nn.Parameter(torch.ones(d_model))
        self.head = nn.Linear(d_model, p, bias=False)

    def rms_norm(self, x, gamma, eps=1e-5):
        return (x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)) * gamma

    def forward(self, x):
        h = self.embedding(x)
        B, N, D = h.shape
        D_metric = self.metric_engine.get_effective_resistance(N)
        
        h_norm1 = self.rms_norm(h, self.gamma_1)
        attn_out, A = self.gha(h_norm1, D_metric)
        h_mid = h + attn_out
        
        h_norm2 = self.rms_norm(h_mid, self.gamma_2)
        ffn_out = self.w_down(F.silu(self.w_gate(h_norm2)) * self.w_up(h_norm2))
        h_ffn = h_mid + ffn_out
        
        L = self.metric_engine.get_laplacian(N)
        h_star, tau_used, K_S = self.tac(h_ffn, L)
        
        logits = self.head(h_star[:, -1, :])
        return logits, A, h_star

    def step_slow_timescale(self, h_star, A_grad, eta_g=0.015):
        self.metric_engine.update_discrete_conductance(
            h_states=h_star,
            A_grad=A_grad,
            eta_g=eta_g,
            mu=0.5,
            gamma=0.1
        )

# ==============================================================================
# 2. RIGOROUS DATASET SPLIT (50% HELD-OUT UNSEEN TEST)
# ==============================================================================

def generate_modular_split(p=23, train_ratio=0.50, seed=42):
    """
    Generates exact train/test split.
    Total pairs: p^2 = 529.
    Train: 264 pairs (50%).
    Test: 265 pairs (50% completely unseen).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    all_pairs = []
    all_labels = []
    for a in range(p):
        for b in range(p):
            all_pairs.append([a, b, p]) # p is the '=' token
            all_labels.append((a + b) % p)
            
    pairs_tensor = torch.tensor(all_pairs, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    n_total = len(pairs_tensor)
    perm = torch.randperm(n_total)
    
    n_train = int(n_total * train_ratio)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    return pairs_tensor[train_idx], labels_tensor[train_idx], pairs_tensor[test_idx], labels_tensor[test_idx]

# ==============================================================================
# 3. TRAINING & EVALUATION HARNESS
# ==============================================================================

def train_and_eval_model(model_cls, model_name, X_train, y_train, X_test, y_test, epochs=140, lr=0.025, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_cls(p=23, d_model=32, n_heads=2)
    optimizer = DualPathKramersOptimizer(model.parameters(), lr=lr, alpha_tau=0.25, d_0=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_acc_curve = []
    test_acc_curve = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, A, h_out = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        
        cal_signal = math.sin(epoch * 0.15)
        optimizer.step(calendar_signal=cal_signal)
        
        # Adaptive metric update if model supports it
        if hasattr(model, "step_slow_timescale") and A is not None and A.grad is not None and h_out is not None:
            model.step_slow_timescale(h_out.detach(), A.grad, eta_g=0.015)
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            tr_pred = logits.argmax(dim=-1)
            tr_acc = (tr_pred == y_train).float().mean().item() * 100.0
            
            te_logits, _, _ = model(X_test)
            te_pred = te_logits.argmax(dim=-1)
            te_acc = (te_pred == y_test).float().mean().item() * 100.0
            
        train_acc_curve.append(tr_acc)
        test_acc_curve.append(te_acc)
        
    final_tr_acc = train_acc_curve[-1]
    final_te_acc = test_acc_curve[-1]
    peak_te_acc = max(test_acc_curve)
    
    return {
        "model_name": model_name,
        "seed": seed,
        "final_train_acc": final_tr_acc,
        "final_test_acc": final_te_acc,
        "peak_test_acc": peak_te_acc,
        "train_curve": train_acc_curve,
        "test_curve": test_acc_curve
    }

# ==============================================================================
# 4. MASTER EXPERIMENTAL EXECUTION (5 RANDOM SEEDS)
# ==============================================================================

def run_decisive_experiment():
    print("=" * 84)
    print("   DECISIVE SCIENTIFIC BENCHMARK: ADAPTIVE PROPAGATION GEOMETRY VS FIXED BASELINES")
    print("   Testing Rule Learning & Generalization on 50% Held-Out Unseen Test Pairs (p=23)")
    print("=" * 84)
    
    seeds = [42, 101, 2024, 777, 999]
    models_to_test = [
        ("Fixed_Euclidean", FixedEuclideanTransformer),
        ("Static_Metric", StaticMetricTransformer),
        ("Adaptive_Metric", AdaptiveMetricTransformer),
        ("Full_Q_GeoFormer", FullAdaptiveGeoFormer),
    ]
    
    results = {name: [] for name, _ in models_to_test}
    sample_curves = {name: None for name, _ in models_to_test}
    
    for s_idx, seed in enumerate(seeds):
        print(f"\n--- SEED {seed} ({s_idx + 1}/{len(seeds)}) ---")
        X_train, y_train, X_test, y_test = generate_modular_split(p=23, train_ratio=0.50, seed=seed)
        
        for name, cls in models_to_test:
            res = train_and_eval_model(cls, name, X_train, y_train, X_test, y_test, epochs=130, lr=0.025, seed=seed)
            results[name].append(res)
            print(f"  [{name:<18}] Final Train: {res['final_train_acc']:>5.1f}% │ Final Test: {res['final_test_acc']:>5.1f}% │ Peak Test: {res['peak_test_acc']:>5.1f}%")
            if s_idx == 0:
                sample_curves[name] = res
                
    # ==============================================================================
    # 5. STATISTICAL HYPOTHESIS TESTING
    # ==============================================================================
    print("\n" + "=" * 84)
    print("                        STATISTICAL HYPOTHESIS TEST RESULTS")
    print("=" * 84)
    print(f"{'Model Architecture':<22} | {'Final Train Acc':<16} | {'Final Test Acc (Unseen)':<24} | {'Peak Test Acc':<16}")
    print("-" * 84)
    
    summary_stats = {}
    for name, _ in models_to_test:
        tr_accs = [r["final_train_acc"] for r in results[name]]
        te_accs = [r["final_test_acc"] for r in results[name]]
        pk_accs = [r["peak_test_acc"] for r in results[name]]
        
        tr_mean, tr_std = np.mean(tr_accs), np.std(tr_accs)
        te_mean, te_std = np.mean(te_accs), np.std(te_accs)
        pk_mean, pk_std = np.mean(pk_accs), np.std(pk_accs)
        
        summary_stats[name] = {
            "tr_mean": tr_mean, "tr_std": tr_std,
            "te_mean": te_mean, "te_std": te_std,
            "pk_mean": pk_mean, "pk_std": pk_std,
            "te_accs": te_accs
        }
        
        print(f"{name:<22} | {tr_mean:>5.1f}% ± {tr_std:>4.1f}%   | {te_mean:>5.1f}% ± {te_std:>4.1f}%              | {pk_mean:>5.1f}% ± {pk_std:>4.1f}%")
        
    print("-" * 84)
    
    # Paired t-tests against the Fixed-Euclidean control
    ctrl_te = summary_stats["Fixed_Euclidean"]["te_accs"]
    print("\nPairwise Statistical Significance vs. Fixed-Euclidean Control (Two-Sided t-test):")
    for name in ["Static_Metric", "Adaptive_Metric", "Full_Q_GeoFormer"]:
        t_stat, p_val = compute_paired_ttest(summary_stats[name]["te_accs"], ctrl_te)
        diff = summary_stats[name]["te_mean"] - summary_stats["Fixed_Euclidean"]["te_mean"]
        signif = "REJECT H0 (p < 0.05)" if p_val < 0.05 else "FAIL TO REJECT H0"
        print(f"  • {name:<18} vs Fixed: Diff = {diff:>+5.1f}% │ t = {t_stat:>6.3f} │ p-value = {p_val:.4e} │ [{signif}]")

    # ==============================================================================
    # 6. GENERATE SCIENTIFIC PUBLICATION FIGURE
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#ffffff")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")
            spine.set_linewidth(1.0)
            
    colors = {
        "Fixed_Euclidean": "#64748b",    # Slate gray
        "Static_Metric":   "#f59e0b",    # Amber
        "Adaptive_Metric": "#0284c7",    # Sky blue
        "Full_Q_GeoFormer": "#059669"    # Emerald green
    }
    
    labels = {
        "Fixed_Euclidean": "Fixed Euclidean Control (Flat)",
        "Static_Metric":   "Static Metric Prior (No Warping)",
        "Adaptive_Metric": "Adaptive Metric (Plastic W^(b))",
        "Full_Q_GeoFormer": "Full Q-GeoFormer (Warping + TAC + S5)"
    }
    
    # Left Plot: Generalization Trajectories (Seed 42)
    ax1 = axes[0]
    epochs_range = range(1, len(sample_curves["Fixed_Euclidean"]["test_curve"]) + 1)
    models_to_test_names = [m[0] for m in models_to_test]
    for name in models_to_test_names:
        ax1.plot(epochs_range, sample_curves[name]["test_curve"], label=labels[name], color=colors[name], linewidth=2.2)
        ax1.plot(epochs_range, sample_curves[name]["train_curve"], color=colors[name], linestyle=":", alpha=0.45)
        
    ax1.set_title("A. Test Generalization on Unseen Pairs vs. Epoch (Seed 42)\n[Solid: Unseen Test Acc | Dotted: Train Acc]", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
    ax1.set_xlabel("Training Epoch", fontsize=10, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold", color="#0f172a")
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
    ax1.legend(loc="lower right", framealpha=0.95, edgecolor="#cbd5e1", fontsize=9)
    
    # Right Plot: Statistical Summary (Mean ± Std over 5 Seeds)
    ax2 = axes[1]
    x_pos = np.arange(len(models_to_test_names))
    means = [summary_stats[name]["te_mean"] for name in models_to_test_names]
    stds = [summary_stats[name]["te_std"] for name in models_to_test_names]
    bar_colors = [colors[name] for name in models_to_test_names]
    
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.88, edgecolor="#0f172a", linewidth=1.2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["Fixed\nEuclidean", "Static\nMetric", "Adaptive\nMetric", "Full Q-GeoFormer\n(Warping+TAC)"], fontsize=9.5, fontweight="bold", color="#0f172a")
    ax2.set_title("B. Unseen Test Accuracy Across 5 Seeds (Mean ± Std)\n[Statistical Generalization Across Unseen Test Instances]", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
    ax2.set_ylabel("Unseen Test Accuracy (%)", fontsize=10, fontweight="bold", color="#0f172a")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y", color="#cbd5e1")
    
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                    ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=11)
        
    plt.tight_layout()
    plot_path = "diagrams/figure_7_decisive_adaptive_vs_fixed_benchmark.png"
    plt.savefig(plot_path, dpi=300, facecolor="#ffffff")
    plt.savefig("diagrams/figure_7_decisive_adaptive_vs_fixed_benchmark.svg", facecolor="#ffffff")
    print(f"\nSaved scientific figure to: {plot_path}")
    print("=" * 84)

if __name__ == "__main__":
    run_decisive_experiment()
