"""
Scientific Diagram & Architectural Figure Generator (Clean Light / Academic Paper Edition)
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Generates 6 publication-quality, non-overlapping scientific diagrams with:
  - Clean pure white background (#ffffff)
  - Elegant light/pastel card fills (#f8fafc, #f1f5f9) with subtle borders
  - High-contrast, crystal-clear dark typography (#0f172a)
  - Zero overlapping labels, generous padding, and spacious layout
  - High-resolution SVG and PNG outputs in diagrams/
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Clean Light Theme Palette
LIGHT_BG = "#ffffff"
CARD_BG = "#f8fafc"
CARD_BG_BLUE = "#f0f9ff"
CARD_BG_GOLD = "#fffbeb"
CARD_BG_PURPLE = "#faf5ff"
CARD_BG_GREEN = "#f0fdf4"
BORDER_COLOR = "#cbd5e1"
TEXT_DARK = "#0f172a"
TEXT_BODY = "#334155"
TEXT_MUTED = "#64748b"

# Accent Colors (Clean, high-contrast on white)
SKY_BLUE = "#0284c7"
DEEP_BLUE = "#1d4ed8"
AMBER_GOLD = "#d97706"
PURPLE = "#7c3aed"
EMERALD = "#059669"
ROSE = "#dc2626"
INDIGO = "#4f46e5"

plt.rcParams.update({
    "figure.facecolor": LIGHT_BG,
    "axes.facecolor": LIGHT_BG,
    "axes.edgecolor": BORDER_COLOR,
    "axes.labelcolor": TEXT_DARK,
    "xtick.color": TEXT_BODY,
    "ytick.color": TEXT_BODY,
    "text.color": TEXT_DARK,
    "grid.color": "#e2e8f0",
    "grid.linestyle": ":",
    "grid.alpha": 0.8,
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif"
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 1: Geodesic Highway Emergence & Distance Contraction (Theorem Q-1)
# -----------------------------------------------------------------------------
def plot_figure_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle("FIGURE 1: DISCRETE GEODESIC HIGHWAY EMERGENCE & DISTANCE CONTRACTION (THEOREM Q-1)",
                 fontsize=14, fontweight="bold", color=TEXT_DARK, y=0.98)
    
    # 1A: Metric Distance vs Conductance
    W_levels = np.linspace(0.0, 1.0, 16) # 4-bit lattice levels
    eps = 0.05
    d_metric = 1.0 / (W_levels + eps)
    
    ax1.plot(W_levels, d_metric, color=SKY_BLUE, linewidth=2.8, marker='o', markersize=7,
             markerfacecolor=DEEP_BLUE, markeredgecolor=LIGHT_BG, markeredgewidth=1.5,
             label=r"Geodesic Distance $d_{ij} = \frac{1}{W_{ij} + \epsilon_{\mathrm{dist}}}$")
    ax1.fill_between(W_levels, 0, d_metric, color=SKY_BLUE, alpha=0.08)
    
    # Non-overlapping callout boxes
    ax1.annotate("Geodesic Highway Emergence\n(Low Resistance, $d_{ij} \\to 0.95$)",
                 xy=(W_levels[-1], d_metric[-1]), xytext=(0.45, 6.5),
                 arrowprops=dict(arrowstyle="->", color=EMERALD, lw=2.0),
                 color=TEXT_DARK, fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG_GREEN, edgecolor=EMERALD, lw=1.5))
    
    ax1.annotate("Decayed Barrier State\n(High Impedance, $d_{ij} \\to 20.0$)",
                 xy=(W_levels[0], d_metric[0]), xytext=(0.20, 17.0),
                 arrowprops=dict(arrowstyle="->", color=ROSE, lw=2.0),
                 color=TEXT_DARK, fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff1f2", edgecolor=ROSE, lw=1.5))
    
    ax1.set_title("Effective Resistance Metric vs Discrete Conductance Level", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Discrete Conductance Level $W_{ij}$ (4-Bit Lattice Grid)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Effective Resistance Geodesic Distance $d_{ij}$", fontsize=10, labelpad=8)
    ax1.set_ylim(0, 22)
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True, facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9.5)
    
    # 1B: Contraction Dynamics Over Slow Steps
    steps = np.arange(0, 50)
    w_active = 0.05 + 0.95 * (1.0 - np.exp(-0.08 * steps))
    d_active = 1.0 / (w_active + eps)
    
    w_idle = 0.5 * np.exp(-0.05 * steps) + 0.02
    d_idle = 1.0 / (w_idle + eps)
    
    ax2.plot(steps, d_active, color=EMERALD, linewidth=2.8,
             label=r"Active Pathway ($a_{ij} > \frac{\gamma}{\mu} W_{ij}$): $\dot{d}_{ij} < 0$ (Contracting)")
    ax2.plot(steps, d_idle, color=ROSE, linewidth=2.2, linestyle="--",
             label=r"Idle Pathway ($a_{ij} \to 0$): Resistance Decays to Barrier")
    
    ax2.fill_between(steps, d_active, d_idle, color=AMBER_GOLD, alpha=0.06, label="Plastic Adaptation Margin")
    
    ax2.set_title("Two-Timescale Distance Evolution Across Learning Steps", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Slow Metric Learning Steps $t$", fontsize=10, labelpad=8)
    ax2.set_ylabel("Geodesic Distance $d_{ij}(t)$", fontsize=10, labelpad=8)
    ax2.set_ylim(0, 22)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_1_geodesic_highway_contraction.svg"), format="svg", facecolor=LIGHT_BG)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_1_geodesic_highway_contraction.png"), dpi=250, facecolor=LIGHT_BG)
    plt.close()
    print("  [✓] Generated Figure 1 (Clean Light Edition)")

# -----------------------------------------------------------------------------
# Figure 2: S_5 Galois Irreducible Representations & Kolmogorov Bound (Theorem Q-6)
# -----------------------------------------------------------------------------
def plot_figure_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle("FIGURE 2: S_5 GALOIS CAYLEY DECOMPOSITION & DESCRIPTION COMPLEXITY BOUND (THEOREM Q-6)",
                 fontsize=14, fontweight="bold", color=TEXT_DARK, y=0.98)
    
    irreps = [r"$\rho_1\,(5)$", r"$\rho_2\,(1^5)$", r"$\rho_3\,(4,1)$", r"$\rho_4\,(2,1^3)$",
              r"$\rho_5\,(3,2)$", r"$\rho_6\,(2^2,1)$", r"$\rho_7\,(3,1^2)$"]
    dims = [1, 1, 4, 4, 5, 5, 6]
    squares = [d**2 for d in dims] # [1, 1, 16, 16, 25, 25, 36]
    c_bits = [0.0 if d == 1 else d * math.log2(d) for d in dims]
    
    x_pos = np.arange(len(irreps))
    
    # 2A: Burnside Squares Bar Chart
    colors_bars = [SKY_BLUE, SKY_BLUE, INDIGO, INDIGO, PURPLE, PURPLE, DEEP_BLUE]
    bars = ax1.bar(x_pos, squares, color=colors_bars, alpha=0.85, edgecolor=BORDER_COLOR, width=0.55)
    
    for b, d, sq in zip(bars, dims, squares):
        ax1.text(b.get_x() + b.get_width()/2, sq + 1.2, f"$d={d}$\n$d^2={sq}$",
                 ha='center', va='bottom', fontsize=9, color=TEXT_DARK, fontweight="bold")
        
    ax1.set_title("Burnside Dimension Sum: $\sum_{\\rho=1}^7 d_\\rho^2 = 120 = |S_5|$", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(irreps, fontsize=9.5)
    ax1.set_ylabel(r"Irreducible Representation Dimension Squared $d_\rho^2$", fontsize=10, labelpad=8)
    ax1.set_ylim(0, 50)
    ax1.grid(True)
    
    # Annotate Burnside sum in clean box
    ax1.text(0.05, 0.90, "Burnside's Identity:\n$1 + 1 + 16 + 16 + 25 + 25 + 36 = 120$\nOrthogonality Certified (Z3 `unsat`)",
             transform=ax1.transAxes, fontsize=9, color=TEXT_DARK,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG_PURPLE, edgecolor=PURPLE, lw=1.2))
    
    # 2B: Entropy Contributions and Cumulative Bound
    cum_bits = np.cumsum(c_bits)
    ax2.bar(x_pos, c_bits, color=AMBER_GOLD, alpha=0.55, edgecolor=AMBER_GOLD, width=0.50,
            label=r"Per-Irrep Capacity $d_\rho \log_2 d_\rho$")
    ax2.plot(x_pos, cum_bits, color=DEEP_BLUE, marker='s', markersize=8, linewidth=2.8,
             label=r"Cumulative Channel Capacity $\sum_{k=1}^\rho d_k \log_2 d_k$")
    
    ax2.axhline(y=54.68, color=ROSE, linestyle="--", linewidth=2.2,
                label=r"Universal Kolmogorov Ceiling $K(S) \leq 54.68\ \mathrm{bits}$")
    ax2.fill_between(x_pos, 0, 54.68, color=ROSE, alpha=0.04)
    
    ax2.set_title("Internal Description Complexity Upper Bound", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(irreps, fontsize=9.5)
    ax2.set_ylabel("Information Capacity (Bits)", fontsize=10, labelpad=8)
    ax2.set_ylim(0, 70)
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=True, facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_2_s5_cayley_irrep_spectrum.svg"), format="svg", facecolor=LIGHT_BG)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_2_s5_cayley_irrep_spectrum.png"), dpi=250, facecolor=LIGHT_BG)
    plt.close()
    print("  [✓] Generated Figure 2 (Clean Light Edition)")

# -----------------------------------------------------------------------------
# Figure 3: Quantized Kramers Grokking Potential Landscape (Theorem Q-5)
# -----------------------------------------------------------------------------
def plot_figure_3():
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle("FIGURE 3: QUANTIZED KRAMERS GROKKING POTENTIAL LANDSCAPE (THEOREM Q-5)",
                 fontsize=14, fontweight="bold", color=TEXT_DARK, y=0.98)
    
    theta = np.linspace(-3.2, 3.2, 600)
    V_smooth = 0.45 * (theta**2 - 1.6)**2 + 0.25 * theta
    V_quant = V_smooth + 0.12 * np.sin(18 * theta)
    
    ax.plot(theta, V_quant, color=DEEP_BLUE, linewidth=2.6, label=r"Quantized Piecewise-Constant Loss $\mathcal{F}_{\mathrm{quant}}(\theta)$")
    ax.plot(theta, V_smooth, color=TEXT_MUTED, linewidth=1.4, linestyle=":", label="Underlying Continuous Action Manifold")
    
    # Highlight Local Trap (Memorization) and Global Basin (Generalization)
    ax.scatter([-1.35], [0.18], color=ROSE, s=180, edgecolor=LIGHT_BG, linewidth=2, zorder=5)
    ax.scatter([1.20], [0.85], color=EMERALD, s=180, edgecolor=LIGHT_BG, linewidth=2, zorder=5)
    
    # Clean Callouts with Ample Spacing
    ax.annotate("MEMORIZATION TRAP\n(Training Acc = 100%, Test Acc ~ Chance)\nDeterministic Gradient Trapped $\\nabla \\mathcal{F} = 0$",
                xy=(-1.35, 0.25), xytext=(-3.1, 2.2),
                arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.8),
                color=TEXT_DARK, fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff1f2", edgecolor=ROSE, lw=1.5))
    
    ax.annotate("ALGORITHMIC ATTRACTOR (GROKKING)\n(Generalization Acc = 100%)\nLow-Rank Induction Head Formed",
                xy=(1.20, 0.95), xytext=(0.8, 3.2),
                arrowprops=dict(arrowstyle="->", color=EMERALD, lw=1.8),
                color=TEXT_DARK, fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG_GREEN, edgecolor=EMERALD, lw=1.5))
    
    # Kramers Barrier Height annotation
    ax.annotate("", xy=(0.0, 1.15), xytext=(0.0, 2.2),
                arrowprops=dict(arrowstyle="<->", color=AMBER_GOLD, lw=2.2))
    ax.text(0.12, 1.65, r"Discrete Barrier $\Delta \mathcal{F}_b$", color=AMBER_GOLD, fontsize=11, fontweight="bold")
    
    # Tunneling Path Arrow
    ax.annotate(r"Dual Path Kramers Escape: $\Gamma_{\mathrm{dual}} = \frac{\omega_0 \omega_b}{2\pi \gamma_d} e^{-\Delta \mathcal{F}_b / (\alpha_\tau D)} > 0$" + "\n" + r"Finite Grokking Time $\tau_{\mathrm{grok}} < \infty$ under Langevin Field 2 Diffusion",
                xy=(0.95, 1.1), xytext=(-1.8, 3.6),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.25", color=INDIGO, lw=2.5),
                color=TEXT_DARK, fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG_BLUE, edgecolor=INDIGO, lw=1.5))
    
    ax.set_title("Thermally-Assisted Stochastic Action Diffusion Across Non-Smooth Step Barriers", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(r"Quantized Synaptic Parameter Coordinate $\theta \in \{-1, 0, +1\}$", fontsize=10, labelpad=8)
    ax.set_ylabel(r"Variational Loss Landscape Potential $\mathcal{F}(\theta)$", fontsize=10, labelpad=8)
    ax.set_ylim(-0.2, 4.6)
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_quantized_kramers_grokking.svg"), format="svg", facecolor=LIGHT_BG)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_quantized_kramers_grokking.png"), dpi=250, facecolor=LIGHT_BG)
    plt.close()
    print("  [✓] Generated Figure 3 (Clean Light Edition)")

# -----------------------------------------------------------------------------
# Figure 4: The Bit-Neuron Pareto Frontier (Theorem Q-3)
# -----------------------------------------------------------------------------
def plot_figure_4():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.suptitle("FIGURE 4: THE BIT-NEURON PARETO OPTIMIZATION FRONTIER (THEOREM Q-3)",
                 fontsize=14, fontweight="bold", color=TEXT_DARK, y=0.98)
    
    b_theta = np.linspace(1, 16, 120)
    b_W = np.linspace(1, 8, 120)
    B_theta, B_W = np.meshgrid(b_theta, b_W)
    
    N = 32
    Total_Bits = (N * 64 * B_theta + (N * (N - 1) / 2) * B_W)
    Capability = 100.0 / (1.0 + np.exp(-(B_W - 2.8))) * (1.0 - np.exp(-B_theta * 0.75))
    Efficiency = (Capability / (Total_Bits / 1000.0))
    
    # Light aesthetic colormap: YlGnBu
    cp = ax.contourf(B_theta, B_W, Efficiency, levels=25, cmap="YlGnBu", alpha=0.9)
    cbar = fig.colorbar(cp, ax=ax, pad=0.02)
    cbar.set_label(r"ISL Efficiency $\eta_{\mathrm{ISL}} = C / T$ (Capability per Unit Existential Cost)", color=TEXT_DARK, fontsize=9.5)
    
    # Mark Optimal Sweet Spot
    ax.scatter([1.58], [4.0], color=ROSE, s=280, edgecolor=LIGHT_BG, linewidth=2.5, zorder=10)
    ax.annotate("OPTIMAL PARETO VALLEY\nTernary Weights ($b_\\theta = 1.58$ bits) + 4-Bit Metric Lattice ($b_W = 4$)\nTopological degrees of freedom substitute for float mantissa!\nEmergence Verified at N = 32 Neurons",
                xy=(1.58, 4.0), xytext=(3.5, 6.0),
                arrowprops=dict(arrowstyle="->", color=ROSE, lw=2.2),
                color=TEXT_DARK, fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff1f2", edgecolor=ROSE, lw=1.5))
    
    # Mark FP16 Standard
    ax.scatter([16.0], [1.0], color=TEXT_BODY, s=180, edgecolor=LIGHT_BG, linewidth=2.0, zorder=10)
    ax.annotate("Conventional AI Regime (FP16)\nZero Plastic Metric ($b_W = 0$)\nExtreme Parameter Bloat",
                xy=(16.0, 1.0), xytext=(9.5, 1.8),
                arrowprops=dict(arrowstyle="->", color=TEXT_BODY, lw=1.8),
                color=TEXT_DARK, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_BG, edgecolor=BORDER_COLOR, lw=1.2))
    
    ax.set_title("Information Allocation: Fast Numerical Precision vs Slow Metric Precision", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(r"Fast Synaptic Parameter Precision $b_\theta$ (Bits)", fontsize=10, labelpad=8)
    ax.set_ylabel(r"Slow Metric Conductance Precision $b_W$ (Bits)", fontsize=10, labelpad=8)
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_4_bit_neuron_pareto_frontier.svg"), format="svg", facecolor=LIGHT_BG)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_4_bit_neuron_pareto_frontier.png"), dpi=250, facecolor=LIGHT_BG)
    plt.close()
    print("  [✓] Generated Figure 4 (Clean Light Edition)")

# -----------------------------------------------------------------------------
# Figure 5: Two-Timescale Metric Warping Dynamical Engine
# -----------------------------------------------------------------------------
def plot_figure_5():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    fig.suptitle("FIGURE 5: TWO-TIMESCALE METRIC WARPING DYNAMICAL ENGINE",
                 fontsize=14, fontweight="bold", color=TEXT_DARK, y=0.98)
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Fast Timescale Card (Left)
    fast_card = patches.FancyBboxPatch((1.0, 4.0), 5.8, 4.5, boxstyle="round,pad=0.3",
                                      edgecolor=DEEP_BLUE, facecolor=CARD_BG_BLUE, linewidth=2.0)
    ax.add_patch(fast_card)
    ax.text(3.9, 7.8, r"FAST TIMESCALE ($\tau_{\mathrm{fast}} \sim 1$)",
            ha="center", fontsize=11, fontweight="bold", color=DEEP_BLUE)
    ax.text(3.9, 5.8,
            "• Token Projection Matrices ($W_Q, W_K, W_V$)\n"
            "• Ternary Parameters: $\\theta \\in \\{-1, 0, +1\\} \\cdot \\alpha$\n"
            "• Instantaneous Geodesic Highway Attention (Q-GHA)\n"
            "• Continuous Q-TAC Relaxation Loop ($O(1)$ Hardware Time)\n"
            "• Latent Equilibrium State: $h^* \\in \\Delta^N$",
            ha="center", fontsize=9.5, color=TEXT_BODY, linespacing=1.6)
    
    # Slow Timescale Card (Right)
    slow_card = patches.FancyBboxPatch((9.2, 4.0), 5.8, 4.5, boxstyle="round,pad=0.3",
                                      edgecolor=AMBER_GOLD, facecolor=CARD_BG_GOLD, linewidth=2.0)
    ax.add_patch(slow_card)
    ax.text(12.1, 7.8, r"SLOW TIMESCALE ($\tau_{\mathrm{slow}} \sim 1/\eta_g \gg 1$)",
            ha="center", fontsize=11, fontweight="bold", color=AMBER_GOLD)
    ax.text(12.1, 5.8,
            "• Discrete Conductance Tensor: $\mathbf{W}^{(b)} \\in \mathbb{L}_W^{N \\times N}$\n"
            "• Composite Interaction Field: $a_{ij} = \\alpha_a g_{ij} + (1-\\alpha_a) c_{ij}$\n"
            "• Hebbian Co-activation + Task Gradient Sensitivity\n"
            "• Discrete Markov Lattice Jump Law: $\\Delta W_{ij} = \\pm \\Delta w$\n"
            "• Permanent Geodesic Highway Memory: $d_{ij} \\to 0$",
            ha="center", fontsize=9.5, color=TEXT_BODY, linespacing=1.6)
    
    # Top Coupling Arrow: Fast -> Slow (Flux Ingestion)
    ax.annotate("", xy=(9.2, 6.7), xytext=(6.8, 6.7),
                arrowprops=dict(arrowstyle="->", color=ROSE, lw=2.8))
    ax.text(8.0, 7.2, "Signal Flux $a_{ij}$\n& Gradients $g_{ij}$",
            ha="center", va="center", fontsize=9.5, color=ROSE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1f2", edgecolor=ROSE, lw=1.2))
    
    # Bottom Coupling Arrow: Slow -> Fast (Metric Warping)
    ax.annotate("", xy=(6.8, 5.0), xytext=(9.2, 5.0),
                arrowprops=dict(arrowstyle="->", color=EMERALD, lw=2.8))
    ax.text(8.0, 4.4, "Geodesic Resistance\nDistances $d_{ij}^{(b)}$",
            ha="center", va="center", fontsize=9.5, color=EMERALD, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG_GREEN, edgecolor=EMERALD, lw=1.2))
    
    # Bottom Guarantee Banner (Tikhonov Decoupling)
    tikh_card = patches.FancyBboxPatch((2.0, 0.8), 12.0, 2.3, boxstyle="round,pad=0.3",
                                      edgecolor=PURPLE, facecolor=CARD_BG_PURPLE, linewidth=2.0)
    ax.add_patch(tikh_card)
    ax.text(8.0, 2.3, "TIKHONOV SINGULAR PERTURBATION THEOREM (THEOREMS PG-2 & Q-1 `[Z3 VERIFIED]`)",
            ha="center", fontsize=11, fontweight="bold", color=PURPLE)
    ax.text(8.0, 1.5,
            "The fast hidden state reaches physical quasi-equilibrium on the manifold before metric conductance $\mathbf{W}$ updates.\n"
            "This mathematical decoupling eliminates chaotic positive feedback, preventing catastrophic forgetting and gradient livelock.",
            ha="center", fontsize=9.5, color=TEXT_BODY, linespacing=1.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_two_timescale_metric_warping.svg"), format="svg", facecolor=LIGHT_BG)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_two_timescale_metric_warping.png"), dpi=250, facecolor=LIGHT_BG)
    plt.close()
    print("  [✓] Generated Figure 5 (Clean Light Edition)")

# -----------------------------------------------------------------------------
# Figure 6: Quantized GeoFormer Block Circuit Architecture Pipeline
# -----------------------------------------------------------------------------
def plot_figure_6():
    fig, ax = plt.subplots(figsize=(16, 9.5))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    fig.suptitle("FIGURE 6: QUANTIZED GEOFORMER NEURAL CIRCUIT ARCHITECTURE",
                 fontsize=14, fontweight="bold", color=TEXT_DARK, y=0.98)
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Row 1: Main Feed-Forward Pipeline (y = 6.2)
    stages = [
        ("Input State $h$\n$h \\in \\mathbb{Q}_{b_h}^{B \\times N \\times D}$", 1.8, 6.2, SKY_BLUE, CARD_BG_BLUE),
        ("Pre-RMSNorm 1\n$\\gamma_1 \\in \\mathbb{R}^D$", 4.6, 6.2, TEXT_BODY, CARD_BG),
        ("Quantized GHA\n$S_{ij} = \\frac{Q_i K_j^\\top}{\\sqrt{d_k}} - \\lambda_{\\mathrm{geo}} d_{ij}^{(b)}$", 8.0, 6.2, AMBER_GOLD, CARD_BG_GOLD),
        ("Pre-RMSNorm 2\n$\\gamma_2 \\in \\mathbb{R}^D$", 11.4, 6.2, TEXT_BODY, CARD_BG),
        ("Ternary SwiGLU\n$\\theta \\in \\{-1, 0, +1\\} \\cdot \\alpha$", 14.2, 6.2, EMERALD, CARD_BG_GREEN),
    ]
    
    for text, x, y, stroke, fill in stages:
        box = patches.FancyBboxPatch((x - 1.15, y - 0.75), 2.3, 1.5, boxstyle="round,pad=0.2",
                                     edgecolor=stroke, facecolor=fill, linewidth=2.0)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.8, color=TEXT_DARK, fontweight="bold", linespacing=1.4)
        
    # Arrows between Row 1 stages
    ax.annotate("", xy=(3.35, 6.2), xytext=(2.95, 6.2), arrowprops=dict(arrowstyle="->", color=TEXT_DARK, lw=2.0))
    ax.annotate("", xy=(6.75, 6.2), xytext=(5.75, 6.2), arrowprops=dict(arrowstyle="->", color=TEXT_DARK, lw=2.0))
    ax.annotate("", xy=(10.15, 6.2), xytext=(9.15, 6.2), arrowprops=dict(arrowstyle="->", color=TEXT_DARK, lw=2.0))
    ax.annotate("", xy=(13.0, 6.2), xytext=(12.55, 6.2), arrowprops=dict(arrowstyle="->", color=TEXT_DARK, lw=2.0))
    
    # Metric Lattice Box feeding GHA from above (y = 8.8)
    metric_box = patches.FancyBboxPatch((6.6, 8.1), 2.8, 1.4, boxstyle="round,pad=0.2",
                                        edgecolor=AMBER_GOLD, facecolor=CARD_BG_GOLD, linewidth=2.0)
    ax.add_patch(metric_box)
    ax.text(8.0, 8.8, r"Discrete Metric Lattice $\mathbf{W}^{(b)}$" + "\n" + r"$d_{ij} = (W_{ij} + \epsilon_{\mathrm{dist}})^{-1}$",
            ha="center", va="center", fontsize=8.8, color=AMBER_GOLD, fontweight="bold", linespacing=1.3)
    ax.annotate("", xy=(8.0, 7.0), xytext=(8.0, 8.1),
                arrowprops=dict(arrowstyle="->", color=AMBER_GOLD, lw=2.2))
    
    # Row 2: Q-TAC Continuous Relaxation & Output (y = 2.4)
    # Downward connection from SwiGLU to Q-TAC
    ax.annotate("", xy=(14.2, 3.2), xytext=(14.2, 5.4),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=2.5))
    
    tac_box = patches.FancyBboxPatch((6.0, 1.4), 8.8, 1.8, boxstyle="round,pad=0.3",
                                     edgecolor=PURPLE, facecolor=CARD_BG_PURPLE, linewidth=2.2)
    ax.add_patch(tac_box)
    ax.text(10.4, 2.5, "QUANTIZED TAC CONTINUOUS RELAXATION LOOP (O(1) Physical Relaxation)",
            ha="center", fontsize=10, fontweight="bold", color=PURPLE)
    ax.text(10.4, 1.8,
            r"$\nabla_h \mathcal{F}_{\mathrm{latent}} = (h - h_{\mathrm{ffn}}) + \lambda_{\mathrm{Dir}} \mathbf{L}^{(b)} h + \lambda_{S_5} (h - P_{S_5} h)$" + "\n" +
            r"Enforces: Dirichlet Smoothness + $S_5$ Galois Invariance ($K(S) \leq 54.68\ \mathrm{bits}$) + Clamped Quantization",
            ha="center", fontsize=8.5, color=TEXT_BODY, linespacing=1.4)
    
    # Q-TAC to Final Output
    ax.annotate("", xy=(3.0, 2.3), xytext=(6.0, 2.3),
                arrowprops=dict(arrowstyle="->", color=DEEP_BLUE, lw=2.5))
    
    out_box = patches.FancyBboxPatch((0.8, 1.5), 2.2, 1.6, boxstyle="round,pad=0.2",
                                    edgecolor=DEEP_BLUE, facecolor=CARD_BG_BLUE, linewidth=2.0)
    ax.add_patch(out_box)
    ax.text(1.9, 2.3, "Output State $h^*$\n$h^* \\in \\Delta^N$\n(Attractor State)",
            ha="center", va="center", fontsize=9, color=DEEP_BLUE, fontweight="bold", linespacing=1.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_6_q_geoformer_architecture_pipeline.svg"), format="svg", facecolor=LIGHT_BG)
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_6_q_geoformer_architecture_pipeline.png"), dpi=250, facecolor=LIGHT_BG)
    plt.close()
    print("  [✓] Generated Figure 6 (Clean Light Edition)")

if __name__ == "__main__":
    print("=" * 80)
    print("   GENERATING CLEAN LIGHT SCIENTIFIC SCHEMATICS (ACADEMIC PAPER EDITION)")
    print("=" * 80)
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    print("=" * 80)
    print(f"All 6 Figures successfully re-rendered in: {OUTPUT_DIR}")
    print("=" * 80)
