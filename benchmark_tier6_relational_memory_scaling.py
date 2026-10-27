"""
Tier 6 Benchmark: Hippocampal Relational Working Memory (RML) Scaling
====================================================================
Evaluates whether a Hippocampal Relational Working Memory Tensor eliminates
branching ambiguity and higher-order path attenuation across deep 2-hop, 3-hop,
and 4-hop zero-shot reasoning chains.

Replications: 5 Seeds [42, 101, 2024, 777, 999] across 128 entities.
Compares:
  1. Fixed Euclidean Transformer Control (Flat Attention)
  2. Standard Adaptive Q-GeoFormer (Scalar Conductance Lattice)
  3. Relational Memory-Augmented Q-GeoFormer (+ Hippocampal Relational Memory Kernel)

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from quantized_geoformer.quantized_ops import QuantizedLinear


# ==============================================================================
# 1. RELATIONAL DEEP KNOWLEDGE GRAPH GENERATOR
# ==============================================================================

def generate_relational_deep_chains(num_entities=128, num_chains=30, num_rel_types=4, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    chains = []
    used_edges = set()
    attempts = 0

    while len(chains) < num_chains and attempts < 40000:
        attempts += 1
        nodes = random.sample(range(num_entities), 5)
        rel_type = random.randint(0, num_rel_types - 1)
        a, b, c, d, e = nodes

        # Edge tuple: ((src, tgt), rel_type)
        edges = [((a, b), rel_type), ((b, c), rel_type), ((c, d), rel_type), ((d, e), rel_type)]
        if all(edge not in used_edges for edge in edges):
            for edge in edges:
                used_edges.add(edge)
            chains.append((nodes, rel_type))

    # Training Set: ONLY 1-hop facts with relation context
    one_hop_facts = []
    for nodes, r in chains:
        a, b, c, d, e = nodes
        one_hop_facts.append(([a, r], b))
        one_hop_facts.append(([b, r], c))
        one_hop_facts.append(([c, r], d))
        one_hop_facts.append(([d, r], e))

    test_2hop = []
    test_3hop = []
    test_4hop = []

    for nodes, r in chains:
        a, b, c, d, e = nodes
        test_2hop.append(([a, r], c))
        test_3hop.append(([a, r], d))
        test_4hop.append(([a, r], e))

    random.shuffle(one_hop_facts)
    X_tr = torch.tensor([item[0] for item in one_hop_facts], dtype=torch.long)
    y_tr = torch.tensor([item[1] for item in one_hop_facts], dtype=torch.long)

    X_te2 = torch.tensor([item[0] for item in test_2hop], dtype=torch.long)
    y_te2 = torch.tensor([item[1] for item in test_2hop], dtype=torch.long)

    X_te3 = torch.tensor([item[0] for item in test_3hop], dtype=torch.long)
    y_te3 = torch.tensor([item[1] for item in test_3hop], dtype=torch.long)

    X_te4 = torch.tensor([item[0] for item in test_4hop], dtype=torch.long)
    y_te4 = torch.tensor([item[1] for item in test_4hop], dtype=torch.long)

    return X_tr, y_tr, (X_te2, y_te2), (X_te3, y_te3), (X_te4, y_te4), chains


# ==============================================================================
# 2. ARCHITECTURES
# ==============================================================================

class FixedEuclideanControlTier6(nn.Module):
    """Control: Flat Euclidean Transformer with Ternary Weights."""
    def __init__(self, num_entities=128, num_relations=4, d_model=512, n_heads=8):
        super().__init__()
        self.num_entities = num_entities
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.entity_emb   = nn.Embedding(num_entities, d_model)
        self.relation_emb = nn.Embedding(num_relations, d_model)

        self.w_q = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_k = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_v = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_o = QuantizedLinear(d_model, d_model, mode="ternary")

        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")

        self.head = nn.Linear(2 * d_model, num_entities, bias=False)

    def forward(self, x, hops=1):
        B = x.shape[0]
        h_e = self.entity_emb(x[:, 0]).unsqueeze(1)
        h_r = self.relation_emb(x[:, 1]).unsqueeze(1)
        h = torch.cat([h_e, h_r], dim=1)
        h = F.layer_norm(h, (self.d_model,))

        Q = self.w_q(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = F.softmax(scores, dim=-1)
        attn_out = self.w_o(torch.matmul(A, V).transpose(1, 2).contiguous().view(B, 2, self.d_model))
        h = F.layer_norm(h + attn_out, (self.d_model,))

        ffn_out = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        h = F.layer_norm(h + ffn_out, (self.d_model,))

        return self.head(h.view(B, -1))


class StandardAdaptiveQGeoFormerTier6(nn.Module):
    """Standard Adaptive Q-GeoFormer with Scalar Metric Powers W^k (Tier 5 Baseline)."""
    def __init__(self, num_entities=128, num_relations=4, d_model=512, n_heads=8):
        super().__init__()
        self.num_entities = num_entities
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.entity_emb   = nn.Embedding(num_entities, d_model)
        self.relation_emb = nn.Embedding(num_relations, d_model)

        self.w_q = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_k = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_v = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_o = QuantizedLinear(d_model, d_model, mode="ternary")

        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")

        self.head = nn.Linear(2 * d_model, num_entities, bias=False)
        self.register_buffer('W', torch.zeros(num_entities, num_entities))
        self.delta_w = 1.0 / 15.0

    def forward(self, x, hops=1):
        B = x.shape[0]
        h_e = self.entity_emb(x[:, 0]).unsqueeze(1)
        h_r = self.relation_emb(x[:, 1]).unsqueeze(1)
        h = torch.cat([h_e, h_r], dim=1)
        h = F.layer_norm(h, (self.d_model,))

        Q = self.w_q(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = F.softmax(scores, dim=-1)
        attn_out = self.w_o(torch.matmul(A, V).transpose(1, 2).contiguous().view(B, 2, self.d_model))
        h = F.layer_norm(h + attn_out, (self.d_model,))

        ffn_out = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        h = F.layer_norm(h + ffn_out, (self.d_model,))

        base_logits = self.head(h.view(B, -1))

        if hops == 1:
            return base_logits

        # Multi-hop via scalar conductance matrix powers W^k
        src_oh = F.one_hot(x[:, 0], self.num_entities).float()
        col_sums = torch.clamp(self.W.sum(dim=0, keepdim=True), min=1.0)
        W_norm = self.W / col_sums

        z = src_oh.T
        for _ in range(hops):
            z = torch.matmul(W_norm, z)
        geo_logits = z.T * 20.0
        return geo_logits

    def step_slow_timescale(self, batch_entities, target_entities):
        with torch.no_grad():
            for i in range(len(target_entities)):
                src = batch_entities[i, 0].item()
                tgt = target_entities[i].item()
                cur_w = self.W[tgt, src].item()
                self.W[tgt, src] = min(1.0, cur_w + self.delta_w * 3)


class RelationalMemoryQGeoFormerTier6(nn.Module):
    """
    Treatment: Adaptive Q-GeoFormer with Hippocampal Relational Working Memory (RML).
    Maintains a Relational Conductance Tensor W[r, tgt, src] that resolves branching crossroads
    by contextualizing the transition kernel with relation identity r.
    """
    def __init__(self, num_entities=128, num_relations=4, d_model=512, n_heads=8):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.entity_emb   = nn.Embedding(num_entities, d_model)
        self.relation_emb = nn.Embedding(num_relations, d_model)

        self.w_q = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_k = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_v = QuantizedLinear(d_model, d_model, mode="ternary")
        self.w_o = QuantizedLinear(d_model, d_model, mode="ternary")

        self.w_gate = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_up   = QuantizedLinear(d_model, d_model * 2, mode="ternary")
        self.w_down = QuantizedLinear(d_model * 2, d_model, mode="ternary")

        self.head = nn.Linear(2 * d_model, num_entities, bias=False)

        # Relational Working Memory Tensor: [R, N, N]
        self.register_buffer('W_rel', torch.zeros(num_relations, num_entities, num_entities))
        self.delta_w = 1.0 / 15.0

    def forward(self, x, hops=1):
        B = x.shape[0]
        h_e = self.entity_emb(x[:, 0]).unsqueeze(1)
        h_r = self.relation_emb(x[:, 1]).unsqueeze(1)
        h = torch.cat([h_e, h_r], dim=1)
        h = F.layer_norm(h, (self.d_model,))

        Q = self.w_q(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(h).view(B, 2, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = F.softmax(scores, dim=-1)
        attn_out = self.w_o(torch.matmul(A, V).transpose(1, 2).contiguous().view(B, 2, self.d_model))
        h = F.layer_norm(h + attn_out, (self.d_model,))

        ffn_out = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        h = F.layer_norm(h + ffn_out, (self.d_model,))

        base_logits = self.head(h.view(B, -1))

        if hops == 1:
            return base_logits

        # Relational Working Memory Path Tracing
        geo_logits = torch.zeros_like(base_logits)
        for i in range(B):
            r = x[i, 1].item()
            src = x[i, 0].item()
            W_r = self.W_rel[r]

            # Normalize for probability flow along relational manifold
            col_sums = torch.clamp(W_r.sum(dim=0, keepdim=True), min=1.0)
            W_r_norm = W_r / col_sums

            e_src = F.one_hot(torch.tensor(src), self.num_entities).float()
            z = e_src
            for _ in range(hops):
                z = torch.matmul(W_r_norm, z)
            geo_logits[i] = z * 20.0

        return geo_logits

    def step_slow_timescale(self, batch_entities, target_entities):
        with torch.no_grad():
            for i in range(len(target_entities)):
                src = batch_entities[i, 0].item()
                r   = batch_entities[i, 1].item()
                tgt = target_entities[i].item()
                cur_w = self.W_rel[r, tgt, src].item()
                self.W_rel[r, tgt, src] = min(1.0, cur_w + self.delta_w * 3)


# ==============================================================================
# 3. TRAINING & EVALUATION HARNESS
# ==============================================================================

def train_and_eval(model, X_tr, y_tr, test_splits, epochs=250, lr=0.008):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr, hops=1)
        loss = criterion(logits, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if hasattr(model, "step_slow_timescale"):
            model.step_slow_timescale(X_tr, y_tr)

    model.eval()
    with torch.no_grad():
        results = {}
        for depth_name, ((X_te, y_te), h_val) in test_splits.items():
            out = model(X_te, hops=h_val)
            acc = (out.argmax(dim=-1) == y_te).float().mean().item() * 100.0
            results[depth_name] = acc
    return results


# ==============================================================================
# 4. STATISTICAL SIMPSON STUDENT-T TEST
# ==============================================================================

def student_t_pvalue(diffs):
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    s_d = float(np.std(diffs, ddof=1))
    if s_d < 1e-12:
        return (float('inf') if mean_d > 0 else 0.0), 0.0
    t_stat = mean_d / (s_d / math.sqrt(n))
    df = n - 1

    def t_pdf(x):
        coeff = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        return coeff * (1 + (x ** 2) / df) ** (-(df + 1) / 2)

    limit = 25.0
    t_abs = abs(t_stat)
    if t_abs >= limit:
        return t_stat, 1e-7

    steps = 1000
    h = (limit - t_abs) / steps
    total = t_pdf(t_abs) + t_pdf(limit)
    for i in range(1, steps):
        x = t_abs + i * h
        total += 4 * t_pdf(x) if (i % 2 == 1) else 2 * t_pdf(x)
    one_tailed = (h / 3) * total
    return t_stat, float(min(1.0, 2.0 * one_tailed))


# ==============================================================================
# 5. BENCHMARK EXECUTION ACROSS 5 RANDOM SEEDS
# ==============================================================================

def run_tier6_benchmark():
    seeds = [42, 101, 2024, 777, 999]
    num_entities = 128
    d_model = 512

    ctrl_2hop, ctrl_3hop, ctrl_4hop = [], [], []
    geo_2hop, geo_3hop, geo_4hop = [], [], []
    mem_2hop, mem_3hop, mem_4hop = [], [], []

    print("=" * 80)
    print("TIER 6 BENCHMARK: HIPPOCAMPAL RELATIONAL WORKING MEMORY IN DEEP REASONING")
    print(f"Scale: {num_entities} Entities, d={d_model}, Ternary Weights {{-1, 0, +1}}")
    print(f"Replications: 5 Seeds {seeds} across 2-Hop, 3-Hop, and 4-Hop Zero-Shot Tasks")
    print("=" * 80)

    for seed in seeds:
        print(f"\n---> Running Replication Seed {seed}...")
        X_tr, y_tr, te2, te3, te4, _ = generate_relational_deep_chains(num_entities=num_entities, num_chains=30, num_rel_types=4, seed=seed)
        test_splits = {"2-Hop": (te2, 2), "3-Hop": (te3, 3), "4-Hop": (te4, 4)}

        # 1. Fixed Control (No metric, no memory)
        torch.manual_seed(seed)
        m_ctrl = FixedEuclideanControlTier6(num_entities=num_entities, d_model=d_model)
        r_ctrl = train_and_eval(m_ctrl, X_tr, y_tr, test_splits)
        ctrl_2hop.append(r_ctrl["2-Hop"])
        ctrl_3hop.append(r_ctrl["3-Hop"])
        ctrl_4hop.append(r_ctrl["4-Hop"])

        # 2. Standard Adaptive Q-GeoFormer (Scalar Conductance, no relational memory)
        torch.manual_seed(seed)
        m_geo = StandardAdaptiveQGeoFormerTier6(num_entities=num_entities, d_model=d_model)
        r_geo = train_and_eval(m_geo, X_tr, y_tr, test_splits)
        geo_2hop.append(r_geo["2-Hop"])
        geo_3hop.append(r_geo["3-Hop"])
        geo_4hop.append(r_geo["4-Hop"])

        # 3. Relational Memory-Augmented Q-GeoFormer (+ Hippocampal Relational Memory)
        torch.manual_seed(seed)
        m_mem = RelationalMemoryQGeoFormerTier6(num_entities=num_entities, d_model=d_model)
        r_mem = train_and_eval(m_mem, X_tr, y_tr, test_splits)
        mem_2hop.append(r_mem["2-Hop"])
        mem_3hop.append(r_mem["3-Hop"])
        mem_4hop.append(r_mem["4-Hop"])

        print(f"  [Seed {seed}] Control:    2-Hop={r_ctrl['2-Hop']:4.1f}%, 3-Hop={r_ctrl['3-Hop']:4.1f}%, 4-Hop={r_ctrl['4-Hop']:4.1f}%")
        print(f"  [Seed {seed}] GeoFormer:  2-Hop={r_geo['2-Hop']:4.1f}%, 3-Hop={r_geo['3-Hop']:4.1f}%, 4-Hop={r_geo['4-Hop']:4.1f}%")
        print(f"  [Seed {seed}] Memory-Geo: 2-Hop={r_mem['2-Hop']:4.1f}%, 3-Hop={r_mem['3-Hop']:4.1f}%, 4-Hop={r_mem['4-Hop']:4.1f}%")

    def stats(arr):
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    c2_m, c2_s = stats(ctrl_2hop); c3_m, c3_s = stats(ctrl_3hop); c4_m, c4_s = stats(ctrl_4hop)
    g2_m, g2_s = stats(geo_2hop);  g3_m, g3_s = stats(geo_3hop);  g4_m, g4_s = stats(geo_4hop)
    m2_m, m2_s = stats(mem_2hop);  m3_m, m3_s = stats(mem_3hop);  m4_m, m4_s = stats(mem_4hop)

    # Significance Tests: Memory vs Standard GeoFormer
    t2, p2 = student_t_pvalue(np.array(mem_2hop) - np.array(geo_2hop))
    t3, p3 = student_t_pvalue(np.array(mem_3hop) - np.array(geo_3hop))
    t4, p4 = student_t_pvalue(np.array(mem_4hop) - np.array(geo_4hop))

    print("\n" + "=" * 96)
    print("FINAL QUANTITATIVE RESULTS: MULTI-HOP GENERALIZATION BY REASONING DEPTH")
    print("=" * 96)
    print(f"{'Reasoning Task':<16} | {'Flat Control':<16} | {'Standard GeoFormer':<20} | {'Memory-Augmented':<20} | {'Memory Gain':<12} | {'Student-t & p-value'}")
    print("-" * 96)
    print(f"{'2-Hop (A->C)':<16} | {c2_m:5.1f}% ± {c2_s:4.1f}% | {g2_m:5.1f}% ± {g2_s:4.1f}%      | {m2_m:5.1f}% ± {m2_s:4.1f}%      | +{m2_m - g2_m:4.1f}%      | t={t2:5.2f}, p={p2:.2e}")
    print(f"{'3-Hop (A->D)':<16} | {c3_m:5.1f}% ± {c3_s:4.1f}% | {g3_m:5.1f}% ± {g3_s:4.1f}%      | {m3_m:5.1f}% ± {m3_s:4.1f}%      | +{m3_m - g3_m:4.1f}%      | t={t3:5.2f}, p={p3:.2e}")
    print(f"{'4-Hop (A->E)':<16} | {c4_m:5.1f}% ± {c4_s:4.1f}% | {g4_m:5.1f}% ± {g4_s:4.1f}%      | {m4_m:5.1f}% ± {m4_s:4.1f}%      | +{m4_m - g4_m:4.1f}%      | t={t4:5.2f}, p={p4:.2e}")
    print("=" * 96)

    # Render Publication Figure 12
    render_figure_12(
        (c2_m, c2_s, c3_m, c3_s, c4_m, c4_s),
        (g2_m, g2_s, g3_m, g3_s, g4_m, g4_s),
        (m2_m, m2_s, m3_m, m3_s, m4_m, m4_s)
    )


def render_figure_12(ctrl_stats, geo_stats, mem_stats):
    c2_m, c2_s, c3_m, c3_s, c4_m, c4_s = ctrl_stats
    g2_m, g2_s, g3_m, g3_s, g4_m, g4_s = geo_stats
    m2_m, m2_s, m3_m, m3_s, m4_m, m4_s = mem_stats

    hops = [2, 3, 4]
    c_means = [c2_m, c3_m, c4_m]
    c_errs  = [c2_s, c3_s, c4_s]
    g_means = [g2_m, g3_m, g4_m]
    g_errs  = [g2_s, g3_s, g4_s]
    m_means = [m2_m, m3_m, m4_m]
    m_errs  = [m2_s, m3_s, m4_s]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    x = np.arange(len(hops))
    width = 0.26

    rects1 = ax.bar(x - width, c_means, width, yerr=c_errs, capsize=5, label='Fixed Euclidean Control (Flat Attention)', color='#94a3b8', edgecolor='#64748b', linewidth=1.2)
    rects2 = ax.bar(x, g_means, width, yerr=g_errs, capsize=5, label='Standard Adaptive Q-GeoFormer (Scalar Conductance)', color='#3b82f6', edgecolor='#1d4ed8', linewidth=1.2)
    rects3 = ax.bar(x + width, m_means, width, yerr=m_errs, capsize=5, label='Memory-Augmented Q-GeoFormer (+ Hippocampal Relational Memory)', color='#10b981', edgecolor='#047857', linewidth=1.2)

    ax.set_ylabel('Zero-Shot Generalization Accuracy (%)', fontsize=12, fontweight='bold', color='#0f172a')
    ax.set_xlabel('Reasoning Depth (Transitive Path Length k)', fontsize=12, fontweight='bold', color='#0f172a')
    ax.set_title('Figure 12: Eliminating Higher-Order Path Attenuation via Hippocampal Relational Working Memory', fontsize=12.5, fontweight='bold', color='#0f172a', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['2-Hop ($A \\to C$)', '3-Hop ($A \\to D$)', '4-Hop ($A \\to E$)'], fontsize=11, fontweight='bold', color='#0f172a')
    ax.set_ylim(0, 100)

    # Reference chance line
    ax.axhline(0.78, color='#ef4444', linestyle='--', linewidth=1.5, label='Random Chance Floor (1/128 = 0.78%)')

    ax.grid(axis='y', linestyle=':', alpha=0.6, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#0f172a')
    ax.spines['bottom'].set_color('#0f172a')
    ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=9.5, loc='upper right')

    for bar in rects2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1d4ed8')

    for bar in rects3:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#047857')

    plt.tight_layout()
    plt.savefig('diagrams/figure_12_relational_memory_transitive_scaling.png', dpi=300)
    plt.savefig('diagrams/figure_12_relational_memory_transitive_scaling.svg', format='svg')
    print("\n[SUCCESS] Rendered diagrams/figure_12_relational_memory_transitive_scaling.png and .svg")


if __name__ == "__main__":
    run_tier6_benchmark()
