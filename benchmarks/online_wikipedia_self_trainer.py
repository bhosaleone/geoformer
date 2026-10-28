"""
Online Wikipedia Self-Trainer with Endogenous Dirichlet Gating & Rig-Safe Fail-Safe.
====================================================================================
Streams encyclopedic knowledge online from the official Wikipedia REST API,
allowing GeoFormer's own metric manifold to autonomously decide what to admit
via Endogenous Dirichlet Gating, while continuously carving 4-bit Geodesic Highways.

Fail-Safe & Rig-Safe Architecture:
  1. CPU Governor: Limits PyTorch to 3 threads (1 core permanently free).
  2. Memory Sentinel: Tracks RAM via psutil; forces gc.collect() if RSS > 3.5 GB.
  3. Network Shield: Exponential backoff on HTTP 429/timeouts for Wikipedia API.
  4. Graceful Signal Interception: Intercepts SIGINT/SIGTERM to save atomic state.
  5. Auto-Resume: Automatically restores from checkpoints/wiki_trainer_state.json.

Author: Shrikant Bhosale (Atmabhan Pandit), TWIST POOL Labs
"""

import os
import sys
import time
import json
import math
import signal
import gc
import random
import psutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.abspath("src"))
from geoformer.quant.ternary_linear import QuantizedLinear
from geoformer.layers.endogenous_gate import EndogenousDirichletGate
from geoformer.data.wiki_streamer import WikipediaStreamer


# ==============================================================================
# 1. RIG GOVERNOR & HARDWARE CONSTRAINTS
# ==============================================================================

# Limit PyTorch CPU threads so 1 CPU core is always free for OS/IDE
torch.set_num_threads(3)
MAX_RSS_GB = 3.5  # Memory ceiling guard for 15 GB rig


def check_rig_memory():
    """Monitors process RSS; forces garbage collection if approaching memory threshold."""
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / (1024 ** 3)
    if rss_gb > MAX_RSS_GB:
        gc.collect()
    return rss_gb


# ==============================================================================
# 2. TIER 8 AUTONOMOUS MODEL WITH ENDOGENOUS DIRICHLET GATING
# ==============================================================================

class OnlineGeoFormerSynthesizer(nn.Module):
    """
    Tier 8 Online Synthesizer (N=4096 capacity, d=512 for responsive online CPU updates).
    Features:
      • 1.58-bit Ternary Synapses (BitNet b1.58 via QuantizedLinear)
      • 4-bit Plastic Metric Conductance Lattice W (16 discrete levels, delta_w = 1/15)
      • Endogenous Dirichlet Gating Layer (autonomous self-purification)
      • Continuous Q-TAC Relaxation Loop (tau = 4)
    """
    def __init__(self, max_entities=4096, max_relations=32, d_model=512, tau_max=4, b_w=4):
        super().__init__()
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.d_model = d_model
        self.tau_max = tau_max
        self.b_w = b_w
        self.delta_w = 1.0 / ((1 << b_w) - 1)  # 1/15 = 0.0667

        # 1. Fast Ternary Synapses
        self.entity_emb = nn.Embedding(max_entities, d_model)
        self.relation_emb = nn.Embedding(max_relations, d_model)
        self.w_proj = QuantizedLinear(d_model * 2, d_model, mode="ternary")
        self.ln_proj = nn.LayerNorm(d_model)

        # 2. 4-Bit Discrete Plastic Metric Conductance Lattice W [4096, 4096]
        # Initialized with uniform conductive floor to preserve algebraic connectivity lambda_2
        self.register_buffer("W", torch.full((max_entities, max_entities), self.delta_w))

        # 3. Endogenous Dirichlet Gating Layer (The Model IS the Purifier)
        self.endogenous_gate = EndogenousDirichletGate(d_model=d_model, lambda_dir=0.40, lambda_ent=0.15)

    def carve_geodesic_highway(self, src_idx: int, tgt_idx: int):
        """
        Carves/reinforces a low-resistance Geodesic Highway on the 4-bit lattice W:
        W[i, j] <- clip(round((W[i, j] + delta_w)/delta_w)*delta_w, 0, 1)
        """
        with torch.no_grad():
            cur_w = self.W[src_idx, tgt_idx].item()
            new_w = min(1.0, cur_w + self.delta_w)
            quant_w = round(new_w / self.delta_w) * self.delta_w
            self.W[src_idx, tgt_idx] = quant_w
            self.W[tgt_idx, src_idx] = quant_w  # Symmetrized undirected conductance

    def forward_query(self, h_idx: torch.Tensor, r_idx: torch.Tensor) -> torch.Tensor:
        h_e = self.entity_emb(h_idx)
        h_r = self.relation_emb(r_idx)
        x = torch.cat([h_e, h_r], dim=-1)
        return self.ln_proj(self.w_proj(x))

    def evaluate_candidate_association(self, head_idx: int, tail_idx: int, snippet: str):
        """
        Evaluates candidate semantic association through Endogenous Dirichlet Gate.
        """
        h_tensor = torch.tensor([head_idx], dtype=torch.long)
        t_tensor = torch.tensor([tail_idx], dtype=torch.long)
        h_rep = self.entity_emb(h_tensor)
        t_rep = self.entity_emb(t_tensor)
        is_admitted, gate_score, e_dir = self.endogenous_gate(h_rep, t_rep, snippet)
        return is_admitted.item(), gate_score.item(), e_dir

    def forward_qtac_diffusion(self, start_idx: int, steps: int = 2) -> torch.Tensor:
        """
        Executes continuous Q-TAC mass diffusion across the learned metric lattice W.
        h(tau) = W^tau @ e_start
        """
        with torch.no_grad():
            vec = F.one_hot(torch.tensor([start_idx]), self.max_entities).float()
            state = vec
            for _ in range(steps):
                next_state = torch.matmul(state, self.W)
                row_sum = next_state.sum(dim=-1, keepdim=True) + 1e-8
                state = next_state / row_sum
            return state.squeeze(0)


# ==============================================================================
# 3. FAIL-SAFE CHECKPOINT & AUTO-RESTART ENGINE
# ==============================================================================

class SelfTrainerFailSafe:
    """Manages atomic checkpoints, signal interception, and instant resume."""

    CHECKPOINT_DIR = "checkpoints"
    STATE_FILE = "checkpoints/wiki_trainer_state.json"
    MODEL_FILE = "checkpoints/wiki_trainer_model.pt"

    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        self.shutdown_requested = False

    def register_signal_handlers(self, save_callback):
        """Intercepts SIGINT (Ctrl+C) and SIGTERM for graceful shutdown."""
        def handle_signal(sig, frame):
            print("\n\n" + "!" * 80)
            print("[!] SHUTDOWN SIGNAL DETECTED. Executing atomic fail-safe save...")
            print("!" * 80)
            self.shutdown_requested = True
            save_callback()
            print("[✓] State and model successfully saved. Exiting cleanly. Resume anytime!")
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def load_existing_state(self):
        if os.path.exists(self.STATE_FILE) and os.path.exists(self.MODEL_FILE):
            try:
                with open(self.STATE_FILE, "r") as f:
                    state_data = json.load(f)
                model_data = torch.load(self.MODEL_FILE, map_location="cpu")
                return state_data, model_data
            except Exception as e:
                print(f"[!] Warning: Failed to load previous state ({e}). Starting fresh.")
        return None, None

    def save_atomic_state(self, model, optimizer, state_dict):
        # 1. Save state metadata
        temp_state = self.STATE_FILE + ".tmp"
        with open(temp_state, "w") as f:
            json.dump(state_dict, f, indent=2)
        os.replace(temp_state, self.STATE_FILE)

        # 2. Save model checkpoint
        temp_model = self.MODEL_FILE + ".tmp"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": time.time()
        }, temp_model)
        os.replace(temp_model, self.MODEL_FILE)


# ==============================================================================
# 4. AUTONOMOUS ONLINE LEARNING SUPERVISOR
# ==============================================================================

class OnlineSelfTrainer:
    """
    Orchestrates continuous reading from Wikipedia, endogenous filtering,
    plastic metric deformation, and periodic multi-hop validation.
    """
    def __init__(self, target_steps: int = 50):
        self.target_steps = target_steps
        self.streamer = WikipediaStreamer(request_delay=0.15)
        self.failsafe = SelfTrainerFailSafe()

        # Vocabulary Mappings
        self.entity2idx = {}
        self.idx2entity = {}
        self.relation2idx = {
            "relates_to": 0, "is_a": 1, "part_of": 2, "developed": 3,
            "causes": 4, "discovered": 5, "studied_by": 6
        }

        # Model and Optimizer
        self.model = OnlineGeoFormerSynthesizer(max_entities=4096, max_relations=32, d_model=512)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Metrics & History
        self.step = 0
        self.total_admitted = 0
        self.total_rejected = 0
        self.history_triples = []
        self.recent_hops = []

        # Check for resume
        self._check_and_resume()
        self.failsafe.register_signal_handlers(self.save_checkpoint)

    def _get_entity_idx(self, name: str) -> int:
        clean = name.strip().title()
        if clean not in self.entity2idx:
            idx = len(self.entity2idx)
            if idx < self.model.max_entities:
                self.entity2idx[clean] = idx
                self.idx2entity[idx] = clean
                return idx
            else:
                return random.randint(0, self.model.max_entities - 1)
        return self.entity2idx[clean]

    def _check_and_resume(self):
        state_data, model_data = self.failsafe.load_existing_state()
        if state_data and model_data:
            print("=" * 80)
            print("⚡ RESUMING FROM PREVIOUS CHECKPOINT:")
            self.step = state_data.get("step", 0)
            self.total_admitted = state_data.get("total_admitted", 0)
            self.total_rejected = state_data.get("total_rejected", 0)
            self.entity2idx = state_data.get("entity2idx", {})
            self.idx2entity = {int(k): v for k, v in state_data.get("idx2entity", {}).items()}
            self.streamer.visited = set(state_data.get("visited", []))
            self.streamer.frontier = state_data.get("frontier", self.streamer.DEFAULT_SEEDS)
            self.history_triples = state_data.get("history_triples", [])

            self.model.load_state_dict(model_data["model_state_dict"])
            self.optimizer.load_state_dict(model_data["optimizer_state_dict"])
            print(f"    • Restored Step:         {self.step}")
            print(f"    • Active Discovered Ents:{len(self.entity2idx)}")
            print(f"    • Total Admitted Facts:  {self.total_admitted}")
            print(f"    • Total Rejected Noise:  {self.total_rejected}")
            print("=" * 80)

    def save_checkpoint(self):
        state_dict = {
            "step": self.step,
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
            "entity2idx": self.entity2idx,
            "idx2entity": self.idx2entity,
            "visited": list(self.streamer.visited),
            "frontier": self.streamer.frontier[:100],
            "history_triples": self.history_triples[-200:],
            "timestamp": time.time()
        }
        self.failsafe.save_atomic_state(self.model, self.optimizer, state_dict)

    def extract_semantic_candidates(self, title: str, extract: str):
        """Extracts candidate entity co-activations directly from raw sentences."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        candidates = []
        head = title.replace("_", " ").title()

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 20:
                continue
            # Find capitalized entity terms
            terms = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", s_clean)
            for t in terms:
                t_clean = t.strip().title()
                if t_clean != head and len(t_clean) >= 3 and t_clean.lower() not in ("the", "this", "that"):
                    candidates.append((head, "relates_to", t_clean, s_clean))
        return candidates[:6]

    def run_training_loop(self):
        print("\n" + "=" * 80)
        print("🚀 STARTING AUTONOMOUS ONLINE SELF-TRAINING ON WIKIPEDIA")
        print(f"   Model: Tier 8 Universal Synthesizer (N={self.model.max_entities}, d={self.model.d_model})")
        print("   Gating Engine: Endogenous Dirichlet Minimization (Model-Autonomous)")
        print("   Rig Governor: Max 3 CPU Threads | RAM Guard Active | Checkpoints Atomic")
        print("=" * 80)

        while self.step < self.target_steps:
            if self.failsafe.shutdown_requested:
                break

            self.step += 1
            article = self.streamer.next_article()
            if not article:
                print("[!] Frontier exhausted. Refreshing default seeds...")
                self.streamer.frontier.extend(self.streamer.DEFAULT_SEEDS)
                continue

            title = article["title"]
            extract = article["extract"]
            rss = check_rig_memory()

            print(f"\n[Step {self.step:03d}/{self.target_steps:03d}] 📖 Ingesting: '{title}' (RAM: {rss:.2f} GB)")

            candidates = self.extract_semantic_candidates(title, extract)
            admitted_this_step = 0

            for h_str, r_str, t_str, sent in candidates:
                h_idx = self._get_entity_idx(h_str)
                t_idx = self._get_entity_idx(t_str)
                r_idx = self.relation2idx.get(r_str, 0)

                # Endogenous Dirichlet Gate: Model self-decides!
                is_admitted, gate_score, e_dir = self.model.evaluate_candidate_association(h_idx, t_idx, sent)

                if is_admitted:
                    self.total_admitted += 1
                    admitted_this_step += 1
                    # 1. Carve Geodesic Highway in 4-bit lattice W
                    self.model.carve_geodesic_highway(h_idx, t_idx)
                    self.history_triples.append((h_idx, r_idx, t_idx))

                    # 2. Online contrastive update on ternary parameters
                    self.model.train()
                    h_tensor = torch.tensor([h_idx], dtype=torch.long)
                    r_tensor = torch.tensor([r_idx], dtype=torch.long)
                    t_tensor = torch.tensor([t_idx], dtype=torch.long)

                    q_rep = self.model.forward_query(h_tensor, r_tensor)
                    pos_emb = self.model.entity_emb(t_tensor)
                    negs = torch.randint(0, len(self.entity2idx), (1, 7), dtype=torch.long)
                    neg_embs = self.model.entity_emb(negs)

                    cands = torch.cat([pos_emb.unsqueeze(1), neg_embs], dim=1)
                    scores = torch.bmm(cands, q_rep.unsqueeze(-1)).squeeze(-1)
                    loss = F.cross_entropy(scores, torch.zeros(1, dtype=torch.long))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    new_w = self.model.W[h_idx, t_idx].item()
                    print(f"    ✓ ADMITTED (G={gate_score:.2f}, E_dir={e_dir:.3f}): ({h_str}) ──[{r_str}]──► ({t_str}) | Highway W={new_w:.4f}")
                else:
                    self.total_rejected += 1
                    print(f"    ✗ REJECTED (G={gate_score:.2f}, E_dir={e_dir:.3f}): ({h_str}) ──[{r_str}]──► ({t_str}) [Metric Dissonance]")

            # Micro-sleep to keep CPU cool and respect rate limits
            time.sleep(0.08)

            # Periodic Multi-Hop Verification & Checkpointing every 5 steps
            if self.step % 5 == 0:
                self._run_periodic_verification()
                self.save_checkpoint()
                print(f"    [💾] Checkpoint automatically saved (Total Ents: {len(self.entity2idx)}, Admitted: {self.total_admitted}).")

        print("\n" + "=" * 80)
        print("🏁 AUTONOMOUS SELF-TRAINING RUN COMPLETE!")
        print(f"   • Total Articles Read:   {self.step}")
        print(f"   • Discovered Vocabulary: {len(self.entity2idx)} Entities")
        print(f"   • Certified Admissions:  {self.total_admitted}")
        print(f"   • Filtered Dissonance:   {self.total_rejected}")
        print(f"   • Acceptance Ratio:      {(self.total_admitted / max(1, self.total_admitted + self.total_rejected))*100:.1f}%")
        print("=" * 80)
        self.save_checkpoint()

    def _run_periodic_verification(self):
        """Tests zero-shot 2-hop transitive deduction across learned Geodesic Highways."""
        if len(self.history_triples) < 4:
            return

        print("\n    " + "-" * 70)
        print("    [🔍] REAL-TIME ZERO-SHOT MULTI-HOP GEODESIC DEDUCTION TEST:")
        # Find a 2-hop candidate chain: A -> B and B -> C
        pairs = {}
        for h, r, t in self.history_triples[-40:]:
            pairs[h] = t

        chains = []
        for a, b in pairs.items():
            if b in pairs and pairs[b] != a:
                chains.append((a, b, pairs[b]))

        if not chains:
            print("    • No 2-hop chains formed yet. Continuing knowledge ingestion...")
            print("    " + "-" * 70)
            return

        sample_chain = chains[-1]
        a_idx, b_idx, c_idx = sample_chain
        a_name = self.idx2entity.get(a_idx, f"ID_{a_idx}")
        b_name = self.idx2entity.get(b_idx, f"ID_{b_idx}")
        c_name = self.idx2entity.get(c_idx, f"ID_{c_idx}")

        # Test Q-TAC continuous relaxation over 2 steps
        pred_dist = self.model.forward_qtac_diffusion(a_idx, steps=2)
        top5_ranked = torch.argsort(pred_dist, descending=True)[:5].tolist()

        is_hit = (c_idx in top5_ranked)
        top1_name = self.idx2entity.get(top5_ranked[0], f"ID_{top5_ranked[0]}")

        print(f"    • Chain Ground Truth: ({a_name}) ──► ({b_name}) ──► ({c_name})")
        print(f"    • Continuous Q-TAC Top-1 Deduction: ({top1_name})")
        print(f"    • Multi-Hop Target Hit in Top-5:    {'✅ SUCCESS' if is_hit else '❌ PENDING CONSOLIDATION'}")
        print("    " + "-" * 70)


if __name__ == "__main__":
    trainer = OnlineSelfTrainer(target_steps=15)
    trainer.run_training_loop()
