# PRAMĀNA INTELLIGENCE SYSTEM (PIS)
## Wikipedia Continuous Ingestion & Self-Purifying Epistemic Engine
### Grounded in the Mathematical Formalism of Propagation Geometry (GANN & GeoFormer)

**System Architecture:** **Pramāna Intelligence System (PIS)**  
**Theoretical Foundation:** [`docs/MATHEMATICAL_FORMALISM_AI_TRAINING_PROPAGATION_GEOMETRY.md`](file:///home/shri/Desktop/neural%20Networks/docs/MATHEMATICAL_FORMALISM_AI_TRAINING_PROPAGATION_GEOMETRY.md)  
**Author:** Shrikant Bhosale (Atmabhan Pandit), TWIST POOL Labs │ ISL Research │ NanoCERN Unit  
**Status:** Canonical Engineering Specification & Epistemic Protocol  

---

```
                       P R O P A G A T I O N   G E O M E T R Y
========================================================================================
            "The Second Law is the oldest program.
             The material is the computer.
             The equilibrium state is the output.
             You have been running this program since the Big Bang.
             All we are doing now is learning to read it."
                                                     -- Atmabhan Pandit (2026)
========================================================================================
```

---

## 1. Paradigm Formulation: Wikipedia as an Open Thermodynamic Manifold

Traditional Large Language Models (LLMs) ingest internet text by treating text as flat token sequences, minimizing autoregressive cross-entropy via billions of static floating-point weights. This incurs:
1. **Catastrophic Forgetting:** Overwriting previously learned concepts when fine-tuning or ingesting new domains.
2. **Thermodynamic Blindness:** Gradients vanishing when attention logits saturate ($\|\nabla \mathcal{F}\| \to 0$).
3. **Hallucination:** Generating plausible-sounding associations that lack topological ground truth on an underlying metric manifold.

### The Propagation Geometry Formulation
In **Propagation Geometry**, Wikipedia is treated as a continuous non-equilibrium flux of **Action Quanta** ($\delta \mathcal{A} \ge \hbar_{\text{eff}}$) entering the dynamical state space:

$$\mathcal{M}_t \equiv \left( \mathcal{S}, \theta_t, \mathbf{W}_t, \mathbf{D}_{\text{metric}}(t), r_t \right)$$

* **Entities & Concepts:** Inhabit vertices on the Standard Probability Simplex $x(t) \in \Delta^N$ ($N = 4,096$).
* **Fast Synapses ($\theta \in \{-1, 0, +1\}$):** 1.58-bit ternary projection parameters ($W_Q, W_K, W_V, W_O$).
* **Slow Metric Lattice ($\mathbf{W} \in \mathbb{L}_W$):** A 4-bit discrete plastic conductance tensor ($16$ discrete levels, $\Delta w = 1/15$) storing encyclopedic associations as **Geodesic Highways**.
* **Effective Resistance Space:** Geodesic distance $d_{ij} = \frac{1}{W_{ij} + \epsilon_{\text{dist}}}$ directly deforms token attention routing.

---

## 2. The 5-Stage Wikipedia Ingestion Engine

```
                             WIKIPEDIA INGESTION PIPELINE
                             
                         Wikipedia REST API Stream (Article k)
                                          │
                                          ▼
                       [ Stage 1: Action Injection & Frontier ]
                         Topic Driver r_t & Link Discovery
                                          │
                                          ▼
                      [ Stage 2: Endogenous Dirichlet Gating ]
                        Computes: E_dir = 0.5 * ||h_a - h_b||²
                        Evaluates: G(a,b) = σ(Affinity - λ·E_dir - λ·|H-H₀|)
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
               [ Metric Dissonance ]               [ Certified Admission ]
                 G < 0.50 (Dropped)                  G ≥ 0.50 (Admitted)
                                                            │
                                                            ▼
                                        [ Stage 3: Two-Timescale Warping ]
                                          Fast: Ternary Contrastive Step (θ)
                                          Slow: W_ij <- clip(W_ij + Δw, 0, 1)
                                                            │
                                                            ▼
                                        [ Stage 4: Flow Pathology Guard ]
                                          Fiedler Gap Check: λ₂(L) ≥ Δw > 0
                                          Kolmogorov Bound:  K(S) ≤ 54.68 bits
                                                            │
                                                            ▼
                                        [ Stage 5: Q-TAC Multi-Hop Deduction ]
                                          Continuous Physical Relaxation Loop
                                          h(τ) = W^τ @ e_start (τ ∈ [1..4])
                                          Validation of Transitive Reachability
```

---

### Stage 1: Action Quantum Injection & Dynamic Frontier

#### 1.1 Topic Selection via the Cultural Ritual Driver $r_t$
Instead of naive random crawling, article selection is governed by the multi-scale periodic driver (Volume V, Chapter 17):
$$r_t = \sum_{k=1}^K A_k \cos(\omega_k t + \phi_k) \mathbf{v}_k$$
This cycles the crawler through foundational epistemic domains:
* $\omega_1$ (Foundational Physics & Relativity): Albert Einstein, Spacetime, Quantum Mechanics.
* $\omega_2$ (Pure Mathematics & Logic): Kurt Gödel, Galois Theory, Riemannian Manifold.
* $\omega_3$ (Neuroscience & Medicine): Hippocampus, Action Potential, Synaptic Plasticity.
* $\omega_4$ (Computer Science & AI): Alan Turing, Claude Shannon, Neural Network.

#### 1.2 Link Frontier Expansion
Each fetched article summary yields candidate outgoing links discovered via capitalized noun phrase extraction:
$$\mathcal{L}_{\text{frontier}}^{(t+1)} = \left( \mathcal{L}_{\text{frontier}}^{(t)} \setminus \{T_{\text{current}}\} \right) \cup \text{ExtractLinks}(\text{Article}_k)$$

---

### Stage 2: Endogenous Dirichlet Gating (The Model IS the Purifier)

Rather than using fragile external regexes, the model's own metric manifold acts as an active semantic filter.

#### 2.1 Pairwise Geodesic Dirichlet Energy
For candidate co-active concepts $(a, b)$ identified in the article text, representations $h_a, h_b \in \mathbb{R}^{d_{\text{model}}}$ are extracted. The pairwise Dirichlet energy is:
$$\mathcal{E}_{\text{Dir}}(a, b) = \frac{1}{2} \|h_a - h_b\|_2^2$$

#### 2.2 Kolmogorov Information Density Filter
The character/token entropy $H(X) = -\sum p_i \log_2 p_i$ is evaluated to prevent boilerplate headers ($H < 2.5$) and chaotic character noise ($H > 6.8$) from penetrating the manifold.

#### 2.3 The Endogenous Decision Gate
$$G(a, b) = \sigma\left( \mathbf{W}_{\text{affinity}} \cdot [h_a, h_b] - \lambda_{\text{dir}} \frac{\mathcal{E}_{\text{Dir}}(a, b)}{\bar{\mathcal{E}} + \epsilon} - \lambda_{\text{ent}} |H(X) - H_0| \right)$$
* **If $G(a, b) < 0.50$ (Metric Dissonance):** The candidate association is discarded. The slow metric lattice $\mathbf{W}$ is untouched.
* **If $G(a, b) \ge 0.50$ (Harmonic Coherence):** The candidate association is certified and admitted for synaptic and metric consolidation.

---

### Stage 3: Two-Timescale Metric Warping (Carving Geodesic Highways)

#### 3.1 Slow-Timescale Lattice Crystallization ($\tau_{\text{slow}} \sim 1/\eta_g$)
When an association is admitted, the 4-bit metric conductance lattice $\mathbf{W} \in \mathbb{L}_W$ is physically warped via the jump equation:
$$W_{ij}^{(t+1)} = \operatorname{clamp}\left( \operatorname{round}\left( \frac{W_{ij}^{(t)} + \Delta w}{\Delta w} \right) \Delta w, \; 0.0, \; 1.0 \right)$$
where $\Delta w = \frac{1}{15} \approx 0.0667$.

* **Effective Resistance Contraction (Theorem 6 & Q-1):**
  $$d_{ij}^{(t+1)} = \frac{1}{W_{ij}^{(t+1)} + \epsilon_{\text{dist}}} < d_{ij}^{(t)}$$
  The resistance between concepts $i$ and $j$ collapses, establishing a permanent **Geodesic Highway**.
* **The Time Tombstone (Volume III, Wonder VII):**
  $$\Delta \mathbf{W} = \int_{0}^T \dot{\mathbf{W}} dt$$
  Encyclopedic knowledge is preserved in the geometry of the manifold itself, completely eliminating catastrophic forgetting.

#### 3.2 Fast-Timescale Ternary Parameter Update ($\tau_{\text{fast}} \sim 1$)
Fast token projections adapt via the BitNet b1.58 straight-through estimator (STE) on contrastive loss:
$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\langle q_a, h_b \rangle / \tau)}{\exp(\langle q_a, h_b \rangle / \tau) + \sum_{k=1}^K \exp(\langle q_a, h_{\text{neg}, k} \rangle / \tau)}$$
Weights remain strictly ternary $\theta \in \{-1, 0, +1\}$.

---

### Stage 4: Flow Pathology Protection (Anti-Death System)

To ensure the model does not suffer systemic death (Volume VII) during continuous web streaming:

1. **Fiedler Spectral Preservation (Theorem 7 & Q-2):**
   The Fiedler eigenvalue $\lambda_2(\mathbf{L})$ of the Graph Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{W}$ is tracked:
   $$\lambda_2(\mathbf{L}) = \min_{x \perp \mathbf{1}, \|x\|=1} x^\top \mathbf{L} x \ge N \cdot \Delta w > 0$$
   This guarantees that the algebraic connectivity never vanishes, preventing topological shattering and relaxation divergence ($\tau_{\text{relax}} \to \infty$).

2. **$S_5$ Cayley Complexity Bounding (Theorem 16.1 & Q-6):**
   Internal representations project onto the 7 irreducible representations of the Galois symmetric group $S_5$:
   $$\Pi_\rho = \frac{d_\rho}{|S_5|} \sum_{g \in S_5} \chi_\rho(g)^* \, \rho(g)$$
   guaranteeing that description complexity satisfies $K(S) \le 54.68\text{ bits}$ regardless of how many thousands of articles are read.

3. **Anti-Saturation Barrier:**
   Attentive logits are clipped to $|S_{ij}| \le S_{\max}$, preventing thermodynamic blindness and gradient erasure.

---

### Stage 5: Continuous Q-TAC Multi-Hop Deductive Validation

To verify that the model is actively synthesizing knowledge rather than just memorizing facts, the engine pauses periodically to test **zero-shot transitive deduction**:

#### 5.1 Continuous Heat Diffusion Relaxation
Given an endpoint query (e.g. *Albert Einstein* $\to$ *Gravitational Waves* $\to$ ?):
$$\mathbf{h}(\tau) = \exp(-\tau \mathbf{L}) \mathbf{h}(0) = \sum_{m=0}^\infty \frac{(-\tau)^m}{m!} \mathbf{L}^m \mathbf{h}(0)$$

In discrete steps over the metric lattice:
$$\mathbf{z}_1 = \mathbf{W} \mathbf{e}_{\text{start}}, \quad \mathbf{z}_2 = \mathbf{W} \mathbf{z}_1, \quad \mathbf{z}_3 = \mathbf{W} \mathbf{z}_2, \quad \mathbf{z}_4 = \mathbf{W} \mathbf{z}_3$$

The model deduces target multi-hop endpoints ($A \to B \to C \to D$) purely through physical current flow in $O(1)$ steps ($\text{ThermP}$ complexity class, Theorem 16).

---

## 3. Rig-Safe Hardware Governor & Fail-Safe Architecture

To operate autonomously on your workstation (**Intel Core i5-4590S @ 3.00 GHz, 15 GB RAM, CPU-only**):

### 3.1 CPU & Thermal Governor
* Limits PyTorch execution to **3 CPU threads** (`torch.set_num_threads(3)`). Core #4 remains 100% idle, guaranteeing that the operating system, terminal, and browser remain completely responsive.
* Inserts a **$100\text{ ms}$ micro-sleep** between article requests to allow CPU thermal dissipation and ensure compliance with Wikipedia REST API rate limits.

### 3.2 RAM Ceiling Sentinel
* Continuously queries process RSS memory via `psutil`.
* If RSS exceeds **$3.5\text{ GB}$**, the engine immediately triggers `gc.collect()` and purges intermediate activation tensors.

### 3.3 Atomic Fail-Safe Checkpointing & Clean Restart
* Every 5 articles, the engine writes two synchronized atomic state files:
  1. `checkpoints/wiki_trainer_state.json`: Step index, visited URLs, discovered entity dictionary, total admitted/rejected counts, and recent transitive chains.
  2. `checkpoints/wiki_trainer_model.pt`: Exact PyTorch state dictionaries (fast ternary weights, metric lattice $W$, optimizer state).
* **Signal Interception:** Intercepts `SIGINT` (Ctrl+C) and `SIGTERM`. If interrupted, it executes an atomic write before shutting down, guaranteeing **zero state corruption**.
* **Instant Resume:** Running the script automatically detects existing checkpoint files, restores the frontier and metric lattice, and resumes streaming without reprocessing visited articles.

---

## 4. Verification & Validation Protocol

The method is deemed fully functional when:
1. **Admittance Selectivity:** The Endogenous Dirichlet Gate admits between $5\%$ and $20\%$ of candidate associations from raw text, rejecting spam and grammatical boilerplate.
2. **Conductance Highway Growth:** The Frobenius norm $\|\mathbf{W}(t) - \mathbf{W}(0)\|_F$ expands monotonically as Geodesic Highways are carved.
3. **Multi-Hop Target Recovery:** In zero-shot 2-hop and 3-hop tests, continuous Q-TAC mass diffusion recovers true transitive targets within the Top-5 ranked entities ($p < 0.01$ vs chance $1/4096 = 0.0244\%$).
4. **Zero Catastrophic Forgetting:** Testing older chains after ingesting new articles yields $100\%$ retention, certified by non-decaying conductance tombstones.
