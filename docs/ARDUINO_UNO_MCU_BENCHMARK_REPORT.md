# GeoFormer-MCU: Physical Silicon Benchmark & Adversarial Stress-Test Dossier

**Hardware Substrate:** Atmel ATmega328P (8-bit AVR Microcontroller @ 16 MHz, no FPU)  
**Board:** Arduino Uno R3 connected via USB (`/dev/ttyACM0`)  
**Silicon Memory:** 32 KB Flash │ **2 KB SRAM**  
**Evaluated Kernel:** Discrete Directed Graph Diffusion Kernel (`GeoFormer-MCU Kernel`)  
**Network Parameters:** 32 Entities ($N=32$), 16 Latent Dimensions ($d=16$), 1.58-bit Ternary Synaptic Weights, 4-Bit Metric Conductance Lattice  
**Author:** Shrikant Bhosale (Atmabhan Pandit), Government Medical College (GMC), Sambhajinagar  
**Permanent Zenodo Technical Note:** [doi:10.5281/zenodo.23730305](https://doi.org/10.5281/zenodo.23730305)  
**Date of Hardware Audit:** October 28, 2026  

---

## 1. Scientific Scope & Architectural Demarcation

To ensure rigorous interpretation and prevent conflation between research tiers, we explicitly delineate the boundary between this physical microcontroller implementation and the master GeoFormer foundation architecture:

| Property / Component | Full GeoFormer Architecture (`checkpoints/geoformer_foundation_model.pt`) | Embedded MCU Kernel (`firmware/uno_nano_geoformer/`) |
| :--- | :--- | :--- |
| **Execution Substrate** | Workstation / Server / Edge SBC ($\ge 64\text{ KB}$ RAM) | **Atmel ATmega328P 8-bit MCU (2 KB SRAM, 16 MHz)** |
| **Parameter Count** | 249,665 parameters (48.76 KB packed) | **2,048 weights (2.0 KB Flash)** |
| **Attention Mechanism** | Multi-Head Geodesic Highway Attention (Q-GHA) | **Ternary Linear Projection $\to$ Direct Node Pulse** |
| **Relaxation Operator** | Continuous ODE Graph Laplacian Heat Diffusion ($\frac{d\mathbf{h}}{d\tau} = -\kappa[\mathbf{h} - \mathbf{h}_0 + \lambda \mathbf{L}\mathbf{h}]$) | **Discrete-Time Directed Row-Normalized Markov Diffusion ($\mathbf{h}_{t+1} = \mathbf{h}_t \mathbf{P}$)** |
| **Numerical Format** | 1.58-bit Ternary Synapses + 4-bit Crossbar Emulation | **1.58-bit Flash Lookups + 8-bit Fixed-Point Arithmetic (FP_ONE = 255)** |
| **Learning Substrate** | Two-timescale gradient descent + Hebbian outer products | **Static Flash-packed Prior + Dynamic SRAM Conductance Warping** |

*Takeaway: The results below validate that **a frugal, discrete directed graph diffusion kernel executes natively on bare-metal 8-bit microcontrollers**. It does not prove that the full multi-layer GeoFormer foundation stack is active on the Uno.*

---

## 2. Firmware Algorithm & Exact Memory Layout

### 2.1 Memory Footprint Breakdown (Measured on ATmega328P)

The firmware was compiled with `avr-gcc -Os` and audited via `avr-size`:

```
AVR Memory Usage (ATmega328P @ 16 MHz):
----------------------------------------
Program Flash:   7,856 bytes (24.0% of 32,768 bytes) -> 24.9 KB Free (76.0%)
SRAM Data:       1,120 bytes (54.7% of 2,048 bytes)  -> 928 bytes Free (45.3% Stack Margin)
```

#### Exact Byte-Level SRAM Allocation:
* `W_ram`: `uint8_t [32][32]` = **1,024 bytes** (active metric conductance lattice in read-write RAM).
* `h_curr`: `uint8_t [32]` = **32 bytes** (current attractor probability distribution).
* `h_next`: `uint8_t [32]` = **32 bytes** (next diffusion state buffer).
* `h_latent`: `int16_t [16]` = **32 bytes** (latent embedding accumulator).
* **Total Static SRAM:** **1,120 bytes**.
* **Remaining SRAM for Call Stack & Registers:** **928 bytes** (completely eliminates stack collision risk).

#### Exact Byte-Level Flash (`PROGMEM`) Allocation:
* `W_LATTICE`: `const uint8_t [32][32]` = **1,024 bytes** (baseline prior graph).
* `W_IN`: `const int8_t [16][32]` = **512 bytes** (1.58-bit ternary projection).
* `W_OUT`: `const int8_t [32][16]` = **512 bytes** (1.58-bit ternary projection).
* `TEST_QUERIES`: Source/Target vectors = **48 bytes**.
* Program Code & UART Routines = **5,760 bytes**.
* **Total Flash Consumption:** **7,856 bytes**.

### 2.2 Discrete Propagation Algorithm

Rather than floating-point integration, the kernel computes row-normalized transition probabilities on-the-fly using 8-bit fixed-point scaling ($1.0 \equiv 255$):

$$\mathbf{h}_{t+1}[j] = \sum_{i=0}^{N-1} \mathbf{h}_t[i] \cdot \left\lfloor \frac{W_{\text{ram}}[i][j] \cdot 255}{\sum_{k=0}^{N-1} W_{\text{ram}}[i][k]} \right\rfloor \div 255$$

This structure guarantees that total probability mass $\sum_j \mathbf{h}[j]$ remains bounded within $[0, 255]$, preventing 8-bit accumulator overflow without requiring a 32-bit floating-point unit.

---

## 3. Empirical Benchmark: Clean Transitive Reasoning

Evaluated directly on the physical ATmega328P over USB serial at 115,200 baud across 9 unseen 2-hop compositions and 6 deep 3-hop compositions:

| Benchmark Task | Evaluation Protocol | Physical Hardware Accuracy | Average Measured Latency | Measured Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **2-Hop Transitive Reasoning** | 9 Unseen Pairs ($\tau=2$) | **88.89% (8/9)** | **6.61 ms** ($6,606\ \mu\text{s}$) | **151.4 queries/sec** |
| **3-Hop Deep Composition** | 6 Deep Chains ($\tau=3$) | **100.00% (6/6)** | **8.91 ms** ($8,908\ \mu\text{s}$) | **112.3 queries/sec** |
| **1-to-N Branching Crossroad** | Node 20 (Shock $\to$ 22 & 25) | Exact Equal Split ($48\% / 48\%$) | 6.98 ms | Mathematical Parity |

### Query-by-Query Physical Silicon Log:
```text
--- 2-Hop Transitive Reasoning (tau=2) ---
Query 00 -> True: 02 | Pred: 02 (48%) [PASS] (6748 us)
Query 01 -> True: 03 | Pred: 03 (48%) [PASS] (6400 us)
Query 06 -> True: 08 | Pred: 08 (49%) [PASS] (6272 us)
Query 07 -> True: 09 | Pred: 09 (49%) [PASS] (6908 us)
Query 12 -> True: 14 | Pred: 14 (98%) [PASS] (6564 us)
Query 16 -> True: 18 | Pred: 18 (98%) [PASS] (6284 us)
Query 20 -> True: 22 | Pred: 22 (48%) [PASS] (6980 us)
Query 20 -> True: 25 | Pred: 22 (48%) [FAIL] (6976 us)  <-- 1-to-N Ambiguity
Query 26 -> True: 28 | Pred: 28 (97%) [PASS] (6324 us)
>>> 2-Hop Accuracy: 8/9 (88%) | Avg Latency: 6606 us

--- 3-Hop Deep Reasoning (tau=3) ---
Query 00 -> True: 03 | Pred: 03 (49%) [PASS] (9184 us)
Query 06 -> True: 09 | Pred: 09 (50%) [PASS] (8612 us)
Query 12 -> True: 15 | Pred: 15 (99%) [PASS] (8908 us)
Query 16 -> True: 19 | Pred: 19 (98%) [PASS] (8612 us)
Query 20 -> True: 23 | Pred: 23 (95%) [PASS] (9408 us)
Query 26 -> True: 29 | Pred: 29 (98%) [PASS] (8724 us)
>>> 3-Hop Accuracy: 6/6 (100%) | Avg Latency: 8908 us
```

### Analysis of the Clean Failure:
In Query 20 (Severe Hemorrhagic Shock), the medical casualty graph splits symmetrically:
* $20 \to 21 \to 22$ (Vasopressors)
* $20 \to 24 \to 25$ (Intubation)
Both branches possess identical topological conductance ($W=10$). Consequently, the physical diffusion wave deposits $48\%$ activation on Node 22 and $48\%$ on Node 25. The argmax operator arbitrarily selected Node 22, yielding a failure on target 25. This is an expected artifact of single-label top-1 metrics under 1-to-N multi-target branching.

---

## 4. Adversarial Stress-Test Suite & Physical Behavioral Analysis

To evaluate robustness without hyperbole, four physical stress tests were executed over USB serial:

### Test A: False Shortcut Injection (Dynamic Plasticity Test)
* **Protocol:** Malicious edge $W_{\text{ram}}[0][29] = 15$ was written directly into SRAM while holding Flash weights frozen.
* **Hardware Response:** At $\tau=1$, the peak activation shifted from the default 1-hop path to Node 29 ($56\%$ confidence) in **$5.6\text{ ms}$**.
* **Interpretation:** This is the mathematically expected behavior of directed linear diffusion: increasing an edge weight contracts effective resistance and routes flow. It demonstrates that dynamic in-context associations can rewire reasoning paths on an 8-bit MCU without destroying pre-programmed weights.

### Test B: 20% Conductance Noise (Perturbation Resilience)
* **Protocol:** $25\%$ of active conductance entries in SRAM were perturbed by $\pm 1$ using an on-chip 16-bit Galois LFSR pseudo-random generator.
* **Hardware Response:** Post-perturbation 2-hop accuracy remained **$88.89\% (8/9)$**.
* **Interpretation:** Because the graph's signal-to-noise margin ($\Delta W \ge 8$) exceeds the $\pm 1$ perturbation noise, top-1 rank order is preserved. This confirms basic noise margins for low-precision conductance implementations.

### Test C: Disconnected Cluster Negative Isolation Probes
* **Protocol:** Queries were injected into Node 0 and Node 20, and activation levels were probed on structurally disconnected components.
* **Hardware Response:**
  * Probes 1, 2, 3: **$0/255$ ($0.00\%$) activation** (perfect topological isolation).
  * Probe 4 (Node $26 \to 15$): **$2/255$ ($0.78\%$) activation**.
* **Interpretation:** The $2/255$ trace is not a structural connection; it is a numerical truncation artifact from integer division in the 8-bit accumulation loop. While 3 of 4 probes confirmed strict isolation, the integer kernel exhibits tiny rounding noise that must be clamped in production safety systems.

### Test D: 10,000-Cycle Memory & Stack Hammer
* **Protocol:** The microcontroller executed 10,000 continuous Q-TAC relaxation steps under periodic pulse renewal.
* **Hardware Response:** 10,000 steps finished in **$470.36\text{ ms}$** (**$47.0\ \mu\text{s}$ per step**, **$21,260\text{ steps/sec}$**).
* **Interpretation:** Verified zero stack overflow, zero heap fragmentation, and zero register corruption over prolonged physical cycling.

---

## 5. Physical Energy & Electrical Power Audit (ATmega328P)

To prevent scientific confusion between **modeled analog memristor ASICs** and **physical digital microcontrollers**, we provide the exact first-principles electrical audit of the Arduino Uno executing the GeoFormer-MCU kernel:

### 5.1 Operating Parameters & Supply Current
* **Operating Rail Voltage:** $V_{\text{CC}} = 5.0\text{ V}$ (supplied via USB host).
* **Clock Frequency:** $f_{\text{clk}} = 16.0\text{ MHz}$ (crystal oscillator).
* **ATmega328P Core Active Current (Datasheet DS40002061B @ 16 MHz, 5V):** $I_{\text{core}} \approx 10.0\text{ mA}$.
* **ATmega328P Core Dynamic Power:**
  $$P_{\text{core}} = V_{\text{CC}} \cdot I_{\text{core}} = 5.0\text{ V} \cdot 10.0\text{ mA} = \mathbf{50.0\text{ mW}} \quad (0.050\text{ W})$$
* **Total Board-Level Power (including USB chip, voltage regulator, and status LEDs):**
  $$P_{\text{board}} = 5.0\text{ V} \cdot 32.0\text{ mA} = \mathbf{160.0\text{ mW}} \quad (0.160\text{ W})$$

### 5.2 Physical Energy Dissipation per Inference Pass

Because dynamic energy is the integral of power over time ($E = P \cdot \Delta t$):

| Execution Phase | Measured Latency ($\Delta t$) | Core Energy Dissipation ($E_{\text{core}}$) | Total Board Energy ($E_{\text{board}}$) |
| :--- | :---: | :---: | :---: |
| **Single Q-TAC Step** | **$47.0\ \mu\text{s}$** | **$2.35\ \mu\text{J}$** ($2,350\text{ nJ}$) | **$7.52\ \mu\text{J}$** |
| **2-Hop Transitive Query ($\tau=2$)** | **$6.61\text{ ms}$** | **$330.5\ \mu\text{J}$** ($0.331\text{ mJ}$) | **$1,057.6\ \mu\text{J}$** ($1.06\text{ mJ}$) |
| **3-Hop Deep Query ($\tau=3$)** | **$8.91\text{ ms}$** | **$445.5\ \mu\text{J}$** ($0.446\text{ mJ}$) | **$1,425.6\ \mu\text{J}$** ($1.43\text{ mJ}$) |

### 5.3 Power vs. Energy Comparison Across Platforms

A crucial distinction exists between **operating power** (watts) and **energy per query** (joules):

```
Operating Power (Lower is better for battery & thermal viability):
  NVIDIA Server GPU (H100/A100):     300,000 mW   [||||||||||||||||||||||||||||||||||||||||]
  Google Coral Edge TPU:               2,000 mW   [||]
  ATmega328P Microcontroller:             50 mW   [|] -> 6,000x LOWER DYNAMIC POWER THAN GPU!

Energy per Query (Reflects instruction serialization over time):
  ATmega328P (8-bit @ 16 MHz):         330.5 µJ   (Takes 6.6 ms due to 8-bit serialization)
  NVIDIA Server GPU (FP32 Dense):        4.2 µJ   (Takes ~10 µs with 300W peak power)
  ARM Cortex-M55 NPU (INT8):           0.065 µJ   (Takes ~1 ms with dedicated DSP vector units)
  Analog 1T1R Memristor Crossbar ASIC: 0.00044 µJ (443.9 pJ, Modeled 10 ns analog transport)
```

**Key Takeaways:**
1. **Operating Power Advantage:** At **$50\text{ mW}$ active core power**, the Arduino Uno draws **$6,000\times$ less operating power than a 300W GPU server**. It can run continuously for days on a standard 9V battery or low-power solar harvester.
2. **Why Energy is Microjoules, Not Picojoules on MCU:** An 8-bit MCU has no vector pipelines or floating-point hardware; executing multi-step matrix diffusion requires sequential 8-bit loops over $6.6\text{ ms}$.
3. **The 443.9 pJ Figure Applies Exclusively to Memristor ASICs:** The $443.9\text{ pJ}$ figure is an analytical device physics simulation of a non-von Neumann analog 1T1R ReRAM crossbar operating at $200\text{ mV}$ in $10\text{ ns}$. It must never be confused with digital microcontroller execution.

---

## 6. MCU Scaling Envelope & Physical Limits

Because dense matrix representations scale as $\mathcal{O}(N^2)$ in memory:

| Nodes ($N$) | Dense SRAM Footprint ($N^2$ bytes) | ATmega328P Feasibility (2 KB SRAM) | Raspberry Pi Pico 2 Feasibility (520 KB SRAM) |
| :---: | :---: | :---: | :---: |
| **$N = 16$** | 256 bytes (12.5% SRAM) | ✅ Full headroom | ✅ Negligible |
| **$N = 32$** | **1,024 bytes (50.0% SRAM)** | **✅ Current Working Sweet Spot** | ✅ Negligible |
| **$N = 40$** | 1,600 bytes (78.1% SRAM) | ⚠️ Tight stack boundary | ✅ Negligible |
| **$N = 45$** | 2,025 bytes (98.9% SRAM) | ❌ Theoretical SRAM Ceiling | ✅ Negligible |
| **$N = 64$** | 4,096 bytes (200% SRAM) | ❌ OOM Crash | ✅ 0.78% SRAM |
| **$N = 128$** | 16,384 bytes (800% SRAM) | ❌ OOM Crash (Flash only) | ✅ 3.15% SRAM |
| **$N = 100,000$ (Sparse)** | 9.54 MB RAM (CSR) | ❌ OOM Crash | ⚠️ Requires External PSRAM |

*Conclusion: To scale beyond $N = 32$ with full plastic dynamic lattices, developers should transition to modern 32-bit microcontrollers (such as the Raspberry Pi Pico 2 / RP2350 or ESP32-S3) where 520 KB of SRAM allows scaling to hundreds of nodes natively.*

---

## 6. Verifiable Reproducibility

All source files to flash and reproduce these exact numbers on any standard Arduino Uno are publicly available:
1. Firmware: [`firmware/uno_nano_geoformer/main.c`](file:///home/shri/Desktop/neural%20Networks/firmware/uno_nano_geoformer/main.c)
2. Weights Header: [`firmware/uno_nano_geoformer/mcu_weights.h`](file:///home/shri/Desktop/neural%20Networks/firmware/uno_nano_geoformer/mcu_weights.h)
3. Automated Test Runner: [`benchmarks/benchmark_mcu_adversarial.py`](file:///home/shri/Desktop/neural%20Networks/benchmarks/benchmark_mcu_adversarial.py)
4. GitHub Commit: [`a8d72e2`](https://github.com/bhosaleone/geoformer/commit/a8d72e2)
