"""
Empirical Benchmark: Modular Arithmetic & Grokking in Finite Quantized Networks
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Task:
Modular Addition: (a + b) mod p, with p = 17.
Tests whether a defined network of N = 32 neurons with ternary weights
can escape memorization basins via Dual Path Kramers tunneling (Theorem Q-5)
and achieve algorithmic generalization (grokking).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantized_geoformer import (
    QuantizedGeoFormerBlock,
    DualPathKramersOptimizer
)

class ModularArithmeticGeoFormer(nn.Module):
    def __init__(self, p=17, d_model=32, n_neurons=32, b_w=4):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_neurons = n_neurons
        
        # Two operand tokens [a, b] + 1 equal sign token = sequence length 3
        self.embedding = nn.Embedding(p + 1, d_model)
        self.block = QuantizedGeoFormerBlock(
            d_model=d_model,
            n_heads=2,
            n_neurons=n_neurons,
            b_w=b_w,
            mode="ternary",
            lambda_geo=0.2
        )
        self.head = nn.Linear(d_model, p, bias=False)

    def forward(self, x):
        h = self.embedding(x)
        h_star, A, L, tau, K_S = self.block(h)
        # Prediction at position 2 (after the equal sign)
        logits = self.head(h_star[:, -1, :])
        return logits, h_star, A, L, tau, K_S

def create_modular_dataset(p=17, train_split=0.6, seed=42):
    torch.manual_seed(seed)
    pairs = []
    labels = []
    for a in range(p):
        for b in range(p):
            pairs.append([a, b, p]) # p is the '=' symbol token
            labels.append((a + b) % p)
            
    pairs = torch.tensor(pairs, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    n_total = len(pairs)
    perm = torch.randperm(n_total)
    n_train = int(n_total * train_split)
    
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    return pairs[train_idx], labels[train_idx], pairs[test_idx], labels[test_idx]

def run_grokking_benchmark(epochs=120, lr=0.02):
    print("=" * 80)
    print("   BENCHMARK: MODULAR ARITHMETIC GROKKING (p=17, N=32 NEURONS, TERNARY WEIGHTS)")
    print("=" * 80)
    
    p = 17
    X_train, y_train, X_test, y_test = create_modular_dataset(p=p, train_split=0.65)
    print(f"Dataset: {len(X_train)} Train Pairs, {len(X_test)} Test Pairs.")
    
    model = ModularArithmeticGeoFormer(p=p, d_model=32, n_neurons=32, b_w=4)
    optimizer = DualPathKramersOptimizer(model.parameters(), lr=lr, alpha_tau=0.3, d_0=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, h_star, A, L, tau, K_S = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        
        # Dual Path step with thermal pulse
        cal_signal = torch.sin(torch.tensor(epoch * 0.15)).item()
        optimizer.step(calendar_signal=cal_signal)
        
        # Metric warping update
        if A.grad is not None:
            model.block.step_slow_timescale(h_star.detach(), A.grad, eta_g=0.01)
            
        if epoch % 15 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                train_preds = logits.argmax(dim=-1)
                train_acc = (train_preds == y_train).float().mean().item() * 100.0
                
                test_logits, _, _, _, _, test_K_S = model(X_test)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == y_test).float().mean().item() * 100.0
                
                fiedler = model.block.metric_engine.get_fiedler_eigenvalue(32)
                
            print(f"Epoch {epoch:3d}/{epochs:3d} │ Train: {train_acc:>5.1f}% │ Test: {test_acc:>5.1f}% │ Fiedler λ2: {fiedler:.4f} │ K(S): {test_K_S.item():.2f}b (≤54.68b) │ Loss: {loss.item():.4f}")
            
    print("=" * 80)
    print("Modular arithmetic grokking benchmark finished.")

if __name__ == "__main__":
    run_grokking_benchmark()
