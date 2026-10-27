"""
Empirical Benchmark: Emergence of Algorithmic Induction in N <= 16 Neurons
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Task:
In-Context Prefix Induction (Copying):
Given a sequence [x_1, x_2, ..., x_k, x_1], the model must predict [x_2].
Tests whether a defined network of N = 16 neurons with ternary weights and
discrete metric conductances can form an induction circuit and generalize.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantized_geoformer import (
    QuantizedGeoFormerBlock,
    DualPathKramersOptimizer
)

class MicroInductionModel(nn.Module):
    """Minimal model with N=16 defined neurons and single Quantized GeoFormer block."""
    def __init__(self, vocab_size=8, d_model=32, n_neurons=16, b_w=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_neurons = n_neurons
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.block = QuantizedGeoFormerBlock(
            d_model=d_model,
            n_heads=2,
            n_neurons=n_neurons,
            b_w=b_w,
            mode="ternary",
            lambda_geo=0.25
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embedding(x) # [B, N, D]
        h_star, A, L, tau, K_S = self.block(h)
        logits = self.head(h_star) # [B, N, vocab_size]
        return logits, A, L, tau, K_S

def generate_induction_batch(batch_size=32, seq_len=16, vocab_size=8):
    """
    Generates sequences [x_0, x_1, ..., x_{seq_len-2}, x_target_prev]
    where the last token is identical to a preceding token x_i (i < seq_len - 2),
    and the target is the token following x_i.
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = x[:, 1:].clone()
    
    # Force induction cue: pick a random position i in [0, seq_len-4]
    for b in range(batch_size):
        cue_pos = torch.randint(0, seq_len - 4, (1,)).item()
        # Make the second-to-last token identical to x[b, cue_pos]
        x[b, seq_len - 2] = x[b, cue_pos]
        # The expected next token is x[b, cue_pos + 1]
        targets[b, seq_len - 3] = x[b, cue_pos + 1]
        
    return x, targets

def run_induction_experiment(epochs=80, lr=0.015):
    print("=" * 80)
    print("   BENCHMARK: EMERGENCE OF IN-CONTEXT INDUCTION IN N=16 DEFINED NEURONS")
    print("=" * 80)
    
    model = MicroInductionModel(vocab_size=8, d_model=32, n_neurons=16, b_w=4)
    optimizer = DualPathKramersOptimizer(model.parameters(), lr=lr, alpha_tau=0.25, d_0=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    initial_fiedler = model.block.metric_engine.get_fiedler_eigenvalue(16)
    print(f"Substrate Initial Fiedler Gap lambda_2: {initial_fiedler:.4f}")
    
    for epoch in range(1, epochs + 1):
        x, targets = generate_induction_batch(batch_size=32, seq_len=16, vocab_size=8)
        
        optimizer.zero_grad()
        logits, A, L, tau, K_S = model(x)
        
        # We evaluate prediction at the induction test position
        pred_logits = logits[:, :-1, :].reshape(-1, 8)
        target_labels = targets.reshape(-1)
        
        loss = criterion(pred_logits, target_labels)
        loss.backward()
        
        # Dual Path Kramers step with sinusoidal calendar modulation
        cal_signal = torch.sin(torch.tensor(epoch * 0.1)).item()
        optimizer.step(calendar_signal=cal_signal)
        
        # Slow-timescale metric warping update
        if A.grad is not None:
            model.block.step_slow_timescale(logits.detach(), A.grad, eta_g=0.02)
            
        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                x_val, targets_val = generate_induction_batch(batch_size=64, seq_len=16, vocab_size=8)
                val_logits, _, _, _, val_K_S = model(x_val)
                val_preds = val_logits[:, :-1, :].argmax(dim=-1)
                acc = (val_preds == targets_val).float().mean().item() * 100.0
                fiedler = model.block.metric_engine.get_fiedler_eigenvalue(16)
                
            print(f"Epoch {epoch:2d}/{epochs:2d} │ Loss: {loss.item():.4f} │ Val Acc: {acc:>5.1f}% │ Fiedler λ2: {fiedler:.4f} │ K(S): {val_K_S.item():.2f}b (≤54.68b) │ TAC τ: {tau}")
            
    print("=" * 80)
    print("In-context induction benchmark completed.")

if __name__ == "__main__":
    run_induction_experiment()
