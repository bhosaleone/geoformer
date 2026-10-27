"""
Empirical Benchmark: Description Complexity Bound under S_5 Cayley Projector
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Formal Inquiry:
Verifies that adding discrete S_5 permutation group Cayley projectors directly
into the Q-TAC relaxation loop strictly bounds internal description complexity:
    K(S) <= sum_{rho=1}^7 d_rho * log2(d_rho) <= 54.68 bits
under extreme ternary {-1, 0, +1} quantization.
"""

import torch
import torch.nn as nn
from quantized_geoformer import (
    QuantizedGeoFormerBlock,
    DualPathKramersOptimizer,
    S5CayleyProjector
)

def run_s5_complexity_benchmark(epochs=40, batch_size=16, seq_len=16, d_model=30):
    print("=" * 90)
    print("   BENCHMARK: S_5 CAYLEY PROJECTOR DESCRIPTION COMPLEXITY AUDIT UNDER TERNARY QUANTIZATION")
    print("   Theoretical Bound: K(S) <= 54.68 bits (Theorem 16.1 / Q-6)")
    print("=" * 90)
    
    # 1. Model WITH S_5 Cayley Projection in Q-TAC
    model_with_s5 = QuantizedGeoFormerBlock(
        d_model=d_model,
        n_heads=2,
        n_neurons=seq_len,
        b_w=4,
        mode="ternary",
        use_s5=True,
        lambda_s5=0.2
    )
    
    # 2. Model WITHOUT S_5 Cayley Projection (for comparison)
    model_no_s5 = QuantizedGeoFormerBlock(
        d_model=d_model,
        n_heads=2,
        n_neurons=seq_len,
        b_w=4,
        mode="ternary",
        use_s5=False
    )
    
    # Standalone S_5 auditor for measuring unconstrained representation complexity
    auditor = S5CayleyProjector(d_model=d_model)
    
    opt_with = DualPathKramersOptimizer(model_with_s5.parameters(), lr=0.01, alpha_tau=0.25, d_0=1e-3)
    opt_no   = DualPathKramersOptimizer(model_no_s5.parameters(), lr=0.01, alpha_tau=0.25, d_0=1e-3)
    
    max_k_with = 0.0
    all_bounded_with = True
    
    print(f"{'Epoch':<6} │ {'K(S) With S_5':<16} │ {'Status':<14} │ {'K(S) No S_5':<16} │ {'Fiedler λ2':<12} │ {'TAC τ':<6}")
    print("-" * 90)
    
    torch.manual_seed(42)
    for epoch in range(1, epochs + 1):
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Train With S_5
        opt_with.zero_grad()
        h_star_w, A_w, L_w, tau_w, K_S_w = model_with_s5(x)
        loss_w = (h_star_w ** 2).mean()
        loss_w.backward()
        opt_with.step()
        if A_w.grad is not None:
            model_with_s5.step_slow_timescale(h_star_w.detach(), A_w.grad, eta_g=0.01)
            
        # Train Without S_5
        opt_no.zero_grad()
        h_star_no, A_no, L_no, tau_no, _ = model_no_s5(x)
        loss_no = (h_star_no ** 2).mean()
        loss_no.backward()
        opt_no.step()
        if A_no.grad is not None:
            model_no_s5.step_slow_timescale(h_star_no.detach(), A_no.grad, eta_g=0.01)
            
        # Audit unconstrained representation complexity
        _, K_S_no, _ = auditor(h_star_no.detach())
        
        k_val_w = K_S_w.item()
        k_val_no = K_S_no.item()
        max_k_with = max(max_k_with, k_val_w)
        
        # Test assertion: K(S) <= 54.68 bits
        bounded = (k_val_w <= 54.68)
        if not bounded:
            all_bounded_with = False
            
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            status_str = "✓ BOUNDED" if bounded else "X VIOLATED"
            fiedler = model_with_s5.metric_engine.get_fiedler_eigenvalue(seq_len)
            print(f"{epoch:<6} │ {k_val_w:>6.2f} bits      │ {status_str:<14} │ {k_val_no:>6.2f} bits      │ {fiedler:>8.4f}   │ {tau_w:<6}")
            
    print("=" * 90)
    print(f"Audit Result: Max K(S) Observed under S_5 Q-TAC = {max_k_with:.2f} bits.")
    print(f"Universal Upper Bound = 54.68 bits.")
    print(f"Hypothesis Verification: {'CONFIRMED (Strictly Bounded <= 54.68 bits)' if all_bounded_with else 'FAILED'}")
    print("=" * 90)

if __name__ == "__main__":
    run_s5_complexity_benchmark()
