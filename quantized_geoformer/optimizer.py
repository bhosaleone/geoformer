"""
Dual Path Kramers Optimizer with Modulated Langevin Action Diffusion
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Combines deterministic momentum drift with stochastic field diffusion:
    theta_{t+1} = theta_t - eta * [(1 - alpha_tau) * m_t + alpha_tau * g_t]
                  + sqrt(2 * alpha_tau * D(t) * eta) * xi_t
Enables tunneling across non-smooth discrete energy barriers (Theorems 4 & Q-5),
powering delayed algorithmic generalization (grokking) in quantized networks.
"""

import math
import torch
from torch.optim.optimizer import Optimizer

class DualPathKramersOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, alpha_tau=0.2, d_0=1e-4, gamma_r=0.5, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, beta1=beta1, alpha_tau=alpha_tau, d_0=d_0, gamma_r=gamma_r, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, calendar_signal=0.0):
        """
        Performs a single optimization step.
        calendar_signal: external scalar r_t in [-1, 1] modulating thermal diffusion.
        """
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            alpha_tau = group['alpha_tau']
            d_0 = group['d_0']
            gamma_r = group['gamma_r']
            weight_decay = group['weight_decay']
            
            # Modulated diffusion coefficient: D(t) = D_0 * (1 + gamma_r * r_t)
            d_t = max(1e-9, d_0 * (1.0 + gamma_r * calendar_signal))
            noise_std = math.sqrt(2.0 * alpha_tau * d_t * lr)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Decoupled weight decay (L2 regularization to destabilize memorization basins)
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                state['step'] += 1

                # Momentum update
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Combined drift: (1 - alpha_tau) * momentum + alpha_tau * raw_gradient
                drift = (1.0 - alpha_tau) * exp_avg + alpha_tau * grad
                p.add_(drift, alpha=-lr)

                # Stochastic Langevin Field 2 action diffusion
                if noise_std > 0.0:
                    noise = torch.randn_like(p) * noise_std
                    p.add_(noise)

