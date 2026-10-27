"""
Straight-Through Estimators (STE) for Quantized Propagation Geometry.
Implements BitNet b1.58 Ternary and Int4/Int8 Straight-Through Estimators.
"""

import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients unchanged
        return grad_output


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class AbsMaxQuantizer:
    """Utility class for absolute-maximum dynamic quantization scaling."""
    @staticmethod
    def quantize_weight_ternary(w, eps=1e-8):
        # BitNet b1.58 ternary scaling: gamma = mean(|w|)
        gamma = torch.mean(torch.abs(w)) + eps
        w_scaled = w / gamma
        w_quant = torch.clamp(RoundSTE.apply(w_scaled), -1.0, 1.0)
        return w_quant, gamma

    @staticmethod
    def quantize_activation_int(x, bits=8, eps=1e-8):
        q_max = (1 << (bits - 1)) - 1
        scale = q_max / (torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + eps)
        x_quant = torch.clamp(RoundSTE.apply(x * scale), -q_max, q_max)
        return x_quant, scale
