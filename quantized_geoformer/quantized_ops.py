"""
Quantized Tensor Operators & Straight-Through Estimators (STE)
Monograph: Quantized Propagation Geometry (Q-PG)
Author: Atmabhan Pandit (Shrikant Bhosale)

Provides:
  - TernaryQuantizeSTE: Ternary weights {-1, 0, +1} * alpha
  - BinaryQuantizeSTE: 1-bit weights {-1, +1} * alpha (XNOR-Net style)
  - INT4QuantizeSTE: 4-bit uniform integer quantization
  - QuantizedLinear: Modular PyTorch Linear layer with STE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Ternary Quantization with Straight-Through Estimator (STE)
    using BitNet b1.58 formulation:
        gamma = mean(|W|)
        W_quant = round(clip(W / gamma, -1, 1)) * gamma
    Values strictly in {-1, 0, +1} * gamma, preserving variance through layers.
    """
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        gamma = weight.abs().mean().clamp(min=1e-5)
        scaled_w = weight / gamma
        ternary_w = torch.round(torch.clamp(scaled_w, -1.0, 1.0))
        return ternary_w * gamma

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_input = grad_output.clone()
        gamma = weight.abs().mean().clamp(min=1e-5)
        grad_input[weight.abs() > 1.5 * gamma] = 0.0
        return grad_input

class BinaryQuantizeSTE(torch.autograd.Function):
    """1-bit Binary Quantization {-1, +1} scaled by mean(|W|)."""
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        scale = weight.abs().mean().clamp(min=1e-5)
        binary_w = torch.sign(weight)
        binary_w[binary_w == 0] = 1.0
        return binary_w * scale

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = weight.abs().mean().clamp(min=1e-5)
        grad_input[weight.abs() > 1.5 * scale] = 0.0
        return grad_input

class QuantizedLinear(nn.Module):
    """
    Linear layer supporting ternary (1.58-bit) and binary (1-bit) weights
    with backward pass mediated by Straight-Through Estimators.
    """
    def __init__(self, in_features, out_features, mode="ternary", bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        
        # Proper Kaiming normal initialization preserving forward/backward variance
        std = math.sqrt(2.0 / in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        if self.mode == "ternary":
            w_quant = TernaryQuantizeSTE.apply(self.weight)
        elif self.mode == "binary":
            w_quant = BinaryQuantizeSTE.apply(self.weight)
        else:
            w_quant = self.weight
            
        return F.linear(x, w_quant, self.bias)
