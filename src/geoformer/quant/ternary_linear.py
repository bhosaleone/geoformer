"""
QuantizedLinear Module with BitNet b1.58 Ternary Straight-Through Estimation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ste import RoundSTE, AbsMaxQuantizer
from .init import variance_preserving_init_

class QuantizedLinear(nn.Module):
    """
    Quantized Linear Layer implementing BitNet b1.58 {-1, 0, +1} weight quantization
    and optional low-bit integer activation quantization.
    """
    def __init__(self, in_features, out_features, bias=False, mode="ternary", act_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.act_bits = act_bits

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        variance_preserving_init_(self.weight, mode=self.mode)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.mode == "fp32":
            return F.linear(x, self.weight, self.bias)

        # 1. Weight Quantization via BitNet b1.58 STE
        if self.mode == "ternary":
            w_quant, gamma = AbsMaxQuantizer.quantize_weight_ternary(self.weight)
            # Reconstruct effective weight tensor differentiably with STE
            w_eff = (self.weight + (w_quant * gamma - self.weight).detach())
        elif self.mode in ("int4", "int8"):
            bits = 4 if self.mode == "int4" else 8
            q_max = (1 << (bits - 1)) - 1
            alpha = torch.mean(torch.abs(self.weight)) + 1e-8
            w_scaled = self.weight / alpha
            w_quant = torch.clamp(RoundSTE.apply(w_scaled), -q_max, q_max)
            w_eff = (self.weight + (w_quant * alpha - self.weight).detach())
        else:
            w_eff = self.weight

        # 2. Optional Activation Quantization
        if self.act_bits is not None and self.act_bits < 32 and self.mode != "fp32":
            x_quant, scale = AbsMaxQuantizer.quantize_activation_int(x, bits=self.act_bits)
            x_eff = (x + (x_quant / scale - x).detach())
        else:
            x_eff = x

        return F.linear(x_eff, w_eff, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, mode='{self.mode}', act_bits={self.act_bits}"
