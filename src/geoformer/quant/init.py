"""
Variance-Preserving Initialization for Quantized Weights.
Prevents exponential signal attenuation and explosion under ternary {-1, 0, +1} constraints.
"""

import math
import torch
import torch.nn as nn

def variance_preserving_init_(tensor, mode="ternary"):
    """
    Initializes a weight tensor to preserve signal variance after BitNet b1.58 quantization.
    For ternary weights {-1, 0, +1}, the expected variance is 2/3 * gamma^2.
    """
    with torch.no_grad():
        fan_in = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        if mode == "ternary":
            std = math.sqrt(1.5 / max(1, fan_in))
            tensor.normal_(0.0, std)
        elif mode.startswith("int"):
            std = math.sqrt(2.0 / max(1, fan_in))
            tensor.normal_(0.0, std)
        else:
            nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
    return tensor
