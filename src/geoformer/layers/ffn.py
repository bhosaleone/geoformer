"""
Ternary SwiGLU Feed-Forward Network Layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant.ternary_linear import QuantizedLinear

class TernarySwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network operating with BitNet b1.58 ternary linear layers.
    Out = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, d_model, d_ff=None, mode="ternary"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(8 / 3 * d_model)
        
        self.w_gate = QuantizedLinear(d_model, self.d_ff, bias=False, mode=mode)
        self.w_up = QuantizedLinear(d_model, self.d_ff, bias=False, mode=mode)
        self.w_down = QuantizedLinear(self.d_ff, d_model, bias=False, mode=mode)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)
