"""
geoformer-sdk: Quantized Propagation Geometry & Neuromorphic GeoFormer Framework.
Author: Atmabhan Pandit (Shrikant Bhosale)
TWIST POOL Labs │ ISL Research │ NanoCERN Unit, Sambhaji Nagar / Pune
"""

__version__ = "0.1.0"

# 1. Quantization Sub-system
from .quant.ste import RoundSTE, SignSTE, AbsMaxQuantizer
from .quant.init import variance_preserving_init_
from .quant.ternary_linear import QuantizedLinear

# 2. Metric Geometry & Graph Dynamics
from .core.laplacian import GraphLaplacian
from .core.metric_warping import DiscreteMetricWarpingEngine

# 3. Geometric Attention & Continuous Relaxation Layers
from .layers.ffn import TernarySwiGLU
from .layers.cayley import S5CayleyProjector
from .layers.attention import QuantizedGHA
from .layers.tac import QuantizedTAC

# 4. Neural Network Models
from .models.block import QGeoFormerBlock, RMSNorm
from .models.relational import RelationalQGeoFormer
from .models.transitive import DeepTransitiveModel

# 5. Neuromorphic Memristor Hardware & Energy
from .hardware.crossbar import PhysicalMemristorCrossbar
from .hardware.emulator import NeuromorphicQGeoFormer
from .hardware.energy import NanojouleEnergyAuditor

# 6. Optimization
from .optim.kramers import DualPathKramersOptimizer

__all__ = [
    "__version__",
    "RoundSTE",
    "SignSTE",
    "AbsMaxQuantizer",
    "variance_preserving_init_",
    "QuantizedLinear",
    "GraphLaplacian",
    "DiscreteMetricWarpingEngine",
    "TernarySwiGLU",
    "S5CayleyProjector",
    "QuantizedGHA",
    "QuantizedTAC",
    "QGeoFormerBlock",
    "RMSNorm",
    "RelationalQGeoFormer",
    "DeepTransitiveModel",
    "PhysicalMemristorCrossbar",
    "NeuromorphicQGeoFormer",
    "NanojouleEnergyAuditor",
    "DualPathKramersOptimizer",
]
