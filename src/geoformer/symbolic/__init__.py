"""
GeoFormer Neuro-Symbolic Hybridization Engine.
Combines discrete Domain-Specific Language (DSL) program synthesis
with continuous GeoFormer metric manifold heuristic evaluation.
"""

from .dsl import ARCPrimitives
from .synthesizer import ARCSynthesizer

__all__ = ["ARCPrimitives", "ARCSynthesizer"]
