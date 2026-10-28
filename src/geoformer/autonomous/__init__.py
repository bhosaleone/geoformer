"""
GeoFormer Autonomous Module:
Multi-scale Q-TAC schedules, quasi-crystalline encodings, and bounded endogenous consolidation.
"""

from .schedules import (
    continued_fraction_convergents,
    fibonacci_schedule,
    make_relaxation_schedule,
)
from .encodings import (
    quasi_crystalline_encoding,
    encoding_collision_rate,
)
from .endogenous_engine import (
    ConsolidationConfig,
    ConsolidationReport,
    EndogenousConsolidator,
)

__all__ = [
    "continued_fraction_convergents",
    "fibonacci_schedule",
    "make_relaxation_schedule",
    "quasi_crystalline_encoding",
    "encoding_collision_rate",
    "ConsolidationConfig",
    "ConsolidationReport",
    "EndogenousConsolidator",
]
