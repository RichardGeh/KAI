"""
component_44_resonance_engine.py

Cognitive Resonance Core - Wellenförmige Aktivierung mit Resonanz-Verstärkung

FACADE FILE - Maintains backward compatibility after modular refactoring.
All functionality delegated to focused modules:
- component_44_resonance_data_structures.py (130 lines) - Data classes
- component_44_resonance_core.py (632 lines) - Core spreading activation
- component_44_adaptive_resonance.py (330 lines) - Adaptive tuning

Teil von Phase 4 Architecture Refactoring (2025-11-29)
Original: 1060 lines -> Split into 3 modules + facade

Author: KAI Development Team
Created: 2025-11-07
Refactored: 2025-11-29
"""

# Re-export all public classes and functions for backward compatibility
from component_44_adaptive_resonance import AdaptiveResonanceEngine
from component_44_resonance_core import ResonanceEngine
from component_44_resonance_data_structures import (
    ActivationMap,
    ActivationType,
    ReasoningPath,
    ResonancePoint,
)

__all__ = [
    # Data structures
    "ActivationType",
    "ReasoningPath",
    "ResonancePoint",
    "ActivationMap",
    # Engine classes
    "ResonanceEngine",
    "AdaptiveResonanceEngine",
]
