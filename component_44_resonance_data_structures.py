"""
component_44_resonance_data_structures.py

Data structures for Cognitive Resonance Engine

Provides:
- ActivationType enum
- ReasoningPath dataclass
- ResonancePoint dataclass
- ActivationMap dataclass

Teil von Phase 4 Architecture Refactoring (2025-11-29)
Split from component_44_resonance_engine.py (1060 lines -> modular)

Author: KAI Development Team
Created: 2025-11-29
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class ActivationType(Enum):
    """Typ der Aktivierung für besseres Tracking"""

    DIRECT = "direct"  # Start-Konzept
    PROPAGATED = "propagated"  # Via Spreading
    RESONANCE = "resonance"  # Via Resonanz-Verstärkung


@dataclass
class ReasoningPath:
    """
    Ein Pfad von Quelle zu Ziel im Aktivierungsnetzwerk

    Attributes:
        source: Start-Konzept des Pfades
        target: Ziel-Konzept des Pfades
        relations: Liste der Beziehungstypen im Pfad
        confidence_product: Produkt aller Confidences im Pfad
        wave_depth: Bei welcher Wave wurde dieser Pfad entdeckt?
        activation_contribution: Wie viel trägt dieser Pfad zur finalen Aktivierung bei?
    """

    source: str
    target: str
    relations: List[str]
    confidence_product: float
    wave_depth: int
    activation_contribution: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Path({self.source} -> {self.target}, "
            f"relations={self.relations}, conf={self.confidence_product:.3f}, "
            f"wave={self.wave_depth})"
        )


@dataclass
class ResonancePoint:
    """
    Ein Konzept mit Resonanz-Verstärkung

    Attributes:
        concept: Das Konzept
        resonance_boost: Stärke der Resonanz-Verstärkung
        wave_depth: Wann trat Resonanz auf?
        num_paths: Anzahl konvergierender Pfade
    """

    concept: str
    resonance_boost: float
    wave_depth: int
    num_paths: int = 1

    def __repr__(self) -> str:
        return f"Resonance({self.concept}, boost={self.resonance_boost:.3f}, paths={self.num_paths})"


@dataclass
class ActivationMap:
    """
    Snapshot der Aktivierungszustände nach Spreading Activation

    Attributes:
        activations: Konzept -> Aktivierungslevel
        wave_history: Aktivierungen pro Wave (für Animation/Debugging)
        reasoning_paths: Alle entdeckten Pfade
        resonance_points: Konzepte mit Resonanz-Verstärkung
        max_activation: Höchste erreichte Aktivierung
        concepts_activated: Anzahl aktivierter Konzepte
        waves_executed: Anzahl durchgeführter Waves
        activation_types: Konzept -> Typ der Aktivierung
    """

    activations: Dict[str, float] = field(default_factory=dict)
    wave_history: List[Dict[str, float]] = field(default_factory=list)
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)
    resonance_points: List[ResonancePoint] = field(default_factory=list)
    max_activation: float = 0.0
    concepts_activated: int = 0
    waves_executed: int = 0
    activation_types: Dict[str, ActivationType] = field(default_factory=dict)

    def get_top_concepts(self, n: int = 10) -> List[Tuple[str, float]]:
        """Gibt die Top-N aktivierten Konzepte zurück"""
        return sorted(self.activations.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_paths_to(self, concept: str) -> List[ReasoningPath]:
        """Gibt alle Pfade zurück, die zu diesem Konzept führen"""
        return [p for p in self.reasoning_paths if p.target == concept]

    def is_resonance_point(self, concept: str) -> bool:
        """Prüft, ob ein Konzept ein Resonanz-Punkt ist"""
        return any(rp.concept == concept for rp in self.resonance_points)
