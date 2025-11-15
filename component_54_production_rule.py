"""
Component 54: Production System - Production Rule

ProductionRule class representing a single IF-THEN rule.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory


@dataclass
class ProductionRule:
    """
    Eine Produktionsregel im Produktionssystem.

    Format: IF condition(working_memory) THEN action(working_memory)

    Attributes:
        name: Eindeutiger Name der Regel
        category: Kategorie (ContentSelection, Lexicalization, Discourse, Syntax)
        condition: Callable, das Working Memory prüft und True/False zurückgibt
        action: Callable, das Working Memory modifiziert
        utility: Statische Utility (Präferenz, handkodiert)
        specificity: Spezifität (wird automatisch berechnet)
        metadata: Zusätzliche Informationen (Tags, Beschreibung)
    """

    name: str
    category: RuleCategory
    condition: Callable[[ResponseGenerationState], bool]
    action: Callable[[ResponseGenerationState], None]
    utility: float = 1.0
    specificity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistiken (für Learning, später)
    application_count: int = 0
    success_count: int = 0
    last_applied: Optional[datetime] = None

    def matches(self, state: ResponseGenerationState) -> bool:
        """
        Prüft, ob die Regel auf den aktuellen State anwendbar ist.

        Returns:
            True wenn Condition erfüllt ist, False sonst
        """
        try:
            return self.condition(state)
        except Exception as e:
            logging.error(f"Error evaluating condition for rule {self.name}: {e}")
            return False

    def apply(self, state: ResponseGenerationState) -> None:
        """
        Wendet die Regel auf den State an.

        Modifiziert den State in-place und aktualisiert Statistiken.
        """
        try:
            self.action(state)
            self.application_count += 1
            self.last_applied = datetime.now()
            logging.debug(
                f"Applied production rule: {self.name} (category={self.category.value})"
            )
        except Exception as e:
            logging.error(f"Error applying rule {self.name}: {e}")
            raise

    def get_priority(self) -> float:
        """
        Berechnet die Gesamt-Priorität der Regel für Conflict Resolution.

        Priority = Utility * Specificity

        Returns:
            Float-Wert der Priorität
        """
        return self.utility * self.specificity

    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Regel in ein Dict für Persistierung.

        Note: Callables können nicht serialisiert werden, nur Metadaten.
        """
        return {
            "name": self.name,
            "category": self.category.value,
            "utility": self.utility,
            "specificity": self.specificity,
            "metadata": self.metadata,
            "application_count": self.application_count,
            "success_count": self.success_count,
            "last_applied": (
                self.last_applied.isoformat() if self.last_applied else None
            ),
        }
