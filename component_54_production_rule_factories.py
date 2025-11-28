"""
Component 54: Production System - Rule Factory Utilities

Utility functions for creating production rules with automatic specificity calculation.

Author: KAI Development Team
Date: 2025-11-21
"""

from typing import Callable

from component_54_production_rule import ProductionRule
from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory

# ============================================================================
# Configuration Constants
# ============================================================================

# Fact Selection Thresholds
MAX_PENDING_FACTS = 3
MAX_SENTENCES = 3
MAX_TRANSITIONS = 2

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_MIN = 0.70
MEDIUM_CONFIDENCE_MAX = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.40
VERY_HIGH_CONFIDENCE_THRESHOLD = 0.95

# Multi-Source Aggregation
MULTI_SOURCE_BOOST_BASE = 0.02
MULTI_SOURCE_BOOST_MAX = 0.05

# German Grammar Utilities
GERMAN_ARTICLES = {
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "den",
    "dem",
    "des",
    "einer",
    "einem",
    "einen",
}


# ============================================================================
# German Grammar Utility Functions
# ============================================================================


def determine_german_article(noun: str, case: str = "nominative") -> str:
    """
    Determine German article based on noun ending (heuristic).

    Args:
        noun: The German noun
        case: Grammatical case (nominative, accusative, dative)

    Returns:
        Appropriate article (ein/eine/einen/einer/einem)
    """
    # Common feminine endings
    is_feminine = noun.endswith(("e", "ung", "heit", "keit", "ion", "schaft"))

    if case == "nominative":
        return "eine" if is_feminine else "ein"
    elif case == "accusative":
        return "eine" if is_feminine else "einen"
    elif case == "dative":
        return "einer" if is_feminine else "einem"

    return "ein"  # Default fallback


def pluralize_german_noun(noun: str) -> str:
    """
    Simple German pluralization heuristic.

    Args:
        noun: The German noun to pluralize

    Returns:
        Pluralized form
    """
    # Already plural-like or ends with n
    if noun.endswith("n"):
        return noun

    # Common pluralization: add 'n'
    return noun + "n"


# ============================================================================
# Specificity Calculation
# ============================================================================


def calculate_specificity(
    condition_func: Callable[[ResponseGenerationState], bool],
) -> float:
    """
    Berechnet die Spezifität einer Condition-Funktion.

    Heuristik: Zähle Anzahl der Checks im Condition-Code.
    Mehr Checks = höhere Spezifität

    Args:
        condition_func: Condition-Callable mit ResponseGenerationState Parameter

    Returns:
        Float zwischen 1.0 (unspezifisch) und 10.0 (sehr spezifisch)
    """
    try:
        import inspect

        source = inspect.getsource(condition_func)

        # Zähle Checks
        check_count = source.count("if ") + source.count("and ") + source.count("or ")

        # Mehr Checks = höhere Spezifität (cap bei 10.0)
        specificity = min(1.0 + check_count * 0.5, 10.0)

        return specificity
    except Exception:
        # Fallback
        return 1.0


def create_production_rule(
    name: str,
    category: RuleCategory,
    condition: Callable[[ResponseGenerationState], bool],
    action: Callable[[ResponseGenerationState], None],
    utility: float = 1.0,
    auto_calculate_specificity: bool = True,
    **metadata,
) -> ProductionRule:
    """
    Factory-Funktion zum Erstellen einer ProductionRule.

    Args:
        name: Regelname
        category: Regelkategorie
        condition: Condition-Callable
        action: Action-Callable
        utility: Statische Utility (default=1.0)
        auto_calculate_specificity: Automatisch Spezifität berechnen
        **metadata: Zusätzliche Metadaten

    Returns:
        ProductionRule Instanz
    """
    specificity = (
        calculate_specificity(condition) if auto_calculate_specificity else 1.0
    )

    return ProductionRule(
        name=name,
        category=category,
        condition=condition,
        action=action,
        utility=utility,
        specificity=specificity,
        metadata=metadata,
    )
