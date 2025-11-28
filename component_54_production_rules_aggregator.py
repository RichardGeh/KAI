"""
Component 54: Production System - Rule Aggregator

Aggregates all production rules from specialized modules into complete rule sets.

This module provides convenience functions for loading rule collections:
- create_all_content_selection_rules() - 15 content selection rules
- create_all_lexicalization_rules() - 15 lexicalization rules
- create_all_discourse_management_rules() - 12 discourse management rules
- create_all_syntactic_realization_rules() - 12 syntactic realization rules
- create_all_phase3_rules() - Combined Phase 3 (Lex + Discourse)
- create_all_phase4_rules() - Combined Phase 4 (Syntax)
- create_complete_production_system() - All 54 rules

Author: KAI Development Team
Date: 2025-11-21
"""

from typing import List

from component_54_production_rule import ProductionRule
from component_54_production_rules_content import create_all_content_selection_rules
from component_54_production_rules_discourse import (
    create_all_discourse_management_rules,
)
from component_54_production_rules_lexical import create_all_lexicalization_rules
from component_54_production_rules_syntax import create_all_syntactic_realization_rules


def create_all_phase3_rules() -> List[ProductionRule]:
    """
    Erstellt alle 27 PHASE 3 Rules (Lexicalization + Discourse Management).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return create_all_lexicalization_rules() + create_all_discourse_management_rules()


def create_all_phase4_rules() -> List[ProductionRule]:
    """
    Erstellt alle 12 PHASE 4 Rules (Syntactic Realization).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return create_all_syntactic_realization_rules()


def create_complete_production_system() -> List[ProductionRule]:
    """
    Erstellt ein vollständiges Produktionssystem mit allen Regeln aus allen Phasen.

    Umfasst:
    - PHASE 2: 15 Content Selection Rules
    - PHASE 3: 15 Lexicalization Rules + 12 Discourse Management Rules
    - PHASE 4: 12 Syntactic Realization Rules

    Gesamt: 54 Produktionsregeln

    Returns:
        Liste aller ProductionRule Instanzen
    """
    return (
        create_all_content_selection_rules()
        + create_all_lexicalization_rules()
        + create_all_discourse_management_rules()
        + create_all_syntactic_realization_rules()
    )
