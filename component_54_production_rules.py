"""
Component 54: Production System - Rule Factories

Factory functions for creating production rules.

Contains all create_*_rule() functions for:
- Content Selection (what to say)
- Lexicalization (how to phrase)
- Discourse Management (structure)
- Syntactic Realization (grammar)

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from typing import Callable, List

from component_54_production_rule import ProductionRule
from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory


def calculate_specificity(condition_func: Callable) -> float:
    """
    Berechnet die Spezifität einer Condition-Funktion.

    Heuristik: Zähle Anzahl der Checks im Condition-Code.
    Mehr Checks = höhere Spezifität

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


# ============================================================================
# PHASE 2: Content Selection Rules (Fact Selection + Confidence Filtering)
# ============================================================================


def create_select_is_a_fact_rule() -> ProductionRule:
    """
    SELECT_IS_A_FACT (utility: 0.95)

    Condition: IS_A Fakt in available_facts und pending_facts leer
    Action: Füge IS_A Fakt mit höchster Confidence zu pending_facts hinzu

    IS_A Fakten haben höchste Priorität, da sie taxonomische Basisinformation liefern.
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Bedingungen:
        # 1. Noch keine pending_facts (noch nichts ausgewählt)
        # 2. IS_A Fakt in available_facts vorhanden
        if len(state.discourse.pending_facts) > 0:
            return False

        isa_facts = [
            f for f in state.available_facts if f.get("relation_type") == "IS_A"
        ]
        return len(isa_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        # Finde IS_A Fakten
        isa_facts = [
            f for f in state.available_facts if f.get("relation_type") == "IS_A"
        ]

        if isa_facts:
            # Wähle Fakt mit höchster Confidence
            best_fact = max(isa_facts, key=lambda f: f.get("confidence", 0.0))
            state.discourse.pending_facts.append(best_fact)

            # Entferne aus available_facts (bereits ausgewählt)
            state.available_facts.remove(best_fact)

            logging.debug(f"Selected IS_A fact: {best_fact}")

    return create_production_rule(
        name="SELECT_IS_A_FACT",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.95,
        description="Wählt IS_A Fakt mit höchster Confidence aus",
        priority_reason="Taxonomische Basisinformation",
    )


def create_select_property_fact_rule() -> ProductionRule:
    """
    SELECT_PROPERTY_FACT (utility: 0.90)

    Condition: HAS_PROPERTY Fakt in available_facts und noch Platz in pending_facts
    Action: Füge HAS_PROPERTY Fakt zu pending_facts hinzu
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Bedingungen:
        # 1. Weniger als 3 pending_facts (Platz für mehr Fakten)
        # 2. HAS_PROPERTY Fakt verfügbar
        if len(state.discourse.pending_facts) >= 3:
            return False

        property_facts = [
            f for f in state.available_facts if f.get("relation_type") == "HAS_PROPERTY"
        ]
        return len(property_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        property_facts = [
            f for f in state.available_facts if f.get("relation_type") == "HAS_PROPERTY"
        ]

        if property_facts:
            # Wähle Fakt mit höchster Confidence
            best_fact = max(property_facts, key=lambda f: f.get("confidence", 0.0))
            state.discourse.pending_facts.append(best_fact)
            state.available_facts.remove(best_fact)

            logging.debug(f"Selected HAS_PROPERTY fact: {best_fact}")

    return create_production_rule(
        name="SELECT_PROPERTY_FACT",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.90,
        description="Wählt HAS_PROPERTY Fakt mit höchster Confidence aus",
    )


def create_select_capability_fact_rule() -> ProductionRule:
    """
    SELECT_CAPABILITY_FACT (utility: 0.88)

    Condition: CAPABLE_OF Fakt in available_facts
    Action: Füge CAPABLE_OF Fakt zu pending_facts hinzu
    """

    def condition(state: ResponseGenerationState) -> bool:
        if len(state.discourse.pending_facts) >= 3:
            return False

        capability_facts = [
            f for f in state.available_facts if f.get("relation_type") == "CAPABLE_OF"
        ]
        return len(capability_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        capability_facts = [
            f for f in state.available_facts if f.get("relation_type") == "CAPABLE_OF"
        ]

        if capability_facts:
            best_fact = max(capability_facts, key=lambda f: f.get("confidence", 0.0))
            state.discourse.pending_facts.append(best_fact)
            state.available_facts.remove(best_fact)

            logging.debug(f"Selected CAPABLE_OF fact: {best_fact}")

    return create_production_rule(
        name="SELECT_CAPABILITY_FACT",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.88,
        description="Wählt CAPABLE_OF Fakt aus",
    )


def create_select_location_fact_rule() -> ProductionRule:
    """
    SELECT_LOCATION_FACT (utility: 0.87)

    Condition: LOCATED_IN Fakt in available_facts
    Action: Füge LOCATED_IN Fakt zu pending_facts hinzu
    """

    def condition(state: ResponseGenerationState) -> bool:
        if len(state.discourse.pending_facts) >= 3:
            return False

        location_facts = [
            f for f in state.available_facts if f.get("relation_type") == "LOCATED_IN"
        ]
        return len(location_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        location_facts = [
            f for f in state.available_facts if f.get("relation_type") == "LOCATED_IN"
        ]

        if location_facts:
            best_fact = max(location_facts, key=lambda f: f.get("confidence", 0.0))
            state.discourse.pending_facts.append(best_fact)
            state.available_facts.remove(best_fact)

            logging.debug(f"Selected LOCATED_IN fact: {best_fact}")

    return create_production_rule(
        name="SELECT_LOCATION_FACT",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.87,
        description="Wählt LOCATED_IN Fakt aus",
    )


def create_select_part_of_fact_rule() -> ProductionRule:
    """
    SELECT_PART_OF_FACT (utility: 0.85)

    Condition: PART_OF Fakt in available_facts
    Action: Füge PART_OF Fakt zu pending_facts hinzu
    """

    def condition(state: ResponseGenerationState) -> bool:
        if len(state.discourse.pending_facts) >= 3:
            return False

        partof_facts = [
            f for f in state.available_facts if f.get("relation_type") == "PART_OF"
        ]
        return len(partof_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        partof_facts = [
            f for f in state.available_facts if f.get("relation_type") == "PART_OF"
        ]

        if partof_facts:
            best_fact = max(partof_facts, key=lambda f: f.get("confidence", 0.0))
            state.discourse.pending_facts.append(best_fact)
            state.available_facts.remove(best_fact)

            logging.debug(f"Selected PART_OF fact: {best_fact}")

    return create_production_rule(
        name="SELECT_PART_OF_FACT",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.85,
        description="Wählt PART_OF Fakt aus",
    )


def create_prioritize_high_confidence_rule() -> ProductionRule:
    """
    PRIORITIZE_HIGH_CONFIDENCE (utility: 0.92)

    Condition: Fakt mit confidence >= 0.85 in available_facts UND noch nicht sortiert
    Action: Bevorzuge diesen Fakt vor anderen (pre-selection)

    Diese Regel sortiert available_facts nach Confidence, sodass nachfolgende
    Select-Regeln automatisch hochwertige Fakten wählen.
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur anwenden, wenn noch nicht sortiert
        if state.constraints.get("facts_prioritized", False):
            return False

        # Nur anwenden, wenn available_facts unsortiert sind
        # und mindestens ein High-Confidence Fakt vorhanden
        if len(state.available_facts) < 2:
            return False

        high_conf_facts = [
            f for f in state.available_facts if f.get("confidence", 0.0) >= 0.85
        ]
        return len(high_conf_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        # Sortiere available_facts nach Confidence (absteigend)
        state.available_facts.sort(key=lambda f: f.get("confidence", 0.0), reverse=True)

        # Markiere als sortiert (verhindert erneute Anwendung)
        state.constraints["facts_prioritized"] = True

        logging.debug("Prioritized high-confidence facts")

    return create_production_rule(
        name="PRIORITIZE_HIGH_CONFIDENCE",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.92,
        description="Sortiert Fakten nach Confidence (High-Confidence first)",
    )


def create_skip_low_confidence_rule() -> ProductionRule:
    """
    SKIP_LOW_CONFIDENCE (utility: 0.80)

    Condition: Fakt mit confidence < 0.40 in available_facts UND noch nicht gefiltert
    Action: Entferne Fakt aus available_facts (wird nicht verwendet)
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur anwenden, wenn noch nicht gefiltert
        if state.constraints.get("low_confidence_filtered", False):
            return False

        low_conf_facts = [
            f for f in state.available_facts if f.get("confidence", 0.0) < 0.40
        ]
        return len(low_conf_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        # Entferne alle Low-Confidence Fakten
        low_conf_facts = [
            f for f in state.available_facts if f.get("confidence", 0.0) < 0.40
        ]

        for fact in low_conf_facts:
            state.available_facts.remove(fact)
            logging.debug(f"Skipped low-confidence fact: {fact}")

        # Markiere als gefiltert
        state.constraints["low_confidence_filtered"] = True

    return create_production_rule(
        name="SKIP_LOW_CONFIDENCE",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.80,
        description="Entfernt Fakten mit Confidence < 0.40",
    )


def create_select_synonym_rule() -> ProductionRule:
    """
    SELECT_SYNONYM (utility: 0.78)

    Condition: Synonym-Fakt in available_facts (relation_type == "SYNONYM")
    Action: Füge Synonym zu pending_facts hinzu (für Variation)
    """

    def condition(state: ResponseGenerationState) -> bool:
        if len(state.discourse.pending_facts) >= 3:
            return False

        synonym_facts = [
            f for f in state.available_facts if f.get("relation_type") == "SYNONYM"
        ]
        return len(synonym_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        synonym_facts = [
            f for f in state.available_facts if f.get("relation_type") == "SYNONYM"
        ]

        if synonym_facts:
            best_fact = max(synonym_facts, key=lambda f: f.get("confidence", 0.0))
            state.discourse.pending_facts.append(best_fact)
            state.available_facts.remove(best_fact)

            logging.debug(f"Selected SYNONYM fact: {best_fact}")

    return create_production_rule(
        name="SELECT_SYNONYM",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.78,
        description="Wählt Synonym-Fakt für sprachliche Variation",
    )


def create_select_definition_rule() -> ProductionRule:
    """
    SELECT_DEFINITION (utility: 0.93)

    Condition: Definition-Fakt in available_facts (relation_type == "DEFINITION")
    Action: Füge Definition zu pending_facts hinzu (höchste Priorität nach IS_A)
    """

    def condition(state: ResponseGenerationState) -> bool:
        if len(state.discourse.pending_facts) > 0:
            return False

        definition_facts = [
            f for f in state.available_facts if f.get("relation_type") == "DEFINITION"
        ]
        return len(definition_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        definition_facts = [
            f for f in state.available_facts if f.get("relation_type") == "DEFINITION"
        ]

        if definition_facts:
            best_fact = max(definition_facts, key=lambda f: f.get("confidence", 0.0))
            state.discourse.pending_facts.append(best_fact)
            state.available_facts.remove(best_fact)

            logging.debug(f"Selected DEFINITION fact: {best_fact}")

    return create_production_rule(
        name="SELECT_DEFINITION",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.93,
        description="Wählt DEFINITION Fakt (explizite Bedeutung)",
    )


def create_finish_content_selection_rule() -> ProductionRule:
    """
    FINISH_CONTENT_SELECTION (utility: 0.70)

    Condition: Genug Fakten in pending_facts ODER keine available_facts mehr
    Action: Markiere Content-Selection als abgeschlossen

    Diese Regel beendet die Content-Selection Phase.
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Beende Selection wenn:
        # 1. Mindestens 1 Fakt ausgewählt UND (keine available_facts mehr ODER >= 3 pending_facts)
        # 2. ODER: Keine available_facts und keine pending_facts (leere Antwort)

        if len(state.discourse.pending_facts) >= 1 and len(state.available_facts) == 0:
            return True

        if len(state.discourse.pending_facts) >= 3:
            return True

        return False

    def action(state: ResponseGenerationState) -> None:
        # Markiere Selection als abgeschlossen
        # (Setze Marker in state.constraints)
        state.constraints["content_selection_finished"] = True

        logging.info(
            f"Content selection finished: {len(state.discourse.pending_facts)} facts selected"
        )

    return create_production_rule(
        name="FINISH_CONTENT_SELECTION",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.70,
        description="Beendet Content Selection Phase",
    )


# ============================================================================
# PHASE 2: Confidence-based Filtering Rules
# ============================================================================


def create_require_high_confidence_rule() -> ProductionRule:
    """
    REQUIRE_HIGH_CONFIDENCE (confidence >= 0.85)

    Condition: Fakt in pending_facts mit confidence < 0.85 UND hochwertige Alternative verfügbar
    Action: Ersetze Fakt durch hochwertige Alternative
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Gibt es Low-Confidence Fakt in pending_facts?
        low_conf_pending = [
            f for f in state.discourse.pending_facts if f.get("confidence", 0.0) < 0.85
        ]

        if not low_conf_pending:
            return False

        # Gibt es High-Confidence Alternative in available_facts?
        high_conf_available = [
            f for f in state.available_facts if f.get("confidence", 0.0) >= 0.85
        ]

        return len(high_conf_available) > 0

    def action(state: ResponseGenerationState) -> None:
        low_conf_pending = [
            f for f in state.discourse.pending_facts if f.get("confidence", 0.0) < 0.85
        ]
        high_conf_available = [
            f for f in state.available_facts if f.get("confidence", 0.0) >= 0.85
        ]

        if low_conf_pending and high_conf_available:
            # Ersetze ersten Low-Confidence Fakt
            old_fact = low_conf_pending[0]
            new_fact = high_conf_available[0]

            state.discourse.pending_facts.remove(old_fact)
            state.available_facts.remove(new_fact)
            state.discourse.pending_facts.append(new_fact)

            # Füge alten Fakt zurück zu available_facts
            state.available_facts.append(old_fact)

            logging.debug(
                f"Replaced low-confidence fact with high-confidence alternative"
            )

    return create_production_rule(
        name="REQUIRE_HIGH_CONFIDENCE",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.92,
        description="Ersetzt Low-Confidence Fakten durch High-Confidence Alternativen",
    )


def create_warn_medium_confidence_rule() -> ProductionRule:
    """
    WARN_MEDIUM_CONFIDENCE (0.70 <= confidence < 0.85)

    Condition: Fakt in pending_facts mit Medium Confidence
    Action: Füge Unsicherheitsmarker zu Fakt hinzu
    """

    def condition(state: ResponseGenerationState) -> bool:
        medium_conf = [
            f
            for f in state.discourse.pending_facts
            if 0.70 <= f.get("confidence", 0.0) < 0.85
        ]

        # Nur anwenden, wenn Fakt noch keinen Marker hat
        return any(not f.get("uncertainty_marked", False) for f in medium_conf)

    def action(state: ResponseGenerationState) -> None:
        medium_conf = [
            f
            for f in state.discourse.pending_facts
            if 0.70 <= f.get("confidence", 0.0) < 0.85
            and not f.get("uncertainty_marked", False)
        ]

        for fact in medium_conf:
            fact["uncertainty_marked"] = True
            fact["hedging_phrase"] = (
                "möglicherweise"  # z.B. "Ein Apfel ist möglicherweise rot"
            )

            logging.debug(f"Marked medium-confidence fact with uncertainty: {fact}")

    return create_production_rule(
        name="WARN_MEDIUM_CONFIDENCE",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.85,
        description="Markiert Medium-Confidence Fakten mit Unsicherheitshinweis",
    )


def create_skip_uncertain_facts_rule() -> ProductionRule:
    """
    SKIP_UNCERTAIN_FACTS (confidence < 0.40)

    Condition: Fakt in pending_facts mit confidence < 0.40
    Action: Entferne Fakt aus pending_facts (zu unsicher)

    Diese Regel ist strenger als SKIP_LOW_CONFIDENCE, da sie bereits ausgewählte
    Fakten noch einmal prüft und entfernt, falls zu unsicher.
    """

    def condition(state: ResponseGenerationState) -> bool:
        uncertain = [
            f for f in state.discourse.pending_facts if f.get("confidence", 0.0) < 0.40
        ]
        return len(uncertain) > 0

    def action(state: ResponseGenerationState) -> None:
        uncertain = [
            f for f in state.discourse.pending_facts if f.get("confidence", 0.0) < 0.40
        ]

        for fact in uncertain:
            state.discourse.pending_facts.remove(fact)
            logging.warning(f"Removed uncertain fact from pending_facts: {fact}")

    return create_production_rule(
        name="SKIP_UNCERTAIN_FACTS",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.88,
        description="Entfernt bereits ausgewählte Fakten mit Confidence < 0.40",
    )


def create_aggregate_multi_source_rule() -> ProductionRule:
    """
    AGGREGATE_MULTI_SOURCE (mehrere Fakten)

    Condition: Mehrere Fakten mit gleichem Inhalt aber unterschiedlichen Confidences UND noch nicht aggregiert
    Action: Aggregiere zu einem Fakt mit erhöhter Confidence
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur anwenden, wenn noch nicht aggregiert
        if state.constraints.get("facts_aggregated", False):
            return False

        # Prüfe ob es Fakten mit gleichem relation_type und object gibt
        facts = state.available_facts + state.discourse.pending_facts

        if len(facts) < 2:
            return False

        # Suche Duplikate
        seen = {}
        for fact in facts:
            key = (fact.get("relation_type"), fact.get("object"))
            if key in seen:
                return True  # Duplikat gefunden
            seen[key] = fact

        return False

    def action(state: ResponseGenerationState) -> None:
        # Aggregiere Duplikate
        all_facts = state.available_facts + state.discourse.pending_facts

        # Gruppiere nach (relation_type, object)
        groups = {}
        for fact in all_facts:
            key = (fact.get("relation_type"), fact.get("object"))
            if key not in groups:
                groups[key] = []
            groups[key].append(fact)

        # Aggregiere Gruppen mit mehreren Fakten
        for key, group in groups.items():
            if len(group) > 1:
                # Berechne durchschnittliche Confidence
                avg_confidence = sum(f.get("confidence", 0.0) for f in group) / len(
                    group
                )

                # Boost für Multi-Source (max 0.05)
                boost = min(0.05, (len(group) - 1) * 0.02)
                final_confidence = min(1.0, avg_confidence + boost)

                # Erstelle aggregierten Fakt
                aggregated_fact = {
                    "relation_type": key[0],
                    "object": key[1],
                    "confidence": final_confidence,
                    "aggregated_from": len(group),
                    "sources": [f.get("source", "unknown") for f in group],
                }

                # Entferne Original-Fakten
                for fact in group:
                    if fact in state.available_facts:
                        state.available_facts.remove(fact)
                    if fact in state.discourse.pending_facts:
                        state.discourse.pending_facts.remove(fact)

                # Füge aggregierten Fakt hinzu (zu available_facts für spätere Selektion)
                state.available_facts.append(aggregated_fact)

                logging.info(
                    f"Aggregated {len(group)} facts: {key[0]}={key[1]} "
                    f"(confidence={final_confidence:.2f})"
                )

        # Markiere als aggregiert
        state.constraints["facts_aggregated"] = True

    return create_production_rule(
        name="AGGREGATE_MULTI_SOURCE",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.90,
        description="Aggregiert Fakten aus mehreren Quellen mit Confidence-Boost",
    )


def create_prefer_direct_fact_rule() -> ProductionRule:
    """
    PREFER_DIRECT_FACT (1-Hop vs. Multi-Hop)

    Condition: Sowohl direkter (1-Hop) als auch indirekter (Multi-Hop) Fakt verfügbar UND noch nicht gefiltert
    Action: Bevorzuge direkten Fakt
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur anwenden, wenn noch nicht gefiltert
        if state.constraints.get("direct_facts_preferred", False):
            return False

        # Prüfe ob es sowohl direkte als auch indirekte Fakten gibt
        direct_facts = [f for f in state.available_facts if f.get("hop_count", 1) == 1]
        indirect_facts = [f for f in state.available_facts if f.get("hop_count", 1) > 1]

        return len(direct_facts) > 0 and len(indirect_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        # Entferne indirekte Fakten wenn direktes Equivalent vorhanden
        direct_facts = [f for f in state.available_facts if f.get("hop_count", 1) == 1]
        indirect_facts = [f for f in state.available_facts if f.get("hop_count", 1) > 1]

        # Erstelle Set von direkten Fakt-Keys
        direct_keys = {(f.get("relation_type"), f.get("object")) for f in direct_facts}

        # Entferne indirekte Fakten mit direktem Equivalent
        removed_count = 0
        for fact in indirect_facts:
            key = (fact.get("relation_type"), fact.get("object"))
            if key in direct_keys:
                state.available_facts.remove(fact)
                removed_count += 1
                logging.debug(
                    f"Removed indirect fact (direct equivalent exists): {fact}"
                )

        if removed_count > 0:
            logging.info(
                f"Preferred {removed_count} direct facts over indirect equivalents"
            )

        # Markiere als gefiltert
        state.constraints["direct_facts_preferred"] = True

    return create_production_rule(
        name="PREFER_DIRECT_FACT",
        category=RuleCategory.CONTENT_SELECTION,
        condition=condition,
        action=action,
        utility=0.86,
        description="Bevorzugt direkte (1-Hop) Fakten über indirekte (Multi-Hop)",
    )


# ============================================================================
# Convenience Function: Load All Content Selection Rules
# ============================================================================


def create_all_content_selection_rules() -> List[ProductionRule]:
    """
    Erstellt alle 15 Content Selection Rules (Fact Selection + Confidence Filtering).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return [
        # Fact Selection Rules (10)
        create_select_is_a_fact_rule(),
        create_select_property_fact_rule(),
        create_select_capability_fact_rule(),
        create_select_location_fact_rule(),
        create_select_part_of_fact_rule(),
        create_prioritize_high_confidence_rule(),
        create_skip_low_confidence_rule(),
        create_select_synonym_rule(),
        create_select_definition_rule(),
        create_finish_content_selection_rule(),
        # Confidence Filtering Rules (5)
        create_require_high_confidence_rule(),
        create_warn_medium_confidence_rule(),
        create_skip_uncertain_facts_rule(),
        create_aggregate_multi_source_rule(),
        create_prefer_direct_fact_rule(),
    ]


# ============================================================================
# PHASE 3: Lexicalization Rules (Fact → Natural Language)
# ============================================================================


def create_verbalize_is_a_simple_rule() -> ProductionRule:
    """
    VERBALIZE_IS_A_SIMPLE (utility: 0.90)

    Condition: IS_A Fakt in pending_facts UND noch nicht verbalisiert
    Action: Erzeuge Satz "X ist ein/eine Y"

    Beispiel: {"relation_type": "IS_A", "object": "frucht"} → "Ein Apfel ist eine Frucht."
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Bedingungen:
        # 1. Content Selection abgeschlossen
        # 2. IS_A Fakt in pending_facts vorhanden
        # 3. Noch nicht alle Fakten verbalisiert
        if not state.constraints.get("content_selection_finished", False):
            return False

        isa_facts = [
            f for f in state.discourse.pending_facts if f.get("relation_type") == "IS_A"
        ]
        return len(isa_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        # Finde ersten IS_A Fakt
        isa_facts = [
            f for f in state.discourse.pending_facts if f.get("relation_type") == "IS_A"
        ]

        if isa_facts:
            fact = isa_facts[0]
            subject = state.primary_goal.target_entity or "Das"
            object_noun = fact.get("object", "etwas")

            # Artikel-Bestimmung (heuristisch)
            article = (
                "eine"
                if object_noun.endswith(("e", "ung", "heit", "keit", "ion"))
                else "ein"
            )

            # Satz-Konstruktion
            sentence = (
                f"Ein {subject.capitalize()} ist {article} {object_noun.capitalize()}."
            )

            # Füge Unsicherheitsmarker hinzu falls vorhanden
            if fact.get("uncertainty_marked", False):
                hedge = fact.get("hedging_phrase", "möglicherweise")
                sentence = f"Ein {subject.capitalize()} ist {hedge} {article} {object_noun.capitalize()}."

            state.add_sentence(sentence)
            state.mention_entity(subject)
            state.mention_entity(object_noun)

            # Entferne verarbeiteten Fakt
            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Verbalized IS_A fact: {sentence}")

    return create_production_rule(
        name="VERBALIZE_IS_A_SIMPLE",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.90,
        description="Erzeuge einfachen IS_A Satz: 'X ist ein/eine Y'",
    )


def create_verbalize_is_a_variant_1_rule() -> ProductionRule:
    """
    VERBALIZE_IS_A_VARIANT_1 (utility: 0.75)

    Condition: IS_A Fakt vorhanden UND bereits ein IS_A Satz erzeugt (Variation)
    Action: Erzeuge "X gehört zu Y"

    Beispiel: "Ein Apfel gehört zu den Früchten."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Nur anwenden, wenn bereits ein IS_A Satz existiert (Variation)
        isa_facts = [
            f for f in state.discourse.pending_facts if f.get("relation_type") == "IS_A"
        ]
        has_isa_sentence = any(
            "ist ein" in s or "ist eine" in s for s in state.text.completed_sentences
        )

        return len(isa_facts) > 0 and has_isa_sentence

    def action(state: ResponseGenerationState) -> None:
        isa_facts = [
            f for f in state.discourse.pending_facts if f.get("relation_type") == "IS_A"
        ]

        if isa_facts:
            fact = isa_facts[0]
            subject = state.primary_goal.target_entity or "Das"
            object_noun = fact.get("object", "etwas")

            # Plural-Form (heuristisch)
            object_plural = (
                object_noun + "n" if not object_noun.endswith("n") else object_noun
            )

            sentence = f"Ein {subject.capitalize()} gehört zu den {object_plural.capitalize()}."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Verbalized IS_A (variant 1): {sentence}")

    return create_production_rule(
        name="VERBALIZE_IS_A_VARIANT_1",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.75,
        description="Variante für IS_A: 'X gehört zu Y'",
    )


def create_verbalize_is_a_variant_2_rule() -> ProductionRule:
    """
    VERBALIZE_IS_A_VARIANT_2 (utility: 0.72)

    Condition: IS_A Fakt vorhanden UND bereits 2+ IS_A Sätze erzeugt
    Action: Erzeuge "X zählt zu Y"

    Beispiel: "Ein Apfel zählt zu den Früchten."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        isa_facts = [
            f for f in state.discourse.pending_facts if f.get("relation_type") == "IS_A"
        ]

        # Zähle IS_A Sätze
        isa_sentence_count = sum(
            1
            for s in state.text.completed_sentences
            if any(marker in s for marker in ["ist ein", "ist eine", "gehört zu"])
        )

        return len(isa_facts) > 0 and isa_sentence_count >= 2

    def action(state: ResponseGenerationState) -> None:
        isa_facts = [
            f for f in state.discourse.pending_facts if f.get("relation_type") == "IS_A"
        ]

        if isa_facts:
            fact = isa_facts[0]
            subject = state.primary_goal.target_entity or "Das"
            object_noun = fact.get("object", "etwas")

            object_plural = (
                object_noun + "n" if not object_noun.endswith("n") else object_noun
            )

            sentence = (
                f"Ein {subject.capitalize()} zählt zu den {object_plural.capitalize()}."
            )

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Verbalized IS_A (variant 2): {sentence}")

    return create_production_rule(
        name="VERBALIZE_IS_A_VARIANT_2",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.72,
        description="Variante für IS_A: 'X zählt zu Y'",
    )


def create_verbalize_has_property_rule() -> ProductionRule:
    """
    VERBALIZE_HAS_PROPERTY (utility: 0.88)

    Condition: HAS_PROPERTY Fakt in pending_facts
    Action: Erzeuge "X ist Y" (Adjektiv)

    Beispiel: {"relation_type": "HAS_PROPERTY", "object": "rot"} → "Äpfel sind rot."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        property_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "HAS_PROPERTY"
        ]
        return len(property_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        property_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "HAS_PROPERTY"
        ]

        if property_facts:
            fact = property_facts[0]
            subject = state.primary_goal.target_entity or "Es"
            property_adj = fact.get("object", "besonders")

            # Plural-Form für Subject (heuristisch)
            subject_plural = subject if subject.endswith("n") else subject + "n"

            # Satz mit Unsicherheitsmarker falls vorhanden
            hedge = ""
            if fact.get("uncertainty_marked", False):
                hedge = fact.get("hedging_phrase", "möglicherweise") + " "

            sentence = f"{subject_plural.capitalize()} sind {hedge}{property_adj}."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Verbalized HAS_PROPERTY: {sentence}")

    return create_production_rule(
        name="VERBALIZE_HAS_PROPERTY",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.88,
        description="Erzeuge Property-Satz: 'X ist Y' (Adjektiv)",
    )


def create_verbalize_capable_of_rule() -> ProductionRule:
    """
    VERBALIZE_CAPABLE_OF (utility: 0.87)

    Condition: CAPABLE_OF Fakt in pending_facts
    Action: Erzeuge "X kann Y"

    Beispiel: {"relation_type": "CAPABLE_OF", "object": "fliegen"} → "Ein Vogel kann fliegen."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        capability_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "CAPABLE_OF"
        ]
        return len(capability_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        capability_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "CAPABLE_OF"
        ]

        if capability_facts:
            fact = capability_facts[0]
            subject = state.primary_goal.target_entity or "Es"
            capability = fact.get("object", "etwas tun")

            sentence = f"Ein {subject.capitalize()} kann {capability}."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Verbalized CAPABLE_OF: {sentence}")

    return create_production_rule(
        name="VERBALIZE_CAPABLE_OF",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.87,
        description="Erzeuge Capability-Satz: 'X kann Y'",
    )


def create_verbalize_located_in_rule() -> ProductionRule:
    """
    VERBALIZE_LOCATED_IN (utility: 0.85)

    Condition: LOCATED_IN Fakt in pending_facts
    Action: Erzeuge "X liegt in Y" oder "X befindet sich in Y"

    Beispiel: {"relation_type": "LOCATED_IN", "object": "europa"} → "Deutschland liegt in Europa."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        location_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "LOCATED_IN"
        ]
        return len(location_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        location_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "LOCATED_IN"
        ]

        if location_facts:
            fact = location_facts[0]
            subject = state.primary_goal.target_entity or "Es"
            location = fact.get("object", "einem Ort")

            sentence = f"{subject.capitalize()} liegt in {location.capitalize()}."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Verbalized LOCATED_IN: {sentence}")

    return create_production_rule(
        name="VERBALIZE_LOCATED_IN",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.85,
        description="Erzeuge Location-Satz: 'X liegt in Y'",
    )


def create_verbalize_part_of_rule() -> ProductionRule:
    """
    VERBALIZE_PART_OF (utility: 0.83)

    Condition: PART_OF Fakt in pending_facts
    Action: Erzeuge "X ist Teil von Y" oder "X gehört zu Y"

    Beispiel: {"relation_type": "PART_OF", "object": "baum"} → "Ein Blatt ist Teil eines Baums."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        partof_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "PART_OF"
        ]
        return len(partof_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        partof_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "PART_OF"
        ]

        if partof_facts:
            fact = partof_facts[0]
            subject = state.primary_goal.target_entity or "Es"
            whole = fact.get("object", "etwas")

            # Artikel für Genitiv (heuristisch)
            article_gen = "eines" if whole.endswith(("er", "el", "en")) else "einer"

            sentence = f"Ein {subject.capitalize()} ist Teil {article_gen} {whole.capitalize()}s."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Verbalized PART_OF: {sentence}")

    return create_production_rule(
        name="VERBALIZE_PART_OF",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.83,
        description="Erzeuge Part-Of-Satz: 'X ist Teil von Y'",
    )


def create_vary_copula_verb_rule() -> ProductionRule:
    """
    VARY_COPULA_VERB (utility: 0.70)

    Condition: Mehrere Sätze mit "ist" bereits vorhanden UND weiterer Fakt zu verbalisieren
    Action: Verwende Variation wie "gilt als", "stellt dar", "bezeichnet"

    Beispiel: "Ein Apfel gilt als gesund."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Zähle "ist"-Vorkommen
        ist_count = sum(s.count(" ist ") for s in state.text.completed_sentences)

        # Nur anwenden wenn viele "ist" Sätze vorhanden
        if ist_count < 2:
            return False

        # Noch Fakten zum Verbalisieren?
        return len(state.discourse.pending_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        if state.discourse.pending_facts:
            fact = state.discourse.pending_facts[0]
            subject = state.primary_goal.target_entity or "Es"

            # Wähle Copula-Variation
            copula_variant = "gilt als"

            # Je nach Relation-Type
            if fact.get("relation_type") == "HAS_PROPERTY":
                property_adj = fact.get("object", "besonders")
                sentence = (
                    f"Ein {subject.capitalize()} {copula_variant} {property_adj}."
                )
            elif fact.get("relation_type") == "IS_A":
                object_noun = fact.get("object", "etwas")
                sentence = f"Ein {subject.capitalize()} {copula_variant} {object_noun.capitalize()}."
            else:
                # Fallback
                sentence = f"Ein {subject.capitalize()} {copula_variant} relevant."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Varied copula verb: {sentence}")

    return create_production_rule(
        name="VARY_COPULA_VERB",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.70,
        description="Variiere Kopula-Verb: 'gilt als', 'stellt dar'",
    )


def create_combine_facts_conjunction_rule() -> ProductionRule:
    """
    COMBINE_FACTS_CONJUNCTION (utility: 0.78)

    Condition: Mindestens 2 ähnliche Fakten (gleicher Typ) in pending_facts
    Action: Kombiniere zu einem Satz mit "und"

    Beispiel: "Äpfel sind rot und süß." (aus 2 HAS_PROPERTY Fakten)
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Zähle Fakten nach Typ
        type_counts = {}
        for fact in state.discourse.pending_facts:
            rel_type = fact.get("relation_type")
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        # Mindestens 2 Fakten gleichen Typs?
        return any(count >= 2 for count in type_counts.values())

    def action(state: ResponseGenerationState) -> None:
        # Finde Typ mit mehreren Fakten
        type_counts = {}
        for fact in state.discourse.pending_facts:
            rel_type = fact.get("relation_type")
            if rel_type not in type_counts:
                type_counts[rel_type] = []
            type_counts[rel_type].append(fact)

        # Wähle Typ mit >= 2 Fakten
        for rel_type, facts in type_counts.items():
            if len(facts) >= 2:
                subject = state.primary_goal.target_entity or "Es"

                if rel_type == "HAS_PROPERTY":
                    # Kombiniere Properties
                    properties = [f.get("object", "X") for f in facts[:2]]
                    combined = " und ".join(properties)

                    subject_plural = subject if subject.endswith("n") else subject + "n"
                    sentence = f"{subject_plural.capitalize()} sind {combined}."

                    state.add_sentence(sentence)
                    state.mention_entity(subject)

                    # Entferne beide Fakten
                    for fact in facts[:2]:
                        state.discourse.pending_facts.remove(fact)

                    logging.debug(f"Combined facts with conjunction: {sentence}")
                    break

    return create_production_rule(
        name="COMBINE_FACTS_CONJUNCTION",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.78,
        description="Kombiniere ähnliche Fakten mit 'und'",
    )


def create_avoid_repetition_rule() -> ProductionRule:
    """
    AVOID_REPETITION (utility: 0.82)

    Condition: Entität wurde bereits erwähnt UND neuer Fakt über sie
    Action: Verwende Pronomen statt vollen Namen

    Beispiel: "Ein Apfel ist rot. Er ist süß." (statt "Ein Apfel ist süß.")
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Wurde target_entity bereits erwähnt?
        target = state.primary_goal.target_entity
        if not target:
            return False

        already_mentioned = target in state.discourse.mentioned_entities

        # Gibt es noch Fakten zu verbalisieren?
        return already_mentioned and len(state.discourse.pending_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        if state.discourse.pending_facts:
            fact = state.discourse.pending_facts[0]
            subject = state.primary_goal.target_entity or "Es"

            # Verwende Pronomen
            pronoun = "Er" if not subject.endswith("e") else "Sie"

            # Je nach Relation
            if fact.get("relation_type") == "HAS_PROPERTY":
                property_adj = fact.get("object", "besonders")
                sentence = f"{pronoun} ist {property_adj}."
            elif fact.get("relation_type") == "CAPABLE_OF":
                capability = fact.get("object", "etwas tun")
                sentence = f"{pronoun} kann {capability}."
            else:
                # Fallback
                object_noun = fact.get("object", "etwas")
                sentence = f"{pronoun} hat Bezug zu {object_noun}."

            state.add_sentence(sentence)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Avoided repetition with pronoun: {sentence}")

    return create_production_rule(
        name="AVOID_REPETITION",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.82,
        description="Vermeide Wiederholung durch Pronomen",
    )


def create_select_formal_style_rule() -> ProductionRule:
    """
    SELECT_FORMAL_STYLE (utility: 0.75)

    Condition: Constraint "style=formal" gesetzt
    Action: Verwende formale Sprache

    Beispiel: "Es handelt sich bei einem Apfel um eine Frucht."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Nur anwenden wenn formal style gefordert
        is_formal = state.constraints.get("style") == "formal"

        return is_formal and len(state.discourse.pending_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        if state.discourse.pending_facts:
            fact = state.discourse.pending_facts[0]
            subject = state.primary_goal.target_entity or "das Objekt"

            if fact.get("relation_type") == "IS_A":
                object_noun = fact.get("object", "etwas")
                sentence = f"Es handelt sich bei einem {subject.capitalize()} um eine {object_noun.capitalize()}."
            elif fact.get("relation_type") == "HAS_PROPERTY":
                property_adj = fact.get("object", "charakteristisch")
                sentence = f"Ein {subject.capitalize()} zeichnet sich durch die Eigenschaft '{property_adj}' aus."
            else:
                object_noun = fact.get("object", "etwas")
                sentence = (
                    f"Ein {subject.capitalize()} steht in Beziehung zu {object_noun}."
                )

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Formal style: {sentence}")

    return create_production_rule(
        name="SELECT_FORMAL_STYLE",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.75,
        description="Verwende formale Sprache",
    )


def create_select_casual_style_rule() -> ProductionRule:
    """
    SELECT_CASUAL_STYLE (utility: 0.73)

    Condition: Constraint "style=casual" gesetzt
    Action: Verwende lockere Sprache

    Beispiel: "Äpfel? Klar, das sind Früchte!"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        is_casual = state.constraints.get("style") == "casual"

        return is_casual and len(state.discourse.pending_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        if state.discourse.pending_facts:
            fact = state.discourse.pending_facts[0]
            subject = state.primary_goal.target_entity or "das"

            if fact.get("relation_type") == "IS_A":
                object_noun = fact.get("object", "etwas")
                sentence = f"{subject.capitalize()}? Klar, das sind {object_noun}!"
            elif fact.get("relation_type") == "HAS_PROPERTY":
                property_adj = fact.get("object", "cool")
                sentence = f"{subject.capitalize()} sind total {property_adj}!"
            else:
                object_noun = fact.get("object", "etwas")
                sentence = f"{subject.capitalize()} haben was mit {object_noun} zu tun."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            state.discourse.pending_facts.remove(fact)

            logging.debug(f"Casual style: {sentence}")

    return create_production_rule(
        name="SELECT_CASUAL_STYLE",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.73,
        description="Verwende lockere Sprache",
    )


def create_add_elaboration_rule() -> ProductionRule:
    """
    ADD_ELABORATION (utility: 0.68)

    Condition: Satz wurde erzeugt UND Elaboration-Flag gesetzt
    Action: Füge Zusatzinformation hinzu

    Beispiel: "Äpfel sind rot. Sie wachsen an Bäumen und sind beliebt."
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur wenn mindestens ein Satz erzeugt wurde
        if len(state.text.completed_sentences) < 1:
            return False

        # Und Elaboration gewünscht
        wants_elaboration = state.constraints.get("elaboration", False)

        return wants_elaboration and not state.constraints.get(
            "elaboration_added", False
        )

    def action(state: ResponseGenerationState) -> None:
        subject = state.primary_goal.target_entity or "Sie"

        # Generische Elaboration
        elaboration = f"{subject.capitalize()} sind in vielen Regionen verbreitet und haben verschiedene Eigenschaften."

        state.add_sentence(elaboration)

        # Markiere als hinzugefügt
        state.constraints["elaboration_added"] = True

        logging.debug(f"Added elaboration: {elaboration}")

    return create_production_rule(
        name="ADD_ELABORATION",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.68,
        description="Füge Zusatzinformation hinzu",
    )


def create_compress_similar_facts_rule() -> ProductionRule:
    """
    COMPRESS_SIMILAR_FACTS (utility: 0.80)

    Condition: Mehrere Fakten mit ähnlichen Objekten UND noch nicht komprimiert
    Action: Fasse zu einem Satz zusammen

    Beispiel: "Äpfel können rot, grün oder gelb sein." (aus 3 HAS_PROPERTY Fakten)
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Bereits komprimiert?
        if state.constraints.get("facts_compressed", False):
            return False

        # Mehrere HAS_PROPERTY Fakten?
        property_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "HAS_PROPERTY"
        ]

        return len(property_facts) >= 3

    def action(state: ResponseGenerationState) -> None:
        property_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("relation_type") == "HAS_PROPERTY"
        ]

        if len(property_facts) >= 3:
            subject = state.primary_goal.target_entity or "Sie"
            subject_plural = subject if subject.endswith("n") else subject + "n"

            # Extrahiere Properties
            properties = [f.get("object", "X") for f in property_facts[:3]]

            # Kombiniere mit "oder"
            combined = ", ".join(properties[:-1]) + " oder " + properties[-1]

            sentence = f"{subject_plural.capitalize()} können {combined} sein."

            state.add_sentence(sentence)
            state.mention_entity(subject)

            # Entferne alle verwendeten Fakten
            for fact in property_facts[:3]:
                state.discourse.pending_facts.remove(fact)

            # Markiere als komprimiert
            state.constraints["facts_compressed"] = True

            logging.debug(f"Compressed facts: {sentence}")

    return create_production_rule(
        name="COMPRESS_SIMILAR_FACTS",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.80,
        description="Fasse ähnliche Fakten zusammen",
    )


def create_finish_lexicalization_rule() -> ProductionRule:
    """
    FINISH_LEXICALIZATION (utility: 0.65)

    Condition: Keine pending_facts mehr ODER max. Satzanzahl erreicht
    Action: Markiere Lexicalization als abgeschlossen

    Diese Regel beendet die Lexicalization Phase.
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Beende wenn:
        # 1. Content Selection abgeschlossen UND keine pending_facts mehr
        # 2. ODER: Genug Sätze erzeugt (>= 3)

        content_done = state.constraints.get("content_selection_finished", False)

        if not content_done:
            return False

        no_pending = len(state.discourse.pending_facts) == 0
        enough_sentences = len(state.text.completed_sentences) >= 3

        return no_pending or enough_sentences

    def action(state: ResponseGenerationState) -> None:
        # Markiere Lexicalization als abgeschlossen
        state.constraints["lexicalization_finished"] = True

        logging.info(
            f"Lexicalization finished: {len(state.text.completed_sentences)} sentences generated"
        )

    return create_production_rule(
        name="FINISH_LEXICALIZATION",
        category=RuleCategory.LEXICALIZATION,
        condition=condition,
        action=action,
        utility=0.65,
        description="Beendet Lexicalization Phase",
    )


# ============================================================================
# PHASE 3: Discourse Management Rules (Response Structuring & Qualification)
# ============================================================================


def create_introduce_with_context_rule() -> ProductionRule:
    """
    INTRODUCE_WITH_CONTEXT (utility: 0.85)

    Condition: Noch kein Satz erzeugt UND komplexe Frage erkannt
    Action: Füge kontextuelle Einleitung hinzu: "Um das zu beantworten..."

    Beispiel: "Um das zu beantworten: Ein Apfel ist eine Frucht."
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur am Anfang (kein Satz erzeugt)
        if len(state.text.completed_sentences) > 0:
            return False

        # Lexicalization muss begonnen haben
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Nur bei komplexer Frage (z.B. Multi-Hop)
        is_complex = state.constraints.get("complex_query", False)

        return is_complex

    def action(state: ResponseGenerationState) -> None:
        intro = "Um das zu beantworten:"

        # Speichere Intro in discourse markers
        state.discourse.discourse_markers_used.append("introduce_context")

        # Füge als Text-Fragment hinzu (nicht als vollständiger Satz)
        state.text.sentence_fragments.append(intro)

        logging.debug(f"Added contextual introduction: {intro}")

    return create_production_rule(
        name="INTRODUCE_WITH_CONTEXT",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.85,
        description="Kontextuelle Einleitung: 'Um das zu beantworten...'",
    )


def create_introduce_simple_rule() -> ProductionRule:
    """
    INTRODUCE_SIMPLE (utility: 0.80)

    Condition: Noch kein Satz erzeugt UND einfache Frage
    Action: Füge einfache Einleitung hinzu: "Das weiß ich über X:"

    Beispiel: "Das weiß ich über Äpfel: Sie sind Früchte."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if len(state.text.completed_sentences) > 0:
            return False

        if not state.constraints.get("content_selection_finished", False):
            return False

        # Nur bei einfacher Frage (kein complex_query Flag)
        is_simple = not state.constraints.get("complex_query", False)

        # Noch kein Discourse Marker gesetzt
        no_intro = "introduce_context" not in state.discourse.discourse_markers_used

        return is_simple and no_intro

    def action(state: ResponseGenerationState) -> None:
        target = state.primary_goal.target_entity or "das Thema"

        intro = f"Das weiß ich über {target.capitalize()}:"

        state.discourse.discourse_markers_used.append("introduce_simple")

        state.text.sentence_fragments.append(intro)

        logging.debug(f"Added simple introduction: {intro}")

    return create_production_rule(
        name="INTRODUCE_SIMPLE",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.80,
        description="Einfache Einleitung: 'Das weiß ich über X:'",
    )


def create_signal_uncertainty_rule() -> ProductionRule:
    """
    SIGNAL_UNCERTAINTY (utility: 0.90)

    Condition: Fakt mit confidence < 0.7 in pending_facts UND noch nicht markiert
    Action: Füge "ich vermute" oder ähnliches hinzu

    Beispiel: "Ich vermute, dass Äpfel grün sein können."
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur wenn Sätze erzeugt werden (Lexicalization läuft)
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Gibt es unsichere Fakten?
        uncertain_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 1.0) < 0.70
            and not f.get("uncertainty_signaled", False)
        ]

        return len(uncertain_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        uncertain_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 1.0) < 0.70
            and not f.get("uncertainty_signaled", False)
        ]

        if uncertain_facts:
            # Markiere ersten unsicheren Fakt
            fact = uncertain_facts[0]
            fact["uncertainty_signaled"] = True
            fact["uncertainty_phrase"] = "ich vermute"

            state.discourse.discourse_markers_used.append("uncertainty")

            logging.debug(f"Signaled uncertainty for fact: {fact}")

    return create_production_rule(
        name="SIGNAL_UNCERTAINTY",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.90,
        description="Signalisiere Unsicherheit: 'ich vermute'",
    )


def create_signal_high_confidence_rule() -> ProductionRule:
    """
    SIGNAL_HIGH_CONFIDENCE (utility: 0.88)

    Condition: Fakt mit confidence >= 0.95 in pending_facts
    Action: Füge "Mit hoher Sicherheit..." hinzu

    Beispiel: "Mit hoher Sicherheit ist ein Apfel eine Frucht."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Gibt es hochsichere Fakten?
        high_conf_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 0.0) >= 0.95
            and not f.get("high_confidence_signaled", False)
        ]

        return len(high_conf_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        high_conf_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 0.0) >= 0.95
            and not f.get("high_confidence_signaled", False)
        ]

        if high_conf_facts:
            fact = high_conf_facts[0]
            fact["high_confidence_signaled"] = True
            fact["confidence_phrase"] = "Mit hoher Sicherheit"

            state.discourse.discourse_markers_used.append("high_confidence")

            logging.debug(f"Signaled high confidence for fact: {fact}")

    return create_production_rule(
        name="SIGNAL_HIGH_CONFIDENCE",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.88,
        description="Signalisiere hohe Sicherheit",
    )


def create_explain_reasoning_path_rule() -> ProductionRule:
    """
    EXPLAIN_REASONING_PATH (utility: 0.82)

    Condition: Multi-Hop Fakt (hop_count > 1) in pending_facts
    Action: Füge Erklärung hinzu: "Durch komplexe Schlussfolgerung..."

    Beispiel: "Durch komplexe Schlussfolgerung kann ich ableiten, dass..."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Gibt es Multi-Hop Fakten?
        multihop_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("hop_count", 1) > 1 and not f.get("reasoning_explained", False)
        ]

        return len(multihop_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        multihop_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("hop_count", 1) > 1 and not f.get("reasoning_explained", False)
        ]

        if multihop_facts:
            fact = multihop_facts[0]
            fact["reasoning_explained"] = True
            fact["reasoning_phrase"] = "Durch komplexe Schlussfolgerung"

            state.discourse.discourse_markers_used.append("reasoning_path")

            logging.debug(f"Explained reasoning path for multi-hop fact: {fact}")

    return create_production_rule(
        name="EXPLAIN_REASONING_PATH",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.82,
        description="Erkläre Reasoning-Pfad bei Multi-Hop",
    )


def create_mark_hypothesis_rule() -> ProductionRule:
    """
    MARK_HYPOTHESIS (utility: 0.85)

    Condition: Fakt aus abduktiver Schlussfolgerung (type="hypothesis")
    Action: Füge Markierung hinzu: "Basierend auf abduktiver Schlussfolgerung..."

    Beispiel: "Basierend auf abduktiver Schlussfolgerung vermute ich, dass..."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Gibt es Hypothesen-Fakten?
        hypothesis_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("type") == "hypothesis" and not f.get("hypothesis_marked", False)
        ]

        return len(hypothesis_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        hypothesis_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("type") == "hypothesis" and not f.get("hypothesis_marked", False)
        ]

        if hypothesis_facts:
            fact = hypothesis_facts[0]
            fact["hypothesis_marked"] = True
            fact["hypothesis_phrase"] = "Basierend auf abduktiver Schlussfolgerung"

            state.discourse.discourse_markers_used.append("hypothesis")

            logging.debug(f"Marked hypothesis: {fact}")

    return create_production_rule(
        name="MARK_HYPOTHESIS",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.85,
        description="Markiere Hypothese aus Abduktion",
    )


def create_add_confidence_qualifier_rule() -> ProductionRule:
    """
    ADD_CONFIDENCE_QUALIFIER (utility: 0.83)

    Condition: Medium-Confidence Fakt (0.70 <= conf < 0.85) UND noch kein Qualifier
    Action: Füge Qualifier hinzu: "wahrscheinlich", "vermutlich"

    Beispiel: "Ein Apfel ist wahrscheinlich rot."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Medium-Confidence Fakten ohne Qualifier
        medium_conf = [
            f
            for f in state.discourse.pending_facts
            if 0.70 <= f.get("confidence", 1.0) < 0.85
            and not f.get("qualifier_added", False)
        ]

        return len(medium_conf) > 0

    def action(state: ResponseGenerationState) -> None:
        medium_conf = [
            f
            for f in state.discourse.pending_facts
            if 0.70 <= f.get("confidence", 1.0) < 0.85
            and not f.get("qualifier_added", False)
        ]

        if medium_conf:
            fact = medium_conf[0]
            fact["qualifier_added"] = True
            fact["qualifier"] = "wahrscheinlich"

            state.discourse.discourse_markers_used.append("confidence_qualifier")

            logging.debug(f"Added confidence qualifier to fact: {fact}")

    return create_production_rule(
        name="ADD_CONFIDENCE_QUALIFIER",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.83,
        description="Füge Confidence-Qualifier hinzu",
    )


def create_mention_evidence_source_rule() -> ProductionRule:
    """
    MENTION_EVIDENCE_SOURCE (utility: 0.70)

    Condition: Fakt mit expliziter Source UND source_mention erwünscht
    Action: Erwähne Quelle: "Laut meinem Wissen...", "Aus der Quelle X..."

    Beispiel: "Laut meinem Wissen ist ein Apfel eine Frucht."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Quelle erwünscht?
        wants_source = state.constraints.get("mention_source", False)

        if not wants_source:
            return False

        # Fakten mit Source
        facts_with_source = [
            f
            for f in state.discourse.pending_facts
            if f.get("source") and not f.get("source_mentioned", False)
        ]

        return len(facts_with_source) > 0

    def action(state: ResponseGenerationState) -> None:
        facts_with_source = [
            f
            for f in state.discourse.pending_facts
            if f.get("source") and not f.get("source_mentioned", False)
        ]

        if facts_with_source:
            fact = facts_with_source[0]
            fact["source_mentioned"] = True

            source_name = fact.get("source", "meinem Wissen")
            fact["source_phrase"] = f"Laut {source_name}"

            state.discourse.discourse_markers_used.append("evidence_source")

            logging.debug(f"Mentioned evidence source: {fact}")

    return create_production_rule(
        name="MENTION_EVIDENCE_SOURCE",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.70,
        description="Erwähne Evidenz-Quelle",
    )


def create_structure_multi_part_answer_rule() -> ProductionRule:
    """
    STRUCTURE_MULTI_PART_ANSWER (utility: 0.77)

    Condition: Mehrere verschiedene Relation-Types zu verbalisieren
    Action: Strukturiere in Absätze: "Erstens..., Zweitens..."

    Beispiel: "Erstens: Ein Apfel ist eine Frucht. Zweitens: Äpfel sind rot."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("content_selection_finished", False):
            return False

        # Mehrere Relation-Types?
        relation_types = set(
            f.get("relation_type") for f in state.discourse.pending_facts
        )

        # Noch nicht strukturiert
        already_structured = state.constraints.get("multi_part_structured", False)

        return len(relation_types) >= 3 and not already_structured

    def action(state: ResponseGenerationState) -> None:
        # Markiere als strukturiert
        state.constraints["multi_part_structured"] = True

        state.discourse.discourse_markers_used.append("multi_part_structure")

        logging.debug("Structured multi-part answer")

    return create_production_rule(
        name="STRUCTURE_MULTI_PART_ANSWER",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.77,
        description="Strukturiere mehrteilige Antwort",
    )


def create_add_transition_rule() -> ProductionRule:
    """
    ADD_TRANSITION (utility: 0.72)

    Condition: Mindestens 2 Sätze erzeugt UND Themenwechsel
    Action: Füge Übergang hinzu: "Außerdem", "Darüber hinaus", "Zusätzlich"

    Beispiel: "Ein Apfel ist eine Frucht. Außerdem sind Äpfel rot."
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Mindestens 2 Sätze vorhanden
        if len(state.text.completed_sentences) < 2:
            return False

        # Noch Fakten zu verbalisieren
        if len(state.discourse.pending_facts) == 0:
            return False

        # Noch keine Transition hinzugefügt (in diesem Zyklus)
        recent_transitions = [
            m for m in state.discourse.discourse_markers_used if m == "transition"
        ]

        return len(recent_transitions) < 2  # Max 2 Transitions

    def action(state: ResponseGenerationState) -> None:
        # Wähle Transition-Wort
        transitions = ["Außerdem", "Darüber hinaus", "Zusätzlich"]

        transition_count = len(
            [m for m in state.discourse.discourse_markers_used if m == "transition"]
        )
        transition_word = transitions[min(transition_count, len(transitions) - 1)]

        # Füge als Fragment hinzu
        state.text.sentence_fragments.append(transition_word)

        state.discourse.discourse_markers_used.append("transition")

        logging.debug(f"Added transition: {transition_word}")

    return create_production_rule(
        name="ADD_TRANSITION",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.72,
        description="Füge Übergang hinzu: 'Außerdem', 'Darüber hinaus'",
    )


def create_conclude_answer_rule() -> ProductionRule:
    """
    CONCLUDE_ANSWER (utility: 0.75)

    Condition: Lexicalization abgeschlossen UND noch kein Abschluss
    Action: Füge Abschluss hinzu: "Zusammenfassend lässt sich sagen..."

    Beispiel: "Zusammenfassend lässt sich sagen, dass Äpfel Früchte sind."
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Lexicalization abgeschlossen?
        lexicalization_done = state.constraints.get("lexicalization_finished", False)

        if not lexicalization_done:
            return False

        # Noch kein Abschluss
        already_concluded = "conclusion" in state.discourse.discourse_markers_used

        # Mindestens 2 Sätze vorhanden
        enough_sentences = len(state.text.completed_sentences) >= 2

        return not already_concluded and enough_sentences

    def action(state: ResponseGenerationState) -> None:
        conclusion = "Zusammenfassend lässt sich sagen"

        state.text.sentence_fragments.append(conclusion)

        state.discourse.discourse_markers_used.append("conclusion")

        logging.debug(f"Added conclusion: {conclusion}")

    return create_production_rule(
        name="CONCLUDE_ANSWER",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.75,
        description="Füge Abschluss hinzu",
    )


def create_offer_elaboration_rule() -> ProductionRule:
    """
    OFFER_ELABORATION (utility: 0.60)

    Condition: Antwort abgeschlossen UND Elaboration möglich
    Action: Füge Angebot hinzu: "Möchten Sie mehr erfahren?"

    Beispiel: "Äpfel sind Früchte. Möchten Sie mehr über Äpfel erfahren?"
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Lexicalization abgeschlossen
        lexicalization_done = state.constraints.get("lexicalization_finished", False)

        if not lexicalization_done:
            return False

        # Noch kein Angebot
        already_offered = "elaboration_offer" in state.discourse.discourse_markers_used

        # Constraint erlaubt Angebot
        allow_offer = state.constraints.get("allow_elaboration_offer", True)

        return not already_offered and allow_offer

    def action(state: ResponseGenerationState) -> None:
        target = state.primary_goal.target_entity or "das Thema"

        offer = f"Möchten Sie mehr über {target} erfahren?"

        state.add_sentence(offer)

        state.discourse.discourse_markers_used.append("elaboration_offer")

        # Markiere als abgeschlossen
        state.primary_goal.completed = True

        logging.debug(f"Offered elaboration: {offer}")

    return create_production_rule(
        name="OFFER_ELABORATION",
        category=RuleCategory.DISCOURSE,
        condition=condition,
        action=action,
        utility=0.60,
        description="Biete weitere Elaboration an",
    )


# ============================================================================
# Convenience Functions: Load PHASE 3 Rules
# ============================================================================


def create_all_lexicalization_rules() -> List[ProductionRule]:
    """
    Erstellt alle 15 Lexicalization Rules (Fact → Natural Language).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return [
        # Basic Verbalization (7)
        create_verbalize_is_a_simple_rule(),
        create_verbalize_is_a_variant_1_rule(),
        create_verbalize_is_a_variant_2_rule(),
        create_verbalize_has_property_rule(),
        create_verbalize_capable_of_rule(),
        create_verbalize_located_in_rule(),
        create_verbalize_part_of_rule(),
        # Stylistic Variation (8)
        create_vary_copula_verb_rule(),
        create_combine_facts_conjunction_rule(),
        create_avoid_repetition_rule(),
        create_select_formal_style_rule(),
        create_select_casual_style_rule(),
        create_add_elaboration_rule(),
        create_compress_similar_facts_rule(),
        create_finish_lexicalization_rule(),
    ]


def create_all_discourse_management_rules() -> List[ProductionRule]:
    """
    Erstellt alle 12 Discourse Management Rules (Response Structuring & Qualification).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return [
        # Introduction (2)
        create_introduce_with_context_rule(),
        create_introduce_simple_rule(),
        # Confidence Signaling (5)
        create_signal_uncertainty_rule(),
        create_signal_high_confidence_rule(),
        create_explain_reasoning_path_rule(),
        create_mark_hypothesis_rule(),
        create_add_confidence_qualifier_rule(),
        # Source & Structure (5)
        create_mention_evidence_source_rule(),
        create_structure_multi_part_answer_rule(),
        create_add_transition_rule(),
        create_conclude_answer_rule(),
        create_offer_elaboration_rule(),
    ]


def create_all_phase3_rules() -> List[ProductionRule]:
    """
    Erstellt alle 27 PHASE 3 Rules (Lexicalization + Discourse Management).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return create_all_lexicalization_rules() + create_all_discourse_management_rules()


def create_all_syntactic_realization_rules() -> List[ProductionRule]:
    """
    Erstellt alle 12 Syntactic Realization Rules (Grammatical Correctness).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return [
        # Article Rules (3)
        create_add_article_nominative_rule(),
        create_add_article_accusative_rule(),
        create_add_article_dative_rule(),
        # Capitalization & Punctuation (3)
        create_capitalize_sentence_start_rule(),
        create_capitalize_nouns_rule(),
        create_add_period_rule(),
        # Syntax Correctness (5)
        create_add_comma_conjunction_rule(),
        create_fix_verb_agreement_rule(),
        create_ensure_gender_agreement_rule(),
        create_insert_preposition_rule(),
        create_order_sentence_elements_rule(),
        # Finish (1)
        create_finish_sentence_rule(),
    ]


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


# ============================================================================
# PHASE 4: Syntactic Realization Rules (Grammatical Correctness)
# ============================================================================


def create_add_article_nominative_rule() -> ProductionRule:
    """
    ADD_ARTICLE_NOMINATIVE (utility: 0.99)

    Condition: Satzfragment enthält Nomen ohne Artikel UND Nominativ-Kontext
    Action: Füge korrekten Artikel im Nominativ hinzu (der/die/das/ein/eine)

    Beispiel: "Apfel ist Frucht" → "Ein Apfel ist eine Frucht"
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur anwenden, wenn Lexicalization abgeschlossen
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe ob Sätze ohne Artikel existieren
        for sentence in state.text.completed_sentences:
            # Einfache Heuristik: Satz beginnt mit Großbuchstaben direkt gefolgt von Verb
            # z.B. "Apfel ist eine Frucht" (fehlt "Ein")
            if sentence and sentence[0].isupper():
                words = sentence.split()
                if len(words) >= 2:
                    # Kein Artikel am Anfang
                    first_word = words[0].lower()
                    if first_word not in [
                        "der",
                        "die",
                        "das",
                        "ein",
                        "eine",
                        "einer",
                        "einem",
                        "einen",
                    ]:
                        # Noch nicht korrigiert
                        if not state.constraints.get(
                            f"article_nominative_{sentence[:20]}", False
                        ):
                            return True

        return False

    def action(state: ResponseGenerationState) -> None:
        # Finde ersten Satz ohne Artikel
        for i, sentence in enumerate(state.text.completed_sentences):
            if sentence and sentence[0].isupper():
                words = sentence.split()
                if len(words) >= 2:
                    first_word = words[0].lower()
                    if first_word not in [
                        "der",
                        "die",
                        "das",
                        "ein",
                        "eine",
                        "einer",
                        "einem",
                        "einen",
                    ]:
                        # Füge Artikel hinzu (heuristisch: "Ein" oder "Eine")
                        noun = words[0]
                        article = (
                            "Eine"
                            if noun.endswith(("e", "ung", "heit", "keit", "ion"))
                            else "Ein"
                        )

                        # Konstruiere neuen Satz
                        new_sentence = f"{article} {sentence}"

                        # Ersetze alten Satz
                        state.text.completed_sentences[i] = new_sentence

                        # Markiere als korrigiert
                        state.constraints[f"article_nominative_{sentence[:20]}"] = True

                        logging.debug(
                            f"Added nominative article: {sentence} → {new_sentence}"
                        )
                        break

    return create_production_rule(
        name="ADD_ARTICLE_NOMINATIVE",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.99,
        description="Füge Artikel im Nominativ hinzu",
    )


def create_add_article_accusative_rule() -> ProductionRule:
    """
    ADD_ARTICLE_ACCUSATIVE (utility: 0.99)

    Condition: Objekt ohne Artikel in Akkusativ-Position
    Action: Füge korrekten Artikel im Akkusativ hinzu (den/die/das/einen/eine)

    Beispiel: "Ich sehe Apfel" → "Ich sehe einen Apfel"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe auf Akkusativ-Kontext (z.B. nach "sehe", "habe", "kenne")
        for sentence in state.text.completed_sentences:
            if any(
                verb in sentence.lower()
                for verb in ["sehe", "habe", "kenne", "liebe", "esse"]
            ):
                # Noch nicht korrigiert
                if not state.constraints.get(
                    f"article_accusative_{sentence[:20]}", False
                ):
                    return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            if any(
                verb in sentence.lower()
                for verb in ["sehe", "habe", "kenne", "liebe", "esse"]
            ):
                # Einfache Heuristik: Füge "einen/eine" nach Verb hinzu falls fehlend
                words = sentence.split()

                # Suche Verb-Position
                for j, word in enumerate(words):
                    if word.lower() in ["sehe", "habe", "kenne", "liebe", "esse"]:
                        # Prüfe ob nächstes Wort Artikel ist
                        if j + 1 < len(words):
                            next_word = words[j + 1].lower()
                            if next_word not in [
                                "den",
                                "die",
                                "das",
                                "einen",
                                "eine",
                                "ein",
                            ]:
                                # Füge Artikel hinzu
                                noun = words[j + 1]
                                article = (
                                    "eine"
                                    if noun.lower().endswith(("e", "ung", "heit"))
                                    else "einen"
                                )

                                # Rekonstruiere Satz
                                new_words = words[: j + 1] + [article] + words[j + 1 :]
                                new_sentence = " ".join(new_words)

                                state.text.completed_sentences[i] = new_sentence
                                state.constraints[
                                    f"article_accusative_{sentence[:20]}"
                                ] = True

                                logging.debug(
                                    f"Added accusative article: {sentence} → {new_sentence}"
                                )
                                break
                break

    return create_production_rule(
        name="ADD_ARTICLE_ACCUSATIVE",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.99,
        description="Füge Artikel im Akkusativ hinzu",
    )


def create_add_article_dative_rule() -> ProductionRule:
    """
    ADD_ARTICLE_DATIVE (utility: 0.98)

    Condition: Dativ-Objekt ohne Artikel
    Action: Füge korrekten Artikel im Dativ hinzu (dem/der/einem/einer)

    Beispiel: "Ich gebe Apfel" → "Ich gebe dem Apfel"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe auf Dativ-Kontext (z.B. nach "gebe", "helfe", "folge")
        for sentence in state.text.completed_sentences:
            if any(
                verb in sentence.lower()
                for verb in ["gebe", "helfe", "folge", "gehört"]
            ):
                if not state.constraints.get(f"article_dative_{sentence[:20]}", False):
                    return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            if any(
                verb in sentence.lower()
                for verb in ["gebe", "helfe", "folge", "gehört"]
            ):
                words = sentence.split()

                for j, word in enumerate(words):
                    if word.lower() in ["gebe", "helfe", "folge", "gehört"]:
                        if j + 1 < len(words):
                            next_word = words[j + 1].lower()
                            if next_word not in ["dem", "der", "den", "einem", "einer"]:
                                noun = words[j + 1]
                                article = (
                                    "einer"
                                    if noun.lower().endswith(("e", "ung", "heit"))
                                    else "einem"
                                )

                                new_words = words[: j + 1] + [article] + words[j + 1 :]
                                new_sentence = " ".join(new_words)

                                state.text.completed_sentences[i] = new_sentence
                                state.constraints[f"article_dative_{sentence[:20]}"] = (
                                    True
                                )

                                logging.debug(
                                    f"Added dative article: {sentence} → {new_sentence}"
                                )
                                break
                break

    return create_production_rule(
        name="ADD_ARTICLE_DATIVE",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.98,
        description="Füge Artikel im Dativ hinzu",
    )


def create_capitalize_sentence_start_rule() -> ProductionRule:
    """
    CAPITALIZE_SENTENCE_START (utility: 1.0)

    Condition: Satz beginnt mit Kleinbuchstaben
    Action: Großschreibung des ersten Wortes

    Beispiel: "ein Apfel ist rot" → "Ein Apfel ist rot"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe ob Sätze mit Kleinbuchstaben beginnen
        for sentence in state.text.completed_sentences:
            if sentence and sentence[0].islower():
                return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            if sentence and sentence[0].islower():
                # Kapitalisiere ersten Buchstaben
                new_sentence = sentence[0].upper() + sentence[1:]
                state.text.completed_sentences[i] = new_sentence

                logging.debug(
                    f"Capitalized sentence start: {sentence} → {new_sentence}"
                )

    return create_production_rule(
        name="CAPITALIZE_SENTENCE_START",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=1.0,
        description="Großschreibung am Satzanfang",
    )


def create_capitalize_nouns_rule() -> ProductionRule:
    """
    CAPITALIZE_NOUNS (utility: 0.99)

    Condition: Deutsche Nomen in Sätzen ohne Großschreibung
    Action: Kapitalisiere alle Nomen

    Beispiel: "der apfel ist rot" → "der Apfel ist rot"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe ob Nomen klein geschrieben sind
        # Einfache Heuristik: Wörter nach Artikeln sollten groß sein
        for sentence in state.text.completed_sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if word.lower() in [
                    "der",
                    "die",
                    "das",
                    "ein",
                    "eine",
                    "den",
                    "dem",
                    "des",
                ]:
                    if i + 1 < len(words) and words[i + 1][0].islower():
                        return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            words = sentence.split()
            modified = False

            for j, word in enumerate(words):
                if word.lower() in [
                    "der",
                    "die",
                    "das",
                    "ein",
                    "eine",
                    "den",
                    "dem",
                    "des",
                ]:
                    if j + 1 < len(words) and words[j + 1][0].islower():
                        # Kapitalisiere Nomen
                        words[j + 1] = words[j + 1].capitalize()
                        modified = True

            if modified:
                new_sentence = " ".join(words)
                state.text.completed_sentences[i] = new_sentence

                logging.debug(f"Capitalized nouns: {sentence} → {new_sentence}")

    return create_production_rule(
        name="CAPITALIZE_NOUNS",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.99,
        description="Großschreibung deutscher Nomen",
    )


def create_add_period_rule() -> ProductionRule:
    """
    ADD_PERIOD (utility: 0.98)

    Condition: Satz endet nicht mit Interpunktion
    Action: Füge Punkt am Satzende hinzu

    Beispiel: "Ein Apfel ist rot" → "Ein Apfel ist rot."
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe ob Sätze ohne Punkt enden
        for sentence in state.text.completed_sentences:
            if sentence and not sentence.endswith((".", "!", "?")):
                return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            if sentence and not sentence.endswith((".", "!", "?")):
                new_sentence = sentence + "."
                state.text.completed_sentences[i] = new_sentence

                logging.debug(f"Added period: {sentence} → {new_sentence}")

    return create_production_rule(
        name="ADD_PERIOD",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.98,
        description="Füge Punkt am Satzende hinzu",
    )


def create_add_comma_conjunction_rule() -> ProductionRule:
    """
    ADD_COMMA_CONJUNCTION (utility: 0.95)

    Condition: Konjunktion ohne Komma davor (z.B. "aber", "denn", "doch")
    Action: Füge Komma vor Konjunktion hinzu

    Beispiel: "Äpfel sind rot aber Birnen sind grün" → "Äpfel sind rot, aber Birnen sind grün"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe auf Konjunktionen ohne Komma
        for sentence in state.text.completed_sentences:
            for conj in [" aber ", " denn ", " doch ", " sondern ", " jedoch "]:
                if conj in sentence.lower() and f",{conj}" not in sentence.lower():
                    return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            modified_sentence = sentence

            for conj in [" aber ", " denn ", " doch ", " sondern ", " jedoch "]:
                if (
                    conj in modified_sentence.lower()
                    and f",{conj}" not in modified_sentence.lower()
                ):
                    # Füge Komma vor Konjunktion hinzu
                    # Finde Position (case-insensitive)
                    lower_sent = modified_sentence.lower()
                    pos = lower_sent.find(conj)

                    if pos > 0:
                        modified_sentence = (
                            modified_sentence[:pos] + "," + modified_sentence[pos:]
                        )

            if modified_sentence != sentence:
                state.text.completed_sentences[i] = modified_sentence
                logging.debug(
                    f"Added comma before conjunction: {sentence} → {modified_sentence}"
                )

    return create_production_rule(
        name="ADD_COMMA_CONJUNCTION",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.95,
        description="Füge Komma vor Konjunktion hinzu",
    )


def create_fix_verb_agreement_rule() -> ProductionRule:
    """
    FIX_VERB_AGREEMENT (utility: 0.97)

    Condition: Verb stimmt nicht mit Subjekt überein
    Action: Korrigiere Verb-Form (Singular/Plural)

    Beispiel: "Äpfel ist rot" → "Äpfel sind rot"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe auf falsche Verb-Kongruenz
        for sentence in state.text.completed_sentences:
            # Suche nach Plural-Nomen mit "ist" (z.B. "Äpfel ist" oder "äpfel ist")
            words = sentence.split()
            for i, word in enumerate(words):
                # Prüfe ob Wort plural ist (endet auf "n", "en", oder Umlaut+n)
                word_lower = word.lower()
                is_plural = word_lower.endswith(("en", "äpfel", "birnen")) or (
                    word_lower.endswith("n") and len(word) > 2
                )

                if is_plural and i + 1 < len(words) and words[i + 1].lower() == "ist":
                    # Plural-Subjekt mit Singular-Verb gefunden
                    return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            words = sentence.split()
            modified = False

            for j, word in enumerate(words):
                word_lower = word.lower()
                is_plural = word_lower.endswith(("en", "äpfel", "birnen")) or (
                    word_lower.endswith("n") and len(word) > 2
                )

                if is_plural and j + 1 < len(words) and words[j + 1].lower() == "ist":
                    # Ersetze "ist" durch "sind"
                    words[j + 1] = "sind"
                    modified = True
                    break

            if modified:
                new_sentence = " ".join(words)
                state.text.completed_sentences[i] = new_sentence

                logging.debug(f"Fixed verb agreement: {sentence} → {new_sentence}")
                break

    return create_production_rule(
        name="FIX_VERB_AGREEMENT",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.97,
        description="Korrigiere Verb-Subjekt-Kongruenz",
    )


def create_ensure_gender_agreement_rule() -> ProductionRule:
    """
    ENSURE_GENDER_AGREEMENT (utility: 0.96)

    Condition: Artikel und Adjektiv stimmen nicht im Genus überein
    Action: Korrigiere Genus-Kongruenz

    Beispiel: "der rote Haus" → "das rote Haus"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe auf Genus-Inkongruenz (vereinfacht)
        # Dies ist eine sehr komplexe Aufgabe, hier nur Beispiel-Heuristik
        for sentence in state.text.completed_sentences:
            # Beispiel: "der" sollte nicht vor femininen/neutralen Nomen stehen
            words = sentence.split()
            for i, word in enumerate(words):
                if word.lower() == "der" and i + 1 < len(words):
                    next_word = words[i + 1].lower()
                    # Feminine Endungen
                    if next_word.endswith(
                        ("e", "ung", "heit", "keit", "ion", "schaft")
                    ):
                        return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            words = sentence.split()
            modified = False

            for j, word in enumerate(words):
                if word.lower() == "der" and j + 1 < len(words):
                    next_word = words[j + 1].lower()
                    if next_word.endswith(
                        ("e", "ung", "heit", "keit", "ion", "schaft")
                    ):
                        # Ersetze "der" durch "die"
                        words[j] = "die" if word.islower() else "Die"
                        modified = True

            if modified:
                new_sentence = " ".join(words)
                state.text.completed_sentences[i] = new_sentence

                logging.debug(f"Fixed gender agreement: {sentence} → {new_sentence}")

    return create_production_rule(
        name="ENSURE_GENDER_AGREEMENT",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.96,
        description="Korrigiere Genus-Kongruenz",
    )


def create_insert_preposition_rule() -> ProductionRule:
    """
    INSERT_PREPOSITION (utility: 0.94)

    Condition: Fehlende Präposition vor lokalem/temporalem Ausdruck
    Action: Füge korrekte Präposition ein

    Beispiel: "Berlin liegt Deutschland" → "Berlin liegt in Deutschland"
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe auf fehlende Präpositionen
        for sentence in state.text.completed_sentences:
            # "liegt" sollte "in" haben
            if " liegt " in sentence.lower() and " in " not in sentence.lower():
                return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            if " liegt " in sentence.lower() and " in " not in sentence.lower():
                words = sentence.split()

                for j, word in enumerate(words):
                    if word.lower() == "liegt" and j + 1 < len(words):
                        # Füge "in" nach "liegt" ein
                        new_words = words[: j + 1] + ["in"] + words[j + 1 :]
                        new_sentence = " ".join(new_words)

                        state.text.completed_sentences[i] = new_sentence

                        logging.debug(
                            f"Inserted preposition: {sentence} → {new_sentence}"
                        )
                        break

    return create_production_rule(
        name="INSERT_PREPOSITION",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.94,
        description="Füge fehlende Präposition ein",
    )


def create_order_sentence_elements_rule() -> ProductionRule:
    """
    ORDER_SENTENCE_ELEMENTS (utility: 0.93)

    Condition: Falsche Wortstellung (z.B. Verb an falscher Position)
    Action: Korrigiere Satzgliedstellung nach deutschen Regeln

    Beispiel: "Ist ein Apfel eine Frucht" → "Ein Apfel ist eine Frucht"

    Note: Sehr vereinfacht, da deutsche Syntax komplex ist
    """

    def condition(state: ResponseGenerationState) -> bool:
        if not state.constraints.get("lexicalization_finished", False):
            return False

        # Prüfe auf Verb am Satzanfang (außer Fragen)
        for sentence in state.text.completed_sentences:
            words = sentence.split()
            if len(words) > 0:
                first_word = words[0].lower()
                # Verb am Anfang (außer bei Fragen mit "?")
                if first_word in [
                    "ist",
                    "sind",
                    "war",
                    "waren",
                    "hat",
                    "haben",
                ] and not sentence.endswith("?"):
                    return True

        return False

    def action(state: ResponseGenerationState) -> None:
        for i, sentence in enumerate(state.text.completed_sentences):
            words = sentence.split()
            if len(words) > 2:
                first_word = words[0].lower()

                if first_word in [
                    "ist",
                    "sind",
                    "war",
                    "waren",
                    "hat",
                    "haben",
                ] and not sentence.endswith("?"):
                    # Vertausche Verb und Subjekt
                    # "Ist ein Apfel rot" → "Ein Apfel ist rot"
                    verb = words[0]
                    subject = " ".join(words[1:3])  # Nimm 2 Wörter als Subjekt
                    rest = " ".join(words[3:])

                    new_sentence = f"{subject} {verb.lower()} {rest}".strip()

                    # Kapitalisiere Satzanfang
                    new_sentence = new_sentence[0].upper() + new_sentence[1:]

                    state.text.completed_sentences[i] = new_sentence

                    logging.debug(
                        f"Reordered sentence elements: {sentence} → {new_sentence}"
                    )

    return create_production_rule(
        name="ORDER_SENTENCE_ELEMENTS",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.93,
        description="Korrigiere Satzgliedstellung",
    )


def create_finish_sentence_rule() -> ProductionRule:
    """
    FINISH_SENTENCE (utility: 0.85)

    Condition: Lexicalization abgeschlossen UND alle syntaktischen Korrekturen angewendet
    Action: Markiere Satz-Konstruktion als abgeschlossen

    Diese Regel beendet die Syntactic Realization Phase.
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Beende wenn:
        # 1. Lexicalization abgeschlossen
        # 2. Mindestens ein Satz vorhanden
        # 3. Noch nicht als abgeschlossen markiert

        lexicalization_done = state.constraints.get("lexicalization_finished", False)
        has_sentences = len(state.text.completed_sentences) >= 1
        not_finished = not state.constraints.get("syntax_finished", False)

        return lexicalization_done and has_sentences and not_finished

    def action(state: ResponseGenerationState) -> None:
        # Markiere Syntax als abgeschlossen
        state.constraints["syntax_finished"] = True

        # Markiere Goal als completed (Ende der Pipeline)
        state.primary_goal.completed = True

        logging.info(
            f"Syntax realization finished: {len(state.text.completed_sentences)} sentences completed"
        )

    return create_production_rule(
        name="FINISH_SENTENCE",
        category=RuleCategory.SYNTAX,
        condition=condition,
        action=action,
        utility=0.85,
        description="Beendet Satz-Konstruktion",
    )
