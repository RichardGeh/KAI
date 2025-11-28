"""
Component 54: Production System - Content Selection Rules

Factory functions for creating content selection production rules.
These rules determine WHAT to say by selecting facts from available_facts
and managing confidence-based filtering.

Includes:
- Fact Selection Rules (10): Select facts by type (IS_A, HAS_PROPERTY, etc.)
- Confidence Filtering Rules (5): Filter and prioritize based on confidence scores

Author: KAI Development Team
Date: 2025-11-21
"""

import logging
from typing import List

from component_54_production_rule import ProductionRule
from component_54_production_rule_factories import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_PENDING_FACTS,
    MULTI_SOURCE_BOOST_BASE,
    MULTI_SOURCE_BOOST_MAX,
    create_production_rule,
)
from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory

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
        # 1. Weniger als MAX_PENDING_FACTS (Platz für mehr Fakten)
        # 2. HAS_PROPERTY Fakt verfügbar
        if len(state.discourse.pending_facts) >= MAX_PENDING_FACTS:
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
        if len(state.discourse.pending_facts) >= MAX_PENDING_FACTS:
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
        if len(state.discourse.pending_facts) >= MAX_PENDING_FACTS:
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
        if len(state.discourse.pending_facts) >= MAX_PENDING_FACTS:
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

    Condition: Fakt mit confidence >= HIGH_CONFIDENCE_THRESHOLD in available_facts UND noch nicht sortiert
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
            f
            for f in state.available_facts
            if f.get("confidence", 0.0) >= HIGH_CONFIDENCE_THRESHOLD
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

    Condition: Fakt mit confidence < LOW_CONFIDENCE_THRESHOLD in available_facts UND noch nicht gefiltert
    Action: Entferne Fakt aus available_facts (wird nicht verwendet)
    """

    def condition(state: ResponseGenerationState) -> bool:
        # Nur anwenden, wenn noch nicht gefiltert
        if state.constraints.get("low_confidence_filtered", False):
            return False

        low_conf_facts = [
            f
            for f in state.available_facts
            if f.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
        ]
        return len(low_conf_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        # Entferne alle Low-Confidence Fakten
        low_conf_facts = [
            f
            for f in state.available_facts
            if f.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
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
        description=f"Entfernt Fakten mit Confidence < {LOW_CONFIDENCE_THRESHOLD}",
    )


def create_select_synonym_rule() -> ProductionRule:
    """
    SELECT_SYNONYM (utility: 0.78)

    Condition: Synonym-Fakt in available_facts (relation_type == "SYNONYM")
    Action: Füge Synonym zu pending_facts hinzu (für Variation)
    """

    def condition(state: ResponseGenerationState) -> bool:
        if len(state.discourse.pending_facts) >= MAX_PENDING_FACTS:
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
        # 1. Mindestens 1 Fakt ausgewählt UND (keine available_facts mehr ODER >= MAX_PENDING_FACTS)
        # 2. ODER: Keine available_facts und keine pending_facts (leere Antwort)

        if len(state.discourse.pending_facts) >= 1 and len(state.available_facts) == 0:
            return True

        if len(state.discourse.pending_facts) >= MAX_PENDING_FACTS:
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
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 0.0) < HIGH_CONFIDENCE_THRESHOLD
        ]

        if not low_conf_pending:
            return False

        # Gibt es High-Confidence Alternative in available_facts?
        high_conf_available = [
            f
            for f in state.available_facts
            if f.get("confidence", 0.0) >= HIGH_CONFIDENCE_THRESHOLD
        ]

        return len(high_conf_available) > 0

    def action(state: ResponseGenerationState) -> None:
        low_conf_pending = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 0.0) < HIGH_CONFIDENCE_THRESHOLD
        ]
        high_conf_available = [
            f
            for f in state.available_facts
            if f.get("confidence", 0.0) >= HIGH_CONFIDENCE_THRESHOLD
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
            if 0.70 <= f.get("confidence", 0.0) < HIGH_CONFIDENCE_THRESHOLD
        ]

        # Nur anwenden, wenn Fakt noch keinen Marker hat
        return any(not f.get("uncertainty_marked", False) for f in medium_conf)

    def action(state: ResponseGenerationState) -> None:
        medium_conf = [
            f
            for f in state.discourse.pending_facts
            if 0.70 <= f.get("confidence", 0.0) < HIGH_CONFIDENCE_THRESHOLD
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
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
        ]
        return len(uncertain) > 0

    def action(state: ResponseGenerationState) -> None:
        uncertain = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
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
                boost = min(
                    MULTI_SOURCE_BOOST_MAX, (len(group) - 1) * MULTI_SOURCE_BOOST_BASE
                )
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
