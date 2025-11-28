"""
Component 54: Production System - Discourse Management Rules

Factory functions for creating discourse management production rules.
These rules determine response STRUCTURE and add qualifiers.

Includes 12 discourse management rules for context, uncertainty, and flow.

Author: KAI Development Team
Date: 2025-11-21
"""

import logging
from typing import List

from component_54_production_rule import ProductionRule
from component_54_production_rule_factories import (
    MAX_TRANSITIONS,
    MEDIUM_CONFIDENCE_MAX,
    MEDIUM_CONFIDENCE_MIN,
    VERY_HIGH_CONFIDENCE_THRESHOLD,
    create_production_rule,
)
from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory

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
            if f.get("confidence", 1.0) < MEDIUM_CONFIDENCE_MIN
            and not f.get("uncertainty_signaled", False)
        ]

        return len(uncertain_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        uncertain_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 1.0) < MEDIUM_CONFIDENCE_MIN
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
            if f.get("confidence", 0.0) >= VERY_HIGH_CONFIDENCE_THRESHOLD
            and not f.get("high_confidence_signaled", False)
        ]

        return len(high_conf_facts) > 0

    def action(state: ResponseGenerationState) -> None:
        high_conf_facts = [
            f
            for f in state.discourse.pending_facts
            if f.get("confidence", 0.0) >= VERY_HIGH_CONFIDENCE_THRESHOLD
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
            if MEDIUM_CONFIDENCE_MIN <= f.get("confidence", 1.0) < MEDIUM_CONFIDENCE_MAX
            and not f.get("qualifier_added", False)
        ]

        return len(medium_conf) > 0

    def action(state: ResponseGenerationState) -> None:
        medium_conf = [
            f
            for f in state.discourse.pending_facts
            if MEDIUM_CONFIDENCE_MIN <= f.get("confidence", 1.0) < MEDIUM_CONFIDENCE_MAX
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

        return len(recent_transitions) < MAX_TRANSITIONS  # Max 2 Transitions

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


# ============================================================================
# Convenience Function: Load All Discourse Management Rules
# ============================================================================


def create_all_discourse_management_rules() -> List[ProductionRule]:
    """
    Erstellt alle 12 Discourse Management Rules (Response Structuring & Qualification).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return [
        # Introduction Rules (2)
        create_introduce_with_context_rule(),
        create_introduce_simple_rule(),
        # Confidence Signaling Rules (5)
        create_signal_uncertainty_rule(),
        create_signal_high_confidence_rule(),
        create_explain_reasoning_path_rule(),
        create_mark_hypothesis_rule(),
        create_add_confidence_qualifier_rule(),
        # Source & Structure Rules (5)
        create_mention_evidence_source_rule(),
        create_structure_multi_part_answer_rule(),
        create_add_transition_rule(),
        create_conclude_answer_rule(),
        create_offer_elaboration_rule(),
    ]
