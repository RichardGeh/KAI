"""
Component 54: Production System - Lexicalization Rules

Factory functions for creating lexicalization production rules.
These rules determine HOW to phrase selected facts as natural language.

Includes 15 lexicalization rules for fact verbalization and style variation.

Author: KAI Development Team
Date: 2025-11-21
"""

import logging
from typing import List

from component_54_production_rule import ProductionRule
from component_54_production_rule_factories import (
    MAX_SENTENCES,
    create_production_rule,
    determine_german_article,
    pluralize_german_noun,
)
from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory

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
            article = determine_german_article(object_noun)

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
            object_plural = pluralize_german_noun(object_noun)

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

            object_plural = pluralize_german_noun(object_noun)

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
        enough_sentences = len(state.text.completed_sentences) >= MAX_SENTENCES

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
# Convenience Function: Load All Lexicalization Rules
# ============================================================================


def create_all_lexicalization_rules() -> List[ProductionRule]:
    """
    Erstellt alle 15 Lexicalization Rules (Fact Verbalization + Style Variation).

    Returns:
        Liste von ProductionRule Instanzen
    """
    return [
        # Basic Verbalization Rules (7)
        create_verbalize_is_a_simple_rule(),
        create_verbalize_is_a_variant_1_rule(),
        create_verbalize_is_a_variant_2_rule(),
        create_verbalize_has_property_rule(),
        create_verbalize_capable_of_rule(),
        create_verbalize_located_in_rule(),
        create_verbalize_part_of_rule(),
        # Style Variation Rules (5)
        create_vary_copula_verb_rule(),
        create_combine_facts_conjunction_rule(),
        create_avoid_repetition_rule(),
        create_select_formal_style_rule(),
        create_select_casual_style_rule(),
        # Elaboration Rules (3)
        create_add_elaboration_rule(),
        create_compress_similar_facts_rule(),
        create_finish_lexicalization_rule(),
    ]
