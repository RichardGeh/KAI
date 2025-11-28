"""
Component 54: Production System - Syntactic Realization Rules

Factory functions for creating syntactic realization production rules.
These rules ensure grammatical correctness of generated sentences.

Includes 12 syntactic realization rules for articles, agreement, and word order.

Author: KAI Development Team
Date: 2025-11-21
"""

import logging
from typing import List

from component_54_production_rule import ProductionRule
from component_54_production_rule_factories import (
    GERMAN_ARTICLES,
    create_production_rule,
    determine_german_article,
)
from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory


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
                    if first_word not in GERMAN_ARTICLES:
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
                    if first_word not in GERMAN_ARTICLES:
                        # Füge Artikel hinzu (heuristisch: "Ein" oder "Eine")
                        noun = words[0]
                        article = determine_german_article(noun).capitalize()

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
                                article = determine_german_article(
                                    noun, case="accusative"
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
                                article = determine_german_article(noun, case="dative")

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


# ============================================================================
# Convenience Function: Load All Syntactic Realization Rules
# ============================================================================


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
        # Capitalization & Punctuation Rules (3)
        create_capitalize_sentence_start_rule(),
        create_capitalize_nouns_rule(),
        create_add_period_rule(),
        # Syntax Correctness Rules (5)
        create_add_comma_conjunction_rule(),
        create_fix_verb_agreement_rule(),
        create_ensure_gender_agreement_rule(),
        create_insert_preposition_rule(),
        create_order_sentence_elements_rule(),
        # Finishing Rule (1)
        create_finish_sentence_rule(),
    ]
