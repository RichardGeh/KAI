# component_19_pattern_recognition_char.py
"""
Buchstaben-Ebene Mustererkennung für Tippfehler-Korrektur.

Features:
- QWERTZ-Layout-basierte Edit Distance
- Kandidaten-Suche in bekannten Wörtern
- Confidence-Scoring basierend auf:
  * Edit Distance
  * Keyboard-Nachbarschaft
  * Wort-Häufigkeit (aus Neo4j)
"""

from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger
from component_25_adaptive_thresholds import AdaptiveThresholdManager
from kai_config import get_config

logger = get_logger(__name__)
config = get_config()

# ============================================================================
# QWERTZ KEYBOARD LAYOUT
# ============================================================================

# QWERTZ-Layout (Deutsch) - Nachbarschafts-Matrix
QWERTZ_NEIGHBORS = {
    "q": ["w", "a", "1", "2"],
    "w": ["q", "e", "s", "a", "2", "3"],
    "e": ["w", "r", "d", "s", "3", "4"],
    "r": ["e", "t", "f", "d", "4", "5"],
    "t": ["r", "z", "g", "f", "5", "6"],
    "z": ["t", "u", "h", "g", "6", "7"],
    "u": ["z", "i", "j", "h", "7", "8"],
    "i": ["u", "o", "k", "j", "8", "9"],
    "o": ["i", "p", "l", "k", "9", "0"],
    "p": ["o", "ü", "ö", "l", "0", "ß"],
    "ü": ["p", "ö"],
    "a": ["q", "w", "s", "y"],
    "s": ["a", "w", "e", "d", "x", "y"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "z", "h", "b", "v"],
    "h": ["g", "z", "u", "j", "n", "b"],
    "j": ["h", "u", "i", "k", "m", "n"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p", "ö"],
    "ö": ["l", "p", "ü", "ä"],
    "ä": ["ö"],
    "y": ["a", "s", "x"],
    "x": ["y", "s", "d", "c"],
    "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"],
    "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"],
    "m": ["n", "j", "k"],
}

# Sonderzeichen-Mapping
CHAR_CONFUSIONS = {
    "ß": ["s", "ss"],
    "ä": ["a", "ae"],
    "ö": ["o", "oe"],
    "ü": ["u", "ue"],
}


# ============================================================================
# LEVENSHTEIN DISTANCE mit Keyboard-Gewichtung
# ============================================================================


def keyboard_distance(char1: str, char2: str) -> float:
    """
    Berechnet "Kosten" für Ersetzung basierend auf Tastatur-Nachbarschaft.

    Returns:
        0.3: Nachbar-Tasten (häufiger Tippfehler)
        0.5: Sonderzeichen-Verwechslung (ß/s, ä/a)
        1.0: Keine Nachbarschaft (unwahrscheinlicher)
    """
    char1, char2 = char1.lower(), char2.lower()

    if char1 == char2:
        return 0.0

    # Prüfe QWERTZ-Nachbarschaft
    if char2 in QWERTZ_NEIGHBORS.get(char1, []):
        return 0.3

    # Prüfe Sonderzeichen-Verwechslung
    for special, alternatives in CHAR_CONFUSIONS.items():
        if (char1 == special and char2 in alternatives) or (
            char2 == special and char1 in alternatives
        ):
            return 0.5

    return 1.0


def weighted_levenshtein(s1: str, s2: str) -> float:
    """
    Levenshtein Distance mit Keyboard-Gewichtung.

    Niedrigere Kosten für Tastatur-Nachbarn.
    """
    s1, s2 = s1.lower(), s2.lower()

    if len(s1) < len(s2):
        return weighted_levenshtein(s2, s1)

    if len(s2) == 0:
        return float(len(s1))

    # Dynamic Programming Matrix
    previous_row = [float(x) for x in range(len(s2) + 1)]

    for i, c1 in enumerate(s1):
        current_row = [float(i + 1)]
        for j, c2 in enumerate(s2):
            # Kosten für Operationen
            insertions = previous_row[j + 1] + 1.0
            deletions = current_row[j] + 1.0
            substitutions = previous_row[j] + keyboard_distance(c1, c2)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


# ============================================================================
# TYPO CANDIDATE DETECTION
# ============================================================================


class TypoCandidateFinder:
    """
    Findet Tippfehler-Kandidaten in bekannten Wörtern.
    """

    def __init__(self, netzwerk):
        """
        Args:
            netzwerk: KonzeptNetzwerk Instanz für Wort-Zugriff
        """
        self.netzwerk = netzwerk
        self.adaptive_manager = AdaptiveThresholdManager(netzwerk)

        # Verwende adaptive Threshold (fallback auf Config/Default)
        self.min_word_occurrences = self.adaptive_manager.get_typo_threshold()

    def find_candidates(
        self, typo_word: str, max_candidates: int = 5, max_distance: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Findet Korrektur-Kandidaten für ein potenzielles Tippfehler-Wort.

        Args:
            typo_word: Das möglicherweise falsch geschriebene Wort
            max_candidates: Max. Anzahl Kandidaten
            max_distance: Max. erlaubte Edit Distance

        Returns:
            Liste von Kandidaten mit Confidence-Scores
        """
        typo_word = typo_word.lower()

        # Hole alle bekannten Wörter
        try:
            known_words = self.netzwerk.get_all_known_words()
        except Exception as e:
            logger.warning(
                "Failed to fetch known words from Neo4j, returning empty candidates",
                extra={"error": str(e), "typo_word": typo_word},
            )
            return []

        if not known_words:
            logger.debug("Keine bekannten Wörter im Graph")
            return []

        candidates = []

        # Hole Negative Examples (False Positives aus Feedback)
        negative_examples = {}
        if hasattr(self.netzwerk, "_feedback") and self.netzwerk._feedback:
            for word in known_words[:50]:  # Limitiere auf Top-50 für Performance
                negatives = self.netzwerk._feedback.get_negative_examples(word)
                if negatives:
                    # Berechne Rejection Rate
                    rejection_count = len(negatives)
                    negative_examples[word] = rejection_count

        for word in known_words:
            # Skip wenn identisch
            if word == typo_word:
                continue

            # False-Positive Filter: Skip wenn >50% Rejection Rate
            if word in negative_examples:
                rejection_count = negative_examples[word]
                # Heuristik: Wenn >3 Rejections für dieses Wort, deutlich downgrade
                if rejection_count > 3:
                    logger.debug(
                        "Candidate downgraded wegen hoher Rejection Rate",
                        extra={"candidate": word, "rejection_count": rejection_count},
                    )
                    # Reduziere Candidate-Priorität durch höhere Distance-Penalty
                    # (wird später in Confidence-Berechnung berücksichtigt)

            # Berechne gewichtete Distance
            distance = weighted_levenshtein(typo_word, word)

            # Filter nach max_distance
            if distance > max_distance:
                continue

            # Berechne Base Confidence
            base_confidence = self._calculate_confidence(
                typo_word=typo_word, candidate=word, distance=distance
            )

            # Appliziere Pattern Quality Multiplier (aus Feedback)
            pattern_key = f"{typo_word.lower()}->{word.lower()}"
            pattern_quality_weight = 1.0

            if hasattr(self.netzwerk, "_feedback") and self.netzwerk._feedback:
                pattern_quality_weight = (
                    self.netzwerk._feedback.get_pattern_quality_weight(
                        pattern_type="typo_correction", pattern_key=pattern_key
                    )
                )

                # Multiplier: 0.5 - 1.5x basierend auf Pattern Quality
                # Weight 0.0 -> 0.5x, Weight 0.75 -> 1.0x, Weight 1.0 -> 1.5x
                multiplier = 0.5 + pattern_quality_weight

                # Appliziere Multiplier
                final_confidence = min(1.0, base_confidence * multiplier)

                logger.debug(
                    "Pattern quality applied",
                    extra={
                        "pattern_key": pattern_key,
                        "base_confidence": f"{base_confidence:.3f}",
                        "quality_weight": f"{pattern_quality_weight:.3f}",
                        "multiplier": f"{multiplier:.3f}",
                        "final_confidence": f"{final_confidence:.3f}",
                    },
                )
            else:
                final_confidence = base_confidence

            candidates.append(
                {
                    "word": word,
                    "distance": distance,
                    "confidence": final_confidence,
                    "base_confidence": base_confidence,
                    "pattern_quality_weight": pattern_quality_weight,
                    "reason": self._get_correction_reason(typo_word, word, distance),
                }
            )

        # Sortiere nach Confidence (höchste zuerst)
        candidates.sort(key=lambda x: x["confidence"], reverse=True)

        # Limitiere Ergebnisse
        return candidates[:max_candidates]

    def _calculate_confidence(
        self, typo_word: str, candidate: str, distance: float
    ) -> float:
        """
        Berechnet Confidence-Score für einen Kandidaten.

        Score-Komponenten (gewichtet):
        - Edit Distance: 40% (niedriger = besser)
        - Längen-Ähnlichkeit: 25%
        - First/Last Letter: 15%
        - Word Frequency: 20% (häufigere Wörter = höhere Confidence)
        """
        # 1. Distance Score (0.0 - 1.0, invertiert)
        # Distance 0 = 1.0, Distance 3+ = 0.0
        distance_score = max(0.0, 1.0 - (distance / 3.0))

        # 2. Längen-Ähnlichkeit (0.0 - 1.0)
        len_diff = abs(len(typo_word) - len(candidate))
        max_len = max(len(typo_word), len(candidate))
        if max_len == 0:
            length_score = 1.0  # Beide leer = perfekt
        else:
            length_score = max(0.0, 1.0 - (len_diff / max_len))

        # 3. First/Last Letter Bonus
        first_last_bonus = 0.0
        if len(typo_word) > 0 and len(candidate) > 0:
            if typo_word[0] == candidate[0]:
                first_last_bonus += 0.5  # 50% des Bonus
            if typo_word[-1] == candidate[-1]:
                first_last_bonus += 0.5  # 50% des Bonus

        # 4. Word Frequency Score (0.0 - 1.0)
        try:
            frequency_score = self.netzwerk.get_normalized_word_frequency(candidate)
        except Exception as e:
            logger.warning(
                "Failed to get word frequency, using default 0.0",
                extra={"word": candidate, "error": str(e)},
            )
            frequency_score = 0.0

        # Gewichtete Kombination (neue Gewichtung)
        confidence = (
            0.40 * distance_score
            + 0.25 * length_score
            + 0.15 * first_last_bonus
            + 0.20 * frequency_score
        )

        logger.debug(
            "Typo confidence berechnet",
            extra={
                "typo": typo_word,
                "candidate": candidate,
                "distance_score": f"{distance_score:.2f}",
                "length_score": f"{length_score:.2f}",
                "first_last_bonus": f"{first_last_bonus:.2f}",
                "frequency_score": f"{frequency_score:.2f}",
                "final_confidence": f"{confidence:.2f}",
            },
        )

        return min(1.0, max(0.0, confidence))

    def _get_correction_reason(
        self, typo_word: str, candidate: str, distance: float
    ) -> str:
        """Gibt lesbare Begründung für Korrektur"""
        if distance < 1.0:
            return "keyboard_neighbor"
        elif distance < 2.0:
            return "edit_distance_1"
        elif distance < 3.0:
            return "edit_distance_2"
        else:
            return "similar_spelling"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_typos(
    text: str, netzwerk, auto_correct_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Standalone-Funktion für Tippfehler-Erkennung.

    Args:
        text: Input-Text
        netzwerk: KonzeptNetzwerk Instanz
        auto_correct_threshold: Optional threshold override

    Returns:
        Dict mit original, corrections, decision
    """
    if auto_correct_threshold is None:
        auto_correct_threshold = config.get("typo_auto_correct_threshold", 0.85)

    finder = TypoCandidateFinder(netzwerk)
    words = text.split()
    corrections = []

    # OPTIMIZATION: Fetch known words ONCE, convert to lowercase set for O(1) lookup
    try:
        known_words_raw = netzwerk.get_all_known_words()
        known_words_set = {w.lower() for w in known_words_raw}
    except Exception as e:
        logger.warning(
            "Failed to fetch known words from Neo4j, typo detection disabled",
            extra={"error": str(e)},
        )
        known_words_set = set()

    # WICHTIG: Blacklist häufiger deutscher Funktionswörter
    # Diese sollten NIEMALS als Tippfehler korrigiert werden
    function_words_blacklist = {
        # Artikel
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "einen",
        "einem",
        "eines",
        # Präpositionen
        "in",
        "an",
        "auf",
        "aus",
        "bei",
        "mit",
        "nach",
        "von",
        "zu",
        "vor",
        "über",
        "unter",
        # Pronomen
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
        "mein",
        "dein",
        "sein",
        "ihr",
        "unser",
        "euer",
        "dieser",
        "diese",
        "dieses",
        "jener",
        "jene",
        "jenes",
        # Konjunktionen
        "und",
        "oder",
        "aber",
        "denn",
        "sondern",
        "wenn",
        "als",
        "wie",
        "dass",
        "weil",
        # Häufige Verben
        "ist",
        "sind",
        "war",
        "waren",
        "hat",
        "haben",
        "wird",
        "werden",
        "kann",
        "können",
        # Fragewörter
        "was",
        "wer",
        "wie",
        "wo",
        "wann",
        "warum",
        "wozu",
        "wieso",
        "weshalb",
    }

    for word in words:
        # Ignoriere kurze Wörter und Satzzeichen
        clean_word = word.strip(".,!?;:")
        if len(clean_word) < 3:
            continue

        # WICHTIG: Überspringe Funktionswörter (sollten nie als Typo behandelt werden)
        if clean_word.lower() in function_words_blacklist:
            continue

        # OPTIMIZED: O(1) lookup statt O(n) list comprehension
        if clean_word.lower() in known_words_set:
            continue  # Bekanntes Wort, kein Typo

        # Suche Kandidaten
        candidates = finder.find_candidates(clean_word)

        if candidates and len(candidates) > 0:
            best = candidates[0]

            corrections.append(
                {
                    "original": word,
                    "candidates": candidates,
                    "best_candidate": best,
                    "decision": (
                        "auto_correct"
                        if best["confidence"] >= auto_correct_threshold
                        else "ask_user"
                    ),
                }
            )

    return {
        "original_text": text,
        "corrections": corrections,
        "has_typos": len(corrections) > 0,
    }


def record_typo_correction_feedback(
    netzwerk,
    original_input: str,
    suggested_correction: str,
    user_accepted: bool,
    actual_correction: Optional[str] = None,
    confidence: float = 0.0,
) -> bool:
    """
    Speichert Feedback für eine Typo-Korrektur.

    Sollte nach jeder Typo-Korrektur aufgerufen werden (akzeptiert oder abgelehnt).

    Args:
        netzwerk: KonzeptNetzwerk Instanz
        original_input: Original-Eingabe (z.B. "Ktzae")
        suggested_correction: Von KAI vorgeschlagene Korrektur (z.B. "Katze")
        user_accepted: True wenn User Korrektur akzeptierte
        actual_correction: Falls abgelehnt, was User tatsächlich meinte
        confidence: Confidence-Score zum Zeitpunkt des Vorschlags

    Returns:
        True bei Erfolg
    """
    if not hasattr(netzwerk, "_feedback") or not netzwerk._feedback:
        logger.warning("record_typo_correction_feedback: Kein Feedback-Modul verfügbar")
        return False

    # Speichere Typo-Feedback
    actual_word = actual_correction if actual_correction else suggested_correction
    feedback_id = netzwerk._feedback.store_typo_feedback(
        original_input=original_input,
        suggested_word=suggested_correction,
        actual_word=actual_word,
        user_accepted=user_accepted,
        confidence=confidence,
        correction_reason="user_feedback",
    )

    # Update Pattern Quality
    pattern_key = f"{original_input.lower()}->{suggested_correction.lower()}"
    netzwerk._feedback.update_pattern_quality(
        pattern_type="typo_correction", pattern_key=pattern_key, success=user_accepted
    )

    logger.info(
        "Typo-Korrektur Feedback gespeichert",
        extra={
            "original": original_input,
            "suggested": suggested_correction,
            "actual": actual_word,
            "accepted": user_accepted,
            "feedback_id": feedback_id,
        },
    )

    return feedback_id is not None


if __name__ == "__main__":
    # Test-Code (ohne Neo4j)
    print("=== Keyboard Distance Tests ===\n")

    test_pairs = [
        ("k", "l"),  # Nachbarn auf QWERTZ
        ("k", "i"),  # Nachbarn diagonal
        ("ß", "s"),  # Sonderzeichen
        ("k", "x"),  # Nicht-Nachbarn
    ]

    for c1, c2 in test_pairs:
        dist = keyboard_distance(c1, c2)
        print(f"'{c1}' <-> '{c2}': {dist:.1f}")

    print("\n=== Weighted Levenshtein Tests ===\n")

    test_words = [
        ("Katze", "Katze"),  # Identisch
        ("Katze", "Katzr"),  # Nachbar-Taste (e->r)
        ("Katze", "Katse"),  # Vertauscht (z<->s)
        ("Katze", "Ktzae"),  # Buchstaben vertauscht
    ]

    for w1, w2 in test_words:
        dist = weighted_levenshtein(w1, w2)
        print(f"'{w1}' -> '{w2}': {dist:.2f}")
