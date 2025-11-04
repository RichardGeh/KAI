# component_1_netzwerk_feedback.py
"""
Feedback-Speicherung für Pattern Recognition.

Speichert:
- Positive Feedback: User akzeptiert Vorschlag
- Negative Feedback: "Nein, ich meine X" Korrekturen
- Pattern Quality Metrics

Zweck: KAI lernt aus Fehlern und verbessert Vorschläge.
"""

import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from neo4j import Driver

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class KonzeptNetzwerkFeedback:
    """
    Feedback-Management für Pattern Recognition.
    """

    def __init__(self, driver: Optional[Driver] = None):
        self.driver = driver
        logger.debug("KonzeptNetzwerkFeedback initialisiert")

    def _create_constraints(self):
        """Erstellt Neo4j Constraints für Feedback Nodes"""
        if not self.driver:
            return

        try:
            with self.driver.session(database="neo4j") as session:
                # TypoFeedback braucht unique ID
                constraint_query = """
                CREATE CONSTRAINT TypoFeedbackId IF NOT EXISTS
                FOR (tf:TypoFeedback) REQUIRE tf.id IS UNIQUE
                """
                session.run(constraint_query)
                logger.debug("Constraint 'TypoFeedbackId' erstellt/verifiziert")

        except Exception as e:
            logger.warning(
                "Fehler beim Erstellen von Feedback Constraints",
                extra={"error": str(e)},
            )

    # ========================================================================
    # TYPO CORRECTION FEEDBACK
    # ========================================================================

    def store_typo_feedback(
        self,
        original_input: str,
        suggested_word: str,
        actual_word: str,
        user_accepted: bool,
        confidence: float,
        correction_reason: str = "user_correction",
    ) -> Optional[str]:
        """
        Speichert Feedback für Tippfehler-Korrektur.

        Args:
            original_input: Original-Eingabe (z.B. "Ktzae")
            suggested_word: Von KAI vorgeschlagenes Wort (z.B. "Katze")
            actual_word: Tatsächlich gemeintes Wort (z.B. "Kitze")
            user_accepted: True wenn User Vorschlag akzeptierte
            confidence: Confidence-Score zum Zeitpunkt des Vorschlags
            correction_reason: Grund für Korrektur

        Returns:
            Feedback-ID oder None bei Fehler
        """
        if not self.driver:
            logger.error("store_typo_feedback: Kein DB-Driver verfügbar")
            return None

        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        try:
            with self.driver.session(database="neo4j") as session:
                query = """
                CREATE (tf:TypoFeedback {
                    id: $feedback_id,
                    original_input: $original_input,
                    suggested_word: $suggested_word,
                    actual_word: $actual_word,
                    user_accepted: $user_accepted,
                    confidence: $confidence,
                    correction_reason: $correction_reason,
                    timestamp: $timestamp
                })
                RETURN tf.id AS id
                """

                result = session.run(
                    query,
                    feedback_id=feedback_id,
                    original_input=original_input.lower(),
                    suggested_word=suggested_word.lower(),
                    actual_word=actual_word.lower(),
                    user_accepted=user_accepted,
                    confidence=confidence,
                    correction_reason=correction_reason,
                    timestamp=timestamp,
                )

                record = result.single()
                if record:
                    logger.info(
                        "Typo-Feedback gespeichert",
                        extra={
                            "feedback_id": feedback_id,
                            "original": original_input,
                            "suggested": suggested_word,
                            "actual": actual_word,
                            "accepted": user_accepted,
                        },
                    )
                    return feedback_id
                else:
                    logger.warning("Typo-Feedback konnte nicht erstellt werden")
                    return None

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler beim Speichern von Typo-Feedback",
                original=original_input,
                suggested=suggested_word,
            )
            return None

    def get_typo_feedback_for_input(self, original_input: str) -> List[Dict[str, Any]]:
        """
        Holt alle Feedback-Einträge für eine bestimmte Eingabe.

        Nutzt diese Info, um bei wiederholter Eingabe bessere Vorschläge zu machen.

        Args:
            original_input: Die Original-Eingabe (z.B. "Ktzae")

        Returns:
            Liste von Feedback-Dicts
        """
        if not self.driver:
            logger.error("get_typo_feedback_for_input: Kein DB-Driver verfügbar")
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                query = """
                MATCH (tf:TypoFeedback {original_input: $original_input})
                RETURN tf.id AS id,
                       tf.suggested_word AS suggested_word,
                       tf.actual_word AS actual_word,
                       tf.user_accepted AS user_accepted,
                       tf.confidence AS confidence,
                       tf.timestamp AS timestamp
                ORDER BY tf.timestamp DESC
                """

                result = session.run(query, original_input=original_input.lower())

                feedback_list = []
                for record in result:
                    feedback_list.append(
                        {
                            "id": record["id"],
                            "suggested_word": record["suggested_word"],
                            "actual_word": record["actual_word"],
                            "user_accepted": record["user_accepted"],
                            "confidence": record["confidence"],
                            "timestamp": record["timestamp"],
                        }
                    )

                logger.debug(
                    "Typo-Feedback abgerufen",
                    extra={
                        "original_input": original_input,
                        "feedback_count": len(feedback_list),
                    },
                )

                return feedback_list

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler beim Abrufen von Typo-Feedback",
                original_input=original_input,
            )
            return []

    def get_negative_examples(self, suggested_word: str) -> List[Dict[str, Any]]:
        """
        Holt alle Negativ-Beispiele für ein vorgeschlagenes Wort.

        Beispiel: Wenn "Katze" oft vorgeschlagen, aber abgelehnt wurde,
        sollte es bei zukünftigen Vorschlägen niedriger gewichtet werden.

        Args:
            suggested_word: Das vorgeschlagene Wort

        Returns:
            Liste von Original-Eingaben, die NICHT dieses Wort meinten
        """
        if not self.driver:
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                query = """
                MATCH (tf:TypoFeedback {suggested_word: $suggested_word, user_accepted: false})
                RETURN tf.original_input AS original_input,
                       tf.actual_word AS actual_word,
                       tf.confidence AS confidence,
                       tf.timestamp AS timestamp
                ORDER BY tf.timestamp DESC
                LIMIT 50
                """

                result = session.run(query, suggested_word=suggested_word.lower())

                negative_examples = []
                for record in result:
                    negative_examples.append(
                        {
                            "original_input": record["original_input"],
                            "actual_word": record["actual_word"],
                            "confidence": record["confidence"],
                            "timestamp": record["timestamp"],
                        }
                    )

                return negative_examples

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler beim Abrufen von Negativ-Beispielen",
                suggested_word=suggested_word,
            )
            return []

    # ========================================================================
    # PATTERN QUALITY SCORING
    # ========================================================================

    def update_pattern_quality(
        self, pattern_type: str, pattern_key: str, success: bool
    ) -> bool:
        """
        Aktualisiert Quality-Score eines Patterns mit Bayesian Update.

        Nutzt Beta-Distribution für robuste Schätzung:
        - Prior: Beta(α=1, β=1) = Uniform [0,1]
        - Update: α += 1 bei Success, β += 1 bei Failure
        - Posterior Mean: α / (α + β)

        Args:
            pattern_type: "typo_correction", "sequence_prediction", etc.
            pattern_key: Eindeutiger Pattern-Key (z.B. "Ktzae->Katze")
            success: True wenn User Vorschlag akzeptierte

        Returns:
            True bei Erfolg
        """
        if not self.driver:
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                # MERGE: Erstelle oder update mit Bayesian Approach
                query = """
                MERGE (pq:PatternQuality {
                    pattern_type: $pattern_type,
                    pattern_key: $pattern_key
                })
                ON CREATE SET
                    pq.success_count = CASE WHEN $success THEN 1 ELSE 0 END,
                    pq.failure_count = CASE WHEN $success THEN 0 ELSE 1 END,
                    pq.alpha = CASE WHEN $success THEN 2.0 ELSE 1.0 END,
                    pq.beta = CASE WHEN $success THEN 1.0 ELSE 2.0 END,
                    pq.created_at = datetime()
                ON MATCH SET
                    pq.success_count = pq.success_count + CASE WHEN $success THEN 1 ELSE 0 END,
                    pq.failure_count = pq.failure_count + CASE WHEN $success THEN 0 ELSE 1 END,
                    pq.alpha = pq.alpha + CASE WHEN $success THEN 1.0 ELSE 0.0 END,
                    pq.beta = pq.beta + CASE WHEN $success THEN 0.0 ELSE 1.0 END
                SET pq.last_updated = datetime(),
                    pq.weight = pq.alpha / (pq.alpha + pq.beta),
                    pq.confidence_interval_lower = (pq.alpha - 1.0) / (pq.alpha + pq.beta - 2.0),
                    pq.total_observations = pq.alpha + pq.beta - 2.0
                RETURN pq.weight AS weight,
                       pq.alpha AS alpha,
                       pq.beta AS beta,
                       pq.total_observations AS total_obs
                """

                result = session.run(
                    query,
                    pattern_type=pattern_type,
                    pattern_key=pattern_key,
                    success=success,
                )

                record = result.single()
                if record:
                    weight = record["weight"]
                    alpha = record["alpha"]
                    beta = record["beta"]
                    total_obs = record["total_obs"]

                    logger.debug(
                        "Pattern Quality aktualisiert (Bayesian)",
                        extra={
                            "pattern_type": pattern_type,
                            "pattern_key": pattern_key,
                            "success": success,
                            "new_weight": f"{weight:.3f}",
                            "alpha": f"{alpha:.1f}",
                            "beta": f"{beta:.1f}",
                            "total_observations": int(total_obs),
                        },
                    )
                    return True

                return False

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler beim Aktualisieren von Pattern Quality",
                pattern_type=pattern_type,
                pattern_key=pattern_key,
            )
            return False

    def get_pattern_quality(
        self, pattern_type: str, pattern_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Holt Quality-Metriken für ein Pattern (Bayesian Approach).

        Returns:
            Dict mit weight, alpha, beta, total_observations, confidence_interval
            None wenn Pattern unbekannt
        """
        if not self.driver:
            return None

        try:
            with self.driver.session(database="neo4j") as session:
                query = """
                MATCH (pq:PatternQuality {
                    pattern_type: $pattern_type,
                    pattern_key: $pattern_key
                })
                RETURN pq.weight AS weight,
                       pq.alpha AS alpha,
                       pq.beta AS beta,
                       pq.total_observations AS total_obs,
                       pq.confidence_interval_lower AS ci_lower,
                       pq.success_count AS success_count,
                       pq.failure_count AS failure_count
                """

                result = session.run(
                    query, pattern_type=pattern_type, pattern_key=pattern_key
                )

                record = result.single()
                if record:
                    return {
                        "weight": float(record["weight"]),
                        "alpha": float(record.get("alpha", 1.0)),
                        "beta": float(record.get("beta", 1.0)),
                        "total_observations": int(record.get("total_obs", 0)),
                        "confidence_interval_lower": float(record.get("ci_lower", 0.0)),
                        "success_count": int(record["success_count"]),
                        "failure_count": int(record["failure_count"]),
                    }
                else:
                    return None  # Pattern noch nie gesehen

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler beim Abrufen von Pattern Quality",
                pattern_type=pattern_type,
                pattern_key=pattern_key,
            )
            return None

    def get_pattern_quality_weight(self, pattern_type: str, pattern_key: str) -> float:
        """
        Convenience-Methode: Gibt nur Weight zurück (0.75 als Prior wenn unbekannt).

        Returns:
            Weight zwischen 0.0 und 1.0 (Default: 0.75 für unbekannte Patterns)
        """
        quality = self.get_pattern_quality(pattern_type, pattern_key)
        if quality:
            return quality["weight"]
        else:
            # Prior: Neutral-positive Annahme (Beta(1,1) = 0.5, aber wir nutzen 0.75 als optimistisches Prior)
            return 0.75


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def store_user_correction(
    netzwerk,
    original_input: str,
    kai_suggestion: str,
    user_meant: str,
    confidence: float,
) -> Optional[str]:
    """
    Standalone-Funktion zum Speichern einer User-Korrektur.

    Args:
        netzwerk: KonzeptNetzwerk Instanz
        original_input: Was User getippt hat
        kai_suggestion: Was KAI vorgeschlagen hat
        user_meant: Was User tatsächlich meinte
        confidence: KAI's Confidence beim Vorschlag

    Returns:
        Feedback-ID
    """
    if not hasattr(netzwerk, "_feedback"):
        logger.warning("Netzwerk hat kein Feedback-Modul")
        return None

    return netzwerk._feedback.store_typo_feedback(  # type: ignore[no-any-return]
        original_input=original_input,
        suggested_word=kai_suggestion,
        actual_word=user_meant,
        user_accepted=False,  # User hat korrigiert
        confidence=confidence,
        correction_reason="user_correction",
    )


if __name__ == "__main__":
    # Test-Code
    print("=== Feedback Storage Test ===\n")
    print("Modul erfolgreich geladen.")
    print("Für Tests mit Neo4j: Nutze pytest tests/test_pattern_recognition.py")
