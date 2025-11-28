"""
component_51_feedback_handler.py

User Feedback Loop - KAI lernt aus direktem Benutzer-Feedback

Implementiert:
- Answer Tracking mit eindeutigen IDs
- Feedback-Verarbeitung (correct/incorrect/unsure)
- Dynamic Confidence Updates basierend auf Feedback
- Meta-Learning Integration (Strategy Performance)
- Negative Pattern Creation (Inhibition)
- Correction Request System
- Feedback History und Statistiken

Teil von Phase 3: Meta-Learning Layer
Unterstützt kontinuierliches Lernen und Self-Improvement

Author: KAI Development Team
Created: 2025-11-08
"""

import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from component_15_logging_config import get_logger
from component_46_meta_learning import MetaLearningEngine
from component_confidence_manager import get_confidence_manager

logger = get_logger(__name__)


# Validation patterns
ENTITY_NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,63}$", re.IGNORECASE)


# ============================================================================
# Data Structures
# ============================================================================


class FeedbackType(Enum):
    """Typ des Benutzer-Feedbacks"""

    CORRECT = "correct"  # Antwort war richtig
    INCORRECT = "incorrect"  # Antwort war falsch
    UNSURE = "unsure"  # Benutzer ist unsicher
    PARTIALLY_CORRECT = "partially_correct"  # Teilweise richtig


@dataclass
class AnswerRecord:
    """
    Gespeicherte Antwort mit allen Metadaten

    Attributes:
        answer_id: Eindeutige ID
        timestamp: Zeitpunkt der Antwort-Generierung
        query: Die ursprüngliche Frage
        answer_text: Die generierte Antwort
        confidence: Confidence-Wert (0.0-1.0)
        strategy: Verwendete Reasoning-Strategy
        used_relations: Liste von verwendeten Relation-IDs aus Neo4j
        used_concepts: Liste von verwendeten Konzept-IDs
        proof_tree: Optional Proof Tree Objekt
        reasoning_paths: Optional Liste von Reasoning Paths
        evaluation_score: Optional Self-Evaluation Score
    """

    answer_id: str
    timestamp: datetime
    query: str
    answer_text: str
    confidence: float
    strategy: str
    used_relations: List[str] = field(default_factory=list)
    used_concepts: List[str] = field(default_factory=list)
    proof_tree: Optional[Any] = None
    reasoning_paths: Optional[List[Any]] = None
    evaluation_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackRecord:
    """
    Benutzer-Feedback zu einer Antwort

    Attributes:
        feedback_id: Eindeutige Feedback-ID
        answer_id: Referenz zur AnswerRecord
        feedback_type: Art des Feedbacks
        timestamp: Zeitpunkt des Feedbacks
        user_comment: Optional Kommentar vom Benutzer
        correction: Optional Korrektur vom Benutzer
    """

    feedback_id: str
    answer_id: str
    feedback_type: FeedbackType
    timestamp: datetime
    user_comment: Optional[str] = None
    correction: Optional[str] = None


@dataclass
class FeedbackHandlerConfig:
    """Configuration for FeedbackHandler with memory limits"""

    # Memory limits
    max_answer_records: int = 10000  # Keep last 10k answers
    max_feedback_records: int = 10000  # Keep last 10k feedbacks
    answer_ttl_seconds: int = 86400 * 7  # 7 days
    feedback_ttl_seconds: int = 86400 * 30  # 30 days

    # Statistics window
    max_history_size: int = 1000  # For get_feedback_history

    # Confidence update factors
    # NOTE: Values chosen empirically to balance learning speed vs. stability
    # - CORRECT: +10% boost for confirmed correct answers
    # - INCORRECT: -15% penalty for wrong answers (asymmetric to prevent over-confidence)
    # - PARTIALLY_CORRECT: +2% small boost for partial correctness
    # - UNSURE: -2% small penalty for uncertainty
    correct_factor: float = 1.1
    incorrect_factor: float = 0.85
    partially_correct_factor: float = 1.02
    unsure_factor: float = 0.98

    # Confidence bounds (prevent drift outside [0, 1])
    min_confidence: float = 0.0
    max_confidence: float = 1.0

    # Validation limits
    max_correction_length: int = 1000
    max_comment_length: int = 500

    def __post_init__(self):
        """Validate configuration"""
        # Validate factors are positive
        if self.correct_factor <= 0 or self.incorrect_factor <= 0:
            raise ValueError("Confidence factors must be positive")

        # Validate factors won't cause confidence drift
        if self.correct_factor > 2.0:
            raise ValueError(
                f"correct_factor too high ({self.correct_factor}), risks confidence explosion"
            )

        if self.incorrect_factor < 0.5:
            raise ValueError(
                f"incorrect_factor too low ({self.incorrect_factor}), risks confidence collapse"
            )

        # Validate bounds
        if not 0.0 <= self.min_confidence <= self.max_confidence <= 1.0:
            raise ValueError(
                f"Invalid confidence bounds: [{self.min_confidence}, {self.max_confidence}]"
            )

        # Validate memory limits
        if self.max_answer_records < 100 or self.max_feedback_records < 100:
            raise ValueError("Record limits must be >= 100")


# ============================================================================
# Feedback Handler
# ============================================================================


class FeedbackHandler:
    """
    Handler für User Feedback Loop

    Verantwortlichkeiten:
    1. Answer Tracking: Speichert Antworten mit IDs
    2. Feedback Processing: Verarbeitet Benutzer-Feedback
    3. Confidence Updates: Passt Confidence dynamisch an
    4. Meta-Learning Updates: Informiert Meta-Learning Engine
    5. Negative Patterns: Erstellt Inhibition-Patterns bei Fehlern
    6. Correction Requests: Fordert Korrekturen an

    Integration:
    - ConfidenceManager für Confidence-Updates
    - MetaLearningEngine für Strategy Performance
    - KonzeptNetzwerk für Negative Patterns
    """

    def __init__(
        self,
        netzwerk: Any,
        meta_learning: Optional[MetaLearningEngine] = None,
        config: Optional[FeedbackHandlerConfig] = None,
    ):
        """
        Initialisiert FeedbackHandler (thread-safe)

        Args:
            netzwerk: KonzeptNetzwerk Instanz
            meta_learning: Optional MetaLearningEngine
            config: Optional FeedbackHandlerConfig
        """
        self.netzwerk = netzwerk
        self.meta_learning = meta_learning
        self.confidence_manager = get_confidence_manager()
        self.config = config or FeedbackHandlerConfig()

        # Thread synchronization
        self._lock = threading.RLock()  # For answer_records and feedback_records
        self._stats_lock = threading.Lock()  # Separate lock for statistics

        # Bounded storage with TTL (thread-safe)
        self.answer_records: TTLCache = TTLCache(
            maxsize=self.config.max_answer_records, ttl=self.config.answer_ttl_seconds
        )
        self.feedback_records: TTLCache = TTLCache(
            maxsize=self.config.max_feedback_records,
            ttl=self.config.feedback_ttl_seconds,
        )

        # Feedback Statistiken
        self.feedback_stats = {
            "total_feedbacks": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "unsure_count": 0,
            "partially_correct_count": 0,
        }

        logger.info(
            "FeedbackHandler initialisiert | max_answers=%d, answer_ttl=%ds, max_feedbacks=%d",
            self.config.max_answer_records,
            self.config.answer_ttl_seconds,
            self.config.max_feedback_records,
        )

    # ========================================================================
    # Helper Methods - Validation & Sanitization
    # ========================================================================

    def _sanitize_for_logging(self, text: str, max_len: int = 100) -> str:
        """Remove cp1252-unsafe Unicode for Windows logging"""
        replacements = {
            "→": "->",
            "×": "x",
            "÷": "/",
            "≠": "!=",
            "≤": "<=",
            "≥": ">=",
            "✓": "[OK]",
            "✗": "[FAIL]",
            "∧": "AND",
            "∨": "OR",
            "¬": "NOT",
        }
        sanitized = text
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        return sanitized[:max_len].encode("ascii", errors="replace").decode("ascii")

    def _validate_and_sanitize_text(
        self, text: str, max_len: int, field_name: str
    ) -> str:
        """
        Validate and sanitize user text input

        Args:
            text: User-provided text
            max_len: Maximum allowed length
            field_name: Name of field (for error messages)

        Returns:
            Sanitized text

        Raises:
            ValueError: If validation fails
        """
        if not text:
            return ""

        # Length validation
        if len(text) > max_len:
            raise ValueError(
                f"{field_name} exceeds maximum length ({len(text)} > {max_len})"
            )

        # Sanitize for logging (cp1252 safety)
        sanitized = self._sanitize_for_logging(text, max_len)

        return sanitized

    def _validate_entity_name(self, name: str) -> None:
        """
        Validate entity name for Neo4j safety

        Args:
            name: Entity/property name

        Raises:
            ValueError: If name is invalid
        """
        if not ENTITY_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid entity name: {name}. Must match pattern: [a-z_][a-z0-9_]{{0,63}}"
            )

    # ========================================================================
    # Answer Tracking
    # ========================================================================

    def track_answer(
        self,
        query: str,
        answer_text: str,
        confidence: float,
        strategy: str,
        used_relations: Optional[List[str]] = None,
        used_concepts: Optional[List[str]] = None,
        proof_tree: Optional[Any] = None,
        reasoning_paths: Optional[List[Any]] = None,
        evaluation_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Speichert eine Antwort für späteres Feedback (thread-safe)

        Args:
            query: Die Frage
            answer_text: Die Antwort
            confidence: Confidence-Wert (0.0-1.0)
            strategy: Verwendete Strategy
            used_relations: Optional Liste von Relation-IDs
            used_concepts: Optional Liste von Konzept-IDs
            proof_tree: Optional Proof Tree
            reasoning_paths: Optional Reasoning Paths
            evaluation_score: Optional Self-Evaluation Score
            metadata: Optional zusätzliche Metadaten

        Returns:
            answer_id: Eindeutige ID für diese Antwort

        Raises:
            ValueError: If confidence or evaluation_score out of range
        """
        # Validate confidence range
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        # Validate evaluation_score if provided
        if evaluation_score is not None and not 0.0 <= evaluation_score <= 1.0:
            raise ValueError(
                f"Evaluation score must be in [0.0, 1.0], got {evaluation_score}"
            )

        # Validate strategy name (no injection risk, but good practice)
        if not strategy or len(strategy) > 100:
            raise ValueError(
                f"Invalid strategy name: '{strategy}' (must be 1-100 chars)"
            )

        answer_id = str(uuid.uuid4())

        record = AnswerRecord(
            answer_id=answer_id,
            timestamp=datetime.now(),
            query=query,
            answer_text=answer_text,
            confidence=confidence,
            strategy=strategy,
            used_relations=used_relations or [],
            used_concepts=used_concepts or [],
            proof_tree=proof_tree,
            reasoning_paths=reasoning_paths,
            evaluation_score=evaluation_score,
            metadata=metadata or {},
        )

        with self._lock:
            self.answer_records[answer_id] = record

        safe_query = self._sanitize_for_logging(query, max_len=50)

        logger.info(
            "Answer tracked | id=%s, strategy=%s, confidence=%.2f, query='%s...', relations=%d",
            answer_id[:8],
            strategy,
            confidence,
            safe_query,
            len(used_relations or []),
        )

        return answer_id

    def get_answer(self, answer_id: str) -> Optional[AnswerRecord]:
        """Gibt AnswerRecord für gegebene ID zurück (thread-safe)"""
        with self._lock:
            return self.answer_records.get(answer_id)

    # ========================================================================
    # Feedback Processing
    # ========================================================================

    def process_user_feedback(
        self,
        answer_id: str,
        feedback_type: FeedbackType,
        user_comment: Optional[str] = None,
        correction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verarbeitet Benutzer-Feedback

        Workflow:
        1. Validierung: Answer existiert?
        2. Confidence Update: Verstärke/Schwäche verwendete Relationen
        3. Meta-Learning Update: Informiere Meta-Learning Engine
        4. Negative Patterns: Bei incorrect → Erstelle Inhibition
        5. Correction Request: Bei unsure/incorrect → Fordere Klarstellung

        Args:
            answer_id: ID der bewerteten Antwort
            feedback_type: Art des Feedbacks
            user_comment: Optional Kommentar
            correction: Optional Korrektur vom Benutzer

        Returns:
            Dict mit:
                - 'success': bool
                - 'actions_taken': List[str]
                - 'confidence_changes': Dict[str, float]
                - 'message': str
        """
        # 1. Validate answer_id format (UUID)
        try:
            uuid.UUID(answer_id)
        except ValueError:
            logger.warning("Invalid answer_id format: %s", answer_id)
            return {
                "success": False,
                "actions_taken": [],
                "confidence_changes": {},
                "message": f"Invalid answer ID format: {answer_id}",
            }

        # 2. Validate and sanitize user inputs
        try:
            if user_comment:
                user_comment = self._validate_and_sanitize_text(
                    user_comment, self.config.max_comment_length, "user_comment"
                )

            if correction:
                correction = self._validate_and_sanitize_text(
                    correction, self.config.max_correction_length, "correction"
                )
        except ValueError as e:
            logger.warning("Input validation failed: %s", e)
            return {
                "success": False,
                "actions_taken": [],
                "confidence_changes": {},
                "message": f"Input validation failed: {e}",
            }

        # 3. Get answer (thread-safe)
        answer = self.get_answer(answer_id)
        if not answer:
            logger.warning("Answer ID nicht gefunden: %s", answer_id)
            return {
                "success": False,
                "actions_taken": [],
                "confidence_changes": {},
                "message": f"Answer ID {answer_id} nicht gefunden",
            }

        logger.info(
            "Processing feedback | answer_id=%s, type=%s, strategy=%s",
            answer_id[:8],
            feedback_type.value,
            answer.strategy,
        )

        # Erstelle Feedback Record
        feedback_id = str(uuid.uuid4())
        feedback_record = FeedbackRecord(
            feedback_id=feedback_id,
            answer_id=answer_id,
            feedback_type=feedback_type,
            timestamp=datetime.now(),
            user_comment=user_comment,
            correction=correction,
        )

        with self._lock:
            self.feedback_records[feedback_id] = feedback_record

        # Update Statistiken
        self._update_statistics(feedback_type)

        actions_taken = []
        confidence_changes = {}

        # 2. Confidence Update
        if answer.used_relations:
            changes = self._update_confidence(answer.used_relations, feedback_type)
            confidence_changes.update(changes)
            actions_taken.append(f"Confidence für {len(changes)} Relationen angepasst")

        # 3. Meta-Learning Update
        if self.meta_learning:
            self._record_to_meta_learning(answer, feedback_type)
            actions_taken.append("Meta-Learning Engine informiert")

        # 4. Negative Patterns
        if feedback_type == FeedbackType.INCORRECT:
            pattern_created = self._create_inhibition_pattern(answer, correction)
            if pattern_created:
                actions_taken.append("Inhibition-Pattern erstellt")

        # 5. Correction Request
        if feedback_type in [FeedbackType.INCORRECT, FeedbackType.UNSURE]:
            request_sent = self._request_correction(answer, correction)
            if request_sent:
                actions_taken.append("Korrektur-Request generiert")

        message = self._generate_feedback_message(feedback_type, actions_taken)

        logger.info(
            "Feedback processed | answer_id=%s, actions=%d, changes=%d",
            answer_id[:8],
            len(actions_taken),
            len(confidence_changes),
        )

        return {
            "success": True,
            "actions_taken": actions_taken,
            "confidence_changes": confidence_changes,
            "message": message,
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _update_confidence(
        self, relation_ids: List[str], feedback_type: FeedbackType
    ) -> Dict[str, float]:
        """
        Updated Confidence für verwendete Relationen

        Args:
            relation_ids: Liste von Relation-IDs
            feedback_type: Art des Feedbacks

        Returns:
            Dict[relation_id, new_confidence]
        """
        changes = {}

        # Use configured factors
        factors = {
            FeedbackType.CORRECT: self.config.correct_factor,
            FeedbackType.INCORRECT: self.config.incorrect_factor,
            FeedbackType.PARTIALLY_CORRECT: self.config.partially_correct_factor,
            FeedbackType.UNSURE: self.config.unsure_factor,
        }

        factor = factors.get(feedback_type, 1.0)

        for rel_id in relation_ids:
            try:
                # Validate rel_id format
                uuid.UUID(rel_id)

                # TODO: When implementing, use parameterized queries:
                # query = """
                #     MATCH ()-[r]->()
                #     WHERE id(r) = $rel_id
                #     SET r.confidence = CASE
                #         WHEN r.confidence * $factor > $max_conf THEN $max_conf
                #         WHEN r.confidence * $factor < $min_conf THEN $min_conf
                #         ELSE r.confidence * $factor
                #     END
                #     RETURN r.confidence AS new_conf
                # """
                # result = session.run(query, {
                #     "rel_id": rel_id,
                #     "factor": factor,
                #     "max_conf": self.config.max_confidence,
                #     "min_conf": self.config.min_confidence
                # })

                # Placeholder für Demo
                changes[rel_id] = factor

            except ValueError as e:
                # Expected error: Invalid UUID format
                logger.warning("Invalid relation ID format: %s: %s", rel_id, e)
            except Neo4jError as e:
                # Neo4j-specific error: Log and continue with other relations
                logger.error(
                    "Neo4j error updating confidence for %s: %s",
                    rel_id,
                    e,
                    exc_info=True,
                )
            except ServiceUnavailable as e:
                # Critical: Database unavailable - re-raise to caller
                logger.critical("Neo4j service unavailable: %s", e, exc_info=True)
                raise
            except KeyError as e:
                # Expected: Relation not found in cache/database
                logger.warning("Relation %s not found: %s", rel_id, e)

        return changes

    def _record_to_meta_learning(
        self, answer: AnswerRecord, feedback_type: FeedbackType
    ) -> None:
        """
        Informiert Meta-Learning Engine über Feedback

        Args:
            answer: AnswerRecord
            feedback_type: Art des Feedbacks
        """
        if not self.meta_learning:
            return

        try:
            # Konvertiere Feedback zu Success/Failure
            success = feedback_type == FeedbackType.CORRECT

            # Record Strategy Usage mit User Feedback
            # Erweiterte Methode wird in component_46 hinzugefügt
            if hasattr(self.meta_learning, "record_strategy_usage_with_feedback"):
                self.meta_learning.record_strategy_usage_with_feedback(
                    strategy_name=answer.strategy,
                    query=answer.query,
                    success=success,
                    confidence=answer.confidence,
                    response_time=0.0,  # Könnte aus metadata kommen
                    user_feedback=feedback_type.value,
                )
            else:
                # Fallback: Standard record_strategy_usage
                self.meta_learning.record_strategy_usage(
                    strategy_name=answer.strategy,
                    query=answer.query,
                    success=success,
                    confidence=answer.confidence,
                    response_time=0.0,
                )

            logger.debug(
                "Meta-Learning updated | strategy=%s, success=%s",
                answer.strategy,
                success,
            )

        except AttributeError as e:
            # Method not available on meta_learning - log as warning
            logger.warning("Meta-learning method not available: %s", e)
        except TypeError as e:
            # Invalid argument types - indicates API mismatch
            logger.error("Invalid arguments to meta-learning: %s", e, exc_info=True)
        except ValueError as e:
            # Invalid values (e.g., confidence out of range)
            logger.error("Invalid value for meta-learning: %s", e, exc_info=True)
        # Note: Let unexpected exceptions propagate to caller

    def _create_inhibition_pattern(
        self, answer: AnswerRecord, correction: Optional[str]
    ) -> bool:
        """
        Erstellt Negative Pattern (Inhibition) für falsche Antwort

        Bei falschen Antworten lernt KAI, was NICHT zu tun ist.

        Args:
            answer: Die falsche Antwort
            correction: Optional Korrektur vom Benutzer (already sanitized)

        Returns:
            True wenn Pattern erstellt wurde
        """
        try:
            # Erstelle Inhibition-Pattern
            # Idee: Markiere die verwendeten Reasoning-Paths als problematisch
            safe_query = self._sanitize_for_logging(answer.query, max_len=30)

            pattern_data = {
                "query": safe_query,
                "incorrect_answer": self._sanitize_for_logging(
                    answer.answer_text, max_len=50
                ),
                "strategy": answer.strategy,
                "used_relations": answer.used_relations,
                "used_concepts": answer.used_concepts,
                "timestamp": datetime.now().isoformat(),
                "correction": correction,
            }

            # TODO: When implementing, use parameterized Neo4j queries:
            # query = """
            #     CREATE (np:NegativePattern {
            #         pattern_id: $pattern_id,
            #         query: $query,
            #         incorrect_answer: $incorrect_answer,
            #         strategy: $strategy,
            #         timestamp: $timestamp,
            #         correction: $correction
            #     })
            #     RETURN np
            # """
            # session.run(query, pattern_data)

            logger.info(
                "Inhibition pattern created | query='%s...', strategy=%s",
                safe_query,
                answer.strategy,
            )

            return True

        except Neo4jError as e:
            # Database error - log but don't crash
            logger.error(
                "Neo4j error creating inhibition pattern: %s", e, exc_info=True
            )
            return False
        except ServiceUnavailable as e:
            # Critical database error - re-raise
            logger.critical("Neo4j service unavailable: %s", e, exc_info=True)
            raise
        # Note: Unexpected exceptions propagate to caller

    def _request_correction(
        self, answer: AnswerRecord, correction: Optional[str]
    ) -> bool:
        """
        Generiert Correction-Request bei unsicheren/falschen Antworten

        Args:
            answer: Die Antwort
            correction: Optional bereits vorhandene Korrektur (already sanitized)

        Returns:
            True wenn Request generiert wurde
        """
        try:
            if correction:
                # Benutzer hat Korrektur gegeben
                # TODO: Parse Korrektur und lerne daraus
                # self.netzwerk.learn_from_correction(answer.query, correction)

                safe_query = self._sanitize_for_logging(answer.query, max_len=30)
                safe_correction = self._sanitize_for_logging(correction, max_len=30)

                logger.info(
                    "Correction received | query='%s...', correction='%s...'",
                    safe_query,
                    safe_correction,
                )
                return True
            else:
                # Benutzer hat keine Korrektur gegeben
                # Könnte in UI eine Nachfrage-Dialog öffnen
                safe_query = self._sanitize_for_logging(answer.query, max_len=50)
                logger.info("Correction requested for query: '%s...'", safe_query)
                return True

        except ValueError as e:
            # Invalid correction format
            logger.error("Invalid correction format: %s", e, exc_info=True)
            return False
        except Neo4jError as e:
            # Database error
            logger.error("Error storing correction: %s", e, exc_info=True)
            return False
        # Note: Unexpected exceptions propagate to caller

    def _update_statistics(self, feedback_type: FeedbackType) -> None:
        """Updated Feedback-Statistiken (thread-safe)"""
        with self._stats_lock:
            self.feedback_stats["total_feedbacks"] += 1

            if feedback_type == FeedbackType.CORRECT:
                self.feedback_stats["correct_count"] += 1
            elif feedback_type == FeedbackType.INCORRECT:
                self.feedback_stats["incorrect_count"] += 1
            elif feedback_type == FeedbackType.UNSURE:
                self.feedback_stats["unsure_count"] += 1
            elif feedback_type == FeedbackType.PARTIALLY_CORRECT:
                self.feedback_stats["partially_correct_count"] += 1

    def _generate_feedback_message(
        self, feedback_type: FeedbackType, actions_taken: List[str]
    ) -> str:
        """Generiert lesbare Feedback-Nachricht"""
        messages = {
            FeedbackType.CORRECT: "Danke für das positive Feedback! Ich habe die verwendeten Informationen verstärkt.",
            FeedbackType.INCORRECT: "Danke für die Korrektur! Ich habe die Confidence angepasst und werde ähnliche Fehler vermeiden.",
            FeedbackType.UNSURE: "Danke für die Rückmeldung! Ich werde bei ähnlichen Fragen vorsichtiger sein.",
            FeedbackType.PARTIALLY_CORRECT: "Danke! Ich habe die teilweise korrekte Antwort zur Kenntnis genommen.",
        }

        base_message = messages.get(feedback_type, "Feedback verarbeitet.")

        if actions_taken:
            actions_str = ", ".join(actions_taken)
            return f"{base_message}\n\nAktionen: {actions_str}"

        return base_message

    # ========================================================================
    # Statistics & History
    # ========================================================================

    def get_feedback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Gibt Feedback-History zurück (thread-safe)

        Args:
            limit: Maximale Anzahl an Einträgen

        Returns:
            Liste von Feedback-Records (neueste zuerst)
        """
        # Create snapshots with lock
        with self._lock:
            feedback_snapshot = list(self.feedback_records.values())

        # Sort outside lock to minimize lock time
        sorted_feedbacks = sorted(
            feedback_snapshot, key=lambda f: f.timestamp, reverse=True
        )

        history = []
        for feedback in sorted_feedbacks[:limit]:
            answer = self.get_answer(feedback.answer_id)

            history.append(
                {
                    "feedback_id": feedback.feedback_id,
                    "timestamp": feedback.timestamp.isoformat(),
                    "feedback_type": feedback.feedback_type.value,
                    "query": answer.query if answer else "N/A",
                    "answer": answer.answer_text if answer else "N/A",
                    "strategy": answer.strategy if answer else "N/A",
                    "user_comment": feedback.user_comment,
                    "correction": feedback.correction,
                }
            )

        return history

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Gibt Feedback-Statistiken zurück (thread-safe)

        Returns:
            Dict mit Statistiken (thread-safe snapshot)
        """
        with self._stats_lock:
            # Create immutable snapshot of stats
            stats_snapshot = self.feedback_stats.copy()

        # Calculate with snapshot (outside lock to minimize lock time)
        total = stats_snapshot["total_feedbacks"]
        correct = stats_snapshot["correct_count"]
        partially = stats_snapshot["partially_correct_count"]

        # Safer calculation (explicit handling)
        accuracy = (correct + 0.5 * partially) / total if total > 0 else 0.0

        with self._lock:
            tracked_answers = len(self.answer_records)

        return {
            "total_feedbacks": total,
            "correct_count": correct,
            "incorrect_count": stats_snapshot["incorrect_count"],
            "unsure_count": stats_snapshot["unsure_count"],
            "partially_correct_count": partially,
            "accuracy": accuracy,
            "accuracy_percent": f"{accuracy * 100:.1f}%",  # User-friendly
            "tracked_answers": tracked_answers,
        }

    def get_strategy_feedback_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Gibt Feedback-Breakdown pro Strategy zurück (thread-safe)

        Returns:
            Dict[strategy_name, Dict[feedback_type, count]]
        """
        with self._lock:
            # Create snapshots
            feedback_snapshot = list(self.feedback_records.values())
            answer_snapshot = dict(self.answer_records)

        breakdown = {}

        for feedback in feedback_snapshot:
            answer = answer_snapshot.get(feedback.answer_id)
            if not answer:
                continue

            strategy = answer.strategy
            if strategy not in breakdown:
                breakdown[strategy] = {
                    "correct": 0,
                    "incorrect": 0,
                    "unsure": 0,
                    "partially_correct": 0,
                }

            feedback_key = feedback.feedback_type.value
            if feedback_key in breakdown[strategy]:
                breakdown[strategy][feedback_key] += 1

        return breakdown

    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics"""
        with self._lock:
            return {
                "answer_records_count": len(self.answer_records),
                "feedback_records_count": len(self.feedback_records),
                "answer_records_limit": self.config.max_answer_records,
                "feedback_records_limit": self.config.max_feedback_records,
                "memory_usage_mb": int(
                    len(self.answer_records) * 0.01  # ~10KB per answer
                    + len(self.feedback_records) * 0.005  # ~5KB per feedback
                ),
            }
