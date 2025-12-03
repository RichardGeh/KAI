# kai_response_formatter.py
"""
Response Formatting Module für KAI (FACADE)

Verantwortlichkeiten:
- Haupt-Schnittstelle für Response-Generierung
- Delegiert an spezialisierte Module:
  * kai_response_pipeline: Pipeline-basierte Formatierung
  * kai_response_production: Production System Generierung
  * kai_response_router: A/B Testing Router
- Bietet 100% Backward Compatibility für bestehenden Code

REFACTORED: 2025-12-01
- Reduziert von 1,258 Zeilen auf ~150 Zeilen
- Folgt Facade-Pattern wie kai_inference_handler
- Alle Public APIs bleiben unverändert
"""
import logging
from typing import Any, Dict, List, Optional

from component_50_self_evaluation import (
    RecommendationType,
    SelfEvaluator,
)
from component_confidence_manager import get_confidence_manager
from component_utils_text_normalization import clean_entity as normalize_entity

# Import neue Module
from kai_response_pipeline import KaiResponse, PipelineResponseGenerator
from kai_response_production import ProductionResponseGenerator
from kai_response_router import ResponseGenerationRouter as Router

# Optional: FeedbackHandler für Answer Tracking
try:
    from component_51_feedback_handler import FeedbackHandler

    FEEDBACK_HANDLER_AVAILABLE = True
except ImportError:
    FEEDBACK_HANDLER_AVAILABLE = False
    FeedbackHandler = None

logger = logging.getLogger(__name__)

# Re-export KaiResponse für backward compatibility
__all__ = ["KaiResponse", "KaiResponseFormatter", "ResponseGenerationRouter"]


class KaiResponseFormatter:
    """
    Formatter für KAI-Antworten basierend auf Fragetyp und Wissensstand (FACADE).

    Diese Klasse delegiert an spezialisierte Module:
    - PipelineResponseGenerator: Template-basierte Formatierung
    - ProductionResponseGenerator: Production System Generierung
    - Text-Normalisierung: component_utils_text_normalization

    PHASE: Confidence-Based Learning Integration
    - Verwendet ConfidenceManager für einheitliches Confidence-Feedback
    - Generiert confidence-aware Antworten für alle Reasoning-Typen
    """

    def __init__(self, feedback_handler: Optional[Any] = None):
        """
        Initialisiert den Formatter mit globalem ConfidenceManager und SelfEvaluator.

        Args:
            feedback_handler: Optional FeedbackHandler für Answer-Tracking
        """
        self.confidence_manager = get_confidence_manager()
        self.self_evaluator = SelfEvaluator()
        self.feedback_handler = feedback_handler

        # Initialisiere spezialisierte Generatoren
        self.pipeline_generator = PipelineResponseGenerator(
            self.confidence_manager, self.self_evaluator
        )
        self.production_generator = ProductionResponseGenerator()

        if self.feedback_handler:
            logger.info(
                "KaiResponseFormatter initialisiert mit ConfidenceManager, SelfEvaluator und FeedbackHandler"
            )
        else:
            logger.info(
                "KaiResponseFormatter initialisiert mit ConfidenceManager und SelfEvaluator"
            )

    # ========================================================================
    # DELEGATION: Text Normalization
    # ========================================================================

    @staticmethod
    def clean_entity(entity_text: str) -> str:
        """
        Entfernt führende Artikel, bereinigt den Text und normalisiert Plurale zu Singularen.

        REFACTORED: Delegiert an zentrale component_utils_text_normalization.

        Args:
            entity_text: Der zu bereinigende Text

        Returns:
            Bereinigter und normalisierter Text
        """
        return normalize_entity(entity_text)

    # ========================================================================
    # DELEGATION: Confidence-Based Formatting
    # ========================================================================

    def format_confidence_prefix(
        self, confidence: float, reasoning_type: str = "standard"
    ) -> str:
        """
        Generiert einen Confidence-aware Präfix für Antworten.

        DELEGATION: pipeline_generator.format_confidence_prefix()

        Args:
            confidence: Confidence-Wert (0.0-1.0)
            reasoning_type: Art des Reasoning

        Returns:
            Formatierter Präfix-String
        """
        return self.pipeline_generator.format_confidence_prefix(
            confidence, reasoning_type
        )

    def format_low_confidence_warning(self, confidence: float) -> str:
        """
        Generiert eine Warnung für niedrige Confidence-Werte.

        DELEGATION: pipeline_generator.format_low_confidence_warning()

        Args:
            confidence: Confidence-Wert (0.0-1.0)

        Returns:
            Warnungs-String oder leerer String (bei hoher Confidence)
        """
        return self.pipeline_generator.format_low_confidence_warning(confidence)

    # ========================================================================
    # DELEGATION: Self-Evaluation
    # ========================================================================

    def evaluate_and_enrich_response(
        self,
        question: str,
        answer_text: str,
        confidence: float,
        strategy: str = "unknown",
        used_relations: Optional[List[str]] = None,
        used_concepts: Optional[List[str]] = None,
        proof_tree: Optional[Any] = None,
        reasoning_paths: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        track_for_feedback: bool = True,
    ) -> Dict[str, Any]:
        """
        Führt Self-Evaluation durch und reichert Antwort mit Warnungen an.

        ERWEITERT in Phase 3.4: Answer Tracking für User Feedback Loop

        Args:
            question: Die ursprüngliche Frage
            answer_text: Die generierte Antwort
            confidence: Claimed confidence
            strategy: Verwendete Reasoning-Strategy
            used_relations: Optional Liste von Relation-IDs
            used_concepts: Optional Liste von Konzept-IDs
            proof_tree: Optional ProofTree Objekt
            reasoning_paths: Optional Liste von ReasoningPaths
            context: Optional zusätzlicher Kontext
            track_for_feedback: Ob Antwort für Feedback getrackt werden soll

        Returns:
            Dict mit enriched response data
        """
        # Erstelle Answer-Dict für Evaluator
        answer_dict = {
            "text": answer_text,
            "confidence": confidence,
            "proof_tree": proof_tree,
            "reasoning_paths": reasoning_paths or [],
        }

        # Führe Evaluation durch
        evaluation = self.self_evaluator.evaluate_answer(question, answer_dict, context)

        # Angereicherte Antwort erstellen
        enriched_text = answer_text
        warnings = []

        # 1. Füge Unsicherheiten als Warnungen hinzu
        if evaluation.uncertainties:
            warnings.append("\n[WARNING] UNSICHERHEITEN:")
            for uncertainty in evaluation.uncertainties:
                warnings.append(f"  [BULLET] {uncertainty}")

        # 2. Prüfe Recommendation
        if evaluation.recommendation == RecommendationType.SHOW_WITH_WARNING:
            warnings.append(
                f"\n[INFO] Diese Antwort hat eine Evaluation-Score von {evaluation.overall_score:.0%}. "
                "Bitte mit Vorsicht interpretieren."
            )
        elif evaluation.recommendation == RecommendationType.REQUEST_CLARIFICATION:
            warnings.append(
                "\n[WARNING] Die Antwort erscheint unvollständig. "
                "Bitte stelle die Frage präziser oder mit mehr Kontext."
            )
        elif (
            evaluation.recommendation
            == RecommendationType.RETRY_WITH_DIFFERENT_STRATEGY
        ):
            warnings.append(
                "\n[WARNING] Diese Reasoning-Strategy liefert möglicherweise keine optimale Antwort. "
                "Versuche es mit einer alternativen Formulierung."
            )
        elif evaluation.recommendation == RecommendationType.INSUFFICIENT_KNOWLEDGE:
            warnings.append(
                "\n[WARNING] Mein Wissen zu diesem Thema ist unzureichend. "
                "Bitte lehre mich mehr über dieses Thema."
            )

        # 3. Confidence-Adjustment
        adjusted_confidence = confidence
        if (
            evaluation.confidence_adjusted
            and evaluation.suggested_confidence is not None
        ):
            adjusted_confidence = evaluation.suggested_confidence
            warnings.append(
                f"\n[INFO] Confidence wurde angepasst: {confidence:.0%} -> {adjusted_confidence:.0%} "
                f"(Grund: Beweislage unzureichend)"
            )

        # 4. Füge Standard-Confidence-Warning hinzu
        confidence_warning = self.format_low_confidence_warning(adjusted_confidence)
        if confidence_warning:
            warnings.append(confidence_warning)

        # Kombiniere Antwort mit Warnungen
        if warnings:
            enriched_text = answer_text + "\n" + "\n".join(warnings)

        # 5. Answer Tracking für Feedback (optional)
        answer_id = None
        if track_for_feedback and self.feedback_handler:
            try:
                answer_id = self.feedback_handler.track_answer(
                    query=question,
                    answer_text=enriched_text,
                    confidence=adjusted_confidence,
                    strategy=strategy,
                    used_relations=used_relations,
                    used_concepts=used_concepts,
                    proof_tree=proof_tree,
                    reasoning_paths=reasoning_paths,
                    evaluation_score=evaluation.overall_score,
                    metadata={
                        "original_confidence": confidence,
                        "confidence_adjusted": evaluation.confidence_adjusted,
                        "recommendation": evaluation.recommendation.value,
                    },
                )
                logger.debug(f"Answer tracked for feedback | id={answer_id[:8]}")
            except Exception as e:
                logger.warning(f"Could not track answer for feedback: {e}")

        return {
            "text": enriched_text,
            "confidence": adjusted_confidence,
            "evaluation": evaluation,
            "warnings": warnings,
            "answer_id": answer_id,
            "strategy": strategy,
        }

    # ========================================================================
    # DELEGATION: Pipeline Format Methods
    # ========================================================================

    @staticmethod
    def format_person_answer(
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
    ) -> str:
        """Formatiert Antwort auf Wer-Frage. DELEGATION: PipelineResponseGenerator"""
        return PipelineResponseGenerator.format_person_answer(
            topic, facts, bedeutungen, synonyms
        )

    @staticmethod
    def format_time_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """Formatiert Antwort auf Wann-Frage. DELEGATION: PipelineResponseGenerator"""
        return PipelineResponseGenerator.format_time_answer(topic, facts, bedeutungen)

    @staticmethod
    def format_process_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """Formatiert Antwort auf Wie-Frage. DELEGATION: PipelineResponseGenerator"""
        return PipelineResponseGenerator.format_process_answer(
            topic, facts, bedeutungen
        )

    @staticmethod
    def format_reason_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """Formatiert Antwort auf Warum-Frage. DELEGATION: PipelineResponseGenerator"""
        return PipelineResponseGenerator.format_reason_answer(topic, facts, bedeutungen)

    def format_standard_answer(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query_type: str = "normal",
        backward_chaining_used: bool = False,
        is_hypothesis: bool = False,
        confidence: Optional[float] = None,
    ) -> str:
        """Formatiert Standard-Antwort. DELEGATION: pipeline_generator"""
        return self.pipeline_generator.format_standard_answer(
            topic,
            facts,
            bedeutungen,
            synonyms,
            query_type,
            backward_chaining_used,
            is_hypothesis,
            confidence,
        )

    @staticmethod
    def format_episodic_answer(
        episodes: List[Dict], query_type: str, topic: Optional[str] = None
    ) -> str:
        """Formatiert episodische Antwort. DELEGATION: PipelineResponseGenerator"""
        return PipelineResponseGenerator.format_episodic_answer(
            episodes, query_type, topic
        )

    @staticmethod
    def format_spatial_answer(
        model_type: str,
        spatial_query_type: str,
        entities: List[Dict] = None,
        positions: Dict = None,
        relations: List[Dict] = None,
        plan: List = None,
        plan_length: int = 0,
        reachable: bool = True,
    ) -> str:
        """Formatiert räumliche Antwort. DELEGATION: PipelineResponseGenerator"""
        return PipelineResponseGenerator.format_spatial_answer(
            model_type,
            spatial_query_type,
            entities,
            positions,
            relations,
            plan,
            plan_length,
            reachable,
        )

    # ========================================================================
    # DELEGATION: Production System
    # ========================================================================

    def generate_with_production_system(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query_type: str = "normal",
        confidence: Optional[float] = None,
        production_engine: Optional[Any] = None,
        signals: Optional[Any] = None,
    ) -> KaiResponse:
        """
        Generiert Antwort mit Production System.

        DELEGATION: production_generator.generate_with_production_system()

        Args:
            topic, facts, bedeutungen, synonyms: Standard args
            query_type: Typ der Query
            confidence: Optional Confidence
            production_engine: Optional ProductionSystemEngine
            signals: Optional KaiSignals

        Returns:
            KaiResponse
        """
        response = self.production_generator.generate_with_production_system(
            topic,
            facts,
            bedeutungen,
            synonyms,
            query_type,
            confidence,
            production_engine,
            signals,
        )

        # Fallback to pipeline if production fails
        if response.strategy == "production_error":
            fallback_text = self.format_standard_answer(
                topic, facts, bedeutungen, synonyms, query_type, confidence=confidence
            )
            return KaiResponse(
                text=fallback_text,
                trace=response.trace + ["Fallback to pipeline"],
                confidence=confidence or 0.5,
                strategy="pipeline_fallback",
            )

        return response


# ============================================================================
# Response Generation Router (A/B Testing) - RE-EXPORT
# ============================================================================

# Re-export ResponseGenerationRouter für backward compatibility
ResponseGenerationRouter = Router
