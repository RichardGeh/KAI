# kai_sub_goal_executor.py
"""
Sub-Goal Execution Module für KAI mit Strategy Pattern

Verantwortlichkeiten:
- Dispatching von SubGoals an spezialisierte Strategien
- Implementierung verschiedener Execution-Strategien (Question, Learning, etc.)
- Entkopplung von Orchestrierung und Ausführungslogik

Author: KAI Development Team
Date: 2025-11-14 (Refactored)
"""

import logging
from typing import Any, Dict, Tuple

from component_5_linguistik_strukturen import SubGoal
from kai_sub_goal_strategies_advanced import (
    EpisodicMemoryStrategy,
    IntrospectionStrategy,
    OrchestratedStrategy,
    SharedStrategy,
)
from kai_sub_goal_strategies_core import (
    DefinitionStrategy,
    LearningStrategy,
    QuestionStrategy,
)
from kai_sub_goal_strategies_dialogue import (
    ClarificationStrategy,
    ConfirmationStrategy,
)
from kai_sub_goal_strategies_input import (
    IngestionStrategy,
    PatternLearningStrategy,
)
from kai_sub_goal_strategies_specialized import (
    ArithmeticStrategy,
    FileReaderStrategy,
    SpatialReasoningStrategy,
)

logger = logging.getLogger(__name__)


class SubGoalExecutor:
    """
    Hauptklasse für Sub-Goal Execution mit Strategy-Dispatching.

    Diese Klasse koordiniert die Ausführung von SubGoals, indem sie
    diese an die passenden Strategien weiterleitet.
    """

    def __init__(self, worker):
        """
        Initialisiert den Executor mit allen Strategien.

        Args:
            worker: KaiWorker-Instanz für Zugriff auf Subsysteme
        """
        self.worker = worker

        # Strategien in Prioritätsreihenfolge
        # (Spezifischere Strategien zuerst)
        self.strategies = [
            OrchestratedStrategy(
                worker
            ),  # Orchestrierte Multi-Segment-Verarbeitung (HÖCHSTE PRIORITÄT)
            ConfirmationStrategy(worker),
            ClarificationStrategy(worker),
            IntrospectionStrategy(worker),  # Production Rule Introspection (PHASE 9)
            ArithmeticStrategy(worker),  # Arithmetische Berechnungen (Phase Mathematik)
            SpatialReasoningStrategy(worker),  # Räumliches Reasoning (Phase 9)
            EpisodicMemoryStrategy(worker),  # Episodisches Gedächtnis
            FileReaderStrategy(worker),  # Datei-Ingestion (Phase 4)
            PatternLearningStrategy(worker),
            IngestionStrategy(worker),
            DefinitionStrategy(worker),
            LearningStrategy(worker),
            QuestionStrategy(worker),
            SharedStrategy(worker),  # Fallback für gemeinsame Sub-Goals
        ]

    def execute_sub_goal(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Führt ein Sub-Goal aus, indem es an die passende Strategy dispatched wird.

        Args:
            sub_goal: Das auszuführende SubGoal
            context: Kontext-Dictionary mit vorherigen Ergebnissen

        Returns:
            Tuple (success: bool, result: Dict)
        """
        logger.debug(f"Dispatching SubGoal: '{sub_goal.description[:50]}...'")

        # Finde passende Strategy
        for strategy in self.strategies:
            if strategy.can_handle(sub_goal.description):
                logger.debug(f"  -> Verwendet {strategy.__class__.__name__}")
                return strategy.execute(sub_goal, context)

        # Keine Strategy gefunden
        logger.error(f"Keine Strategy gefunden für SubGoal: '{sub_goal.description}'")
        return False, {"error": f"Unbekannter Sub-Goal-Typ: {sub_goal.description}"}
