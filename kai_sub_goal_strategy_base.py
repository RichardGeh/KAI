# kai_sub_goal_strategy_base.py
"""
Sub-Goal Execution Base Strategy

Abstract base class for all sub-goal execution strategies.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from component_5_linguistik_strukturen import SubGoal

logger = logging.getLogger(__name__)


class SubGoalStrategy(ABC):
    """
    Abstrakte Basisklasse fuer Sub-Goal Execution Strategies.

    Jede Strategy ist verantwortlich fuer die Ausfuehrung einer bestimmten
    Kategorie von Sub-Goals (z.B. Fragen, Pattern Learning, etc.).
    """

    def __init__(self, worker):
        """
        Initialisiert die Strategy mit Referenz zum Worker.

        Args:
            worker: KaiWorker-Instanz fuer Zugriff auf Subsysteme
        """
        self.worker = worker

    @abstractmethod
    def can_handle(self, sub_goal_description: str) -> bool:
        """
        Prueft ob diese Strategy das gegebene Sub-Goal handhaben kann.

        Args:
            sub_goal_description: Beschreibung des Sub-Goals

        Returns:
            True wenn Strategy verantwortlich ist, sonst False
        """

    @abstractmethod
    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Fuehrt das Sub-Goal aus.

        Args:
            sub_goal: Das auszufuehrende SubGoal
            context: Kontext-Dictionary mit vorherigen Ergebnissen

        Returns:
            Tuple (success: bool, result: Dict)
        """
