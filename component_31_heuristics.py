"""
Component 31: Planning Heuristics

Heuristic functions for guided search in state-space planning:
- RelaxedPlanHeuristic: Ignore delete effects approximation
- SetCoverHeuristic: Weighted set cover for goal achievement

Heuristics estimate the cost from a state to the goal, enabling
efficient A* search in the StateSpacePlanner.

Author: KAI Development Team
Date: 2025-01-30
Refactored: 2025-11-29 (Architecture Phase 4)
"""

from typing import List, Set, Tuple

from component_31_planner_core import Action, State

# ============================================================================
# Heuristics
# ============================================================================


class Heuristic:
    """Base class for planning heuristics."""

    def estimate(self, state: State, goal: Set[Tuple[str, ...]]) -> float:
        """Estimate cost from state to goal."""
        raise NotImplementedError


class RelaxedPlanHeuristic(Heuristic):
    """
    Heuristic based on relaxed planning (ignore delete effects).

    Approximates optimal cost by counting actions needed if
    delete effects were ignored.
    """

    def estimate(self, state: State, goal: Set[Tuple[str, ...]]) -> float:
        """Count unsatisfied goal propositions (simple approximation)."""
        unsatisfied = goal - state.propositions
        return float(len(unsatisfied))


class SetCoverHeuristic(Heuristic):
    """
    Heuristic based on set cover approximation.

    Estimates minimum actions needed to achieve all goal propositions
    by treating it as a weighted set cover problem.
    """

    def __init__(self, actions: List[Action]):
        self.actions = actions

    def estimate(self, state: State, goal: Set[Tuple[str, ...]]) -> float:
        """Greedy set cover for goal propositions."""
        uncovered = goal - state.propositions
        if not uncovered:
            return 0.0

        cost = 0.0
        while uncovered:
            # Find action that covers most uncovered goals
            best_action = None
            best_coverage = 0

            for action in self.actions:
                coverage = len(uncovered & action.add_effects)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_action = action

            if best_action is None:
                # No action covers remaining goals (unreachable)
                return float("inf")

            uncovered -= best_action.add_effects
            cost += best_action.cost

        return cost
