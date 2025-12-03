"""
Component 31: State-Space Planner Core

Core planning functionality including:
- State representation (propositions)
- Action models (STRIPS-style preconditions/effects)
- Planning problem definition
- A* search with heuristic guidance
- Plan validation and simulation
- Diagnostic reasoning (root-cause analysis)

This module contains the fundamental planning primitives and search algorithms.

Author: KAI Development Team
Date: 2025-01-30
Refactored: 2025-11-29 (Architecture Phase 4)
Updated: 2025-12-02 | Added BaseReasoningEngine interface (Phase 5)
"""

import heapq
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofStep, ProofTree, StepType
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult

logger = get_logger(__name__)


def safe_str(obj: Any) -> str:
    """
    Convert object to cp1252-safe string for Windows compatibility.

    Replaces characters that cannot be encoded in Windows cp1252 to prevent
    UnicodeEncodeError in logging and console output.

    Args:
        obj: Object to convert to string

    Returns:
        cp1252-safe string representation
    """
    s = str(obj)
    try:
        # Test if string can be encoded in cp1252
        s.encode("cp1252")
        return s
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Replace problematic characters
        return s.encode("cp1252", errors="replace").decode("cp1252")


# ============================================================================
# State Representation
# ============================================================================


@dataclass
class State:
    """
    Represents a world state as a set of propositions.

    Propositions are tuples: (predicate, *args)
    Example: ("on", "A", "B") means "Block A is on Block B"
    """

    propositions: Set[Tuple[str, ...]] = field(default_factory=set)
    timestamp: Optional[int] = None  # For temporal reasoning

    def __hash__(self):
        """Make State hashable for closed set in search."""
        return hash(frozenset(self.propositions))

    def __eq__(self, other):
        """States are equal if propositions match."""
        if not isinstance(other, State):
            return False
        return self.propositions == other.propositions

    def satisfies(self, conditions: Set[Tuple[str, ...]]) -> bool:
        """Check if state satisfies all conditions."""
        return conditions.issubset(self.propositions)

    def copy(self) -> "State":
        """Create deep copy of state."""
        return State(propositions=deepcopy(self.propositions), timestamp=self.timestamp)

    def to_string(self) -> str:
        """Human-readable state description."""
        if not self.propositions:
            return "Empty State"
        props = sorted([safe_str(p) for p in self.propositions])
        return "\n".join(props)


# ============================================================================
# Action Model
# ============================================================================


@dataclass
class Action:
    """
    STRIPS-style action with preconditions and effects.

    Attributes:
        name: Action identifier
        params: List of parameter names (e.g., ["x", "y"])
        preconditions: Set of propositions that must be true
        add_effects: Propositions to add to state
        delete_effects: Propositions to remove from state
        cost: Action cost (default 1.0)
    """

    name: str
    params: List[str] = field(default_factory=list)
    preconditions: Set[Tuple[str, ...]] = field(default_factory=set)
    add_effects: Set[Tuple[str, ...]] = field(default_factory=set)
    delete_effects: Set[Tuple[str, ...]] = field(default_factory=set)
    cost: float = 1.0

    def is_applicable(self, state: State) -> bool:
        """Check if action can be executed in given state."""
        return state.satisfies(self.preconditions)

    def apply(self, state: State) -> State:
        """Apply action to state, returning new state."""
        if not self.is_applicable(state):
            raise ValueError(f"Action {self} not applicable in state")

        new_state = state.copy()
        new_state.propositions -= self.delete_effects
        new_state.propositions |= self.add_effects

        # Increment timestamp for temporal reasoning
        if new_state.timestamp is not None:
            new_state.timestamp += 1

        return new_state

    def instantiate(self, bindings: Dict[str, str]) -> "Action":
        """Create grounded action with specific parameter bindings."""

        def ground(prop: Tuple[str, ...]) -> Tuple[str, ...]:
            return tuple(bindings.get(arg, arg) for arg in prop)

        return Action(
            name=f"{self.name}({', '.join(bindings.values())})",
            params=[],  # Grounded action has no free params
            preconditions={ground(p) for p in self.preconditions},
            add_effects={ground(p) for p in self.add_effects},
            delete_effects={ground(p) for p in self.delete_effects},
            cost=self.cost,
        )

    def __str__(self):
        return self.name


# ============================================================================
# Planning Problem
# ============================================================================


@dataclass
class PlanningProblem:
    """
    Defines a planning problem instance.

    Attributes:
        initial_state: Starting state
        goal: Goal conditions (set of propositions)
        actions: Available action schemas
        objects: Domain objects for grounding actions
    """

    initial_state: State
    goal: Set[Tuple[str, ...]]
    actions: List[Action]
    objects: List[str] = field(default_factory=list)

    def is_goal(self, state: State) -> bool:
        """Check if state satisfies goal conditions."""
        return state.satisfies(self.goal)

    def get_applicable_actions(self, state: State) -> List[Action]:
        """Get all grounded actions applicable in state."""
        applicable = []

        for action_schema in self.actions:
            if not action_schema.params:
                # Already grounded
                if action_schema.is_applicable(state):
                    applicable.append(action_schema)
            else:
                # Need to ground with object bindings
                for grounded in self._ground_action(action_schema):
                    if grounded.is_applicable(state):
                        applicable.append(grounded)

        return applicable

    def _ground_action(self, action: Action) -> List[Action]:
        """Generate all possible groundings of action schema."""
        from itertools import product

        if not action.params:
            return [action]

        # Generate all combinations of objects for parameters
        groundings = []
        for binding_tuple in product(self.objects, repeat=len(action.params)):
            bindings = dict(zip(action.params, binding_tuple))
            groundings.append(action.instantiate(bindings))

        return groundings


# ============================================================================
# Search Node
# ============================================================================


@dataclass(order=True)
class SearchNode:
    """
    Node in search tree for A* algorithm.

    Attributes:
        f_score: Total estimated cost (g + h)
        state: World state at this node
        g_score: Cost from start to this node
        h_score: Heuristic estimate to goal
        parent: Parent node in search tree
        action: Action that led to this node
    """

    f_score: float
    state: State = field(compare=False)
    g_score: float = field(compare=False)
    h_score: float = field(compare=False)
    parent: Optional["SearchNode"] = field(default=None, compare=False)
    action: Optional[Action] = field(default=None, compare=False)

    def reconstruct_plan(self) -> List[Action]:
        """Reconstruct action sequence from root to this node."""
        plan = []
        node = self
        while node.parent is not None:
            if node.action:
                plan.append(node.action)
            node = node.parent
        return list(reversed(plan))


# ============================================================================
# State-Space Planner
# ============================================================================


class StateSpacePlanner(BaseReasoningEngine):
    """
    STRIPS-style planner with A* search.

    Features:
    - Forward search with heuristic guidance
    - Temporal reasoning support
    - Plan validation
    - Diagnostic reasoning (root-cause analysis)
    - BaseReasoningEngine interface for orchestration

    Capabilities:
    - Planning
    - STRIPS action models
    - State-space search
    - Heuristic guidance
    """

    def __init__(
        self,
        heuristic=None,
        max_expansions: int = 10000,
        state_constraint: Optional[Callable[[State], bool]] = None,
    ):
        """
        Initialize planner.

        Args:
            heuristic: Heuristic function for A* (default: RelaxedPlan from component_31_heuristics)
            max_expansions: Maximum nodes to expand before giving up
            state_constraint: Optional function to validate states (returns False to reject)
        """
        # Import here to avoid circular dependency
        if heuristic is None:
            from component_31_heuristics import RelaxedPlanHeuristic

            heuristic = RelaxedPlanHeuristic()

        self.heuristic = heuristic
        self.max_expansions = max_expansions
        self.state_constraint = state_constraint
        self.stats = {"expansions": 0, "generated": 0, "plan_length": 0}

    def solve(self, problem: PlanningProblem) -> Optional[List[Action]]:
        """
        Find plan to achieve goal from initial state.

        Args:
            problem: Planning problem instance

        Returns:
            List of actions (plan) or None if no solution found
        """
        logger.info(
            f"Starting planning from state with {len(problem.initial_state.propositions)} propositions"
        )
        logger.debug(f"Goal: {problem.goal}")

        # Initialize search
        self.stats = {"expansions": 0, "generated": 0, "plan_length": 0}

        initial_node = SearchNode(
            f_score=self.heuristic.estimate(problem.initial_state, problem.goal),
            state=problem.initial_state,
            g_score=0.0,
            h_score=self.heuristic.estimate(problem.initial_state, problem.goal),
        )

        open_list = [initial_node]
        closed_set = set()
        g_scores: Dict[State, float] = {problem.initial_state: 0.0}

        # A* search
        while open_list and self.stats["expansions"] < self.max_expansions:
            # Pop node with lowest f-score
            current = heapq.heappop(open_list)

            # Goal test
            if problem.is_goal(current.state):
                plan = current.reconstruct_plan()
                self.stats["plan_length"] = len(plan)
                logger.info(
                    f"Plan found! Length: {len(plan)}, Expansions: {self.stats['expansions']}"
                )
                return plan

            # Mark as explored
            closed_set.add(current.state)
            self.stats["expansions"] += 1

            # Expand successors
            for action in problem.get_applicable_actions(current.state):
                try:
                    successor_state = action.apply(current.state)
                except ValueError:
                    continue  # Action not actually applicable

                if successor_state in closed_set:
                    continue

                # Check state constraints (e.g., safety constraints)
                if self.state_constraint and not self.state_constraint(successor_state):
                    continue  # State violates constraints

                # Calculate costs
                tentative_g = current.g_score + action.cost

                if (
                    successor_state not in g_scores
                    or tentative_g < g_scores[successor_state]
                ):
                    # Found better path
                    g_scores[successor_state] = tentative_g
                    h_score = self.heuristic.estimate(successor_state, problem.goal)

                    successor_node = SearchNode(
                        f_score=tentative_g + h_score,
                        state=successor_state,
                        g_score=tentative_g,
                        h_score=h_score,
                        parent=current,
                        action=action,
                    )

                    heapq.heappush(open_list, successor_node)
                    self.stats["generated"] += 1

        logger.warning(f"No plan found after {self.stats['expansions']} expansions")
        return None

    def validate_plan(
        self, problem: PlanningProblem, plan: List[Action]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that plan achieves goal from initial state.

        Returns:
            (success, error_message)
        """
        state = problem.initial_state.copy()

        for i, action in enumerate(plan):
            if not action.is_applicable(state):
                return False, f"Action {i} ({action}) not applicable in state"

            try:
                state = action.apply(state)
            except ValueError as e:
                return False, f"Action {i} ({action}) failed: {e}"

        if not problem.is_goal(state):
            return False, "Final state does not satisfy goal"

        return True, None

    def simulate_plan(
        self, problem: PlanningProblem, plan: List[Action]
    ) -> List[State]:
        """
        Execute plan and return state trajectory.

        Args:
            problem: Planning problem
            plan: Action sequence

        Returns:
            List of states (including initial state)
        """
        states = [problem.initial_state.copy()]
        state = problem.initial_state.copy()

        for action in plan:
            state = action.apply(state)
            states.append(state.copy())

        return states

    def diagnose_failure(
        self, problem: PlanningProblem, plan: List[Action]
    ) -> Dict[str, Any]:
        """
        Analyze why plan fails (root-cause analysis).

        Returns:
            Diagnostic information:
            - failed_at: Action index where plan fails
            - failed_action: The failing action
            - missing_preconditions: Preconditions not satisfied
            - state_before: State before failed action
        """
        state = problem.initial_state.copy()

        for i, action in enumerate(plan):
            if not action.is_applicable(state):
                missing = action.preconditions - state.propositions

                return {
                    "failed_at": i,
                    "failed_action": action,
                    "missing_preconditions": missing,
                    "state_before": state,
                    "error": f"Action {action} requires {missing} but state only has {state.propositions}",
                }

            state = action.apply(state)

        # Plan executes but doesn't reach goal
        if not problem.is_goal(state):
            missing_goals = problem.goal - state.propositions
            return {
                "failed_at": len(plan),
                "failed_action": None,
                "missing_preconditions": missing_goals,
                "state_before": state,
                "error": f"Goal not achieved. Missing: {missing_goals}",
            }

        return {"error": None}

    # ========================================================================
    # BaseReasoningEngine Interface Implementation
    # ========================================================================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute planning reasoning on the given query.

        Context should contain:
        - 'planning_problem': PlanningProblem instance with initial state, goal, actions
        - 'enable_proof': Whether to generate proof tree (default: True)

        Args:
            query: Query string describing the planning goal
            context: Context with planning problem and parameters

        Returns:
            ReasoningResult with plan or failure indication
        """
        # Extract planning problem from context
        problem = context.get("planning_problem")
        if problem is None:
            return ReasoningResult(
                success=False,
                answer="No planning problem provided in context",
                confidence=0.0,
                strategy_used="state_space_planner",
                metadata={"error": "missing_planning_problem"},
            )

        # Attempt to solve
        plan = self.solve(problem)

        if plan is not None:
            # Found a plan
            # Validate it
            is_valid, error_msg = self.validate_plan(problem, plan)

            # Create proof tree showing planning steps
            proof_tree = None
            if context.get("enable_proof", True):
                proof_tree = self._create_plan_proof_tree(query, problem, plan)

            answer = f"Plan with {len(plan)} actions: {[str(a) for a in plan]}"

            return ReasoningResult(
                success=True,
                answer=answer,
                confidence=1.0 if is_valid else 0.8,
                proof_tree=proof_tree,
                strategy_used="state_space_planner_astar",
                computation_cost=self.stats["expansions"] / self.max_expansions,
                metadata={
                    "plan": [str(a) for a in plan],
                    "plan_length": len(plan),
                    "expansions": self.stats["expansions"],
                    "generated": self.stats["generated"],
                    "is_valid": is_valid,
                    "validation_error": error_msg,
                },
            )
        else:
            # No plan found
            return ReasoningResult(
                success=False,
                answer="No plan found within search limits",
                confidence=0.0,
                strategy_used="state_space_planner_astar",
                computation_cost=1.0,  # Used full budget
                metadata={
                    "expansions": self.stats["expansions"],
                    "max_expansions": self.max_expansions,
                    "reason": "search_exhausted",
                },
            )

    def _create_plan_proof_tree(
        self, query: str, problem: PlanningProblem, plan: List[Action]
    ) -> ProofTree:
        """
        Create ProofTree documenting the planning process.

        Args:
            query: Original planning query
            problem: Planning problem
            plan: Found plan

        Returns:
            ProofTree with planning steps
        """
        steps = []

        # Initial state
        steps.append(
            ProofStep(
                step_id="plan_initial_state",
                step_type=StepType.PREMISE,
                output=f"Initial state: {problem.initial_state.to_string()[:100]}...",
                explanation_text="Starting state of the planning problem",
                metadata={"state": str(problem.initial_state.propositions)},
                source_component="component_31_planner",
            )
        )

        # Goal
        steps.append(
            ProofStep(
                step_id="plan_goal",
                step_type=StepType.PREMISE,
                output=f"Goal: {problem.goal}",
                explanation_text="Goal conditions to achieve",
                metadata={"goal": str(problem.goal)},
                source_component="component_31_planner",
            )
        )

        # Actions in plan
        parent_step = "plan_initial_state"
        for i, action in enumerate(plan):
            step_id = f"plan_action_{i}"
            steps.append(
                ProofStep(
                    step_id=step_id,
                    step_type=StepType.INFERENCE,
                    output=f"Action {i+1}: {action.name}",
                    explanation_text=f"Apply action: {action.name}",
                    rule_name=action.name,
                    parent_steps=[parent_step],
                    confidence=1.0,
                    metadata={
                        "action_index": i,
                        "preconditions": str(action.preconditions),
                        "effects_add": str(action.add_effects),
                        "effects_delete": str(action.delete_effects),
                        "cost": action.cost,
                    },
                    source_component="component_31_planner",
                )
            )
            parent_step = step_id

        # Goal achieved
        steps.append(
            ProofStep(
                step_id="plan_goal_achieved",
                step_type=StepType.CONCLUSION,
                output="Goal achieved",
                explanation_text=f"Plan successfully achieves goal with {len(plan)} actions",
                parent_steps=[parent_step],
                confidence=1.0,
                metadata={
                    "plan_length": len(plan),
                    "expansions": self.stats["expansions"],
                },
                source_component="component_31_planner",
            )
        )

        return ProofTree(
            query=query,
            root_steps=steps,
            metadata={
                "planner": "StateSpacePlanner",
                "algorithm": "A*",
                "heuristic": type(self.heuristic).__name__,
                "plan_length": len(plan),
                "search_stats": self.stats.copy(),
            },
        )

    def get_capabilities(self) -> List[str]:
        """
        Return state-space planning capabilities.

        Returns:
            List of capability identifiers
        """
        return [
            "planning",
            "strips",
            "state_space_search",
            "astar_search",
            "heuristic_search",
            "forward_planning",
            "action_planning",
            "goal_achievement",
            "plan_validation",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for planning.

        Planning can be expensive depending on:
        - Size of state space
        - Number of actions
        - Branching factor
        - Heuristic quality

        Args:
            query: Query string (may contain hints about problem complexity)

        Returns:
            Estimated cost in [0.0, 1.0+] range
        """
        # Planning is generally medium to expensive
        # Could be more precise with problem analysis
        return 0.6  # Medium-high cost for planning problems
