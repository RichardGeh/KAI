"""
Component 31: State-Space Planner (STRIPS-style Planning)

Implements goal-based planning with:
- State representation (propositions)
- Action models (preconditions/effects)
- Forward/backward search (A*, heuristics)
- Temporal reasoning (state transitions)
- Diagnostic reasoning (root-cause analysis)

Author: KAI Development Team
Date: 2025-01-30
"""

import heapq
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


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
        props = sorted([str(p) for p in self.propositions])
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


# ============================================================================
# State-Space Planner
# ============================================================================


class StateSpacePlanner:
    """
    STRIPS-style planner with A* search.

    Features:
    - Forward search with heuristic guidance
    - Temporal reasoning support
    - Plan validation
    - Diagnostic reasoning (root-cause analysis)
    """

    def __init__(
        self,
        heuristic: Optional[Heuristic] = None,
        max_expansions: int = 10000,
        state_constraint: Optional[Callable[[State], bool]] = None,
    ):
        """
        Initialize planner.

        Args:
            heuristic: Heuristic function for A* (default: RelaxedPlan)
            max_expansions: Maximum nodes to expand before giving up
            state_constraint: Optional function to validate states (returns False to reject)
        """
        self.heuristic = heuristic or RelaxedPlanHeuristic()
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
    ) -> Dict[str, any]:
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


# ============================================================================
# Temporal Reasoning
# ============================================================================


@dataclass
class TemporalConstraint:
    """
    Temporal constraint between actions.

    Types:
    - BEFORE: action1 must occur before action2
    - AFTER: action1 must occur after action2
    - SIMULTANEOUS: actions must occur at same time
    """

    action1: str
    action2: str
    constraint_type: str  # "BEFORE", "AFTER", "SIMULTANEOUS"


class TemporalPlanner:
    """
    Extends StateSpacePlanner with temporal reasoning.

    Handles:
    - Temporal constraints between actions
    - Duration modeling
    - Concurrent action execution
    """

    def __init__(self, base_planner: StateSpacePlanner):
        self.base_planner = base_planner
        self.temporal_constraints: List[TemporalConstraint] = []

    def add_temporal_constraint(self, constraint: TemporalConstraint):
        """Add temporal constraint to planning problem."""
        self.temporal_constraints.append(constraint)

    def validate_temporal_plan(self, plan: List[Action]) -> Tuple[bool, Optional[str]]:
        """
        Check if plan satisfies all temporal constraints.

        Returns:
            (valid, error_message)
        """
        action_positions = {action.name: i for i, action in enumerate(plan)}

        for constraint in self.temporal_constraints:
            if constraint.action1 not in action_positions:
                continue
            if constraint.action2 not in action_positions:
                continue

            pos1 = action_positions[constraint.action1]
            pos2 = action_positions[constraint.action2]

            if constraint.constraint_type == "BEFORE":
                if pos1 >= pos2:
                    return (
                        False,
                        f"{constraint.action1} must occur before {constraint.action2}",
                    )
            elif constraint.constraint_type == "AFTER":
                if pos1 <= pos2:
                    return (
                        False,
                        f"{constraint.action1} must occur after {constraint.action2}",
                    )
            elif constraint.constraint_type == "SIMULTANEOUS":
                if pos1 != pos2:
                    return (
                        False,
                        f"{constraint.action1} and {constraint.action2} must be simultaneous",
                    )

        return True, None


# ============================================================================
# Domain-Specific Builders (for common planning domains)
# ============================================================================


class BlocksWorldBuilder:
    """
    Builder for classic Blocks World domain.

    Actions: stack, unstack, pickup, putdown
    Goal: Achieve specific block configuration
    """

    @staticmethod
    def create_actions() -> List[Action]:
        """Create Blocks World action schemas."""
        actions = []

        # Stack(x, y): Put block x on block y
        actions.append(
            Action(
                name="stack",
                params=["x", "y"],
                preconditions={("holding", "x"), ("clear", "y")},
                add_effects={("on", "x", "y"), ("clear", "x"), ("handempty",)},
                delete_effects={("holding", "x"), ("clear", "y")},
            )
        )

        # Unstack(x, y): Remove block x from block y
        actions.append(
            Action(
                name="unstack",
                params=["x", "y"],
                preconditions={("on", "x", "y"), ("clear", "x"), ("handempty",)},
                add_effects={("holding", "x"), ("clear", "y")},
                delete_effects={("on", "x", "y"), ("clear", "x"), ("handempty",)},
            )
        )

        # Pickup(x): Pick up block x from table
        actions.append(
            Action(
                name="pickup",
                params=["x"],
                preconditions={("ontable", "x"), ("clear", "x"), ("handempty",)},
                add_effects={("holding", "x")},
                delete_effects={("ontable", "x"), ("clear", "x"), ("handempty",)},
            )
        )

        # Putdown(x): Put block x on table
        actions.append(
            Action(
                name="putdown",
                params=["x"],
                preconditions={("holding", "x")},
                add_effects={("ontable", "x"), ("clear", "x"), ("handempty",)},
                delete_effects={("holding", "x")},
            )
        )

        return actions

    @staticmethod
    def create_problem(
        blocks: List[str], initial_config: Dict[str, str], goal_config: Dict[str, str]
    ) -> PlanningProblem:
        """
        Create Blocks World problem instance.

        Args:
            blocks: List of block names
            initial_config: Initial block positions {"A": "table", "B": "A"}
            goal_config: Goal block positions

        Returns:
            PlanningProblem instance
        """
        # Build initial state
        initial_props = {("handempty",)}

        for block in blocks:
            location = initial_config.get(block, "table")

            if location == "table":
                initial_props.add(("ontable", block))
            else:
                initial_props.add(("on", block, location))

            # Block is clear if no other block is on it
            if not any(initial_config.get(b) == block for b in blocks):
                initial_props.add(("clear", block))

        initial_state = State(propositions=initial_props, timestamp=0)

        # Build goal conditions
        goal_props = set()

        for block, location in goal_config.items():
            if location == "table":
                goal_props.add(("ontable", block))
            else:
                goal_props.add(("on", block, location))

        return PlanningProblem(
            initial_state=initial_state,
            goal=goal_props,
            actions=BlocksWorldBuilder.create_actions(),
            objects=blocks,
        )


class GridNavigationBuilder:
    """
    Builder for grid navigation domain.

    Actions: move_up, move_down, move_left, move_right
    Goal: Reach target position while avoiding obstacles
    """

    @staticmethod
    def create_actions(grid_size: Tuple[int, int]) -> List[Action]:
        """Create grid navigation action schemas."""
        max_x, max_y = grid_size
        actions = []

        # Move up
        for x in range(max_x):
            for y in range(max_y - 1):  # Can't move up from top row
                actions.append(
                    Action(
                        name=f"move_up_{x}_{y}",
                        preconditions={("at", str(x), str(y))},
                        add_effects={("at", str(x), str(y + 1))},
                        delete_effects={("at", str(x), str(y))},
                    )
                )

        # Move down
        for x in range(max_x):
            for y in range(1, max_y):  # Can't move down from bottom row
                actions.append(
                    Action(
                        name=f"move_down_{x}_{y}",
                        preconditions={("at", str(x), str(y))},
                        add_effects={("at", str(x), str(y - 1))},
                        delete_effects={("at", str(x), str(y))},
                    )
                )

        # Move right
        for x in range(max_x - 1):
            for y in range(max_y):
                actions.append(
                    Action(
                        name=f"move_right_{x}_{y}",
                        preconditions={("at", str(x), str(y))},
                        add_effects={("at", str(x + 1), str(y))},
                        delete_effects={("at", str(x), str(y))},
                    )
                )

        # Move left
        for x in range(1, max_x):
            for y in range(max_y):
                actions.append(
                    Action(
                        name=f"move_left_{x}_{y}",
                        preconditions={("at", str(x), str(y))},
                        add_effects={("at", str(x - 1), str(y))},
                        delete_effects={("at", str(x), str(y))},
                    )
                )

        return actions

    @staticmethod
    def create_problem(
        grid_size: Tuple[int, int],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: List[Tuple[int, int]] = None,
    ) -> PlanningProblem:
        """
        Create grid navigation problem.

        Args:
            grid_size: (width, height)
            start: Starting position (x, y)
            goal: Goal position (x, y)
            obstacles: List of obstacle positions

        Returns:
            PlanningProblem instance
        """
        obstacles = obstacles or []

        # Initial state
        initial_props = {("at", str(start[0]), str(start[1]))}

        # Add obstacles as negative preconditions
        for ox, oy in obstacles:
            initial_props.add(("obstacle", str(ox), str(oy)))

        initial_state = State(propositions=initial_props, timestamp=0)

        # Goal
        goal_props = {("at", str(goal[0]), str(goal[1]))}

        # Filter out actions that move into obstacles
        all_actions = GridNavigationBuilder.create_actions(grid_size)
        valid_actions = []

        for action in all_actions:
            # Check if action would move into obstacle
            moves_into_obstacle = False
            for effect in action.add_effects:
                if effect[0] == "at":
                    x, y = effect[1], effect[2]
                    if ("obstacle", x, y) in initial_props:
                        moves_into_obstacle = True
                        break

            if not moves_into_obstacle:
                valid_actions.append(action)

        return PlanningProblem(
            initial_state=initial_state, goal=goal_props, actions=valid_actions
        )


class RiverCrossingBuilder:
    """
    Builder for classic river crossing puzzle.

    Problem: Farmer, fox, chicken, grain must cross river.
    Constraint: Fox eats chicken if left alone, chicken eats grain if left alone.
    Only farmer can row boat, boat holds 2 entities (farmer + 1 other).
    """

    @staticmethod
    def create_actions() -> List[Action]:
        """Create river crossing action schemas."""
        actions = []

        # Cross alone
        actions.append(
            Action(
                name="cross_alone",
                preconditions={("at", "farmer", "left"), ("at", "boat", "left")},
                add_effects={("at", "farmer", "right"), ("at", "boat", "right")},
                delete_effects={("at", "farmer", "left"), ("at", "boat", "left")},
            )
        )

        actions.append(
            Action(
                name="cross_alone_back",
                preconditions={("at", "farmer", "right"), ("at", "boat", "right")},
                add_effects={("at", "farmer", "left"), ("at", "boat", "left")},
                delete_effects={("at", "farmer", "right"), ("at", "boat", "right")},
            )
        )

        # Cross with entity
        for entity in ["fox", "chicken", "grain"]:
            # Left to right
            actions.append(
                Action(
                    name=f"cross_with_{entity}",
                    preconditions={
                        ("at", "farmer", "left"),
                        ("at", entity, "left"),
                        ("at", "boat", "left"),
                    },
                    add_effects={
                        ("at", "farmer", "right"),
                        ("at", entity, "right"),
                        ("at", "boat", "right"),
                    },
                    delete_effects={
                        ("at", "farmer", "left"),
                        ("at", entity, "left"),
                        ("at", "boat", "left"),
                    },
                )
            )

            # Right to left
            actions.append(
                Action(
                    name=f"cross_with_{entity}_back",
                    preconditions={
                        ("at", "farmer", "right"),
                        ("at", entity, "right"),
                        ("at", "boat", "right"),
                    },
                    add_effects={
                        ("at", "farmer", "left"),
                        ("at", entity, "left"),
                        ("at", "boat", "left"),
                    },
                    delete_effects={
                        ("at", "farmer", "right"),
                        ("at", entity, "right"),
                        ("at", "boat", "right"),
                    },
                )
            )

        return actions

    @staticmethod
    def is_safe_state(state: State) -> bool:
        """
        Check if state violates safety constraints.

        Returns False if:
        - Fox and chicken on same side without farmer
        - Chicken and grain on same side without farmer
        """
        for side in ["left", "right"]:
            farmer_here = ("at", "farmer", side) in state.propositions
            fox_here = ("at", "fox", side) in state.propositions
            chicken_here = ("at", "chicken", side) in state.propositions
            grain_here = ("at", "grain", side) in state.propositions

            # Check fox-chicken constraint
            if fox_here and chicken_here and not farmer_here:
                return False

            # Check chicken-grain constraint
            if chicken_here and grain_here and not farmer_here:
                return False

        return True

    @staticmethod
    def create_problem() -> PlanningProblem:
        """
        Create river crossing problem instance.

        Initial: All on left side
        Goal: All on right side
        """
        # Initial state: everyone on left
        initial_props = {
            ("at", "farmer", "left"),
            ("at", "fox", "left"),
            ("at", "chicken", "left"),
            ("at", "grain", "left"),
            ("at", "boat", "left"),
        }

        initial_state = State(propositions=initial_props, timestamp=0)

        # Goal: everyone on right
        goal_props = {
            ("at", "farmer", "right"),
            ("at", "fox", "right"),
            ("at", "chicken", "right"),
            ("at", "grain", "right"),
        }

        return PlanningProblem(
            initial_state=initial_state,
            goal=goal_props,
            actions=RiverCrossingBuilder.create_actions(),
        )


# ============================================================================
# Integration with Reasoning Engines
# ============================================================================


class HybridPlanner(StateSpacePlanner):
    """
    Extends StateSpacePlanner with logic engine and CSP integration.

    Features:
    - Infers additional preconditions from logic rules
    - Validates state constraints via CSP solver
    - Enriches action models with inferred effects
    """

    def __init__(
        self,
        logic_engine=None,
        csp_solver=None,
        heuristic: Optional[Heuristic] = None,
        max_expansions: int = 10000,
    ):
        """
        Initialize hybrid planner.

        Args:
            logic_engine: LogikEngine instance for inference
            csp_solver: ConstraintReasoningEngine for state validation
            heuristic: Heuristic for A* search
            max_expansions: Max search nodes
        """
        super().__init__(heuristic, max_expansions)
        self.logic_engine = logic_engine
        self.csp_solver = csp_solver

    def enrich_action_with_logic(self, action: Action, state: State) -> Action:
        """
        Use logic engine to infer additional effects/preconditions.

        Example: If action adds "on(A, B)", logic may infer "above(A, B)"
        """
        if not self.logic_engine:
            return action

        enriched_action = Action(
            name=action.name,
            params=action.params,
            preconditions=action.preconditions.copy(),
            add_effects=action.add_effects.copy(),
            delete_effects=action.delete_effects.copy(),
            cost=action.cost,
        )

        # Infer additional effects from add_effects
        # (This is a simplified version - real implementation would query logic engine)
        for effect in action.add_effects:
            if effect[0] == "on" and len(effect) == 3:
                # Infer transitive "above" relation
                enriched_action.add_effects.add(("above", effect[1], effect[2]))

        return enriched_action

    def validate_state_constraints(self, state: State) -> bool:
        """
        Use CSP solver to validate state consistency.

        Returns False if state violates any constraints.
        """
        if not self.csp_solver:
            return True

        # Convert state to CSP variables
        # (Simplified - real implementation would interface with component_29)
        variables = {}
        for prop in state.propositions:
            if len(prop) >= 2:
                var_name = f"{prop[0]}_{prop[1]}"
                variables[var_name] = prop[1:]

        # Check constraints (placeholder logic)
        # Real implementation would call self.csp_solver.solve()
        return True

    def get_applicable_actions(
        self, problem: PlanningProblem, state: State
    ) -> List[Action]:
        """Override to enrich actions with logic inference."""
        base_actions = problem.get_applicable_actions(state)

        if self.logic_engine:
            return [self.enrich_action_with_logic(a, state) for a in base_actions]

        return base_actions


# ============================================================================
# Main Functions
# ============================================================================


def main():
    """Example usage: Solve simple Blocks World problem."""
    logging.basicConfig(level=logging.INFO)

    # Create simple problem: A on B on table → B on A on table
    problem = BlocksWorldBuilder.create_problem(
        blocks=["A", "B"],
        initial_config={"A": "B", "B": "table"},
        goal_config={"B": "A", "A": "table"},
    )

    # Solve
    planner = StateSpacePlanner()
    plan = planner.solve(problem)

    if plan:
        print("Plan found:")
        for i, action in enumerate(plan):
            print(f"{i+1}. {action}")

        # Validate
        valid, error = planner.validate_plan(problem, plan)
        print(f"\nPlan valid: {valid}")
        if error:
            print(f"Error: {error}")
    else:
        print("No plan found!")


if __name__ == "__main__":
    main()
