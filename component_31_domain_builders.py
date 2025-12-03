"""
Component 31: Domain Builders and Advanced Planning

Domain-specific problem builders:
- BlocksWorldBuilder: Classic blocks world planning
- GridNavigationBuilder: Grid-based pathfinding with obstacles
- RiverCrossingBuilder: Classic river crossing puzzle

Advanced planning capabilities:
- TemporalPlanner: Temporal constraints and reasoning
- HybridPlanner: Integration with logic engines and CSP solvers

Author: KAI Development Team
Date: 2025-01-30
Refactored: 2025-11-29 (Architecture Phase 4)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from component_31_heuristics import Heuristic
from component_31_planner_core import Action, PlanningProblem, State, StateSpacePlanner

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
# Domain-Specific Builders
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
    import logging

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
