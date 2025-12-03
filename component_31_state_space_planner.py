"""
Component 31: State-Space Planner (STRIPS-style Planning)

Facade module providing backward-compatible access to state-space planning functionality.

This module has been refactored into multiple focused modules:
- component_31_planner_core: Core planning primitives and A* search
- component_31_heuristics: Planning heuristics (RelaxedPlan, SetCover)
- component_31_domain_builders: Domain-specific builders and advanced planning

All original functionality is preserved through this facade.

Original implementation: 2025-01-30
Refactored: 2025-11-29 (Architecture Phase 4, Task 12)
Author: KAI Development Team
"""

# ============================================================================
# Import all public classes from split modules
# ============================================================================

# Domain builders and advanced planning
from component_31_domain_builders import (
    BlocksWorldBuilder,
    GridNavigationBuilder,
    HybridPlanner,
    RiverCrossingBuilder,
    TemporalConstraint,
    TemporalPlanner,
)

# Heuristics
from component_31_heuristics import (
    Heuristic,
    RelaxedPlanHeuristic,
    SetCoverHeuristic,
)

# Core planning primitives
from component_31_planner_core import (
    Action,
    PlanningProblem,
    SearchNode,
    State,
    StateSpacePlanner,
    safe_str,
)

# ============================================================================
# Expose all imports for backward compatibility
# ============================================================================

__all__ = [
    # Core primitives
    "State",
    "Action",
    "PlanningProblem",
    "SearchNode",
    "StateSpacePlanner",
    "safe_str",
    # Heuristics
    "Heuristic",
    "RelaxedPlanHeuristic",
    "SetCoverHeuristic",
    # Temporal reasoning
    "TemporalConstraint",
    "TemporalPlanner",
    # Domain builders
    "BlocksWorldBuilder",
    "GridNavigationBuilder",
    "RiverCrossingBuilder",
    # Advanced planning
    "HybridPlanner",
]


# ============================================================================
# Main function for backward compatibility
# ============================================================================


def main():
    """Example usage: Solve simple Blocks World problem."""
    from component_31_domain_builders import main as domain_main

    domain_main()


if __name__ == "__main__":
    main()
