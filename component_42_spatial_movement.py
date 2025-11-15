"""
Component 42: Spatial Reasoning - Movement Planning

Data structures for movement actions and plans.

Author: KAI Development Team
Date: 2025-11-14
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from component_42_spatial_types import Position


@dataclass(frozen=True)
class MovementAction:
    """Represents a single movement action."""

    object_name: str
    from_position: Position
    to_position: Position
    step_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Step {self.step_number}: Move {self.object_name} "
            f"from {self.from_position} to {self.to_position}"
        )


@dataclass
class MovementPlan:
    """Represents a complete movement plan for an object."""

    object_name: str
    grid_name: str
    actions: List[MovementAction]
    total_steps: int
    path_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"MovementPlan({self.object_name} on {self.grid_name}: "
            f"{self.total_steps} steps, path length {self.path_length})"
        )

    def get_final_position(self) -> Optional[Position]:
        """Get the final position after executing all actions."""
        if self.actions:
            return self.actions[-1].to_position
        return None

    def get_path(self) -> List[Position]:
        """Get the complete path as a list of positions."""
        if not self.actions:
            return []
        path = [self.actions[0].from_position]
        for action in self.actions:
            path.append(action.to_position)
        return path
