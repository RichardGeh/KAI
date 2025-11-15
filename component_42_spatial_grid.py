"""
Component 42: Spatial Reasoning - Grid Data Structure

Grid class for representing 2D grid structures (chess boards, Sudoku, etc.)

Author: KAI Development Team
Date: 2025-11-14
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from component_42_spatial_types import NeighborhoodType, Position


@dataclass
class Grid:
    """
    Represents a 2D grid structure.

    Grids are general-purpose and not tied to specific applications.
    They can represent chess boards, Sudoku grids, or any N×M structure.
    """

    width: int  # Number of columns (N)
    height: int  # Number of rows (M)
    name: str = ""  # Unique identifier (e.g., "Schachbrett_1", "Sudoku_9x9")
    neighborhood_type: NeighborhoodType = NeighborhoodType.ORTHOGONAL
    custom_offsets: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate grid parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Grid dimensions must be positive: {self.width}×{self.height}"
            )

        if (
            self.neighborhood_type == NeighborhoodType.CUSTOM
            and not self.custom_offsets
        ):
            raise ValueError("Custom neighborhood type requires custom_offsets")

        # Generate default name if not provided
        if not self.name:
            object.__setattr__(self, "name", f"Grid_{self.width}x{self.height}")

        # Initialize cell data storage
        object.__setattr__(self, "_cell_data", {})

    @property
    def size(self) -> int:
        """Total number of positions in the grid."""
        return self.width * self.height

    def is_valid_position(self, pos: Position) -> bool:
        """Check if a position is within grid bounds."""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def get_all_positions(self) -> List[Position]:
        """Get all positions in the grid."""
        return [Position(x, y) for x in range(self.width) for y in range(self.height)]

    def get_position_name(self, pos: Position) -> str:
        """
        Generate a unique name for a position in this grid.

        Format: "{grid_name}_Pos_{x}_{y}"
        Example: "Schachbrett_Pos_3_4"
        """
        return f"{self.name}_Pos_{pos.x}_{pos.y}"

    def get_cell_count(self) -> int:
        """Get total number of cells in the grid (alias for size property)."""
        return self.size

    def set_cell_data(self, pos: Position, data: Any) -> None:
        """Set data for a specific cell in the grid. Silently ignores out-of-bounds positions."""
        if not hasattr(self, "_cell_data"):
            object.__setattr__(self, "_cell_data", {})
        if not self.is_valid_position(pos):
            return  # Silently ignore out-of-bounds positions
        self._cell_data[pos] = data

    def get_cell_data(self, pos: Position, default: Any = None) -> Any:
        """Get data for a specific cell in the grid."""
        if not hasattr(self, "_cell_data"):
            object.__setattr__(self, "_cell_data", {})
        return self._cell_data.get(pos, default)

    def get_neighbors(
        self, pos: Position, neighborhood_type: Optional[NeighborhoodType] = None
    ) -> List[Position]:
        """
        Get valid neighbors of a position within grid bounds.

        Args:
            pos: Position to get neighbors for
            neighborhood_type: Type of neighborhood (defaults to grid's neighborhood_type)

        Returns:
            List of valid neighboring positions
        """
        nh_type = neighborhood_type or self.neighborhood_type
        neighbors = pos.get_neighbors(nh_type, self.custom_offsets)
        return [n for n in neighbors if self.is_valid_position(n)]

    def __str__(self) -> str:
        return f"Grid({self.name}, {self.width}×{self.height})"
