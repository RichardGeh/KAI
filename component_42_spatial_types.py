"""
Component 42: Spatial Reasoning - Type Definitions

Basic types and data structures for spatial reasoning:
- Enums for spatial relations and neighborhood types
- Position class for 2D coordinates
- SpatialRelation class for representing spatial facts
- SpatialReasoningResult for query results

Author: KAI Development Team
Date: 2025-11-14
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Enums and Constants
# ============================================================================


class SpatialRelationType(Enum):
    """Types of spatial relations supported by the reasoning engine."""

    # Directional relations (cardinal directions)
    NORTH_OF = "NORTH_OF"
    SOUTH_OF = "SOUTH_OF"
    EAST_OF = "EAST_OF"
    WEST_OF = "WEST_OF"

    # Adjacency relations
    ADJACENT_TO = "ADJACENT_TO"  # General neighbor (symmetric)
    NEIGHBOR_ORTHOGONAL = "NEIGHBOR_ORTHOGONAL"  # 4-directional neighbor
    NEIGHBOR_DIAGONAL = "NEIGHBOR_DIAGONAL"  # Diagonal neighbor

    # Hierarchical/containment relations
    INSIDE = "INSIDE"  # A is inside B
    CONTAINS = "CONTAINS"  # A contains B

    # Vertical relations
    ABOVE = "ABOVE"
    BELOW = "BELOW"

    # Positional relations
    BETWEEN = "BETWEEN"  # A is between B and C
    LOCATED_AT = "LOCATED_AT"  # Object is at specific position

    @property
    def is_symmetric(self) -> bool:
        """Check if this relation is symmetric (A R B => B R A)."""
        return self in {
            SpatialRelationType.ADJACENT_TO,
            SpatialRelationType.NEIGHBOR_ORTHOGONAL,
            SpatialRelationType.NEIGHBOR_DIAGONAL,
        }

    @property
    def is_transitive(self) -> bool:
        """Check if this relation is transitive (A R B, B R C => A R C)."""
        return self in {
            SpatialRelationType.NORTH_OF,
            SpatialRelationType.SOUTH_OF,
            SpatialRelationType.EAST_OF,
            SpatialRelationType.WEST_OF,
            SpatialRelationType.INSIDE,
            SpatialRelationType.CONTAINS,
            SpatialRelationType.ABOVE,
            SpatialRelationType.BELOW,
        }

    @property
    def inverse(self) -> Optional["SpatialRelationType"]:
        """Get the inverse relation (if exists)."""
        inverses = {
            SpatialRelationType.NORTH_OF: SpatialRelationType.SOUTH_OF,
            SpatialRelationType.SOUTH_OF: SpatialRelationType.NORTH_OF,
            SpatialRelationType.EAST_OF: SpatialRelationType.WEST_OF,
            SpatialRelationType.WEST_OF: SpatialRelationType.EAST_OF,
            SpatialRelationType.INSIDE: SpatialRelationType.CONTAINS,
            SpatialRelationType.CONTAINS: SpatialRelationType.INSIDE,
            SpatialRelationType.ABOVE: SpatialRelationType.BELOW,
            SpatialRelationType.BELOW: SpatialRelationType.ABOVE,
        }
        return inverses.get(self)


class NeighborhoodType(Enum):
    """Types of neighborhood definitions for grid-based reasoning."""

    ORTHOGONAL = "orthogonal"  # 4-directional (N, S, E, W)
    DIAGONAL = "diagonal"  # Diagonal only (NE, NW, SE, SW)
    MOORE = "moore"  # 8-directional (orthogonal + diagonal)
    CUSTOM = "custom"  # Custom neighborhood pattern (e.g., knight moves in chess)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True, order=True)
class Position:
    """
    Represents a position in a 2D coordinate system.

    Immutable and hashable for use in sets and dictionaries.
    Coordinates are 0-indexed internally but can represent any coordinate system.
    """

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def distance_to(self, other: "Position", metric: str = "euclidean") -> float:
        """
        Calculate distance to another position.

        Args:
            other: Target position
            metric: Distance metric ('euclidean', 'manhattan', 'chebyshev')

        Returns:
            Distance value
        """
        dx = abs(self.x - other.x)
        dy = abs(self.y - other.y)

        if metric == "manhattan":
            return dx + dy
        elif metric == "chebyshev":
            return max(dx, dy)
        else:  # euclidean
            return math.sqrt(dx**2 + dy**2)

    def manhattan_distance_to(self, other: "Position") -> float:
        """Calculate Manhattan distance to another position (convenience method)."""
        return self.distance_to(other, metric="manhattan")

    def euclidean_distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position (convenience method)."""
        return self.distance_to(other, metric="euclidean")

    def direction_to(self, other: "Position") -> Optional[SpatialRelationType]:
        """
        Determine the cardinal direction from this position to another.

        Returns None if positions are diagonal or identical.
        """
        dx = other.x - self.x
        dy = other.y - self.y

        if dx == 0 and dy > 0:
            return SpatialRelationType.NORTH_OF
        elif dx == 0 and dy < 0:
            return SpatialRelationType.SOUTH_OF
        elif dx > 0 and dy == 0:
            return SpatialRelationType.EAST_OF
        elif dx < 0 and dy == 0:
            return SpatialRelationType.WEST_OF

        return None

    def get_neighbors(
        self,
        neighborhood_type: NeighborhoodType = NeighborhoodType.ORTHOGONAL,
        custom_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List["Position"]:
        """
        Get neighboring positions based on neighborhood type.

        Args:
            neighborhood_type: Type of neighborhood
            custom_offsets: Custom offset list for CUSTOM neighborhood type
                           e.g., [(-2, -1), (-2, 1), ...] for knight moves

        Returns:
            List of neighboring positions
        """
        if neighborhood_type == NeighborhoodType.CUSTOM and custom_offsets:
            return [Position(self.x + dx, self.y + dy) for dx, dy in custom_offsets]

        # Standard offsets
        orthogonal = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        diagonal = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # NE, SE, NE, NW

        if neighborhood_type == NeighborhoodType.ORTHOGONAL:
            offsets = orthogonal
        elif neighborhood_type == NeighborhoodType.DIAGONAL:
            offsets = diagonal
        else:  # MOORE
            offsets = orthogonal + diagonal

        return [Position(self.x + dx, self.y + dy) for dx, dy in offsets]


@dataclass
class SpatialRelation:
    """
    Represents a spatial relation between entities.

    Examples:
    - SpatialRelation("König", "Turm", ADJACENT_TO, 0.95)
    - SpatialRelation("Feld_A1", "Feld_A2", NORTH_OF, 1.0)
    """

    subject: str  # Entity A
    object: str  # Entity B
    relation_type: SpatialRelationType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.subject} {self.relation_type.value} {self.object} (conf={self.confidence:.2f})"

    def to_inverse(self) -> Optional["SpatialRelation"]:
        """Create the inverse relation if one exists."""
        # For symmetric relations, return the same relation with swapped subject/object
        if self.relation_type.is_symmetric:
            return SpatialRelation(
                subject=self.object,
                object=self.subject,
                relation_type=self.relation_type,
                confidence=self.confidence,
                metadata=self.metadata.copy(),
            )

        # For directional relations, get the inverse type
        inverse_type = self.relation_type.inverse
        if inverse_type:
            return SpatialRelation(
                subject=self.object,
                object=self.subject,
                relation_type=inverse_type,
                confidence=self.confidence,
                metadata=self.metadata.copy(),
            )
        return None


@dataclass
class SpatialReasoningResult:
    """
    Result of a spatial reasoning query.

    Contains inferred facts, confidence scores, and reasoning trace.
    """

    query: str
    relations: List[SpatialRelation] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if reasoning was successful."""
        return self.error is None and len(self.relations) > 0
