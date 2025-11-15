"""
Component 42: Spatial Reasoning - Geometric Shapes

Geometric shape classes with area and perimeter calculations.

Author: KAI Development Team
Date: 2025-11-14
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from component_42_spatial_types import Position


@dataclass
class GeometricShape:
    """
    Base class for geometric shapes.

    All shapes can have:
    - Name/identifier
    - Properties (color, size, etc.)
    - Position on a grid (optional)
    """

    name: str
    shape_type: str = ""  # e.g., "Dreieck", "Viereck", "Kreis"
    properties: Dict[str, Any] = field(default_factory=dict)

    def calculate_area(self) -> Optional[float]:
        """Calculate the area of the shape. Override in subclasses."""
        return None

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate the perimeter of the shape. Override in subclasses."""
        return None

    def __str__(self) -> str:
        return f"{self.shape_type}({self.name})"


@dataclass
class Polygon(GeometricShape):
    """
    Represents a polygon (multi-sided shape).

    Properties:
    - num_sides: Number of sides
    - vertices: List of vertex positions (optional)
    """

    num_sides: int = 0
    vertices: List[Position] = field(default_factory=list)

    def __post_init__(self):
        """Set default shape_type for polygons."""
        if not self.shape_type:
            self.shape_type = f"Polygon_{self.num_sides}"

        # Validate vertices match num_sides
        if self.vertices and len(self.vertices) != self.num_sides:
            raise ValueError(
                f"Number of vertices ({len(self.vertices)}) doesn't match num_sides ({self.num_sides})"
            )


@dataclass
class Triangle(Polygon):
    """Triangle (3-sided polygon)."""

    def __post_init__(self):
        self.num_sides = 3
        self.shape_type = "Dreieck"
        super().__post_init__()

    def calculate_area(self) -> Optional[float]:
        """Calculate area using Heron's formula if vertices are provided."""
        if len(self.vertices) != 3:
            return None

        # Calculate side lengths
        a = self.vertices[0].distance_to(self.vertices[1])
        b = self.vertices[1].distance_to(self.vertices[2])
        c = self.vertices[2].distance_to(self.vertices[0])

        # Heron's formula
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate perimeter as sum of side lengths."""
        if len(self.vertices) != 3:
            return None

        a = self.vertices[0].distance_to(self.vertices[1])
        b = self.vertices[1].distance_to(self.vertices[2])
        c = self.vertices[2].distance_to(self.vertices[0])

        return a + b + c


@dataclass
class Quadrilateral(Polygon):
    """Quadrilateral (4-sided polygon)."""

    def __post_init__(self):
        self.num_sides = 4
        self.shape_type = "Viereck"
        super().__post_init__()

    def is_rectangle(self) -> bool:
        """Check if this quadrilateral is a rectangle (all angles 90°)."""
        if len(self.vertices) != 4:
            return False

        # Check if all angles are approximately 90 degrees
        # For a rectangle, opposite sides should be parallel and equal
        # We check if diagonals are equal (property of rectangles)
        d1 = self.vertices[0].distance_to(self.vertices[2])
        d2 = self.vertices[1].distance_to(self.vertices[3])

        return abs(d1 - d2) < 0.001  # Tolerance for floating point

    def calculate_area(self) -> Optional[float]:
        """Calculate area for rectangle (if applicable)."""
        if not self.is_rectangle() or len(self.vertices) != 4:
            return None

        # For a rectangle, area = width × height
        width = self.vertices[0].distance_to(self.vertices[1])
        height = self.vertices[1].distance_to(self.vertices[2])

        return width * height

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate perimeter as sum of side lengths."""
        if len(self.vertices) != 4:
            return None

        perimeter = 0
        for i in range(4):
            next_i = (i + 1) % 4
            perimeter += self.vertices[i].distance_to(self.vertices[next_i])

        return perimeter


@dataclass
class Circle(GeometricShape):
    """Circle with center and radius."""

    center: Optional[Position] = None
    radius: float = 0.0

    def __post_init__(self):
        self.shape_type = "Kreis"

    def calculate_area(self) -> Optional[float]:
        """Calculate area: π × r²"""
        if self.radius <= 0:
            return None
        return math.pi * self.radius**2

    def calculate_perimeter(self) -> Optional[float]:
        """Calculate circumference: 2 × π × r"""
        if self.radius <= 0:
            return None
        return 2 * math.pi * self.radius
