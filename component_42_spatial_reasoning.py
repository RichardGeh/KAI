"""
Component 42: Spatial Reasoning (Compatibility Wrapper)

This module provides backwards compatibility by re-exporting all classes and functions
from the refactored spatial reasoning modules.

For new code, prefer importing from the specific modules:
- component_42_spatial_types: Type definitions (Position, SpatialRelation, etc.)
- component_42_spatial_grid: Grid class
- component_42_spatial_shapes: Geometric shapes
- component_42_spatial_movement: Movement planning classes
- component_42_spatial_reasoner: Main reasoning engine

Author: KAI Development Team
Date: 2025-11-14
"""

from component_42_spatial_grid import Grid
from component_42_spatial_movement import MovementAction, MovementPlan
from component_42_spatial_reasoner import SpatialReasoner, SpatialReasoningEngine
from component_42_spatial_shapes import (
    Circle,
    GeometricShape,
    Polygon,
    Quadrilateral,
    Triangle,
)

# Re-export all public classes and functions for backwards compatibility
from component_42_spatial_types import (
    NeighborhoodType,
    Position,
    SpatialReasoningResult,
    SpatialRelation,
    SpatialRelationType,
)

__all__ = [
    # Types
    "SpatialRelationType",
    "NeighborhoodType",
    "Position",
    "SpatialRelation",
    "SpatialReasoningResult",
    # Grid
    "Grid",
    # Shapes
    "GeometricShape",
    "Polygon",
    "Triangle",
    "Quadrilateral",
    "Circle",
    # Movement
    "MovementAction",
    "MovementPlan",
    # Engine
    "SpatialReasoner",
    "SpatialReasoningEngine",
]
