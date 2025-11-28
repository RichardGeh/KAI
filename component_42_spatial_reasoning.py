"""
Component 42: Spatial Reasoning - Facade

Unified facade for spatial reasoning functionality.

This module provides a backward-compatible interface to the split spatial reasoning
components by re-exporting the SpatialReasoner facade class which delegates to:

- component_42_spatial_inference: Core spatial reasoning and relation inference
- component_42_grid_manager: Grid topology and object management
- component_42_path_finder: Path-finding algorithms
- component_42_spatial_neo4j: Neo4j integration for spatial data
- component_42_spatial_patterns: Spatial pattern learning and recognition

The SpatialReasoner class defined in this file maintains backward compatibility
with the original monolithic implementation.

Author: KAI Development Team
Date: 2025-11-27 (Refactored from monolithic component_42_spatial_reasoner.py)
"""

# Re-export type definitions for backward compatibility
from component_42_spatial_grid import Grid
from component_42_spatial_movement import MovementAction, MovementPlan

# Import the facade implementation from the OLD file (renamed)
# This provides the complete SpatialReasoner class with all methods
from component_42_spatial_reasoner_OLD import SpatialReasoner, SpatialReasoningEngine
from component_42_spatial_shapes import (
    Circle,
    GeometricShape,
    Polygon,
    Quadrilateral,
    Triangle,
)
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
    # Engine (Main Facade)
    "SpatialReasoner",
    "SpatialReasoningEngine",
]
