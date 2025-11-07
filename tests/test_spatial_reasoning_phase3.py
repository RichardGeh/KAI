"""
Tests for Component 42: Spatial Reasoning - Phase 3

Tests for:
- Phase 3.1: Shape ontology in graph
- Phase 3.2: Shape recognition and classification
- Phase 3.3: Geometric calculations

Author: KAI Development Team
Date: 2025-11-05
"""

import math
from unittest.mock import Mock

import pytest

from component_42_spatial_reasoning import (
    Circle,
    GeometricShape,
    Polygon,
    Position,
    Quadrilateral,
    SpatialReasoner,
    Triangle,
)

# ============================================================================
# Phase 3.1: Shape Data Structures Tests
# ============================================================================


class TestGeometricShapes:
    """Test geometric shape data structures."""

    def test_create_generic_shape(self):
        """Test creating a generic shape."""
        shape = GeometricShape(
            name="Shape1", shape_type="Generic", properties={"color": "red"}
        )

        assert shape.name == "Shape1"
        assert shape.shape_type == "Generic"
        assert shape.properties["color"] == "red"

    def test_create_triangle(self):
        """Test creating a triangle."""
        vertices = [Position(0, 0), Position(3, 0), Position(0, 4)]
        triangle = Triangle(name="Triangle1", vertices=vertices)

        assert triangle.name == "Triangle1"
        assert triangle.shape_type == "Dreieck"
        assert triangle.num_sides == 3
        assert len(triangle.vertices) == 3

    def test_triangle_invalid_vertices(self):
        """Test that triangle with wrong number of vertices raises error."""
        with pytest.raises(ValueError):
            Triangle(name="Bad", vertices=[Position(0, 0), Position(1, 1)])

    def test_create_quadrilateral(self):
        """Test creating a quadrilateral."""
        vertices = [Position(0, 0), Position(4, 0), Position(4, 3), Position(0, 3)]
        quad = Quadrilateral(name="Quad1", vertices=vertices)

        assert quad.name == "Quad1"
        assert quad.shape_type == "Viereck"
        assert quad.num_sides == 4
        assert len(quad.vertices) == 4

    def test_create_circle(self):
        """Test creating a circle."""
        circle = Circle(name="Circle1", center=Position(5, 5), radius=10.0)

        assert circle.name == "Circle1"
        assert circle.shape_type == "Kreis"
        assert circle.center == Position(5, 5)
        assert circle.radius == 10.0

    def test_generic_polygon(self):
        """Test creating a generic polygon."""
        vertices = [
            Position(0, 0),
            Position(1, 0),
            Position(1, 1),
            Position(0.5, 1.5),
            Position(0, 1),
        ]
        pentagon = Polygon(name="Pentagon", num_sides=5, vertices=vertices)

        assert pentagon.num_sides == 5
        assert len(pentagon.vertices) == 5


# ============================================================================
# Phase 3.3: Geometric Calculations Tests
# ============================================================================


class TestGeometricCalculations:
    """Test geometric property calculations."""

    def test_triangle_area_3_4_5(self):
        """Test triangle area calculation (3-4-5 right triangle)."""
        # 3-4-5 right triangle has area = 6
        vertices = [Position(0, 0), Position(3, 0), Position(0, 4)]
        triangle = Triangle(name="RightTriangle", vertices=vertices)

        area = triangle.calculate_area()

        assert area is not None
        assert abs(area - 6.0) < 0.001

    def test_triangle_perimeter(self):
        """Test triangle perimeter calculation."""
        vertices = [Position(0, 0), Position(3, 0), Position(0, 4)]
        triangle = Triangle(name="Triangle", vertices=vertices)

        perimeter = triangle.calculate_perimeter()

        assert perimeter is not None
        # Perimeter = 3 + 4 + 5 = 12
        assert abs(perimeter - 12.0) < 0.001

    def test_triangle_area_without_vertices(self):
        """Test that triangle without vertices returns None for area."""
        triangle = Triangle(name="NoVertices", vertices=[])

        area = triangle.calculate_area()

        assert area is None

    def test_rectangle_area(self):
        """Test rectangle area calculation."""
        # 4×3 rectangle
        vertices = [Position(0, 0), Position(4, 0), Position(4, 3), Position(0, 3)]
        quad = Quadrilateral(name="Rectangle", vertices=vertices)

        assert quad.is_rectangle() is True

        area = quad.calculate_area()

        assert area is not None
        assert abs(area - 12.0) < 0.001

    def test_rectangle_perimeter(self):
        """Test rectangle perimeter calculation."""
        vertices = [Position(0, 0), Position(4, 0), Position(4, 3), Position(0, 3)]
        quad = Quadrilateral(name="Rectangle", vertices=vertices)

        perimeter = quad.calculate_perimeter()

        assert perimeter is not None
        # Perimeter = 2(4 + 3) = 14
        assert abs(perimeter - 14.0) < 0.001

    def test_non_rectangle_quadrilateral(self):
        """Test that non-rectangle quadrilateral is detected."""
        # Parallelogram (not a rectangle) - clearly different diagonals
        vertices = [Position(0, 0), Position(5, 0), Position(6, 3), Position(1, 3)]
        quad = Quadrilateral(name="Parallelogram", vertices=vertices)

        assert quad.is_rectangle() is False
        assert quad.calculate_area() is None  # No formula for generic quad

    def test_circle_area(self):
        """Test circle area calculation."""
        circle = Circle(name="Circle", radius=5.0)

        area = circle.calculate_area()

        assert area is not None
        # Area = π × 5² = 25π ≈ 78.54
        assert abs(area - 25 * math.pi) < 0.001

    def test_circle_perimeter(self):
        """Test circle circumference calculation."""
        circle = Circle(name="Circle", radius=5.0)

        circumference = circle.calculate_perimeter()

        assert circumference is not None
        # Circumference = 2π × 5 = 10π ≈ 31.42
        assert abs(circumference - 10 * math.pi) < 0.001

    def test_circle_with_zero_radius(self):
        """Test that circle with zero radius returns None."""
        circle = Circle(name="ZeroCircle", radius=0.0)

        assert circle.calculate_area() is None
        assert circle.calculate_perimeter() is None


# ============================================================================
# Phase 3.1 & 3.2: Shape Management Tests
# ============================================================================


class TestShapeManagement:
    """Test shape creation and retrieval in knowledge graph."""

    def test_create_triangle_in_graph(self):
        """Test creating a triangle in the knowledge graph."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        vertices = [Position(0, 0), Position(3, 0), Position(0, 4)]
        triangle = Triangle(name="Triangle1", vertices=vertices)

        success = reasoner.create_shape(triangle)

        assert success is True
        # Should create shape node, property nodes, and vertex nodes
        assert mock_netzwerk.create_wort_if_not_exists.call_count > 0
        assert mock_netzwerk.assert_relation.call_count > 0

    def test_create_circle_in_graph(self):
        """Test creating a circle in the knowledge graph."""
        mock_netzwerk = Mock()
        mock_netzwerk.create_wort_if_not_exists = Mock()
        mock_netzwerk.assert_relation = Mock()

        reasoner = SpatialReasoner(mock_netzwerk)

        circle = Circle(name="Circle1", center=Position(5, 5), radius=10.0)

        success = reasoner.create_shape(circle)

        assert success is True

    def test_get_triangle_from_graph(self):
        """Test retrieving a triangle from knowledge graph."""
        mock_netzwerk = Mock()

        # Mock shape node
        mock_shape_node = {
            "lemma": "Triangle1",
            "shape_type": "Dreieck",
            "num_sides": 3,
            "type": "GeometricShape",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_shape_node)

        # Mock vertices query
        mock_vertices = [
            {"x": 0, "y": 0, "index": 0},
            {"x": 3, "y": 0, "index": 1},
            {"x": 0, "y": 4, "index": 2},
        ]
        mock_netzwerk.session.run = Mock(return_value=mock_vertices)

        reasoner = SpatialReasoner(mock_netzwerk)
        shape = reasoner.get_shape("Triangle1")

        assert shape is not None
        assert isinstance(shape, Triangle)
        assert shape.name == "Triangle1"
        assert len(shape.vertices) == 3

    def test_get_circle_from_graph(self):
        """Test retrieving a circle from knowledge graph."""
        mock_netzwerk = Mock()

        mock_shape_node = {
            "lemma": "Circle1",
            "shape_type": "Kreis",
            "radius": 10.0,
            "center_x": 5,
            "center_y": 5,
            "type": "GeometricShape",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_shape_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        shape = reasoner.get_shape("Circle1")

        assert shape is not None
        assert isinstance(shape, Circle)
        assert shape.name == "Circle1"
        assert shape.radius == 10.0
        assert shape.center == Position(5, 5)

    def test_get_nonexistent_shape(self):
        """Test retrieving a shape that doesn't exist."""
        mock_netzwerk = Mock()
        mock_netzwerk.find_wort_node = Mock(return_value=None)

        reasoner = SpatialReasoner(mock_netzwerk)
        shape = reasoner.get_shape("NonExistent")

        assert shape is None


# ============================================================================
# Phase 3.2: Shape Classification Tests
# ============================================================================


class TestShapeClassification:
    """Test shape classification logic."""

    def test_classify_triangle(self):
        """Test classifying a triangle by sides."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        classification = reasoner.classify_shape(num_sides=3)

        assert classification == "Dreieck"

    def test_classify_quadrilateral(self):
        """Test classifying a quadrilateral."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        classification = reasoner.classify_shape(num_sides=4)

        assert classification == "Viereck"

    def test_classify_rectangle(self):
        """Test classifying a rectangle."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        # Rectangle vertices
        vertices = [Position(0, 0), Position(4, 0), Position(4, 3), Position(0, 3)]

        classification = reasoner.classify_shape(num_sides=4, vertices=vertices)

        assert classification == "Rechteck"

    def test_classify_pentagon(self):
        """Test classifying a pentagon."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        classification = reasoner.classify_shape(num_sides=5)

        assert classification == "Fünfeck"

    def test_classify_hexagon(self):
        """Test classifying a hexagon."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        classification = reasoner.classify_shape(num_sides=6)

        assert classification == "Sechseck"

    def test_classify_circle(self):
        """Test classifying a circle by properties."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        classification = reasoner.classify_shape(properties={"radius": 10.0})

        assert classification == "Kreis"

    def test_classify_unknown_shape(self):
        """Test classifying unknown shape."""
        mock_netzwerk = Mock()
        reasoner = SpatialReasoner(mock_netzwerk)

        classification = reasoner.classify_shape()

        assert classification == "Unbekannte Form"


# ============================================================================
# Phase 3.3: Calculate Shape Properties Tests
# ============================================================================


class TestCalculateShapeProperties:
    """Test calculating properties of stored shapes."""

    def test_calculate_triangle_properties(self):
        """Test calculating properties of a triangle."""
        mock_netzwerk = Mock()

        # Mock triangle
        mock_shape_node = {
            "lemma": "Triangle1",
            "shape_type": "Dreieck",
            "num_sides": 3,
            "type": "GeometricShape",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_shape_node)

        # Mock vertices (3-4-5 triangle)
        mock_vertices = [
            {"x": 0, "y": 0, "index": 0},
            {"x": 3, "y": 0, "index": 1},
            {"x": 0, "y": 4, "index": 2},
        ]
        mock_netzwerk.session.run = Mock(return_value=mock_vertices)

        reasoner = SpatialReasoner(mock_netzwerk)
        properties = reasoner.calculate_shape_properties("Triangle1")

        assert "area" in properties
        assert "perimeter" in properties
        assert abs(properties["area"] - 6.0) < 0.001
        assert abs(properties["perimeter"] - 12.0) < 0.001

    def test_calculate_circle_properties(self):
        """Test calculating properties of a circle."""
        mock_netzwerk = Mock()

        mock_shape_node = {
            "lemma": "Circle1",
            "shape_type": "Kreis",
            "radius": 5.0,
            "type": "GeometricShape",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_shape_node)

        reasoner = SpatialReasoner(mock_netzwerk)
        properties = reasoner.calculate_shape_properties("Circle1")

        assert "area" in properties
        assert "perimeter" in properties
        assert "diameter" in properties
        assert abs(properties["area"] - 25 * math.pi) < 0.001
        assert abs(properties["perimeter"] - 10 * math.pi) < 0.001
        assert abs(properties["diameter"] - 10.0) < 0.001

    def test_calculate_rectangle_properties(self):
        """Test calculating properties of a rectangle."""
        mock_netzwerk = Mock()

        mock_shape_node = {
            "lemma": "Quad1",
            "shape_type": "Viereck",
            "num_sides": 4,
            "type": "GeometricShape",
        }
        mock_netzwerk.find_wort_node = Mock(return_value=mock_shape_node)

        # Mock rectangle vertices
        mock_vertices = [
            {"x": 0, "y": 0, "index": 0},
            {"x": 4, "y": 0, "index": 1},
            {"x": 4, "y": 3, "index": 2},
            {"x": 0, "y": 3, "index": 3},
        ]
        mock_netzwerk.session.run = Mock(return_value=mock_vertices)

        reasoner = SpatialReasoner(mock_netzwerk)
        properties = reasoner.calculate_shape_properties("Quad1")

        assert "area" in properties
        assert "perimeter" in properties
        assert "is_rectangle" in properties
        assert properties["is_rectangle"] is True
        assert abs(properties["area"] - 12.0) < 0.001

    def test_calculate_properties_nonexistent_shape(self):
        """Test calculating properties for nonexistent shape."""
        mock_netzwerk = Mock()
        mock_netzwerk.find_wort_node = Mock(return_value=None)

        reasoner = SpatialReasoner(mock_netzwerk)
        properties = reasoner.calculate_shape_properties("NonExistent")

        assert len(properties) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
