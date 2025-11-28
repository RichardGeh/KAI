"""
Component 42: Spatial Neo4j Integration

Neo4j integration for spatial data persistence.

This module handles:
- Persisting spatial relations to Neo4j
- Storing grid configurations
- Querying spatial patterns
- Movement history tracking
- Spatial extraction rules

Author: KAI Development Team
Date: 2025-11-27
"""

import threading
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger
from component_42_spatial_shapes import Circle, GeometricShape, Polygon, Quadrilateral
from component_42_spatial_types import Position, SpatialRelationType
from infrastructure.neo4j_session_mixin import Neo4jSessionMixin

logger = get_logger(__name__)


class SpatialNeo4jRepository(Neo4jSessionMixin):
    """
    Neo4j repository for spatial data.

    Provides methods for:
    - Storing spatial relations
    - Persisting geometric shapes
    - Managing spatial extraction rules
    - Querying spatial patterns

    Thread Safety:
        This class is thread-safe through Neo4jSessionMixin's built-in locking.
    """

    def __init__(self, driver, netzwerk=None):
        """
        Initialize the spatial Neo4j repository.

        Args:
            driver: Neo4j driver instance
            netzwerk: Optional KonzeptNetzwerk instance for graph access
        """
        super().__init__(driver, enable_cache=False)  # Disable caching for writes
        self.netzwerk = netzwerk
        self._lock = threading.RLock()

        logger.info("SpatialNeo4jRepository initialized")

    def add_spatial_relation(
        self,
        subject: str,
        relation_type: SpatialRelationType,
        target: str,
        confidence: float = 1.0,
    ) -> bool:
        """
        Store a spatial relation between two objects in the knowledge graph.

        Args:
            subject: The subject entity (e.g., "house")
            relation_type: Type of spatial relation (e.g., NORTH_OF, ADJACENT_TO)
            target: The target entity (e.g., "tree")
            confidence: Confidence in this relation (0.0-1.0)

        Returns:
            True if successful, False otherwise
        """
        if not self.netzwerk:
            logger.warning("No knowledge graph available for storing spatial relation")
            return False

        try:
            with self._lock:
                # Ensure both entities exist in the graph
                self.netzwerk.create_wort_if_not_exists(
                    lemma=subject, pos="NOUN", type="SpatialEntity"
                )
                self.netzwerk.create_wort_if_not_exists(
                    lemma=target, pos="NOUN", type="SpatialEntity"
                )

                # Store the spatial relation
                self.netzwerk.assert_relation(
                    from_lemma=subject,
                    to_lemma=target,
                    relation_type=relation_type.value,
                    confidence=confidence,
                )

                logger.info(
                    "Stored spatial relation: %s %s %s (confidence: %.2f)",
                    subject,
                    relation_type.value,
                    target,
                    confidence,
                )

                return True

        except Exception as e:
            logger.error(
                "Failed to store spatial relation %s %s %s: %s",
                subject,
                relation_type.value,
                target,
                str(e),
                exc_info=True,
            )
            return False

    def register_spatial_extraction_rules(self) -> int:
        """
        Register extraction rules for spatial relations in the knowledge graph.

        This enables KAI to learn spatial relations from natural language input.
        Uses German language patterns.

        Returns:
            Number of rules successfully registered
        """
        if not self.netzwerk:
            logger.warning(
                "No knowledge graph available for registering extraction rules"
            )
            return 0

        logger.info("Registering spatial extraction rules in knowledge graph")

        rules_registered = 0

        # Define extraction rules for spatial relations (German patterns)
        spatial_patterns = [
            # Cardinal directions
            ("NORTH_OF", r"^(.+) liegt nördlich von (.+)$"),
            ("NORTH_OF", r"^(.+) ist nördlich von (.+)$"),
            ("SOUTH_OF", r"^(.+) liegt südlich von (.+)$"),
            ("SOUTH_OF", r"^(.+) ist südlich von (.+)$"),
            ("EAST_OF", r"^(.+) liegt östlich von (.+)$"),
            ("EAST_OF", r"^(.+) ist östlich von (.+)$"),
            ("WEST_OF", r"^(.+) liegt westlich von (.+)$"),
            ("WEST_OF", r"^(.+) ist westlich von (.+)$"),
            # Adjacency
            ("ADJACENT_TO", r"^(.+) liegt neben (.+)$"),
            ("ADJACENT_TO", r"^(.+) ist neben (.+)$"),
            ("ADJACENT_TO", r"^(.+) grenzt an (.+)$"),
            ("NEIGHBOR_ORTHOGONAL", r"^(.+) ist direkter Nachbar von (.+)$"),
            ("NEIGHBOR_DIAGONAL", r"^(.+) ist diagonaler Nachbar von (.+)$"),
            # Containment
            ("INSIDE", r"^(.+) ist in (.+)$"),
            ("INSIDE", r"^(.+) liegt in (.+)$"),
            ("INSIDE", r"^(.+) befindet sich in (.+)$"),
            ("CONTAINS", r"^(.+) enthält (.+)$"),
            ("CONTAINS", r"^(.+) beinhaltet (.+)$"),
            # Vertical relations
            ("ABOVE", r"^(.+) ist über (.+)$"),
            ("ABOVE", r"^(.+) liegt über (.+)$"),
            ("BELOW", r"^(.+) ist unter (.+)$"),
            ("BELOW", r"^(.+) liegt unter (.+)$"),
            # Position
            ("LOCATED_AT", r"^(.+) ist bei (.+)$"),
            ("LOCATED_AT", r"^(.+) ist an (.+)$"),
            ("LOCATED_AT", r"^(.+) steht auf (.+)$"),
        ]

        # Register each extraction rule
        for relation_type, pattern in spatial_patterns:
            try:
                self.netzwerk.create_extraction_rule(
                    relation_type=relation_type, regex_pattern=pattern
                )
                rules_registered += 1
                logger.debug(
                    "Registered extraction rule: %s -> %s", relation_type, pattern
                )

            except Exception as e:
                # Rule might already exist, continue
                logger.debug("Could not register rule %s: %s", relation_type, str(e))
                continue

        logger.info(
            "Successfully registered %d spatial extraction rules", rules_registered
        )
        return rules_registered

    def create_shape(self, shape: GeometricShape) -> bool:
        """
        Create a geometric shape in the knowledge graph.

        Args:
            shape: GeometricShape instance (Triangle, Quadrilateral, Circle, etc.)

        Returns:
            True if successful, False otherwise
        """
        if not self.netzwerk:
            logger.warning("No knowledge graph available for storing shape")
            return False

        logger.info("Creating shape in knowledge graph: %s", shape)

        try:
            with self._lock:
                # Step 1: Create shape node with basic properties
                shape_properties = {
                    "shape_type": shape.shape_type,
                    "type": "GeometricShape",
                }

                # Add custom properties
                shape_properties.update(shape.properties)

                # Add shape-specific properties
                if isinstance(shape, Polygon):
                    shape_properties["num_sides"] = shape.num_sides

                if isinstance(shape, Circle):
                    shape_properties["radius"] = shape.radius
                    if shape.center:
                        shape_properties["center_x"] = shape.center.x
                        shape_properties["center_y"] = shape.center.y

                # Create shape node
                self.netzwerk.create_wort_if_not_exists(
                    lemma=shape.name, pos="NOUN", **shape_properties
                )

                # Step 2: Create IS_A hierarchy
                # e.g., "Dreieck1" IS_A "Dreieck" IS_A "Polygon" IS_A "Form"
                self.netzwerk.assert_relation(
                    from_lemma=shape.name,
                    to_lemma=shape.shape_type,
                    relation_type="IS_A",
                )

                # Step 3: Add HAS_PROPERTY relations for computed properties
                area = shape.calculate_area()
                if area is not None:
                    # Create property node for area
                    area_node_name = f"{shape.name}_Fläche"
                    self.netzwerk.create_wort_if_not_exists(
                        lemma=area_node_name,
                        pos="NOUN",
                        value=area,
                        unit="quadrat_einheiten",
                    )
                    self.netzwerk.assert_relation(
                        from_lemma=shape.name,
                        to_lemma=area_node_name,
                        relation_type="HAS_PROPERTY",
                    )

                perimeter = shape.calculate_perimeter()
                if perimeter is not None:
                    # Create property node for perimeter
                    perimeter_node_name = f"{shape.name}_Umfang"
                    self.netzwerk.create_wort_if_not_exists(
                        lemma=perimeter_node_name,
                        pos="NOUN",
                        value=perimeter,
                        unit="einheiten",
                    )
                    self.netzwerk.assert_relation(
                        from_lemma=shape.name,
                        to_lemma=perimeter_node_name,
                        relation_type="HAS_PROPERTY",
                    )

                # Step 4: Store vertices if polygon
                if isinstance(shape, Polygon) and shape.vertices:
                    for i, vertex in enumerate(shape.vertices):
                        vertex_node_name = f"{shape.name}_Ecke_{i}"
                        self.netzwerk.create_wort_if_not_exists(
                            lemma=vertex_node_name,
                            pos="NOUN",
                            x=vertex.x,
                            y=vertex.y,
                            index=i,
                            type="Vertex",
                        )
                        self.netzwerk.assert_relation(
                            from_lemma=shape.name,
                            to_lemma=vertex_node_name,
                            relation_type="HAS_VERTEX",
                        )

                logger.info("Shape created successfully: %s", shape.name)
                return True

        except Exception as e:
            logger.error(
                "Failed to create shape %s: %s", shape.name, str(e), exc_info=True
            )
            return False

    def get_shape(self, shape_name: str) -> Optional[GeometricShape]:
        """
        Retrieve a geometric shape from the knowledge graph.

        Args:
            shape_name: Name of the shape

        Returns:
            GeometricShape instance if found, None otherwise
        """
        if not self.netzwerk:
            logger.warning("No knowledge graph available for retrieving shape")
            return None

        try:
            with self._lock:
                # Query shape node
                shape_node = self.netzwerk.find_wort_node(shape_name)

                if not shape_node:
                    logger.warning("Shape not found: %s", shape_name)
                    return None

                # Extract properties
                props = dict(shape_node)
                shape_type = props.get("shape_type", "Unknown")

                # Reconstruct shape based on type
                if shape_type == "Dreieck":
                    # Get vertices
                    from component_42_spatial_shapes import Triangle

                    vertices = self._get_shape_vertices(shape_name)
                    return Triangle(
                        name=shape_name,
                        vertices=vertices,
                        properties={
                            k: v
                            for k, v in props.items()
                            if k not in ["shape_type", "type", "num_sides"]
                        },
                    )

                elif shape_type == "Viereck":
                    vertices = self._get_shape_vertices(shape_name)
                    return Quadrilateral(
                        name=shape_name,
                        vertices=vertices,
                        properties={
                            k: v
                            for k, v in props.items()
                            if k not in ["shape_type", "type", "num_sides"]
                        },
                    )

                elif shape_type == "Kreis":
                    center = None
                    if "center_x" in props and "center_y" in props:
                        center = Position(props["center_x"], props["center_y"])

                    return Circle(
                        name=shape_name,
                        center=center,
                        radius=props.get("radius", 0.0),
                        properties={
                            k: v
                            for k, v in props.items()
                            if k
                            not in [
                                "shape_type",
                                "type",
                                "radius",
                                "center_x",
                                "center_y",
                            ]
                        },
                    )

                else:
                    # Generic polygon
                    num_sides = props.get("num_sides", 0)
                    vertices = self._get_shape_vertices(shape_name)
                    return Polygon(
                        name=shape_name,
                        shape_type=shape_type,
                        num_sides=num_sides,
                        vertices=vertices,
                        properties={
                            k: v
                            for k, v in props.items()
                            if k not in ["shape_type", "type", "num_sides"]
                        },
                    )

        except Exception as e:
            logger.error("Error retrieving shape %s: %s", shape_name, str(e))
            return None

    def _get_shape_vertices(self, shape_name: str) -> List[Position]:
        """
        Get vertices of a shape from the knowledge graph.

        Args:
            shape_name: Name of the shape

        Returns:
            List of vertex positions sorted by index
        """
        try:
            # Query HAS_VERTEX relations
            query = """
            MATCH (shape)-[:HAS_VERTEX]->(vertex)
            WHERE shape.lemma = $shape_name
            RETURN vertex.x as x, vertex.y as y, vertex.index as index
            ORDER BY vertex.index
            """

            result = self._safe_run(query, "get_shape_vertices", shape_name=shape_name)

            vertices = [Position(record["x"], record["y"]) for record in result]

            return vertices

        except Exception as e:
            logger.error("Error getting vertices for %s: %s", shape_name, str(e))
            return []

    def classify_shape(
        self,
        num_sides: Optional[int] = None,
        vertices: Optional[List[Position]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Classify a shape based on its properties.

        Args:
            num_sides: Number of sides (for polygons)
            vertices: Vertex positions (for detailed classification)
            properties: Additional properties

        Returns:
            Shape classification as string
        """
        properties = properties or {}

        # Classify by number of sides
        if num_sides is not None:
            if num_sides == 3:
                return "Dreieck"
            elif num_sides == 4:
                # Further classify quadrilaterals
                if vertices and len(vertices) == 4:
                    quad = Quadrilateral(name="temp", vertices=vertices)
                    if quad.is_rectangle():
                        return "Rechteck"
                return "Viereck"
            elif num_sides == 5:
                return "Fünfeck"
            elif num_sides == 6:
                return "Sechseck"
            else:
                return f"Polygon_{num_sides}"

        # Classify by properties
        if properties.get("radius"):
            return "Kreis"

        return "Unbekannte Form"

    def calculate_shape_properties(self, shape_name: str) -> Dict[str, float]:
        """
        Calculate geometric properties of a shape.

        Args:
            shape_name: Name of the shape

        Returns:
            Dictionary with calculated properties (area, perimeter, etc.)
        """
        try:
            shape = self.get_shape(shape_name)
            if not shape:
                logger.warning("Shape not found: %s", shape_name)
                return {}

            properties = {}

            area = shape.calculate_area()
            if area is not None:
                properties["area"] = area

            perimeter = shape.calculate_perimeter()
            if perimeter is not None:
                properties["perimeter"] = perimeter

            # Shape-specific properties
            if isinstance(shape, Quadrilateral):
                properties["is_rectangle"] = shape.is_rectangle()

            if isinstance(shape, Circle):
                properties["diameter"] = 2 * shape.radius if shape.radius > 0 else 0

            logger.debug("Calculated properties for %s: %s", shape_name, properties)
            return properties

        except Exception as e:
            logger.error("Error calculating properties for %s: %s", shape_name, str(e))
            return {}
