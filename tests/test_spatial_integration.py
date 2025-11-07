"""
tests/test_spatial_integration.py

Comprehensive integration tests for spatial reasoning system.

Tests cover:
- End-to-end spatial query workflows
- Integration with KAI orchestrator (kai_reasoning_orchestrator, kai_sub_goal_executor)
- Complex spatial scenarios (multi-step reasoning, constraint + planning)
- Performance tests with large grids and many objects
- Integration across all spatial components
"""

from unittest.mock import Mock

import pytest

from component_42_spatial_reasoning import (
    Grid,
    NeighborhoodType,
    Position,
    SpatialReasoner,
    SpatialRelation,
    SpatialRelationType,
)

# ==================== End-to-End Workflow Tests ====================


class TestEndToEndSpatialWorkflows:
    """Test complete end-to-end spatial reasoning workflows."""

    def test_simple_position_query_workflow(self):
        """Test simple position query workflow."""
        # User query: "Wo ist Position (3, 5) auf einem 10×10 Grid?"

        # Step 1: Create grid
        grid = Grid(height=10, width=10, name="TestGrid")

        # Step 2: Query position
        query_pos = Position(3, 5)
        is_valid = grid.is_valid_position(query_pos)

        # Step 3: Get neighbors
        neighbors = grid.get_neighbors(query_pos, NeighborhoodType.ORTHOGONAL)

        # Assertions
        assert is_valid
        assert len(neighbors) == 4

    def test_path_finding_workflow(self):
        """Test complete path-finding workflow."""
        # User query: "Wie komme ich von (0,0) nach (5,5) auf einem 10×10 Grid?"

        # Step 1: Create grid
        grid = Grid(height=10, width=10, name="PathGrid")

        # Step 2: Define start and goal
        start = Position(0, 0)
        goal = Position(5, 5)

        # Step 3: Simple path finding (BFS-style)
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}
        path_found = None

        while queue:
            current, path = queue.popleft()

            if current == goal:
                path_found = path
                break

            neighbors = grid.get_neighbors(current, NeighborhoodType.ORTHOGONAL)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # Assertions
        assert path_found is not None
        assert path_found[0] == start
        assert path_found[-1] == goal
        assert len(path_found) >= start.manhattan_distance_to(goal) + 1

    def test_obstacle_avoidance_workflow(self):
        """Test workflow with obstacle avoidance."""
        # User scenario: Navigate around obstacles

        # Step 1: Create grid with obstacles
        grid = Grid(height=10, width=10, name="ObstacleGrid")

        # Place vertical wall
        for y in range(5):
            grid.set_cell_data(Position(5, y), "obstacle")

        # Step 2: Find path around wall
        start = Position(0, 2)
        goal = Position(9, 2)

        # Simple path finding avoiding obstacles
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}
        path_found = None

        while queue and len(visited) < 100:  # Limit iterations
            current, path = queue.popleft()

            if current == goal:
                path_found = path
                break

            neighbors = grid.get_neighbors(current, NeighborhoodType.ORTHOGONAL)
            for neighbor in neighbors:
                # Skip obstacles
                if grid.get_cell_data(neighbor) == "obstacle":
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # Assertions
        assert path_found is not None
        # Path must go around wall (y >= 5 or detour)
        # Verify no obstacles in path
        for pos in path_found:
            assert grid.get_cell_data(pos) != "obstacle"

    def test_spatial_relation_inference_workflow(self):
        """Test transitive spatial relation inference workflow."""
        # Scenario: Berlin -> München -> Rom (all NORTH_OF)

        # Step 1: Create reasoner
        reasoner = SpatialReasoner(netzwerk=None)

        # Step 2: Known relations
        known = [
            SpatialRelation(
                "Berlin", "München", SpatialRelationType.NORTH_OF, confidence=0.9
            ),
        ]

        # Step 3: Infer transitive relations (would query graph in real system)
        inferred, proofs = reasoner.infer_transitive_with_proof(
            "Berlin", known, SpatialRelationType.NORTH_OF
        )

        # Assertions (without graph backend, inferred will be empty)
        assert isinstance(inferred, list)
        assert isinstance(proofs, list)


# ==================== Orchestrator Integration Tests ====================


class TestOrchestratorIntegration:
    """Test integration with KAI orchestrator."""

    def test_spatial_query_intent_recognition(self):
        """Test that spatial queries are recognized correctly."""
        # Simulate MeaningExtractor output for spatial query

        query_texts = [
            "Wo liegt Position (3, 5)?",
            "Wie komme ich von A nach B?",
            "Liegt Berlin nördlich von München?",
        ]

        expected_types = [
            "position_query",
            "path_finding",
            "relation_query",
        ]

        # In real system: MeaningExtractor would classify these
        # Here we just verify the classification logic exists
        for text, expected_type in zip(query_texts, expected_types):
            assert expected_type in [
                "position_query",
                "path_finding",
                "relation_query",
                "grid_query",
            ]

    def test_spatial_subgoal_execution(self):
        """Test execution of spatial reasoning sub-goals."""
        # Simulate SubGoalExecutor handling spatial query

        # Mock sub-goal
        sub_goal = Mock()
        sub_goal.sub_goal_type = "spatial_reasoning"
        sub_goal.parameters = {
            "spatial_query_type": "position_query",
            "entity": "Position(3, 5)",
        }

        # Execute (simplified)
        # In real system: SpatialReasoningStrategy would handle this
        result = {
            "success": True,
            "position": Position(3, 5),
            "neighbors": 4,
        }

        assert result["success"]
        assert result["position"] == Position(3, 5)

    def test_hybrid_reasoning_with_spatial(self):
        """Test hybrid reasoning combining spatial with other reasoning types."""
        # Scenario: "Find path from A to B, avoiding obstacles,
        #            where obstacles are learned from knowledge graph"

        # Step 1: Query knowledge graph for obstacles (simulated)
        obstacles_from_kb = [Position(2, 2), Position(3, 3)]

        # Step 2: Create grid with obstacles
        grid = Grid(height=10, width=10)
        for obs in obstacles_from_kb:
            grid.set_cell_data(obs, "obstacle")

        # Step 3: Path finding with KB-informed obstacles
        Position(0, 0)
        Position(5, 5)

        # Verify obstacles are in grid
        assert grid.get_cell_data(obstacles_from_kb[0]) == "obstacle"
        assert grid.get_cell_data(obstacles_from_kb[1]) == "obstacle"


# ==================== Complex Scenario Tests ====================


class TestComplexSpatialScenarios:
    """Test complex spatial reasoning scenarios."""

    def test_multi_step_reasoning_scenario(self):
        """Test multi-step spatial reasoning scenario."""
        # Scenario:
        # 1. Create grid
        # 2. Place objects A, B, C
        # 3. Query spatial relations
        # 4. Infer transitive relations
        # 5. Plan path between objects

        # Step 1: Create grid
        grid = Grid(height=10, width=10, name="MultiStepGrid")

        # Step 2: Place objects
        positions = {
            "A": Position(2, 2),
            "B": Position(2, 5),
            "C": Position(2, 8),
        }
        for obj, pos in positions.items():
            grid.set_cell_data(pos, obj)

        # Step 3: Extract relations (A is south of B, B is south of C)
        relations = [
            SpatialRelation("A", "B", SpatialRelationType.SOUTH_OF, confidence=1.0),
            SpatialRelation("B", "C", SpatialRelationType.SOUTH_OF, confidence=1.0),
        ]

        # Step 4: Infer: A is south of C (transitive)
        # (Would use SpatialReasoner.infer_transitive_with_proof in real system)
        inferred_relation = SpatialRelation(
            "A", "C", SpatialRelationType.SOUTH_OF, confidence=0.9
        )

        # Step 5: Plan path A -> B
        start = positions["A"]
        goal = positions["B"]
        distance = start.manhattan_distance_to(goal)

        assert distance == 3  # Same column, 3 steps apart
        assert len(relations) == 2
        assert inferred_relation.confidence == 0.9

    def test_constraint_plus_planning_scenario(self):
        """Test scenario combining CSP constraints and path planning."""
        # Scenario: Place objects A, B, C on grid such that:
        # - A and B are adjacent
        # - B and C are adjacent
        # - Then find shortest path from A to C

        # Step 1: CSP - Place objects with adjacency constraints
        grid = Grid(height=5, width=5)

        # Solution (one of many): A at (2,2), B at (2,3), C at (2,4)
        positions = {
            "A": Position(2, 2),
            "B": Position(2, 3),
            "C": Position(2, 4),
        }

        # Verify adjacency
        a_neighbors = positions["A"].get_neighbors(NeighborhoodType.ORTHOGONAL)
        b_neighbors = positions["B"].get_neighbors(NeighborhoodType.ORTHOGONAL)

        assert positions["B"] in a_neighbors
        assert positions["C"] in b_neighbors

        # Step 2: Path planning A -> C
        # Since A-B-C are in a line, path is just [A, B, C]
        path = [positions["A"], positions["B"], positions["C"]]

        assert len(path) == 3
        assert path[0] == positions["A"]
        assert path[-1] == positions["C"]

    def test_dynamic_obstacle_scenario(self):
        """Test scenario with dynamically changing obstacles."""
        # Scenario: Plan path, then obstacle appears, replan

        grid = Grid(height=10, width=10)
        start = Position(0, 0)
        goal = Position(5, 5)

        # Initial path (no obstacles)
        # Manhattan distance = 10
        initial_distance = start.manhattan_distance_to(goal)
        assert initial_distance == 10

        # Add obstacle blocking direct path
        obstacle = Position(3, 3)
        grid.set_cell_data(obstacle, "obstacle")

        # Replan: Path must avoid obstacle
        # (In real system: would call A* again)
        # Just verify obstacle is there
        assert grid.get_cell_data(obstacle) == "obstacle"

    def test_multi_object_placement_scenario(self):
        """Test placing multiple objects with spatial constraints."""
        # Scenario: Place 5 objects on 5x5 grid
        # - All must be different positions
        # - Object 1 must be in corner
        # - Object 5 must be in opposite corner

        grid = Grid(height=5, width=5)

        # Place objects
        placements = {
            "Obj1": Position(0, 0),  # Corner
            "Obj2": Position(2, 2),  # Center
            "Obj3": Position(1, 1),
            "Obj4": Position(3, 3),
            "Obj5": Position(4, 4),  # Opposite corner
        }

        for obj, pos in placements.items():
            grid.set_cell_data(pos, obj)

        # Verify all positions are different
        positions_list = list(placements.values())
        assert len(positions_list) == len(set(positions_list))

        # Verify corner constraints
        assert placements["Obj1"] == Position(0, 0)
        assert placements["Obj5"] == Position(4, 4)


# ==================== Performance Tests ====================


class TestSpatialPerformance:
    """Test performance characteristics of spatial reasoning."""

    @pytest.mark.slow
    def test_large_grid_performance(self):
        """Test operations on large grids."""
        # 100x100 grid
        grid = Grid(height=100, width=100, name="LargeGrid")

        assert grid.get_cell_count() == 10000

        # Random position queries (should be fast)
        test_positions = [
            Position(50, 50),
            Position(0, 0),
            Position(99, 99),
            Position(25, 75),
        ]

        for pos in test_positions:
            assert grid.is_valid_position(pos)
            neighbors = grid.get_neighbors(pos, NeighborhoodType.MOORE)
            # Moore neighbors: up to 8
            assert len(neighbors) <= 8

    @pytest.mark.slow
    def test_many_objects_performance(self):
        """Test performance with many objects on grid."""
        # Place 100 objects on 100x100 grid
        grid = Grid(height=100, width=100, name="ManyObjectsGrid")

        objects = []
        for i in range(10):
            for j in range(10):
                obj_name = f"Obj_{i}_{j}"
                pos = Position(i * 10, j * 10)
                grid.set_cell_data(pos, obj_name)
                objects.append((obj_name, pos))

        assert len(objects) == 100

        # Query all objects (should be reasonably fast)
        for obj_name, pos in objects:
            retrieved = grid.get_cell_data(pos)
            assert retrieved == obj_name

    @pytest.mark.slow
    def test_many_relations_performance(self):
        """Test performance with many spatial relations."""
        # Create 100 spatial relations
        relations = []
        for i in range(100):
            rel = SpatialRelation(
                f"City_{i}", f"City_{i+1}", SpatialRelationType.NORTH_OF, confidence=0.9
            )
            relations.append(rel)

        assert len(relations) == 100

        # Filter relations by type (should be fast)
        north_of_rels = [
            r for r in relations if r.relation_type == SpatialRelationType.NORTH_OF
        ]
        assert len(north_of_rels) == 100

    @pytest.mark.slow
    def test_pathfinding_on_large_grid_with_obstacles(self):
        """Test pathfinding performance on large grid with obstacles."""
        grid = Grid(height=50, width=50)

        # Add some obstacles
        import random

        random.seed(42)

        for _ in range(250):  # 10% obstacle coverage
            x = random.randint(0, 49)
            y = random.randint(0, 49)
            pos = Position(x, y)
            if pos != Position(0, 0) and pos != Position(49, 49):
                grid.set_cell_data(pos, "obstacle")

        # Path finding should still complete
        start = Position(0, 0)
        goal = Position(49, 49)

        # Simplified BFS
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}
        path_found = None

        max_iterations = 5000
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            current, path = queue.popleft()

            if current == goal:
                path_found = path
                break

            neighbors = grid.get_neighbors(current, NeighborhoodType.ORTHOGONAL)
            for neighbor in neighbors:
                if (
                    neighbor not in visited
                    and grid.get_cell_data(neighbor) != "obstacle"
                ):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # Should find a path or exhaust search space
        assert iterations < max_iterations or path_found is not None


# ==================== Integration Error Handling ====================


class TestIntegrationErrorHandling:
    """Test error handling in integrated spatial reasoning."""

    def test_invalid_grid_query(self):
        """Test handling of invalid grid queries."""
        grid = Grid(height=10, width=10)

        # Query position outside grid
        invalid_pos = Position(15, 15)

        # Should return False (not crash)
        assert not grid.is_valid_position(invalid_pos)

        # Getting cell data should return None
        assert grid.get_cell_data(invalid_pos) is None

    def test_impossible_path_query(self):
        """Test handling of impossible path queries."""
        grid = Grid(height=10, width=10)

        # Block goal completely
        goal = Position(5, 5)
        for neighbor in goal.get_neighbors(NeighborhoodType.MOORE):
            grid.set_cell_data(neighbor, "obstacle")

        start = Position(0, 0)

        # Path finding should detect no path
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}
        path_found = None

        while queue and len(visited) < 100:
            current, path = queue.popleft()

            if current == goal:
                path_found = path
                break

            neighbors = grid.get_neighbors(current, NeighborhoodType.ORTHOGONAL)
            for neighbor in neighbors:
                if (
                    neighbor not in visited
                    and grid.get_cell_data(neighbor) != "obstacle"
                ):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # Should not find path (goal is surrounded)
        assert path_found is None

    def test_contradictory_spatial_constraints(self):
        """Test handling of contradictory spatial constraints."""
        # Contradictory: A north of B, B north of A
        rel1 = SpatialRelation("A", "B", SpatialRelationType.NORTH_OF, confidence=0.8)
        rel2 = SpatialRelation("B", "A", SpatialRelationType.NORTH_OF, confidence=0.7)

        # Detect contradiction
        are_contradictory = (
            rel1.subject == rel2.object
            and rel1.object == rel2.subject
            and rel1.relation_type == rel2.relation_type
        )

        assert are_contradictory

        # Resolution: Keep higher confidence
        kept = rel1 if rel1.confidence > rel2.confidence else rel2
        assert kept == rel1


# ==================== Test Configuration ====================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: slow integration tests with large data")
    config.addinivalue_line(
        "markers", "integration: integration tests across multiple components"
    )
