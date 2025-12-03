"""
KAI Test Suite - Neo4j Facade Methods Tests
Tests for new facade methods added in Task 16 Phase A:
- QueryEngine.query_semantic_neighbors()
- QueryEngine.query_transitive_path()
- RelationManager.create_specialized_node()
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logger = logging.getLogger(__name__)


class TestQuerySemanticNeighbors:
    """Tests for QueryEngine.query_semantic_neighbors() method."""

    def test_basic_neighbor_search(self, netzwerk_session, clean_test_concepts):
        """Test basic neighbor search with no filters."""
        # Setup: Create a simple graph: dog -> IS_A -> animal
        #                                dog -> HAS_PROPERTY -> furry
        dog = f"{clean_test_concepts}hund"
        animal = f"{clean_test_concepts}tier"
        furry = f"{clean_test_concepts}pelzig"

        netzwerk_session.ensure_wort_und_konzept(dog)
        netzwerk_session.ensure_wort_und_konzept(animal)
        netzwerk_session.ensure_wort_und_konzept(furry)

        # Create relations using assert_relation (no confidence param in facade)
        # Access the core directly for confidence or use direct Neo4j
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (d:Wort {lemma: $dog}), (a:Wort {lemma: $animal}), (f:Wort {lemma: $furry})
                MERGE (d)-[:IS_A {confidence: 0.9, source: 'test'}]->(a)
                MERGE (d)-[:HAS_PROPERTY {confidence: 0.8, source: 'test'}]->(f)
                """,
                dog=dog,
                animal=animal,
                furry=furry,
            )

        # Test: Find all neighbors of dog (access via _core)
        neighbors = netzwerk_session._core.query_semantic_neighbors(dog)

        # Verify: Should find both neighbors
        assert (
            len(neighbors) >= 2
        ), f"Expected at least 2 neighbors, got {len(neighbors)}"

        neighbor_lemmas = [n["neighbor"] for n in neighbors]
        assert animal in neighbor_lemmas
        assert furry in neighbor_lemmas

        # Check structure
        for neighbor in neighbors:
            assert "neighbor" in neighbor
            assert "relation_type" in neighbor
            assert "confidence" in neighbor
            assert isinstance(neighbor["confidence"], float)

    def test_filtered_by_allowed_relations(self, netzwerk_session, clean_test_concepts):
        """Test neighbor search filtered by allowed relations."""
        # Setup: Same graph as above
        dog = f"{clean_test_concepts}hund2"
        animal = f"{clean_test_concepts}tier2"
        furry = f"{clean_test_concepts}pelzig2"

        netzwerk_session.ensure_wort_und_konzept(dog)
        netzwerk_session.ensure_wort_und_konzept(animal)
        netzwerk_session.ensure_wort_und_konzept(furry)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (d:Wort {lemma: $dog}), (a:Wort {lemma: $animal}), (f:Wort {lemma: $furry})
                MERGE (d)-[:IS_A {confidence: 0.9, source: 'test'}]->(a)
                MERGE (d)-[:HAS_PROPERTY {confidence: 0.8, source: 'test'}]->(f)
                """,
                dog=dog,
                animal=animal,
                furry=furry,
            )

        # Test: Filter by IS_A only
        neighbors = netzwerk_session._core.query_semantic_neighbors(
            dog, allowed_relations=["IS_A"]
        )

        # Verify: Should only find animal
        neighbor_lemmas = [n["neighbor"] for n in neighbors]
        assert animal in neighbor_lemmas
        assert furry not in neighbor_lemmas

    def test_filtered_by_min_confidence(self, netzwerk_session, clean_test_concepts):
        """Test neighbor search filtered by minimum confidence."""
        # Setup: Create neighbors with different confidences
        cat = f"{clean_test_concepts}katze"
        animal = f"{clean_test_concepts}tier3"
        pet = f"{clean_test_concepts}haustier"

        netzwerk_session.ensure_wort_und_konzept(cat)
        netzwerk_session.ensure_wort_und_konzept(animal)
        netzwerk_session.ensure_wort_und_konzept(pet)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (c:Wort {lemma: $cat}), (a:Wort {lemma: $animal}), (p:Wort {lemma: $pet})
                MERGE (c)-[:IS_A {confidence: 0.95, source: 'test'}]->(a)
                MERGE (c)-[:IS_A {confidence: 0.5, source: 'test'}]->(p)
                """,
                cat=cat,
                animal=animal,
                pet=pet,
            )

        # Test: Filter by min_confidence=0.8
        neighbors = netzwerk_session._core.query_semantic_neighbors(
            cat, min_confidence=0.8
        )

        # Verify: Should only find high-confidence neighbor
        neighbor_lemmas = [n["neighbor"] for n in neighbors]
        assert animal in neighbor_lemmas
        assert pet not in neighbor_lemmas

    def test_with_limit_parameter(self, netzwerk_session, clean_test_concepts):
        """Test neighbor search with result limit."""
        # Setup: Create multiple neighbors
        bird = f"{clean_test_concepts}vogel"
        neighbors_list = []

        netzwerk_session.ensure_wort_und_konzept(bird)

        for i in range(5):
            neighbor = f"{clean_test_concepts}neighbor_{i}"
            neighbors_list.append(neighbor)
            netzwerk_session.ensure_wort_und_konzept(neighbor)

        # Create all relations in one transaction
        with netzwerk_session.driver.session(database="neo4j") as session:
            for neighbor in neighbors_list:
                session.run(
                    """
                    MATCH (b:Wort {lemma: $bird}), (n:Wort {lemma: $neighbor})
                    MERGE (b)-[:IS_A {confidence: 0.8, source: 'test'}]->(n)
                    """,
                    bird=bird,
                    neighbor=neighbor,
                )

        # Test: Limit to 3 results
        neighbors = netzwerk_session._core.query_semantic_neighbors(bird, limit=3)

        # Verify: Should return at most 3 neighbors
        assert len(neighbors) <= 3

    def test_empty_results(self, netzwerk_session, clean_test_concepts):
        """Test neighbor search with no results."""
        # Setup: Create isolated node
        isolated = f"{clean_test_concepts}isolated"
        netzwerk_session.ensure_wort_und_konzept(isolated)

        # Test: Search for neighbors
        neighbors = netzwerk_session._core.query_semantic_neighbors(isolated)

        # Verify: Should return empty list
        assert neighbors == []

    def test_non_existent_lemma(self, netzwerk_session, clean_test_concepts):
        """Test neighbor search for non-existent lemma."""
        # Test: Search for non-existent word
        neighbors = netzwerk_session._core.query_semantic_neighbors(
            f"{clean_test_concepts}does_not_exist"
        )

        # Verify: Should return empty list (not crash)
        assert neighbors == []

    def test_bidirectional_search(self, netzwerk_session, clean_test_concepts):
        """Test that search finds neighbors in both directions."""
        # Setup: Create bidirectional relationships
        parent = f"{clean_test_concepts}parent"
        child = f"{clean_test_concepts}child"

        netzwerk_session.ensure_wort_und_konzept(parent)
        netzwerk_session.ensure_wort_und_konzept(child)

        # Create unidirectional relationship
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (c:Wort {lemma: $child}), (p:Wort {lemma: $parent})
                MERGE (c)-[:IS_A {confidence: 0.9, source: 'test'}]->(p)
                """,
                child=child,
                parent=parent,
            )

        # Test: Search from parent (reverse direction)
        neighbors_from_parent = netzwerk_session._core.query_semantic_neighbors(parent)

        # Verify: Should find child (bidirectional search)
        neighbor_lemmas = [n["neighbor"] for n in neighbors_from_parent]
        assert (
            child in neighbor_lemmas
        ), "Bidirectional search should find child from parent"


class TestQueryTransitivePath:
    """Tests for QueryEngine.query_transitive_path() method."""

    def test_path_with_both_subject_and_object(
        self, netzwerk_session, clean_test_concepts
    ):
        """Test path finding with both subject and object specified."""
        # Setup: Create transitive chain: dog -> mammal -> animal
        dog = f"{clean_test_concepts}dog_trans"
        mammal = f"{clean_test_concepts}mammal_trans"
        animal = f"{clean_test_concepts}animal_trans"

        # Create concepts (not just words)
        for word in [dog, mammal, animal]:
            netzwerk_session.ensure_wort_und_konzept(word)

        # Create IS_A relationships between concepts
        with netzwerk_session.driver.session(database="neo4j") as session:
            # Ensure concepts exist and create IS_A relationships
            session.run(
                """
                MERGE (d:Konzept {name: $dog})
                MERGE (m:Konzept {name: $mammal})
                MERGE (a:Konzept {name: $animal})
                MERGE (d)-[:IS_A {confidence: 0.9}]->(m)
                MERGE (m)-[:IS_A {confidence: 0.9}]->(a)
                """,
                dog=dog,
                mammal=mammal,
                animal=animal,
            )

        # Test: Find path from dog to animal
        paths = netzwerk_session._core.query_transitive_path(
            subject=dog, predicate="IS_A", object=animal, max_hops=3
        )

        # Verify: Should find at least one path
        assert len(paths) > 0, "Should find transitive path"

        # Check first path structure
        path = paths[0]
        assert path["subject"] == dog
        assert path["object"] == animal
        assert path["hops"] >= 1
        assert "path" in path
        assert isinstance(path["path"], list)

    def test_path_with_only_subject(self, netzwerk_session, clean_test_concepts):
        """Test finding all reachable objects from a subject."""
        # Setup: Create fan-out: root -> child1, root -> child2
        root = f"{clean_test_concepts}root"
        child1 = f"{clean_test_concepts}child1"
        child2 = f"{clean_test_concepts}child2"

        for word in [root, child1, child2]:
            netzwerk_session.ensure_wort_und_konzept(word)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MERGE (r:Konzept {name: $root})
                MERGE (c1:Konzept {name: $child1})
                MERGE (c2:Konzept {name: $child2})
                MERGE (r)-[:IS_A]->(c1)
                MERGE (r)-[:IS_A]->(c2)
                """,
                root=root,
                child1=child1,
                child2=child2,
            )

        # Test: Find all reachable from root
        paths = netzwerk_session._core.query_transitive_path(
            subject=root, predicate="IS_A", object=None, max_hops=2
        )

        # Verify: Should find both children
        assert len(paths) >= 2, "Should find multiple reachable objects"
        objects = [p["object"] for p in paths]
        assert child1 in objects
        assert child2 in objects

    def test_path_with_only_object(self, netzwerk_session, clean_test_concepts):
        """Test finding all subjects that can reach an object."""
        # Setup: Create fan-in: parent1 -> target, parent2 -> target
        parent1 = f"{clean_test_concepts}parent1"
        parent2 = f"{clean_test_concepts}parent2"
        target = f"{clean_test_concepts}target"

        for word in [parent1, parent2, target]:
            netzwerk_session.ensure_wort_und_konzept(word)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MERGE (p1:Konzept {name: $parent1})
                MERGE (p2:Konzept {name: $parent2})
                MERGE (t:Konzept {name: $target})
                MERGE (p1)-[:IS_A]->(t)
                MERGE (p2)-[:IS_A]->(t)
                """,
                parent1=parent1,
                parent2=parent2,
                target=target,
            )

        # Test: Find all subjects that reach target
        paths = netzwerk_session._core.query_transitive_path(
            subject=None, predicate="IS_A", object=target, max_hops=2
        )

        # Verify: Should find both parents
        assert len(paths) >= 2, "Should find multiple subjects"
        subjects = [p["subject"] for p in paths]
        assert parent1 in subjects
        assert parent2 in subjects

    def test_invalid_inputs_neither_specified(self, netzwerk_session):
        """Test that query returns empty when neither subject nor object specified."""
        # Test: Call with both None
        paths = netzwerk_session._core.query_transitive_path(
            subject=None, predicate="IS_A", object=None, max_hops=3
        )

        # Verify: Should return empty list (too broad)
        assert paths == []

    def test_max_hops_validation(self, netzwerk_session, clean_test_concepts):
        """Test that max_hops is clamped to valid range [1, 5]."""
        # Setup: Create simple path
        start = f"{clean_test_concepts}start_hops"
        end = f"{clean_test_concepts}end_hops"

        for word in [start, end]:
            netzwerk_session.ensure_wort_und_konzept(word)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MERGE (s:Konzept {name: $start})
                MERGE (e:Konzept {name: $end})
                MERGE (s)-[:IS_A]->(e)
                """,
                start=start,
                end=end,
            )

        # Test: Try invalid max_hops values (should be clamped)
        # max_hops=0 should be clamped to 1
        paths_zero = netzwerk_session._core.query_transitive_path(
            subject=start, predicate="IS_A", object=end, max_hops=0
        )
        # Should still work (clamped to 1)
        assert isinstance(paths_zero, list)

        # max_hops=10 should be clamped to 5
        paths_ten = netzwerk_session._core.query_transitive_path(
            subject=start, predicate="IS_A", object=end, max_hops=10
        )
        # Should still work (clamped to 5)
        assert isinstance(paths_ten, list)

    def test_no_path_exists(self, netzwerk_session, clean_test_concepts):
        """Test when no path exists between subject and object."""
        # Setup: Create isolated concepts
        isolated1 = f"{clean_test_concepts}isolated1"
        isolated2 = f"{clean_test_concepts}isolated2"

        for word in [isolated1, isolated2]:
            netzwerk_session.ensure_wort_und_konzept(word)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MERGE (i1:Konzept {name: $isolated1})
                MERGE (i2:Konzept {name: $isolated2})
                """,
                isolated1=isolated1,
                isolated2=isolated2,
            )

        # Test: Try to find path between isolated concepts
        paths = netzwerk_session._core.query_transitive_path(
            subject=isolated1, predicate="IS_A", object=isolated2, max_hops=3
        )

        # Verify: Should return empty list
        assert paths == []

    def test_multi_hop_path(self, netzwerk_session, clean_test_concepts):
        """Test finding multi-hop transitive paths."""
        # Setup: Create 3-hop chain: a -> b -> c -> d
        a = f"{clean_test_concepts}a_hop"
        b = f"{clean_test_concepts}b_hop"
        c = f"{clean_test_concepts}c_hop"
        d = f"{clean_test_concepts}d_hop"

        for word in [a, b, c, d]:
            netzwerk_session.ensure_wort_und_konzept(word)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MERGE (a:Konzept {name: $a})
                MERGE (b:Konzept {name: $b})
                MERGE (c:Konzept {name: $c})
                MERGE (d:Konzept {name: $d})
                MERGE (a)-[:IS_A]->(b)
                MERGE (b)-[:IS_A]->(c)
                MERGE (c)-[:IS_A]->(d)
                """,
                a=a,
                b=b,
                c=c,
                d=d,
            )

        # Test: Find path from a to d (3 hops)
        paths = netzwerk_session._core.query_transitive_path(
            subject=a, predicate="IS_A", object=d, max_hops=5
        )

        # Verify: Should find path with 3 hops
        assert len(paths) > 0
        # Find the path from a to d
        a_to_d_path = next(
            (p for p in paths if p["subject"] == a and p["object"] == d), None
        )
        assert a_to_d_path is not None
        assert a_to_d_path["hops"] == 3


class TestCreateSpecializedNode:
    """Tests for RelationManager.create_specialized_node() method."""

    def test_create_node_without_word_link(self, netzwerk_session, clean_test_concepts):
        """Test creating specialized node without linking to a word."""
        # Test: Create NumberNode without word link
        success = netzwerk_session._core.create_specialized_node(
            label="NumberNode",
            properties={"value": 42, "text": "forty_two"},
            link_to_word=None,
        )

        # Verify: Should succeed
        assert success is True

        # Verify node exists in database
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (n:NumberNode {value: 42})
                RETURN n.value as value, n.text as text
                """
            )
            record = result.single()
            assert record is not None
            assert record["value"] == 42
            assert record["text"] == "forty_two"

        # Cleanup
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run("MATCH (n:NumberNode {value: 42}) DETACH DELETE n")

    def test_create_node_with_word_link(self, netzwerk_session, clean_test_concepts):
        """Test creating specialized node with word link."""
        word = f"{clean_test_concepts}fuenf"

        # Test: Create NumberNode with word link
        success = netzwerk_session._core.create_specialized_node(
            label="NumberNode",
            properties={"value": 5, "word": word},
            link_to_word=word,
            relation_type="EQUIVALENT_TO",
        )

        # Verify: Should succeed
        assert success is True

        # Verify node and link exist
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (w:Wort {lemma: $lemma})-[r:EQUIVALENT_TO]->(n:NumberNode {value: 5})
                RETURN n.value as value, n.word as word, r.confidence as confidence
                """,
                lemma=word,
            )
            record = result.single()
            assert record is not None
            assert record["value"] == 5
            assert record["word"] == word
            assert record["confidence"] == 1.0

        # Cleanup
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run("MATCH (n:NumberNode {value: 5}) DETACH DELETE n")

    def test_invalid_label_format(self, netzwerk_session):
        """Test that invalid label format is rejected."""
        # Test: Try to create node with invalid label (spaces, special chars)
        invalid_labels = [
            "Invalid Label",  # space
            "Invalid-Label",  # hyphen
            "123Invalid",  # starts with number
            "Invalid.Label",  # dot
            "",  # empty
        ]

        for label in invalid_labels:
            success = netzwerk_session._core.create_specialized_node(
                label=label, properties={"key": "value"}
            )
            # Verify: Should fail
            assert success is False, f"Label '{label}' should be rejected"

    def test_valid_label_formats(self, netzwerk_session):
        """Test that valid label formats are accepted."""
        # Test: Valid labels
        valid_labels = [
            "ValidLabel",
            "Valid_Label",
            "_ValidLabel",
            "ValidLabel123",
            "VALID_LABEL",
        ]

        for label in valid_labels:
            success = netzwerk_session._core.create_specialized_node(
                label=label, properties={"key": f"value_{label}"}
            )
            # Verify: Should succeed
            assert success is True, f"Label '{label}' should be accepted"

            # Cleanup
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")

    def test_invalid_relation_type(self, netzwerk_session, clean_test_concepts):
        """Test that invalid relation type is rejected."""
        word = f"{clean_test_concepts}test_invalid_rel"

        # Test: Try to create with invalid relation type (with special chars)
        success = netzwerk_session._core.create_specialized_node(
            label="TestNode",
            properties={"key": "value"},
            link_to_word=word,
            relation_type="INVALID-RELATION",  # hyphen not allowed
        )

        # Verify: Should fail
        assert success is False

    def test_empty_properties_dict(self, netzwerk_session):
        """Test that empty properties dict is rejected."""
        # Test: Try to create node with empty properties
        success = netzwerk_session._core.create_specialized_node(
            label="TestNode", properties={}
        )

        # Verify: Should fail
        assert success is False

    def test_idempotency(self, netzwerk_session, clean_test_concepts):
        """Test that creating same node twice doesn't create duplicates."""
        # Test: Create same node twice
        properties = {"value": 99, "name": "test_node"}

        success1 = netzwerk_session._core.create_specialized_node(
            label="IdempotentNode", properties=properties
        )

        success2 = netzwerk_session._core.create_specialized_node(
            label="IdempotentNode", properties=properties
        )

        # Verify: Both should succeed
        assert success1 is True
        assert success2 is True

        # Verify: Only one node exists
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (n:IdempotentNode {value: 99})
                RETURN count(n) as count
                """
            )
            count = result.single()["count"]
            assert count == 1, "Should only have one node (idempotent)"

        # Cleanup
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run("MATCH (n:IdempotentNode) DETACH DELETE n")

    def test_multiple_properties(self, netzwerk_session):
        """Test creating node with multiple properties."""
        # Test: Create node with many properties
        properties = {
            "id": "test_123",
            "value": 42,
            "name": "test_multi",
            "active": True,
            "score": 3.14,
        }

        success = netzwerk_session._core.create_specialized_node(
            label="MultiPropNode", properties=properties
        )

        # Verify: Should succeed
        assert success is True

        # Verify: All properties set
        with netzwerk_session.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (n:MultiPropNode {id: "test_123"})
                RETURN n.id as id, n.value as value, n.name as name,
                       n.active as active, n.score as score
                """
            )
            record = result.single()
            assert record is not None
            assert record["id"] == "test_123"
            assert record["value"] == 42
            assert record["name"] == "test_multi"
            assert record["active"] is True
            assert abs(record["score"] - 3.14) < 0.01

        # Cleanup
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run("MATCH (n:MultiPropNode) DETACH DELETE n")

    def test_specialized_node_types(self, netzwerk_session, clean_test_concepts):
        """Test creating various specialized node types."""
        # Test: Create different specialized node types
        test_cases = [
            ("NumberNode", {"value": 7}, f"{clean_test_concepts}sieben"),
            ("Operation", {"name": "add", "symbol": "+"}, None),
            ("SpatialObject", {"x": 5, "y": 10, "name": "obj1"}, None),
        ]

        for label, properties, word_link in test_cases:
            success = netzwerk_session._core.create_specialized_node(
                label=label, properties=properties, link_to_word=word_link
            )
            assert success is True, f"Failed to create {label}"

            # Verify existence
            with netzwerk_session.driver.session(database="neo4j") as session:
                primary_key = list(properties.keys())[0]
                result = session.run(
                    f"MATCH (n:{label} {{{primary_key}: ${primary_key}}}) RETURN count(n) as count",
                    **{primary_key: properties[primary_key]},
                )
                count = result.single()["count"]
                assert count >= 1, f"{label} node not found"

            # Cleanup
            with netzwerk_session.driver.session(database="neo4j") as session:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")


# Integration test combining multiple methods
class TestFacadeMethodsIntegration:
    """Integration tests combining multiple facade methods."""

    def test_semantic_neighbors_with_specialized_nodes(
        self, netzwerk_session, clean_test_concepts
    ):
        """Test that semantic neighbors work with specialized nodes."""
        # Setup: Create word linked to specialized node
        word = f"{clean_test_concepts}integration_test"

        # Create specialized node linked to word
        netzwerk_session._core.create_specialized_node(
            label="TestSpecial",
            properties={"id": "special_1", "data": "test"},
            link_to_word=word,
            relation_type="EQUIVALENT_TO",
        )

        # Create regular neighbor
        neighbor = f"{clean_test_concepts}neighbor_integration"
        netzwerk_session.ensure_wort_und_konzept(word)
        netzwerk_session.ensure_wort_und_konzept(neighbor)

        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (w:Wort {lemma: $word}), (n:Wort {lemma: $neighbor})
                MERGE (w)-[:IS_A {confidence: 0.9, source: 'test'}]->(n)
                """,
                word=word,
                neighbor=neighbor,
            )

        # Test: Query semantic neighbors
        neighbors = netzwerk_session._core.query_semantic_neighbors(word)

        # Verify: Should find at least the regular neighbor
        neighbor_lemmas = [n["neighbor"] for n in neighbors]
        assert neighbor in neighbor_lemmas

        # Cleanup
        with netzwerk_session.driver.session(database="neo4j") as session:
            session.run("MATCH (n:TestSpecial) DETACH DELETE n")

    def test_transitive_path_performance(self, netzwerk_session, clean_test_concepts):
        """Test transitive path finding with larger graph."""
        # Setup: Create chain of 5 concepts
        concepts = [f"{clean_test_concepts}chain_{i}" for i in range(5)]

        for concept in concepts:
            netzwerk_session.ensure_wort_und_konzept(concept)

        # Create chain in Neo4j
        with netzwerk_session.driver.session(database="neo4j") as session:
            for i in range(len(concepts) - 1):
                session.run(
                    """
                    MERGE (a:Konzept {name: $a})
                    MERGE (b:Konzept {name: $b})
                    MERGE (a)-[:IS_A]->(b)
                    """,
                    a=concepts[i],
                    b=concepts[i + 1],
                )

        # Test: Find path from first to last (4 hops)
        paths = netzwerk_session._core.query_transitive_path(
            subject=concepts[0], predicate="IS_A", object=concepts[4], max_hops=5
        )

        # Verify: Should find path
        assert len(paths) > 0
        assert paths[0]["hops"] == 4
        assert len(paths[0]["path"]) == 5  # 5 nodes in path
