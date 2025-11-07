# tests/test_enhanced_schema.py
"""
Tests für Enhanced Graph Schema (component_48_enhanced_schema.py)

Test-Kategorien:
1. Node Properties Tests
2. Relation Properties Tests
3. Schema Manager Tests
4. Integration Tests
"""

from unittest.mock import MagicMock, Mock

import pytest

from component_48_enhanced_schema import (
    EnhancedNodeProperties,
    EnhancedRelationProperties,
    EnhancedSchemaManager,
)

# ==================== FIXTURES ====================


@pytest.fixture
def mock_netzwerk():
    """Mock KonzeptNetzwerk für Tests."""
    netzwerk = Mock()
    netzwerk.driver = Mock()
    return netzwerk


@pytest.fixture
def schema_manager(mock_netzwerk):
    """EnhancedSchemaManager mit Mock-Netzwerk."""
    return EnhancedSchemaManager(mock_netzwerk)


# ==================== NODE PROPERTIES TESTS ====================


def test_node_properties_defaults():
    """Test: Node Properties haben korrekte Defaults."""
    defaults = EnhancedNodeProperties.DEFAULTS

    assert defaults["pos"] is None
    assert defaults["definitions"] == []
    assert defaults["semantic_field"] is None
    assert defaults["abstraction_level"] == 3
    assert defaults["contexts"] == []
    assert defaults["typical_relations"] == {}
    assert defaults["usage_frequency"] == 0
    assert defaults["first_seen"] is None
    assert defaults["last_used"] is None


def test_validate_abstraction_level():
    """Test: Abstraction Level Validierung."""
    # Valid levels
    assert EnhancedNodeProperties.validate_abstraction_level(1) is True
    assert EnhancedNodeProperties.validate_abstraction_level(3) is True
    assert EnhancedNodeProperties.validate_abstraction_level(5) is True

    # Invalid levels
    assert EnhancedNodeProperties.validate_abstraction_level(0) is False
    assert EnhancedNodeProperties.validate_abstraction_level(6) is False
    assert EnhancedNodeProperties.validate_abstraction_level(-1) is False


def test_infer_abstraction_level_from_pos():
    """Test: Abstraction Level Inferenz aus POS Tags."""
    # Eigennamen = konkret
    assert EnhancedNodeProperties.infer_abstraction_level("berlin", "PROPN") == 1

    # Nomen = eher konkret
    assert EnhancedNodeProperties.infer_abstraction_level("hund", "NOUN") == 2

    # Verben = neutral
    assert EnhancedNodeProperties.infer_abstraction_level("laufen", "VERB") == 3

    # Adjektive = abstrakt
    assert EnhancedNodeProperties.infer_abstraction_level("schön", "ADJ") == 4


def test_infer_abstraction_level_from_keywords():
    """Test: Abstraction Level Inferenz aus Keywords."""
    # Abstrakte Konzepte
    assert EnhancedNodeProperties.infer_abstraction_level("freiheit") == 5
    assert EnhancedNodeProperties.infer_abstraction_level("gerechtigkeit") == 5
    assert EnhancedNodeProperties.infer_abstraction_level("wahrheit") == 5

    # Normale Wörter ohne POS
    assert EnhancedNodeProperties.infer_abstraction_level("apfel") == 3


# ==================== RELATION PROPERTIES TESTS ====================


def test_relation_properties_defaults():
    """Test: Relation Properties haben korrekte Defaults."""
    defaults = EnhancedRelationProperties.DEFAULTS

    assert defaults["confidence"] == 0.85
    assert defaults["source_text"] is None
    assert defaults["asserted_at"] is None
    assert defaults["timestamp"] is None
    assert defaults["context"] == []
    assert defaults["bidirectional"] is False
    assert defaults["inference_rule"] is None
    assert defaults["usage_count"] == 0
    assert defaults["last_reinforced"] is None


def test_is_bidirectional():
    """Test: Bidirectional Relations werden erkannt."""
    # Symmetrische Relations
    assert EnhancedRelationProperties.is_bidirectional("SYNONYM_OF") is True
    assert EnhancedRelationProperties.is_bidirectional("SIMILAR_TO") is True
    assert EnhancedRelationProperties.is_bidirectional("RELATED_TO") is True
    assert EnhancedRelationProperties.is_bidirectional("EQUIVALENT_TO") is True
    assert EnhancedRelationProperties.is_bidirectional("OPPOSITE_OF") is True

    # Asymmetrische Relations
    assert EnhancedRelationProperties.is_bidirectional("IS_A") is False
    assert EnhancedRelationProperties.is_bidirectional("HAS_PROPERTY") is False
    assert EnhancedRelationProperties.is_bidirectional("PART_OF") is False


def test_bidirectional_case_insensitive():
    """Test: Bidirectional Check ist case-insensitive."""
    assert EnhancedRelationProperties.is_bidirectional("synonym_of") is True
    assert EnhancedRelationProperties.is_bidirectional("Synonym_Of") is True
    assert EnhancedRelationProperties.is_bidirectional("SYNONYM_OF") is True


# ==================== SCHEMA MANAGER NODE TESTS ====================


def test_init_node_properties_success(schema_manager, mock_netzwerk):
    """Test: Node Properties werden erfolgreich initialisiert."""
    # Mock DB Session
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.init_node_properties(
        lemma="hund",
        pos="NOUN",
        semantic_field="Natur",
    )

    assert success is True
    mock_session.run.assert_called_once()

    # Prüfe Query-Parameter
    call_args = mock_session.run.call_args
    assert "hund" in str(call_args)
    assert "NOUN" in str(call_args) or call_args[1]["pos"] == "NOUN"


def test_init_node_properties_no_driver(schema_manager):
    """Test: Init schlägt fehl wenn kein Driver."""
    schema_manager.netzwerk.driver = None

    success = schema_manager.init_node_properties("test")

    assert success is False


def test_add_definition(schema_manager, mock_netzwerk):
    """Test: Definition wird hinzugefügt."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.add_definition("hund", "Ein Hund ist ein Säugetier")

    assert success is True
    mock_session.run.assert_called_once()


def test_add_context(schema_manager, mock_netzwerk):
    """Test: Kontext wird hinzugefügt."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.add_context("hund", "Im Tierreich")

    assert success is True
    mock_session.run.assert_called_once()


def test_update_usage(schema_manager, mock_netzwerk):
    """Test: Usage wird aktualisiert."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.update_usage("hund")

    assert success is True
    mock_session.run.assert_called_once()

    # Prüfe dass usage_frequency inkrementiert wird
    call_args = mock_session.run.call_args
    query = call_args[0][0]
    assert "usage_frequency" in query
    assert "COALESCE" in query
    assert "+ 1" in query


def test_update_typical_relations(schema_manager, mock_netzwerk):
    """Test: Typical Relations werden aggregiert."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    # Mock Query Results
    mock_result = MagicMock()
    mock_result.__iter__.return_value = [
        {"relation_type": "IS_A", "count": 5},
        {"relation_type": "HAS_PROPERTY", "count": 3},
    ]
    mock_session.run.return_value = mock_result

    success = schema_manager.update_typical_relations("hund")

    assert success is True
    # Zwei Queries: Zählen + Update
    assert mock_session.run.call_count == 2


# ==================== SCHEMA MANAGER RELATION TESTS ====================


def test_init_relation_properties_success(schema_manager, mock_netzwerk):
    """Test: Relation Properties werden erfolgreich initialisiert."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.init_relation_properties(
        subject="hund",
        relation_type="IS_A",
        object_="säugetier",
        source_text="Ein Hund ist ein Säugetier",
        inference_rule="definition_pattern",
    )

    assert success is True
    mock_session.run.assert_called_once()


def test_init_relation_properties_bidirectional(schema_manager, mock_netzwerk):
    """Test: Bidirectional Property wird korrekt gesetzt."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    # Symmetrische Relation
    schema_manager.init_relation_properties(
        subject="hund",
        relation_type="SYNONYM_OF",
        object_="canine",
    )

    call_args = mock_session.run.call_args
    assert call_args[1]["bidirectional"] is True

    # Asymmetrische Relation
    mock_session.reset_mock()
    schema_manager.init_relation_properties(
        subject="hund",
        relation_type="IS_A",
        object_="säugetier",
    )

    call_args = mock_session.run.call_args
    assert call_args[1]["bidirectional"] is False


def test_add_relation_context(schema_manager, mock_netzwerk):
    """Test: Kontext zu Relation hinzufügen."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.add_relation_context(
        subject="hund",
        relation_type="IS_A",
        object_="säugetier",
        context="Taxonomie",
    )

    assert success is True
    mock_session.run.assert_called_once()


def test_reinforce_relation(schema_manager, mock_netzwerk):
    """Test: Relation wird reinforced."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.reinforce_relation(
        subject="hund",
        relation_type="IS_A",
        object_="säugetier",
    )

    assert success is True
    mock_session.run.assert_called_once()

    # Prüfe dass usage_count inkrementiert wird
    call_args = mock_session.run.call_args
    query = call_args[0][0]
    assert "usage_count" in query
    assert "+ 1" in query
    assert "last_reinforced" in query


# ==================== ERROR HANDLING TESTS ====================


def test_init_node_properties_error_handling(schema_manager, mock_netzwerk):
    """Test: Fehlerbehandlung bei Node Init."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.side_effect = Exception("DB Error")
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.init_node_properties("test")

    assert success is False


def test_add_definition_error_handling(schema_manager, mock_netzwerk):
    """Test: Fehlerbehandlung bei add_definition."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.side_effect = Exception("DB Error")
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.add_definition("test", "definition")

    assert success is False


def test_update_typical_relations_error_handling(schema_manager, mock_netzwerk):
    """Test: Fehlerbehandlung bei typical_relations update."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.side_effect = Exception("DB Error")
    mock_netzwerk.driver.session.return_value = mock_session

    success = schema_manager.update_typical_relations("test")

    assert success is False


# ==================== INTEGRATION TESTS (WITH REAL DB) ====================


@pytest.mark.integration
def test_full_node_lifecycle():
    """
    Integration Test: Vollständiger Node Lifecycle.

    Requires: Running Neo4j instance
    """
    pytest.skip("Integration test - requires Neo4j")

    from component_1_netzwerk import KonzeptNetzwerk

    netzwerk = KonzeptNetzwerk()
    manager = EnhancedSchemaManager(netzwerk)

    # 1. Init Node
    success = manager.init_node_properties(
        lemma="test_hund",
        pos="NOUN",
        semantic_field="Natur",
    )
    assert success is True

    # 2. Add Definition
    success = manager.add_definition(
        "test_hund", "Ein Hund ist ein domestiziertes Säugetier"
    )
    assert success is True

    # 3. Add Context
    success = manager.add_context("test_hund", "Haustiere")
    assert success is True

    # 4. Update Usage
    success = manager.update_usage("test_hund")
    assert success is True

    # 5. Verify in DB
    with netzwerk.driver.session(database="neo4j") as session:
        result = session.run(
            """
            MATCH (k:Konzept {name: 'test_hund'})
            RETURN k.pos AS pos,
                   k.semantic_field AS semantic_field,
                   k.definitions AS definitions,
                   k.contexts AS contexts,
                   k.usage_frequency AS usage_frequency
            """
        )
        record = result.single()

        assert record is not None
        assert record["pos"] == "NOUN"
        assert record["semantic_field"] == "Natur"
        assert len(record["definitions"]) > 0
        assert len(record["contexts"]) > 0
        assert record["usage_frequency"] > 0

    # Cleanup
    with netzwerk.driver.session(database="neo4j") as session:
        session.run("MATCH (k:Konzept {name: 'test_hund'}) DETACH DELETE k")


@pytest.mark.integration
def test_full_relation_lifecycle():
    """
    Integration Test: Vollständiger Relation Lifecycle.

    Requires: Running Neo4j instance
    """
    pytest.skip("Integration test - requires Neo4j")

    from component_1_netzwerk import KonzeptNetzwerk

    netzwerk = KonzeptNetzwerk()
    manager = EnhancedSchemaManager(netzwerk)

    # Setup Nodes
    netzwerk.assert_relation("test_pudel", "IS_A", "test_hund")

    # 1. Init Relation Props
    success = manager.init_relation_properties(
        subject="test_pudel",
        relation_type="IS_A",
        object_="test_hund",
        source_text="Ein Pudel ist ein Hund",
        inference_rule="definition_pattern",
    )
    assert success is True

    # 2. Add Context
    success = manager.add_relation_context(
        subject="test_pudel",
        relation_type="IS_A",
        object_="test_hund",
        context="Taxonomie",
    )
    assert success is True

    # 3. Reinforce
    success = manager.reinforce_relation(
        subject="test_pudel",
        relation_type="IS_A",
        object_="test_hund",
    )
    assert success is True

    # 4. Verify
    with netzwerk.driver.session(database="neo4j") as session:
        result = session.run(
            """
            MATCH (s:Konzept {name: 'test_pudel'})-[r:IS_A]->(o:Konzept {name: 'test_hund'})
            RETURN r.context AS context,
                   r.bidirectional AS bidirectional,
                   r.inference_rule AS inference_rule,
                   r.usage_count AS usage_count
            """
        )
        record = result.single()

        assert record is not None
        assert len(record["context"]) > 0
        assert record["bidirectional"] is False
        assert record["inference_rule"] == "definition_pattern"
        assert record["usage_count"] > 0

    # Cleanup
    with netzwerk.driver.session(database="neo4j") as session:
        session.run(
            """
            MATCH (k:Konzept)
            WHERE k.name IN ['test_pudel', 'test_hund']
            DETACH DELETE k
            """
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
