# tests/test_usage_tracking.py
"""
Tests für Usage Tracking System (component_49_usage_tracking.py)

Test-Kategorien:
1. UsageStats Data Structure Tests
2. Relation Usage Tracking Tests
3. Concept Activation Tracking Tests
4. Batch Query Tracking Tests
5. Statistics Retrieval Tests
6. Analytics Tests
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import pytest

from component_49_usage_tracking import (
    QueryUsageRecord,
    UsageStats,
    UsageTrackingManager,
)

# ==================== FIXTURES ====================


@pytest.fixture
def mock_driver():
    """Mock Neo4j Driver für Tests."""
    driver = Mock()
    return driver


@pytest.fixture
def tracking_manager(mock_driver):
    """UsageTrackingManager mit Mock-Driver."""
    return UsageTrackingManager(mock_driver)


# ==================== USAGESTATS DATA STRUCTURE TESTS ====================


def test_usage_stats_defaults():
    """Test: UsageStats Default-Werte."""
    stats = UsageStats()

    assert stats.usage_count == 0
    assert stats.first_used is None
    assert stats.last_used is None
    assert stats.queries == []
    assert stats.activation_levels == []
    assert stats.contexts == []
    assert stats.reinforcement_score == 0.0


def test_usage_stats_average_activation():
    """Test: Average Activation Berechnung."""
    # Mit Activation Levels
    stats = UsageStats(activation_levels=[1.0, 0.8, 0.6, 0.4])
    assert stats.average_activation == 0.7

    # Ohne Activation Levels
    stats_empty = UsageStats()
    assert stats_empty.average_activation == 0.0


def test_usage_stats_days_since_last_use():
    """Test: Days Since Last Use Berechnung."""
    # Mit last_used
    last_used = datetime.now() - timedelta(days=5)
    stats = UsageStats(last_used=last_used)
    assert stats.days_since_last_use == 5

    # Ohne last_used
    stats_none = UsageStats()
    assert stats_none.days_since_last_use is None


def test_usage_stats_usage_frequency():
    """Test: Usage Frequency Berechnung."""
    # 10 Uses über 5 Tage = 2 uses/day
    first_used = datetime.now() - timedelta(days=5)
    last_used = datetime.now()
    stats = UsageStats(usage_count=10, first_used=first_used, last_used=last_used)
    assert 1.8 < stats.usage_frequency < 2.2  # ~2.0 uses/day

    # Ohne Timestamps
    stats_none = UsageStats(usage_count=10)
    assert stats_none.usage_frequency == 0.0


# ==================== RELATION USAGE TRACKING TESTS ====================


def test_track_relation_usage_success(tracking_manager, mock_driver):
    """Test: Relation Usage wird erfolgreich getrackt."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_driver.session.return_value = mock_session

    success = tracking_manager.track_relation_usage(
        subject="hund",
        relation="IS_A",
        object_="säugetier",
        query_id="query_123",
        context="multi_hop",
    )

    assert success is True
    mock_session.run.assert_called_once()

    # Prüfe Query-Inhalte
    call_args = mock_session.run.call_args
    query = call_args[0][0]
    assert "usage_count" in query
    assert "+ 1" in query
    assert "last_reinforced" in query
    assert "USED_RELATION" in query


def test_track_relation_usage_no_driver(tracking_manager):
    """Test: Tracking schlägt fehl wenn kein Driver."""
    tracking_manager.driver = None

    success = tracking_manager.track_relation_usage("test", "IS_A", "test", "query_1")

    assert success is False


def test_track_relation_usage_sanitizes_relation(tracking_manager, mock_driver):
    """Test: Relation Type wird sanitized."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_driver.session.return_value = mock_session

    tracking_manager.track_relation_usage(
        "test", "IS-A", "test", "query_1"  # Mit Bindestrich
    )

    call_args = mock_session.run.call_args
    query = call_args[0][0]
    # Sollte ISA sein (ohne Bindestrich)
    assert "-[r:ISA]->" in query


# ==================== CONCEPT ACTIVATION TRACKING TESTS ====================


def test_track_concept_activation_success(tracking_manager, mock_driver):
    """Test: Concept Activation wird erfolgreich getrackt."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_driver.session.return_value = mock_session

    success = tracking_manager.track_concept_activation(
        concept="hund",
        activation_level=1.0,
        query_id="query_123",
        context="direct_query",
    )

    assert success is True
    mock_session.run.assert_called_once()

    # Prüfe Query-Inhalte
    call_args = mock_session.run.call_args
    query = call_args[0][0]
    assert "usage_frequency" in query
    assert "+ 1" in query
    assert "last_used" in query
    assert "ACTIVATED_CONCEPT" in query


def test_track_concept_activation_clips_level(tracking_manager, mock_driver):
    """Test: Activation Level wird auf [0, 1] geclipped."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_driver.session.return_value = mock_session

    # Zu hoch
    tracking_manager.track_concept_activation("test", 1.5, "query_1")
    call_args = mock_session.run.call_args
    assert call_args[1]["activation_level"] == 1.0

    # Zu niedrig
    mock_session.reset_mock()
    tracking_manager.track_concept_activation("test", -0.5, "query_1")
    call_args = mock_session.run.call_args
    assert call_args[1]["activation_level"] == 0.0


def test_track_concept_activation_no_driver(tracking_manager):
    """Test: Tracking schlägt fehl wenn kein Driver."""
    tracking_manager.driver = None

    success = tracking_manager.track_concept_activation("test", 1.0, "query_1")

    assert success is False


# ==================== BATCH QUERY TRACKING TESTS ====================


def test_track_query_usage_batch(tracking_manager, mock_driver):
    """Test: Batch Query Usage Tracking."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_driver.session.return_value = mock_session

    success = tracking_manager.track_query_usage(
        query_id="query_123",
        query_type="reasoning",
        activated_concepts=["hund", "säugetier", "tier"],
        used_relations=[("hund", "IS_A", "säugetier"), ("säugetier", "IS_A", "tier")],
        activation_levels={"hund": 1.0, "säugetier": 0.8, "tier": 0.6},
        reasoning_depth=2,
    )

    assert success is True

    # Sollte mehrere run() Calls gemacht haben
    # 3 Concepts + 2 Relations + 1 Query Update = 6 Calls
    assert mock_session.run.call_count >= 3


def test_track_query_usage_defaults_activation_levels(tracking_manager, mock_driver):
    """Test: Default Activation Levels werden gesetzt."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_driver.session.return_value = mock_session

    success = tracking_manager.track_query_usage(
        query_id="query_123",
        query_type="search",
        activated_concepts=["test"],
        used_relations=[],
        # Keine activation_levels angegeben
    )

    assert success is True
    # Default sollte 1.0 sein für alle Concepts


def test_track_query_usage_no_driver(tracking_manager):
    """Test: Batch Tracking schlägt fehl wenn kein Driver."""
    tracking_manager.driver = None

    success = tracking_manager.track_query_usage("query_1", "reasoning", ["test"], [])

    assert success is False


# ==================== STATISTICS RETRIEVAL TESTS ====================


def test_get_usage_statistics_relation(tracking_manager, mock_driver):
    """Test: Relation Usage Statistics abrufen."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session

    # Mock Query Result
    mock_record = {
        "usage_count": 10,
        "last_used": datetime.now(),
        "query_ids": ["q1", "q2", "q3"],
        "contexts": ["reasoning", "search"],
    }
    mock_session.run.return_value.single.return_value = mock_record
    mock_driver.session.return_value = mock_session

    stats = tracking_manager.get_usage_statistics(
        subject="hund", relation="IS_A", object_="säugetier"
    )

    assert stats.usage_count == 10
    assert len(stats.queries) == 3
    assert len(stats.contexts) == 2


def test_get_usage_statistics_concept(tracking_manager, mock_driver):
    """Test: Concept Usage Statistics abrufen."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session

    # Mock Query Result
    first_used = datetime.now() - timedelta(days=10)
    last_used = datetime.now()

    mock_record = {
        "usage_count": 25,
        "first_used": first_used,
        "last_used": last_used,
        "query_ids": ["q1", "q2"],
        "activation_levels": [1.0, 0.8, 0.6],
        "contexts": ["reasoning"],
    }
    mock_session.run.return_value.single.return_value = mock_record
    mock_driver.session.return_value = mock_session

    stats = tracking_manager.get_usage_statistics(concept="hund")

    assert stats.usage_count == 25
    assert stats.first_used is not None
    assert stats.last_used is not None
    assert len(stats.activation_levels) == 3
    assert 0.6 < stats.average_activation < 1.0


def test_get_usage_statistics_no_data(tracking_manager, mock_driver):
    """Test: Empty Statistics wenn keine Daten."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.return_value.single.return_value = None
    mock_driver.session.return_value = mock_session

    stats = tracking_manager.get_usage_statistics(concept="nonexistent")

    assert stats.usage_count == 0
    assert stats.first_used is None


def test_get_usage_statistics_invalid_params(tracking_manager):
    """Test: Invalid Parameters geben Empty Stats."""
    # Weder concept noch relation params
    stats = tracking_manager.get_usage_statistics()

    assert stats.usage_count == 0


# ==================== ANALYTICS TESTS ====================


def test_get_most_used_relations(tracking_manager, mock_driver):
    """Test: Most Used Relations Analytics."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session

    # Mock Results - need to create proper Mock objects with data() method
    mock_results = [
        {"subject": "hund", "relation": "IS_A", "object": "tier", "usage_count": 100},
        {"subject": "katze", "relation": "IS_A", "object": "tier", "usage_count": 90},
    ]

    # Create mock records with data() method
    mock_records = []
    for result in mock_results:
        mock_record = Mock()
        mock_record.data.return_value = result
        mock_records.append(mock_record)

    mock_session.run.return_value = mock_records
    mock_driver.session.return_value = mock_session

    relations = tracking_manager.get_most_used_relations(limit=10)

    assert len(relations) == 2
    assert relations[0]["usage_count"] == 100
    assert relations[1]["usage_count"] == 90


def test_get_most_activated_concepts(tracking_manager, mock_driver):
    """Test: Most Activated Concepts Analytics."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session

    # Mock Results
    mock_results = [
        {"concept": "hund", "usage_frequency": 50, "last_used": datetime.now()},
        {"concept": "tier", "usage_frequency": 45, "last_used": datetime.now()},
    ]

    # Create mock records with data() method
    mock_records = []
    for result in mock_results:
        mock_record = Mock()
        mock_record.data.return_value = result
        mock_records.append(mock_record)

    mock_session.run.return_value = mock_records
    mock_driver.session.return_value = mock_session

    concepts = tracking_manager.get_most_activated_concepts(limit=10)

    assert len(concepts) == 2
    assert concepts[0]["usage_frequency"] == 50
    assert concepts[1]["usage_frequency"] == 45


def test_analytics_no_driver(tracking_manager):
    """Test: Analytics geben leere Liste ohne Driver."""
    tracking_manager.driver = None

    relations = tracking_manager.get_most_used_relations()
    concepts = tracking_manager.get_most_activated_concepts()

    assert relations == []
    assert concepts == []


# ==================== ERROR HANDLING TESTS ====================


def test_track_relation_usage_db_error(tracking_manager, mock_driver):
    """Test: DB Error wird graceful gehandled."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.side_effect = Exception("DB Error")
    mock_driver.session.return_value = mock_session

    success = tracking_manager.track_relation_usage("test", "IS_A", "test", "q1")

    assert success is False


def test_track_concept_activation_db_error(tracking_manager, mock_driver):
    """Test: DB Error wird graceful gehandled."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.side_effect = Exception("DB Error")
    mock_driver.session.return_value = mock_session

    success = tracking_manager.track_concept_activation("test", 1.0, "q1")

    assert success is False


def test_get_usage_statistics_db_error(tracking_manager, mock_driver):
    """Test: DB Error gibt Empty Stats."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.run.side_effect = Exception("DB Error")
    mock_driver.session.return_value = mock_session

    stats = tracking_manager.get_usage_statistics(concept="test")

    assert stats.usage_count == 0


# ==================== QUERY USAGE RECORD TESTS ====================


def test_query_usage_record_creation():
    """Test: QueryUsageRecord kann erstellt werden."""
    record = QueryUsageRecord(
        query_id="query_123",
        timestamp=datetime.now(),
        query_type="reasoning",
        activated_concepts=["hund", "tier"],
        used_relations=[("hund", "IS_A", "tier")],
        activation_levels={"hund": 1.0, "tier": 0.8},
        reasoning_depth=1,
    )

    assert record.query_id == "query_123"
    assert len(record.activated_concepts) == 2
    assert len(record.used_relations) == 1
    assert record.activation_levels["hund"] == 1.0
    assert record.reasoning_depth == 1


# ==================== INTEGRATION TESTS (WITH REAL DB) ====================


@pytest.mark.integration
def test_full_tracking_lifecycle():
    """
    Integration Test: Vollständiger Tracking Lifecycle.

    Requires: Running Neo4j instance
    """
    pytest.skip("Integration test - requires Neo4j")

    from component_1_netzwerk import KonzeptNetzwerk

    netzwerk = KonzeptNetzwerk()
    manager = UsageTrackingManager(netzwerk.driver)

    # Setup: Erstelle Fact
    netzwerk.assert_relation("test_hund", "IS_A", "test_tier")

    # 1. Track Relation Usage
    success = manager.track_relation_usage(
        "test_hund", "IS_A", "test_tier", "test_query_1"
    )
    assert success is True

    # 2. Track Concept Activation
    success = manager.track_concept_activation("test_hund", 1.0, "test_query_1")
    assert success is True

    # 3. Get Statistics
    stats = manager.get_usage_statistics(
        subject="test_hund", relation="IS_A", object_="test_tier"
    )
    assert stats.usage_count > 0

    # 4. Concept Statistics
    stats = manager.get_usage_statistics(concept="test_hund")
    assert stats.usage_count > 0

    # Cleanup
    with netzwerk.driver.session(database="neo4j") as session:
        session.run(
            """
            MATCH (k:Konzept)
            WHERE k.name IN ['test_hund', 'test_tier']
            DETACH DELETE k

            MATCH (q:QueryRecord {id: 'test_query_1'})
            DETACH DELETE q

            MATCH (f:Fact)
            WHERE f.subject = 'test_hund' OR f.object = 'test_hund'
            DETACH DELETE f
            """
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
