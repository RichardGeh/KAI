# tests/test_production_rule_repository.py
"""
Tests für Neo4j Production Rule Repository (PHASE 9).

Testet:
- CRUD Operationen für Production Rules
- Statistik-Updates (mit Batching)
- Introspektions-Queries (Filterung, Sortierung)
- Integration mit ProductionSystemEngine
"""

from datetime import datetime

import pytest

from component_1_netzwerk import KonzeptNetzwerk


@pytest.fixture
def netzwerk():
    """Erstelle eine Neo4j-Verbindung für Tests."""
    netz = KonzeptNetzwerk()
    yield netz
    netz.close()


class TestProductionRuleCRUD:
    """Tests für grundlegende CRUD Operationen."""

    def test_create_production_rule(self, netzwerk):
        """Test: Erstelle eine neue Production Rule."""
        success = netzwerk.create_production_rule(
            name="test_rule_create",
            category="content_selection",
            condition_code="ABCD1234",  # Hex-encoded pickle
            action_code="EFGH5678",
            utility=0.8,
            specificity=0.9,
            metadata={"description": "Test rule for creation", "tags": ["test"]},
        )

        assert success, "Rule creation should succeed"

        # Verifiziere durch Laden
        rule = netzwerk.get_production_rule("test_rule_create")
        assert rule is not None
        assert rule["name"] == "test_rule_create"
        assert rule["category"] == "content_selection"
        assert rule["utility"] == 0.8
        assert rule["specificity"] == 0.9
        assert rule["application_count"] == 0
        assert rule["success_count"] == 0

    def test_get_production_rule_not_found(self, netzwerk):
        """Test: Lade nicht-existierende Regel."""
        rule = netzwerk.get_production_rule("nonexistent_rule")
        assert rule is None

    def test_update_production_rule(self, netzwerk):
        """Test: Aktualisiere existierende Regel (ON MATCH)."""
        # Erstelle initial
        netzwerk.create_production_rule(
            name="test_rule_update",
            category="lexicalization",
            condition_code="AAA",
            action_code="BBB",
            utility=0.5,
        )

        # Update mit neuen Werten
        success = netzwerk.create_production_rule(
            name="test_rule_update",
            category="discourse",  # Geändert
            condition_code="CCC",  # Geändert
            action_code="DDD",  # Geändert
            utility=0.9,  # Geändert
        )

        assert success
        rule = netzwerk.get_production_rule("test_rule_update")
        assert rule["category"] == "discourse"
        assert rule["utility"] == 0.9


class TestProductionRuleStatistics:
    """Tests für Statistik-Updates."""

    def test_update_stats_immediate(self, netzwerk):
        """Test: Sofortiges Stats-Update (force_sync=True)."""
        # Erstelle Regel
        netzwerk.create_production_rule(
            name="test_rule_stats",
            category="content_selection",
            condition_code="X",
            action_code="Y",
        )

        # Update Stats
        success = netzwerk.update_production_rule_stats(
            name="test_rule_stats",
            application_count=5,
            success_count=3,
            last_applied=datetime.now(),
            force_sync=True,
        )

        assert success
        rule = netzwerk.get_production_rule("test_rule_stats")
        assert rule["application_count"] == 5
        assert rule["success_count"] == 3
        assert rule["last_applied"] is not None

    def test_update_stats_batched(self, netzwerk):
        """Test: Gebatchte Stats-Updates."""
        # Erstelle Regel
        netzwerk.create_production_rule(
            name="test_rule_batch",
            category="content_selection",
            condition_code="A",
            action_code="B",
        )

        # Update mehrfach (gebatched, force_sync=False)
        for i in range(5):
            netzwerk.update_production_rule_stats(
                name="test_rule_batch",
                application_count=i + 1,
                force_sync=False,  # Batching
            )

        # Noch nicht synchronisiert (gebatched)
        # Flush manuell
        netzwerk._production_rules._flush_pending_stats()

        rule = netzwerk.get_production_rule("test_rule_batch")
        assert rule["application_count"] == 5  # Letzter Wert


class TestProductionRuleQueries:
    """Tests für Introspektions-Queries."""

    @pytest.fixture(autouse=True)
    def setup_test_rules(self, netzwerk):
        """Erstelle Test-Regeln für Query-Tests."""
        test_rules = [
            {
                "name": "content_rule_1",
                "category": "content_selection",
                "utility": 0.8,
                "specificity": 0.9,
            },
            {
                "name": "content_rule_2",
                "category": "content_selection",
                "utility": 0.6,
                "specificity": 0.7,
            },
            {
                "name": "lex_rule_1",
                "category": "lexicalization",
                "utility": 0.9,
                "specificity": 0.8,
            },
            {
                "name": "discourse_rule_1",
                "category": "discourse",
                "utility": 0.3,
                "specificity": 0.5,
            },
        ]

        for rule_data in test_rules:
            netzwerk.create_production_rule(
                name=rule_data["name"],
                category=rule_data["category"],
                condition_code="TEST",
                action_code="TEST",
                utility=rule_data["utility"],
                specificity=rule_data["specificity"],
            )

            # Setze application_count für einige Regeln
            if rule_data["name"] in ["content_rule_1", "lex_rule_1"]:
                netzwerk.update_production_rule_stats(
                    name=rule_data["name"],
                    application_count=(
                        10 if rule_data["name"] == "content_rule_1" else 5
                    ),
                    force_sync=True,
                )

    def test_query_by_category(self, netzwerk):
        """Test: Abfrage nach Kategorie."""
        rules = netzwerk.query_production_rules(category="content_selection")
        assert len(rules) >= 2
        assert all(r["category"] == "content_selection" for r in rules)

    def test_query_by_utility_range(self, netzwerk):
        """Test: Abfrage mit Utility-Filter."""
        rules = netzwerk.query_production_rules(min_utility=0.7)
        assert len(rules) >= 2
        assert all(r["utility"] >= 0.7 for r in rules)

        rules_low = netzwerk.query_production_rules(max_utility=0.5)
        assert len(rules_low) >= 1
        assert all(r["utility"] <= 0.5 for r in rules_low)

    def test_query_order_by_priority(self, netzwerk):
        """Test: Sortierung nach Priorität (utility * specificity)."""
        rules = netzwerk.query_production_rules(order_by="priority")

        # Verifiziere Sortierung (absteigend)
        priorities = [r["utility"] * r["specificity"] for r in rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_query_order_by_usage(self, netzwerk):
        """Test: Sortierung nach application_count."""
        rules = netzwerk.query_production_rules(
            min_application_count=1, order_by="usage"
        )

        # Verifiziere Sortierung (absteigend)
        app_counts = [r["application_count"] for r in rules]
        assert app_counts == sorted(app_counts, reverse=True)

    def test_query_with_limit(self, netzwerk):
        """Test: Limitierung der Ergebnisse."""
        rules = netzwerk.query_production_rules(limit=2)
        assert len(rules) <= 2


class TestProductionRuleStatisticsAggregate:
    """Tests für aggregierte Statistiken."""

    @pytest.fixture(autouse=True)
    def setup_test_rules(self, netzwerk):
        """Erstelle Test-Regeln mit verschiedenen Stats."""
        rules = [
            {
                "name": "high_util",
                "category": "content_selection",
                "utility": 0.9,
                "app_count": 20,
            },
            {
                "name": "mid_util",
                "category": "lexicalization",
                "utility": 0.5,
                "app_count": 10,
            },
            {
                "name": "low_util",
                "category": "discourse",
                "utility": 0.2,
                "app_count": 2,
            },
        ]

        for rule_data in rules:
            netzwerk.create_production_rule(
                name=rule_data["name"],
                category=rule_data["category"],
                condition_code="T",
                action_code="T",
                utility=rule_data["utility"],
            )
            netzwerk.update_production_rule_stats(
                name=rule_data["name"],
                application_count=rule_data["app_count"],
                force_sync=True,
            )

    def test_get_statistics(self, netzwerk):
        """Test: Aggregierte Statistiken."""
        stats = netzwerk.get_production_rule_statistics()

        assert "total_rules" in stats
        assert stats["total_rules"] >= 3

        assert "by_category" in stats
        assert len(stats["by_category"]) >= 3

        assert "most_used" in stats
        assert len(stats["most_used"]) > 0
        # Verifiziere, dass meistverwendete Regel zuerst kommt
        if len(stats["most_used"]) > 1:
            assert stats["most_used"][0]["count"] >= stats["most_used"][1]["count"]

        assert "low_utility" in stats


class TestProductionSystemEngineIntegration:
    """Tests für Integration mit ProductionSystemEngine."""

    def test_engine_load_from_neo4j(self, netzwerk):
        """Test: Engine lädt Stats aus Neo4j beim Start."""
        from component_54_production_system import (
            ProductionRule,
            ProductionSystemEngine,
            RuleCategory,
        )

        # Erstelle Regel in Neo4j
        netzwerk.create_production_rule(
            name="test_engine_load",
            category="content_selection",
            condition_code="ABC",
            action_code="DEF",
            utility=0.7,
        )
        netzwerk.update_production_rule_stats(
            name="test_engine_load", application_count=15, force_sync=True
        )

        # Erstelle Engine mit in-memory Regel (gleicher Name)
        engine = ProductionSystemEngine(neo4j_repository=netzwerk._production_rules)

        test_rule = ProductionRule(
            name="test_engine_load",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda state: True,
            action=lambda state: None,
            utility=0.7,
            specificity=1.0,
        )
        engine.add_rule(test_rule)

        # Lade Stats aus Neo4j
        loaded_count = engine.load_rules_from_neo4j()
        assert loaded_count >= 1

        # Verifiziere Sync
        assert test_rule.application_count == 15

    def test_engine_save_to_neo4j(self, netzwerk):
        """Test: Engine speichert Regeln zu Neo4j."""
        from component_54_production_system import (
            ProductionRule,
            ProductionSystemEngine,
            RuleCategory,
        )

        engine = ProductionSystemEngine(neo4j_repository=netzwerk._production_rules)

        test_rule = ProductionRule(
            name="test_engine_save",
            category=RuleCategory.LEXICALIZATION,
            condition=lambda state: True,
            action=lambda state: None,
            utility=0.85,
            specificity=0.95,
        )
        engine.add_rule(test_rule)

        # Speichere zu Neo4j
        saved_count = engine.save_rules_to_neo4j()
        assert saved_count == 1

        # Verifiziere
        rule = netzwerk.get_production_rule("test_engine_save")
        assert rule is not None
        assert rule["category"] == "lexicalization"
        assert rule["utility"] == 0.85

    def test_engine_sync_stats(self, netzwerk):
        """Test: Engine synchronisiert Stats zu Neo4j."""
        from component_54_production_system import (
            ProductionRule,
            ProductionSystemEngine,
            RuleCategory,
        )

        # Erstelle Regel in Neo4j
        netzwerk.create_production_rule(
            name="test_engine_sync",
            category="discourse",
            condition_code="X",
            action_code="Y",
        )

        # Erstelle Engine
        engine = ProductionSystemEngine(neo4j_repository=netzwerk._production_rules)
        test_rule = ProductionRule(
            name="test_engine_sync",
            category=RuleCategory.DISCOURSE,
            condition=lambda state: True,
            action=lambda state: None,
        )
        engine.add_rule(test_rule)

        # Simuliere Regelanwendung
        test_rule.application_count = 7
        test_rule.success_count = 5
        test_rule.last_applied = datetime.now()

        # Sync zu Neo4j
        success = engine.sync_rule_stats_to_neo4j(force=True)
        assert success

        # Verifiziere in Neo4j
        rule = netzwerk.get_production_rule("test_engine_sync")
        assert rule["application_count"] == 7
        assert rule["success_count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
