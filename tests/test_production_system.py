"""
test_production_system.py

Tests für das Produktionssystem (component_54_production_system.py).

Testet:
- ProductionRule Datenstruktur
- ResponseGenerationState und Sub-Strukturen
- ProductionSystemEngine (Recognize-Act Cycle)
- Conflict Resolution (Utility × Specificity)
- Neo4j Integration für Regel-Persistierung
"""

import pytest

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_54_production_system import (
    DiscourseState,
    GenerationGoal,
    GenerationGoalType,
    PartialTextStructure,
    ProductionRule,
    ProductionSystemEngine,
    ResponseGenerationState,
    RuleCategory,
    calculate_specificity,
    create_production_rule,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_goal():
    """Erstellt ein einfaches GenerationGoal."""
    return GenerationGoal(
        goal_type=GenerationGoalType.ANSWER_QUESTION,
        target_entity="apfel",
        completed=False,
    )


@pytest.fixture
def initial_state(simple_goal):
    """Erstellt einen initialen ResponseGenerationState."""
    state = ResponseGenerationState(
        primary_goal=simple_goal,
        available_facts=[
            {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
            {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.85},
        ],
    )
    return state


@pytest.fixture
def production_engine():
    """Erstellt eine ProductionSystemEngine."""
    return ProductionSystemEngine()


@pytest.fixture
def netzwerk_core():
    """Erstellt KonzeptNetzwerkCore für Neo4j-Tests."""
    netz = KonzeptNetzwerkCore()
    yield netz
    # Cleanup: Entferne Test-Regeln
    try:
        with netz.driver.session(database="neo4j") as session:
            session.run(
                "MATCH (pr:ProductionRule) WHERE pr.name STARTS WITH 'test_' DELETE pr"
            )
    except Exception:
        pass
    netz.close()


# ============================================================================
# Tests: ProductionRule Datenstruktur
# ============================================================================


class TestProductionRule:
    """Tests für ProductionRule Klasse."""

    def test_rule_creation(self):
        """Test: ProductionRule kann erstellt werden."""
        rule = ProductionRule(
            name="test_rule",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,
            utility=1.5,
            specificity=2.0,
            metadata={"description": "Test rule"},
        )

        assert rule.name == "test_rule"
        assert rule.category == RuleCategory.CONTENT_SELECTION
        assert rule.utility == 1.5
        assert rule.specificity == 2.0
        assert rule.metadata["description"] == "Test rule"
        assert rule.application_count == 0

    def test_rule_matches_condition_true(self, initial_state):
        """Test: Regel matcht wenn Condition True zurückgibt."""
        rule = ProductionRule(
            name="always_match",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,
        )

        assert rule.matches(initial_state) is True

    def test_rule_matches_condition_false(self, initial_state):
        """Test: Regel matcht nicht wenn Condition False zurückgibt."""
        rule = ProductionRule(
            name="never_match",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: False,
            action=lambda s: None,
        )

        assert rule.matches(initial_state) is False

    def test_rule_matches_with_state_check(self, initial_state):
        """Test: Regel checkt State-Eigenschaften."""
        rule = ProductionRule(
            name="check_entity",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: s.primary_goal.target_entity == "apfel",
            action=lambda s: None,
        )

        assert rule.matches(initial_state) is True

    def test_rule_apply_modifies_state(self, initial_state):
        """Test: Regel modifiziert State beim Anwenden."""

        def action(state):
            state.add_sentence("Ein Apfel ist eine Frucht.")

        rule = ProductionRule(
            name="add_sentence",
            category=RuleCategory.SYNTAX,
            condition=lambda s: True,
            action=action,
        )

        rule.apply(initial_state)

        assert len(initial_state.text.completed_sentences) == 1
        assert "Ein Apfel ist eine Frucht." in initial_state.text.completed_sentences
        assert rule.application_count == 1
        assert rule.last_applied is not None

    def test_rule_get_priority(self):
        """Test: Priority = Utility × Specificity."""
        rule = ProductionRule(
            name="test_priority",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,
            utility=2.0,
            specificity=3.0,
        )

        assert rule.get_priority() == 6.0

    def test_rule_to_dict(self):
        """Test: Rule kann zu Dict konvertiert werden."""
        rule = ProductionRule(
            name="test_rule",
            category=RuleCategory.LEXICALIZATION,
            condition=lambda s: True,
            action=lambda s: None,
            utility=1.5,
            specificity=2.0,
            metadata={"tags": ["test"]},
        )

        rule_dict = rule.to_dict()

        assert rule_dict["name"] == "test_rule"
        assert rule_dict["category"] == "lexicalization"
        assert rule_dict["utility"] == 1.5
        assert rule_dict["specificity"] == 2.0
        assert rule_dict["metadata"]["tags"] == ["test"]


# ============================================================================
# Tests: ResponseGenerationState
# ============================================================================


class TestResponseGenerationState:
    """Tests für ResponseGenerationState."""

    def test_state_creation(self, simple_goal):
        """Test: State kann erstellt werden."""
        state = ResponseGenerationState(primary_goal=simple_goal)

        assert state.primary_goal.target_entity == "apfel"
        assert state.cycle_count == 0
        assert state.max_cycles == 50
        assert len(state.text.completed_sentences) == 0

    def test_state_add_sentence(self, initial_state):
        """Test: Satz hinzufügen."""
        initial_state.add_sentence("Ein Apfel ist eine Frucht.")

        assert len(initial_state.text.completed_sentences) == 1
        assert initial_state.discourse.sentence_count == 1

    def test_state_mention_entity(self, initial_state):
        """Test: Entität als erwähnt markieren."""
        initial_state.mention_entity("apfel")

        assert "apfel" in initial_state.discourse.mentioned_entities

    def test_state_get_full_text(self, initial_state):
        """Test: Vollständigen Text abrufen."""
        initial_state.add_sentence("Ein Apfel ist eine Frucht.")
        initial_state.add_sentence("Äpfel sind rot.")

        full_text = initial_state.get_full_text()

        assert full_text == "Ein Apfel ist eine Frucht. Äpfel sind rot."

    def test_state_is_goal_completed_false(self, initial_state):
        """Test: Goal ist nicht completed."""
        assert initial_state.is_goal_completed() is False

    def test_state_is_goal_completed_true(self, initial_state):
        """Test: Goal ist completed."""
        initial_state.primary_goal.completed = True

        assert initial_state.is_goal_completed() is True


class TestDiscourseState:
    """Tests für DiscourseState."""

    def test_discourse_initial_state(self):
        """Test: Initialer Discourse State."""
        discourse = DiscourseState()

        assert discourse.current_focus is None
        assert len(discourse.mentioned_entities) == 0
        assert discourse.sentence_count == 0

    def test_discourse_focus_tracking(self):
        """Test: Focus-Tracking."""
        discourse = DiscourseState(current_focus="apfel")

        assert discourse.current_focus == "apfel"


class TestPartialTextStructure:
    """Tests für PartialTextStructure."""

    def test_partial_text_initial(self):
        """Test: Initiale Text-Struktur."""
        text = PartialTextStructure()

        assert len(text.completed_sentences) == 0
        assert len(text.sentence_fragments) == 0
        assert text.current_fragment == ""


# ============================================================================
# Tests: ProductionSystemEngine
# ============================================================================


class TestProductionSystemEngine:
    """Tests für ProductionSystemEngine."""

    def test_engine_creation(self, production_engine):
        """Test: Engine kann erstellt werden."""
        assert len(production_engine.rules) == 0

    def test_add_rule(self, production_engine):
        """Test: Regel hinzufügen."""
        rule = ProductionRule(
            name="test_rule",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,
        )

        production_engine.add_rule(rule)

        assert len(production_engine.rules) == 1
        assert production_engine.rules[0].name == "test_rule"

    def test_add_multiple_rules(self, production_engine):
        """Test: Mehrere Regeln hinzufügen."""
        rules = [
            ProductionRule(
                name=f"rule_{i}",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
            )
            for i in range(5)
        ]

        production_engine.add_rules(rules)

        assert len(production_engine.rules) == 5

    def test_remove_rule(self, production_engine):
        """Test: Regel entfernen."""
        rule = ProductionRule(
            name="removable",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,
        )

        production_engine.add_rule(rule)
        assert len(production_engine.rules) == 1

        removed = production_engine.remove_rule("removable")
        assert removed is True
        assert len(production_engine.rules) == 0

    def test_match_rules_all_matching(self, production_engine, initial_state):
        """Test: Match-Phase findet alle passenden Regeln."""
        rules = [
            ProductionRule(
                name=f"rule_{i}",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,  # Alle matchen
                action=lambda s: None,
            )
            for i in range(3)
        ]

        production_engine.add_rules(rules)
        matching = production_engine.match_rules(initial_state)

        assert len(matching) == 3

    def test_match_rules_partial_matching(self, production_engine, initial_state):
        """Test: Match-Phase findet nur passende Regeln."""
        rules = [
            ProductionRule(
                name="match1",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
            ),
            ProductionRule(
                name="no_match",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: False,
                action=lambda s: None,
            ),
            ProductionRule(
                name="match2",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
            ),
        ]

        production_engine.add_rules(rules)
        matching = production_engine.match_rules(initial_state)

        assert len(matching) == 2
        assert all(r.name in ["match1", "match2"] for r in matching)

    def test_resolve_conflict_by_priority(self, production_engine):
        """Test: Conflict Resolution nach Priority."""
        rules = [
            ProductionRule(
                name="low_priority",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
                utility=1.0,
                specificity=1.0,  # Priority = 1.0
            ),
            ProductionRule(
                name="high_priority",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
                utility=3.0,
                specificity=2.0,  # Priority = 6.0
            ),
        ]

        best = production_engine.resolve_conflict(rules)

        assert best.name == "high_priority"

    def test_apply_rule_increments_cycle(self, production_engine, initial_state):
        """Test: Apply-Phase erhöht Cycle-Count."""
        rule = ProductionRule(
            name="test",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,
        )

        initial_cycle = initial_state.cycle_count
        production_engine.apply_rule(rule, initial_state)

        assert initial_state.cycle_count == initial_cycle + 1

    def test_generate_simple_scenario(self, production_engine, initial_state):
        """Test: Generate-Loop mit einfachem Szenario."""

        # Regel: Wenn noch keine Sätze, füge einen hinzu und markiere Goal als completed
        def action(state):
            if len(state.text.completed_sentences) == 0:
                state.add_sentence("Ein Apfel ist eine Frucht.")
                state.primary_goal.completed = True

        rule = ProductionRule(
            name="generate_sentence",
            category=RuleCategory.SYNTAX,
            condition=lambda s: len(s.text.completed_sentences) == 0,
            action=action,
        )

        production_engine.add_rule(rule)

        final_state = production_engine.generate(initial_state)

        assert final_state.primary_goal.completed is True
        assert len(final_state.text.completed_sentences) == 1
        assert final_state.cycle_count > 0

    def test_generate_max_cycles_limit(self, production_engine, initial_state):
        """Test: Generate stoppt bei Max-Cycles."""
        initial_state.max_cycles = 5

        # Regel die immer matcht aber Goal nie completed
        rule = ProductionRule(
            name="infinite_loop",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,  # Tut nichts
        )

        production_engine.add_rule(rule)

        final_state = production_engine.generate(initial_state)

        assert final_state.cycle_count == 5
        assert final_state.primary_goal.completed is False

    def test_generate_no_matching_rules(self, production_engine, initial_state):
        """Test: Generate stoppt wenn keine Regeln matchen."""
        rule = ProductionRule(
            name="never_match",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: False,
            action=lambda s: None,
        )

        production_engine.add_rule(rule)

        final_state = production_engine.generate(initial_state)

        assert final_state.cycle_count == 0

    def test_get_rules_by_category(self, production_engine):
        """Test: Regeln nach Kategorie filtern."""
        rules = [
            ProductionRule(
                name="content1",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
            ),
            ProductionRule(
                name="syntax1",
                category=RuleCategory.SYNTAX,
                condition=lambda s: True,
                action=lambda s: None,
            ),
            ProductionRule(
                name="content2",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
            ),
        ]

        production_engine.add_rules(rules)

        content_rules = production_engine.get_rules_by_category(
            RuleCategory.CONTENT_SELECTION
        )

        assert len(content_rules) == 2
        assert all(r.category == RuleCategory.CONTENT_SELECTION for r in content_rules)

    def test_get_rule_by_name(self, production_engine):
        """Test: Regel nach Name finden."""
        rule = ProductionRule(
            name="findme",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: True,
            action=lambda s: None,
        )

        production_engine.add_rule(rule)

        found = production_engine.get_rule_by_name("findme")

        assert found is not None
        assert found.name == "findme"

    def test_get_statistics(self, production_engine):
        """Test: Statistiken abrufen."""
        rules = [
            ProductionRule(
                name=f"rule_{i}",
                category=RuleCategory.CONTENT_SELECTION,
                condition=lambda s: True,
                action=lambda s: None,
            )
            for i in range(3)
        ]

        production_engine.add_rules(rules)
        rules[0].application_count = 5

        stats = production_engine.get_statistics()

        assert stats["total_rules"] == 3
        assert stats["total_applications"] == 5
        assert stats["most_used_rule"] == "rule_0"


# ============================================================================
# Tests: Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Tests für Utility Functions."""

    def test_calculate_specificity_simple(self):
        """Test: Spezifität für einfache Condition."""

        def simple_condition(state):
            if state.cycle_count < 10:
                return True
            return False

        specificity = calculate_specificity(simple_condition)

        assert specificity >= 1.0

    def test_create_production_rule_factory(self):
        """Test: Factory-Funktion erstellt Regel."""
        rule = create_production_rule(
            name="factory_rule",
            category=RuleCategory.DISCOURSE,
            condition=lambda s: True,
            action=lambda s: None,
            utility=2.0,
            description="Test rule from factory",
        )

        assert rule.name == "factory_rule"
        assert rule.category == RuleCategory.DISCOURSE
        assert rule.utility == 2.0
        assert rule.metadata["description"] == "Test rule from factory"


# ============================================================================
# Tests: Neo4j Integration
# ============================================================================


class TestNeo4jIntegration:
    """Tests für Neo4j-Persistierung von Production Rules."""

    def test_create_production_rule_in_db(self, netzwerk_core):
        """Test: Produktionsregel in Neo4j erstellen."""
        success = netzwerk_core.create_production_rule(
            name="test_rule_db_1",
            category="content_selection",
            utility=1.5,
            specificity=2.0,
            metadata={"description": "Test rule in DB"},
        )

        assert success is True

    def test_get_production_rules_from_db(self, netzwerk_core):
        """Test: Produktionsregeln aus Neo4j laden."""
        # Erstelle Test-Regeln
        netzwerk_core.create_production_rule(
            name="test_rule_db_2",
            category="lexicalization",
            utility=1.0,
        )

        netzwerk_core.create_production_rule(
            name="test_rule_db_3",
            category="syntax",
            utility=2.0,
        )

        # Lade alle Test-Regeln
        rules = netzwerk_core.get_production_rules()

        test_rules = [r for r in rules if r["name"].startswith("test_rule_db")]

        assert len(test_rules) >= 2

    def test_get_production_rules_filtered_by_category(self, netzwerk_core):
        """Test: Regeln nach Kategorie filtern."""
        netzwerk_core.create_production_rule(
            name="test_rule_db_4",
            category="discourse",
            utility=1.0,
        )

        rules = netzwerk_core.get_production_rules(category="discourse")

        discourse_test_rules = [r for r in rules if r["name"] == "test_rule_db_4"]

        assert len(discourse_test_rules) >= 1
        assert all(r["category"] == "discourse" for r in discourse_test_rules)

    def test_update_rule_stats(self, netzwerk_core):
        """Test: Regel-Statistiken aktualisieren."""
        netzwerk_core.create_production_rule(
            name="test_rule_db_5",
            category="content_selection",
            utility=1.0,
        )

        success = netzwerk_core.update_rule_stats(
            "test_rule_db_5", applied=True, success=True
        )

        assert success is True

        # Lade Regel und prüfe Stats
        rules = netzwerk_core.get_production_rules()
        test_rule = next((r for r in rules if r["name"] == "test_rule_db_5"), None)

        assert test_rule is not None
        assert test_rule["application_count"] >= 1
        assert test_rule["success_count"] >= 1

    def test_get_production_rule_statistics(self, netzwerk_core):
        """Test: Gesamtstatistiken abrufen."""
        # Erstelle mehrere Test-Regeln
        for i in range(3):
            netzwerk_core.create_production_rule(
                name=f"test_rule_db_stats_{i}",
                category="content_selection",
                utility=1.0 + i * 0.5,
            )

        stats = netzwerk_core.get_production_rule_statistics()

        assert stats["total_rules"] >= 3
        assert "categories" in stats
        assert "top_rules" in stats


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integrationstests für vollständige Szenarien."""

    def test_full_generation_pipeline(self):
        """Test: Vollständiger Generierungs-Pipeline."""
        # Setup
        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
                {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.85},
            ],
        )

        engine = ProductionSystemEngine()

        # Regel 1: Wähle IS_A Fakt
        def select_isa_fact(s):
            if len(s.discourse.pending_facts) == 0 and len(s.available_facts) > 0:
                isa_facts = [
                    f for f in s.available_facts if f["relation_type"] == "IS_A"
                ]
                if isa_facts:
                    s.discourse.pending_facts.append(isa_facts[0])

        # Regel 2: Generiere Satz aus Fakt
        def generate_sentence(s):
            if (
                len(s.discourse.pending_facts) > 0
                and len(s.text.completed_sentences) == 0
            ):
                fact = s.discourse.pending_facts.pop(0)
                sentence = (
                    f"Ein {s.primary_goal.target_entity} ist eine {fact['object']}."
                )
                s.add_sentence(sentence)
                s.primary_goal.completed = True

        rule1 = ProductionRule(
            name="select_isa",
            category=RuleCategory.CONTENT_SELECTION,
            condition=lambda s: len(s.discourse.pending_facts) == 0,
            action=select_isa_fact,
            utility=2.0,
        )

        rule2 = ProductionRule(
            name="generate_sentence",
            category=RuleCategory.SYNTAX,
            condition=lambda s: len(s.discourse.pending_facts) > 0,
            action=generate_sentence,
            utility=1.0,
        )

        engine.add_rules([rule1, rule2])

        # Execute
        final_state = engine.generate(state)

        # Assert
        assert final_state.primary_goal.completed is True
        assert len(final_state.text.completed_sentences) == 1
        assert "Ein apfel ist eine frucht." in final_state.get_full_text()


# ============================================================================
# Tests: PHASE 2 Content Selection Rules (Fact Selection)
# ============================================================================


class TestFactSelectionRules:
    """Tests für Fact Selection Rules (10 Regeln)."""

    def test_select_is_a_fact_rule(self):
        """Test: SELECT_IS_A_FACT wählt IS_A Fakt aus."""
        from component_54_production_system import create_select_is_a_fact_rule

        rule = create_select_is_a_fact_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
                {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.80},
            ],
        )

        # Regel sollte matchen
        assert rule.matches(state) is True

        # Regel anwenden
        rule.apply(state)

        # IS_A Fakt sollte in pending_facts sein
        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["relation_type"] == "IS_A"
        assert state.discourse.pending_facts[0]["object"] == "frucht"

    def test_select_property_fact_rule(self):
        """Test: SELECT_PROPERTY_FACT wählt HAS_PROPERTY Fakt aus."""
        from component_54_production_system import create_select_property_fact_rule

        rule = create_select_property_fact_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90},
                {"relation_type": "HAS_PROPERTY", "object": "süß", "confidence": 0.85},
            ],
        )

        # Regel sollte matchen
        assert rule.matches(state) is True

        # Regel anwenden
        rule.apply(state)

        # HAS_PROPERTY Fakt sollte in pending_facts sein
        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["relation_type"] == "HAS_PROPERTY"
        # Sollte den mit höchster Confidence wählen
        assert state.discourse.pending_facts[0]["object"] == "rot"

    def test_select_capability_fact_rule(self):
        """Test: SELECT_CAPABILITY_FACT wählt CAPABLE_OF Fakt aus."""
        from component_54_production_system import create_select_capability_fact_rule

        rule = create_select_capability_fact_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {
                    "relation_type": "CAPABLE_OF",
                    "object": "fliegen",
                    "confidence": 0.88,
                },
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["relation_type"] == "CAPABLE_OF"

    def test_select_location_fact_rule(self):
        """Test: SELECT_LOCATION_FACT wählt LOCATED_IN Fakt aus."""
        from component_54_production_system import create_select_location_fact_rule

        rule = create_select_location_fact_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "LOCATED_IN", "object": "europa", "confidence": 0.87},
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["relation_type"] == "LOCATED_IN"

    def test_select_part_of_fact_rule(self):
        """Test: SELECT_PART_OF_FACT wählt PART_OF Fakt aus."""
        from component_54_production_system import create_select_part_of_fact_rule

        rule = create_select_part_of_fact_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "PART_OF", "object": "baum", "confidence": 0.85},
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["relation_type"] == "PART_OF"

    def test_prioritize_high_confidence_rule(self):
        """Test: PRIORITIZE_HIGH_CONFIDENCE sortiert Fakten nach Confidence."""
        from component_54_production_system import (
            create_prioritize_high_confidence_rule,
        )

        rule = create_prioritize_high_confidence_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.70},
                {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.95},
                {
                    "relation_type": "CAPABLE_OF",
                    "object": "wachsen",
                    "confidence": 0.60,
                },
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # available_facts sollte sortiert sein (absteigend nach Confidence)
        assert state.available_facts[0]["confidence"] == 0.95
        assert state.available_facts[1]["confidence"] == 0.70
        assert state.available_facts[2]["confidence"] == 0.60

    def test_skip_low_confidence_rule(self):
        """Test: SKIP_LOW_CONFIDENCE entfernt Fakten mit Confidence < 0.40."""
        from component_54_production_system import create_skip_low_confidence_rule

        rule = create_skip_low_confidence_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
                {
                    "relation_type": "HAS_PROPERTY",
                    "object": "blau",
                    "confidence": 0.30,
                },  # Low confidence
                {
                    "relation_type": "CAPABLE_OF",
                    "object": "sprechen",
                    "confidence": 0.15,
                },  # Very low
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Nur High-Confidence Fakt sollte übrig sein
        assert len(state.available_facts) == 1
        assert state.available_facts[0]["confidence"] == 0.95

    def test_select_synonym_rule(self):
        """Test: SELECT_SYNONYM wählt Synonym-Fakt aus."""
        from component_54_production_system import create_select_synonym_rule

        rule = create_select_synonym_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "SYNONYM", "object": "obst", "confidence": 0.78},
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["relation_type"] == "SYNONYM"

    def test_select_definition_rule(self):
        """Test: SELECT_DEFINITION wählt Definition aus."""
        from component_54_production_system import create_select_definition_rule

        rule = create_select_definition_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {
                    "relation_type": "DEFINITION",
                    "object": "essbare Frucht",
                    "confidence": 0.93,
                },
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["relation_type"] == "DEFINITION"

    def test_finish_content_selection_rule(self):
        """Test: FINISH_CONTENT_SELECTION beendet Selektion."""
        from component_54_production_system import create_finish_content_selection_rule

        rule = create_finish_content_selection_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[],  # Keine Fakten mehr
        )

        # Füge einen Fakt zu pending_facts hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "IS_A", "object": "frucht"}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Content selection sollte als abgeschlossen markiert sein
        assert state.constraints.get("content_selection_finished") is True


# ============================================================================
# Tests: PHASE 2 Content Selection Rules (Confidence Filtering)
# ============================================================================


class TestConfidenceFilteringRules:
    """Tests für Confidence-based Filtering Rules (5 Regeln)."""

    def test_require_high_confidence_rule(self):
        """Test: REQUIRE_HIGH_CONFIDENCE ersetzt Low-Confidence Fakt."""
        from component_54_production_system import create_require_high_confidence_rule

        rule = create_require_high_confidence_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
            ],
        )

        # Füge Low-Confidence Fakt zu pending_facts hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "grün", "confidence": 0.70}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Low-Confidence Fakt sollte durch High-Confidence ersetzt sein
        assert len(state.discourse.pending_facts) == 1
        assert state.discourse.pending_facts[0]["confidence"] == 0.95

    def test_warn_medium_confidence_rule(self):
        """Test: WARN_MEDIUM_CONFIDENCE markiert Medium-Confidence Fakt."""
        from component_54_production_system import create_warn_medium_confidence_rule

        rule = create_warn_medium_confidence_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
        )

        # Füge Medium-Confidence Fakt zu pending_facts hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "gelb", "confidence": 0.75}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Fakt sollte markiert sein
        assert state.discourse.pending_facts[0].get("uncertainty_marked") is True
        assert (
            state.discourse.pending_facts[0].get("hedging_phrase") == "möglicherweise"
        )

    def test_skip_uncertain_facts_rule(self):
        """Test: SKIP_UNCERTAIN_FACTS entfernt unsichere Fakten aus pending_facts."""
        from component_54_production_system import create_skip_uncertain_facts_rule

        rule = create_skip_uncertain_facts_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
        )

        # Füge unsicheren Fakt zu pending_facts hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "lila", "confidence": 0.30}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # pending_facts sollte leer sein
        assert len(state.discourse.pending_facts) == 0

    def test_aggregate_multi_source_rule(self):
        """Test: AGGREGATE_MULTI_SOURCE aggregiert Duplikate."""
        from component_54_production_system import create_aggregate_multi_source_rule

        rule = create_aggregate_multi_source_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {
                    "relation_type": "IS_A",
                    "object": "frucht",
                    "confidence": 0.90,
                    "source": "source1",
                },
                {
                    "relation_type": "IS_A",
                    "object": "frucht",
                    "confidence": 0.85,
                    "source": "source2",
                },
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Sollte nur noch einen aggregierten Fakt geben
        assert len(state.available_facts) == 1
        aggregated = state.available_facts[0]

        assert aggregated["relation_type"] == "IS_A"
        assert aggregated["object"] == "frucht"
        assert aggregated["aggregated_from"] == 2
        # Confidence sollte erhöht sein (avg + boost)
        assert aggregated["confidence"] > 0.85

    def test_prefer_direct_fact_rule(self):
        """Test: PREFER_DIRECT_FACT bevorzugt direkte Fakten."""
        from component_54_production_system import create_prefer_direct_fact_rule

        rule = create_prefer_direct_fact_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {
                    "relation_type": "IS_A",
                    "object": "frucht",
                    "confidence": 0.95,
                    "hop_count": 1,
                },  # Direkt
                {
                    "relation_type": "IS_A",
                    "object": "frucht",
                    "confidence": 0.80,
                    "hop_count": 2,
                },  # Indirekt
            ],
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Nur direkter Fakt sollte übrig sein
        assert len(state.available_facts) == 1
        assert state.available_facts[0]["hop_count"] == 1


# ============================================================================
# Integration Tests: Content Selection Workflow
# ============================================================================


class TestContentSelectionWorkflow:
    """Integrationstests für Content Selection Workflow."""

    def test_full_content_selection_pipeline(self):
        """Test: Vollständiger Content Selection Workflow mit allen Regeln."""
        from component_54_production_system import create_all_content_selection_rules

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
                {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90},
                {
                    "relation_type": "HAS_PROPERTY",
                    "object": "grün",
                    "confidence": 0.30,
                },  # Low confidence
                {
                    "relation_type": "CAPABLE_OF",
                    "object": "wachsen",
                    "confidence": 0.88,
                },
            ],
        )

        engine = ProductionSystemEngine()
        rules = create_all_content_selection_rules()
        engine.add_rules(rules)

        # Execute
        final_state = engine.generate(state)

        # Validierungen
        # 1. Low-Confidence Fakten sollten entfernt worden sein
        remaining_facts = (
            final_state.available_facts + final_state.discourse.pending_facts
        )
        low_conf = [f for f in remaining_facts if f.get("confidence", 1.0) < 0.40]
        assert len(low_conf) == 0

        # 2. Mindestens ein Fakt sollte ausgewählt sein
        assert len(final_state.discourse.pending_facts) >= 1

        # 3. Content selection sollte abgeschlossen sein
        assert final_state.constraints.get("content_selection_finished") is True

    def test_confidence_prioritization(self):
        """Test: High-Confidence Fakten werden bevorzugt."""
        from component_54_production_system import (
            create_prioritize_high_confidence_rule,
            create_select_is_a_fact_rule,
        )

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.70},
                {"relation_type": "IS_A", "object": "obst", "confidence": 0.95},
            ],
        )

        engine = ProductionSystemEngine()
        engine.add_rule(create_prioritize_high_confidence_rule())
        engine.add_rule(create_select_is_a_fact_rule())

        # Manuell 2 Zyklen ausführen
        state.max_cycles = 2
        final_state = engine.generate(state)

        # High-Confidence Fakt sollte ausgewählt worden sein
        if len(final_state.discourse.pending_facts) > 0:
            selected = final_state.discourse.pending_facts[0]
            assert selected["object"] == "obst"  # Higher confidence
            assert selected["confidence"] == 0.95


# ============================================================================
# Tests: PHASE 3 Lexicalization Rules (15 Regeln)
# ============================================================================


class TestLexicalizationRules:
    """Tests für Lexicalization Rules (15 Regeln)."""

    def test_verbalize_is_a_simple_rule(self):
        """Test: VERBALIZE_IS_A_SIMPLE erzeugt korrekten Satz."""
        from component_54_production_system import create_verbalize_is_a_simple_rule

        rule = create_verbalize_is_a_simple_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="apfel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Füge IS_A Fakt zu pending_facts hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Satz sollte erzeugt worden sein
        assert len(state.text.completed_sentences) == 1
        assert "apfel" in state.text.completed_sentences[0].lower()
        assert "frucht" in state.text.completed_sentences[0].lower()

    def test_verbalize_has_property_rule(self):
        """Test: VERBALIZE_HAS_PROPERTY erzeugt Property-Satz."""
        from component_54_production_system import create_verbalize_has_property_rule

        rule = create_verbalize_has_property_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="apfel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.text.completed_sentences) == 1
        assert "rot" in state.text.completed_sentences[0].lower()

    def test_verbalize_capable_of_rule(self):
        """Test: VERBALIZE_CAPABLE_OF erzeugt Capability-Satz."""
        from component_54_production_system import create_verbalize_capable_of_rule

        rule = create_verbalize_capable_of_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="vogel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        state.discourse.pending_facts.append(
            {"relation_type": "CAPABLE_OF", "object": "fliegen", "confidence": 0.87}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.text.completed_sentences) == 1
        assert "kann" in state.text.completed_sentences[0].lower()
        assert "fliegen" in state.text.completed_sentences[0].lower()

    def test_verbalize_located_in_rule(self):
        """Test: VERBALIZE_LOCATED_IN erzeugt Location-Satz."""
        from component_54_production_system import create_verbalize_located_in_rule

        rule = create_verbalize_located_in_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="berlin"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        state.discourse.pending_facts.append(
            {"relation_type": "LOCATED_IN", "object": "deutschland", "confidence": 0.85}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        assert len(state.text.completed_sentences) == 1
        assert "liegt" in state.text.completed_sentences[0].lower()

    def test_combine_facts_conjunction_rule(self):
        """Test: COMBINE_FACTS_CONJUNCTION kombiniert Fakten mit 'und'."""
        from component_54_production_system import create_combine_facts_conjunction_rule

        rule = create_combine_facts_conjunction_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="apfel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Zwei HAS_PROPERTY Fakten
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90}
        )
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "süß", "confidence": 0.85}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Kombinierter Satz sollte erzeugt worden sein
        assert len(state.text.completed_sentences) == 1
        assert "und" in state.text.completed_sentences[0]
        assert "rot" in state.text.completed_sentences[0].lower()
        assert "süß" in state.text.completed_sentences[0].lower()

    def test_avoid_repetition_rule(self):
        """Test: AVOID_REPETITION verwendet Pronomen."""
        from component_54_production_system import create_avoid_repetition_rule

        rule = create_avoid_repetition_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="apfel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Markiere Entität als erwähnt
        state.mention_entity("apfel")

        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Pronomen sollte verwendet werden
        assert len(state.text.completed_sentences) == 1
        sentence = state.text.completed_sentences[0]
        assert "er" in sentence.lower() or "sie" in sentence.lower()

    def test_compress_similar_facts_rule(self):
        """Test: COMPRESS_SIMILAR_FACTS fasst mehrere Fakten zusammen."""
        from component_54_production_system import create_compress_similar_facts_rule

        rule = create_compress_similar_facts_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="apfel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Drei HAS_PROPERTY Fakten
        for color in ["rot", "grün", "gelb"]:
            state.discourse.pending_facts.append(
                {"relation_type": "HAS_PROPERTY", "object": color, "confidence": 0.85}
            )

        assert rule.matches(state) is True
        rule.apply(state)

        # Komprimierter Satz sollte erzeugt worden sein
        assert len(state.text.completed_sentences) == 1
        assert "oder" in state.text.completed_sentences[0]

    def test_finish_lexicalization_rule(self):
        """Test: FINISH_LEXICALIZATION markiert Phase als abgeschlossen."""
        from component_54_production_system import create_finish_lexicalization_rule

        rule = create_finish_lexicalization_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Keine pending_facts mehr
        state.discourse.pending_facts = []

        assert rule.matches(state) is True
        rule.apply(state)

        # Lexicalization sollte als abgeschlossen markiert sein
        assert state.constraints.get("lexicalization_finished") is True


# ============================================================================
# Tests: PHASE 3 Discourse Management Rules (12 Regeln)
# ============================================================================


class TestDiscourseManagementRules:
    """Tests für Discourse Management Rules (12 Regeln)."""

    def test_introduce_with_context_rule(self):
        """Test: INTRODUCE_WITH_CONTEXT fügt kontextuelle Einleitung hinzu."""
        from component_54_production_system import create_introduce_with_context_rule

        rule = create_introduce_with_context_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True, "complex_query": True},
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Fragment sollte hinzugefügt worden sein
        assert len(state.text.sentence_fragments) == 1
        assert "introduce_context" in state.discourse.discourse_markers_used

    def test_introduce_simple_rule(self):
        """Test: INTRODUCE_SIMPLE fügt einfache Einleitung hinzu."""
        from component_54_production_system import create_introduce_simple_rule

        rule = create_introduce_simple_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="apfel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Fragment sollte hinzugefügt worden sein
        assert len(state.text.sentence_fragments) == 1
        assert "apfel" in state.text.sentence_fragments[0].lower()

    def test_signal_uncertainty_rule(self):
        """Test: SIGNAL_UNCERTAINTY markiert unsichere Fakten."""
        from component_54_production_system import create_signal_uncertainty_rule

        rule = create_signal_uncertainty_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Unsicherer Fakt
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "grün", "confidence": 0.60}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Fakt sollte markiert sein
        fact = state.discourse.pending_facts[0]
        assert fact.get("uncertainty_signaled") is True
        assert fact.get("uncertainty_phrase") == "ich vermute"

    def test_signal_high_confidence_rule(self):
        """Test: SIGNAL_HIGH_CONFIDENCE markiert hochsichere Fakten."""
        from component_54_production_system import create_signal_high_confidence_rule

        rule = create_signal_high_confidence_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Hochsicherer Fakt
        state.discourse.pending_facts.append(
            {"relation_type": "IS_A", "object": "frucht", "confidence": 0.98}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Fakt sollte markiert sein
        fact = state.discourse.pending_facts[0]
        assert fact.get("high_confidence_signaled") is True

    def test_explain_reasoning_path_rule(self):
        """Test: EXPLAIN_REASONING_PATH markiert Multi-Hop Fakten."""
        from component_54_production_system import create_explain_reasoning_path_rule

        rule = create_explain_reasoning_path_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Multi-Hop Fakt
        state.discourse.pending_facts.append(
            {
                "relation_type": "IS_A",
                "object": "lebewesen",
                "confidence": 0.85,
                "hop_count": 3,
            }
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Fakt sollte markiert sein
        fact = state.discourse.pending_facts[0]
        assert fact.get("reasoning_explained") is True

    def test_mark_hypothesis_rule(self):
        """Test: MARK_HYPOTHESIS markiert Hypothesen."""
        from component_54_production_system import create_mark_hypothesis_rule

        rule = create_mark_hypothesis_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Hypothese
        state.discourse.pending_facts.append(
            {
                "relation_type": "IS_A",
                "object": "pflanze",
                "type": "hypothesis",
                "confidence": 0.70,
            }
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Hypothese sollte markiert sein
        fact = state.discourse.pending_facts[0]
        assert fact.get("hypothesis_marked") is True

    def test_add_confidence_qualifier_rule(self):
        """Test: ADD_CONFIDENCE_QUALIFIER fügt Qualifier hinzu."""
        from component_54_production_system import create_add_confidence_qualifier_rule

        rule = create_add_confidence_qualifier_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Medium-Confidence Fakt
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "gelb", "confidence": 0.78}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Qualifier sollte hinzugefügt sein
        fact = state.discourse.pending_facts[0]
        assert fact.get("qualifier_added") is True
        assert fact.get("qualifier") == "wahrscheinlich"

    def test_add_transition_rule(self):
        """Test: ADD_TRANSITION fügt Übergänge hinzu."""
        from component_54_production_system import create_add_transition_rule

        rule = create_add_transition_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
        )

        # Zwei Sätze vorhanden
        state.add_sentence("Ein Apfel ist eine Frucht.")
        state.add_sentence("Äpfel sind rot.")

        # Noch Fakten zu verbalisieren
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "süß", "confidence": 0.85}
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Transition sollte hinzugefügt sein
        assert "transition" in state.discourse.discourse_markers_used
        assert len(state.text.sentence_fragments) > 0

    def test_conclude_answer_rule(self):
        """Test: CONCLUDE_ANSWER fügt Abschluss hinzu."""
        from component_54_production_system import create_conclude_answer_rule

        rule = create_conclude_answer_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Mehrere Sätze vorhanden
        state.add_sentence("Ein Apfel ist eine Frucht.")
        state.add_sentence("Äpfel sind rot.")

        assert rule.matches(state) is True
        rule.apply(state)

        # Abschluss sollte hinzugefügt sein
        assert "conclusion" in state.discourse.discourse_markers_used

    def test_offer_elaboration_rule(self):
        """Test: OFFER_ELABORATION bietet weitere Infos an."""
        from component_54_production_system import create_offer_elaboration_rule

        rule = create_offer_elaboration_rule()

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION, target_entity="apfel"
        )
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        assert rule.matches(state) is True
        rule.apply(state)

        # Angebot sollte hinzugefügt sein
        assert len(state.text.completed_sentences) == 1
        assert "mehr" in state.text.completed_sentences[0].lower()
        assert state.primary_goal.completed is True


# ============================================================================
# Integration Tests: PHASE 3 Workflow
# ============================================================================


class TestPhase3Workflow:
    """Integrationstests für vollständigen PHASE 3 Workflow."""

    def test_lexicalization_workflow(self):
        """Test: Vollständiger Lexicalization Workflow."""
        from component_54_production_system import create_all_lexicalization_rules

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Füge Fakten hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95}
        )
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90}
        )

        engine = ProductionSystemEngine()
        rules = create_all_lexicalization_rules()
        engine.add_rules(rules)

        # Execute
        final_state = engine.generate(state)

        # Validierungen
        # 1. Sätze sollten erzeugt worden sein
        assert len(final_state.text.completed_sentences) >= 1

        # 2. Lexicalization sollte abgeschlossen sein
        assert final_state.constraints.get("lexicalization_finished") is True

    def test_discourse_management_workflow(self):
        """Test: Vollständiger Discourse Management Workflow."""
        from component_54_production_system import create_all_discourse_management_rules

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Füge unsicheren Fakt hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "grün", "confidence": 0.65}
        )

        engine = ProductionSystemEngine()
        rules = create_all_discourse_management_rules()
        engine.add_rules(rules)

        # Execute
        final_state = engine.generate(state)

        # Validierungen
        # 1. Discourse Markers sollten gesetzt sein
        assert len(final_state.discourse.discourse_markers_used) > 0

    def test_complete_phase3_workflow(self):
        """Test: Vollständiger PHASE 3 Workflow (Lexicalization + Discourse)."""
        from component_54_production_system import create_all_phase3_rules

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"content_selection_finished": True},
        )

        # Füge verschiedene Fakten hinzu
        state.discourse.pending_facts.append(
            {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95}
        )
        state.discourse.pending_facts.append(
            {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90}
        )
        state.discourse.pending_facts.append(
            {"relation_type": "CAPABLE_OF", "object": "wachsen", "confidence": 0.88}
        )

        engine = ProductionSystemEngine()
        rules = create_all_phase3_rules()
        engine.add_rules(rules)

        # Execute
        state.max_cycles = 100  # Genug Zyklen
        final_state = engine.generate(state)

        # Validierungen
        # 1. Sätze erzeugt
        assert len(final_state.text.completed_sentences) >= 1

        # 2. Lexicalization abgeschlossen
        assert final_state.constraints.get("lexicalization_finished") is True

    def test_complete_production_system(self):
        """Test: Vollständiges Produktionssystem (Phase 2 + Phase 3)."""
        from component_54_production_system import create_complete_production_system

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
                {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90},
                {
                    "relation_type": "HAS_PROPERTY",
                    "object": "grün",
                    "confidence": 0.30,
                },  # Low conf
            ],
        )

        engine = ProductionSystemEngine()
        rules = create_complete_production_system()
        engine.add_rules(rules)

        # Execute
        state.max_cycles = 150  # Genug Zyklen für alle Phasen
        final_state = engine.generate(state)

        # Validierungen
        # 1. Content Selection durchgeführt
        assert final_state.constraints.get("content_selection_finished") is True

        # 2. Lexicalization durchgeführt
        assert final_state.constraints.get("lexicalization_finished") is True

        # 3. Low-Confidence Fakten entfernt
        remaining_facts = (
            final_state.available_facts + final_state.discourse.pending_facts
        )
        low_conf = [f for f in remaining_facts if f.get("confidence", 1.0) < 0.40]
        assert len(low_conf) == 0

        # 4. Sätze erzeugt
        assert len(final_state.text.completed_sentences) >= 1


# ============================================================================
# Tests: Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Tests für Convenience-Funktionen."""

    def test_create_all_lexicalization_rules(self):
        """Test: create_all_lexicalization_rules gibt 15 Regeln zurück."""
        from component_54_production_system import create_all_lexicalization_rules

        rules = create_all_lexicalization_rules()

        assert len(rules) == 15
        assert all(r.category == RuleCategory.LEXICALIZATION for r in rules)

    def test_create_all_discourse_management_rules(self):
        """Test: create_all_discourse_management_rules gibt 12 Regeln zurück."""
        from component_54_production_system import create_all_discourse_management_rules

        rules = create_all_discourse_management_rules()

        assert len(rules) == 12
        assert all(r.category == RuleCategory.DISCOURSE for r in rules)

    def test_create_all_phase3_rules(self):
        """Test: create_all_phase3_rules gibt 27 Regeln zurück."""
        from component_54_production_system import create_all_phase3_rules

        rules = create_all_phase3_rules()

        assert len(rules) == 27

    def test_create_complete_production_system(self):
        """Test: create_complete_production_system gibt 54 Regeln zurück."""
        from component_54_production_system import create_complete_production_system

        rules = create_complete_production_system()

        # 15 Content Selection + 15 Lexicalization + 12 Discourse + 12 Syntax = 54
        assert len(rules) == 54


# ============================================================================
# Tests: PHASE 4 Syntactic Realization Rules (12 Regeln)
# ============================================================================


class TestSyntacticRealizationRules:
    """Tests für Syntactic Realization Rules (12 Regeln)."""

    def test_capitalize_sentence_start_rule(self):
        """Test: CAPITALIZE_SENTENCE_START kapitalisiert Satzanfang."""
        from component_54_production_system import create_capitalize_sentence_start_rule

        rule = create_capitalize_sentence_start_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Satz mit Kleinbuchstaben am Anfang
        state.add_sentence("ein apfel ist rot")

        assert rule.matches(state) is True
        rule.apply(state)

        # Satzanfang sollte kapitalisiert sein
        assert state.text.completed_sentences[0][0].isupper()
        assert state.text.completed_sentences[0].startswith("Ein")

    def test_add_period_rule(self):
        """Test: ADD_PERIOD fügt Punkt am Satzende hinzu."""
        from component_54_production_system import create_add_period_rule

        rule = create_add_period_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Satz ohne Punkt
        state.add_sentence("Ein Apfel ist rot")

        assert rule.matches(state) is True
        rule.apply(state)

        # Punkt sollte hinzugefügt sein
        assert state.text.completed_sentences[0].endswith(".")

    def test_capitalize_nouns_rule(self):
        """Test: CAPITALIZE_NOUNS kapitalisiert deutsche Nomen."""
        from component_54_production_system import create_capitalize_nouns_rule

        rule = create_capitalize_nouns_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Nomen mit Kleinbuchstaben
        state.add_sentence("der apfel ist rot")

        assert rule.matches(state) is True
        rule.apply(state)

        # Nomen sollte kapitalisiert sein
        assert "Apfel" in state.text.completed_sentences[0]

    def test_add_comma_conjunction_rule(self):
        """Test: ADD_COMMA_CONJUNCTION fügt Komma vor Konjunktion hinzu."""
        from component_54_production_system import create_add_comma_conjunction_rule

        rule = create_add_comma_conjunction_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Satz mit Konjunktion ohne Komma
        state.add_sentence("Äpfel sind rot aber Birnen sind grün")

        assert rule.matches(state) is True
        rule.apply(state)

        # Komma sollte hinzugefügt sein
        assert ", aber" in state.text.completed_sentences[0]

    def test_fix_verb_agreement_rule(self):
        """Test: FIX_VERB_AGREEMENT korrigiert Verb-Kongruenz."""
        from component_54_production_system import create_fix_verb_agreement_rule

        rule = create_fix_verb_agreement_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Plural-Subjekt mit Singular-Verb
        state.add_sentence("Äpfel ist rot")

        assert rule.matches(state) is True
        rule.apply(state)

        # Verb sollte korrigiert sein
        assert "sind" in state.text.completed_sentences[0]

    def test_ensure_gender_agreement_rule(self):
        """Test: ENSURE_GENDER_AGREEMENT korrigiert Genus-Kongruenz."""
        from component_54_production_system import create_ensure_gender_agreement_rule

        rule = create_ensure_gender_agreement_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Falscher Artikel für feminines Nomen (Ordnung endet auf "ung" -> feminin)
        state.add_sentence("der Ordnung ist wichtig")

        assert rule.matches(state) is True
        rule.apply(state)

        # Artikel sollte korrigiert sein
        assert (
            "die Ordnung" in state.text.completed_sentences[0]
            or "Die Ordnung" in state.text.completed_sentences[0]
        )

    def test_insert_preposition_rule(self):
        """Test: INSERT_PREPOSITION fügt fehlende Präposition ein."""
        from component_54_production_system import create_insert_preposition_rule

        rule = create_insert_preposition_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Satz ohne Präposition
        state.add_sentence("Berlin liegt Deutschland")

        assert rule.matches(state) is True
        rule.apply(state)

        # Präposition sollte eingefügt sein
        assert "liegt in" in state.text.completed_sentences[0]

    def test_finish_sentence_rule(self):
        """Test: FINISH_SENTENCE markiert Syntax als abgeschlossen."""
        from component_54_production_system import create_finish_sentence_rule

        rule = create_finish_sentence_rule()

        goal = GenerationGoal(goal_type=GenerationGoalType.ANSWER_QUESTION)
        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Satz vorhanden
        state.add_sentence("Ein Apfel ist rot.")

        assert rule.matches(state) is True
        rule.apply(state)

        # Syntax sollte als abgeschlossen markiert sein
        assert state.constraints.get("syntax_finished") is True
        assert state.primary_goal.completed is True


# ============================================================================
# Integration Tests: PHASE 4 Workflow
# ============================================================================


class TestPhase4Workflow:
    """Integrationstests für vollständigen PHASE 4 Workflow."""

    def test_syntactic_realization_workflow(self):
        """Test: Vollständiger Syntactic Realization Workflow."""
        from component_54_production_system import (
            create_all_syntactic_realization_rules,
        )

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Füge unvollständige Sätze hinzu
        state.add_sentence("apfel ist rot")  # Kleinbuchstaben, kein Punkt

        engine = ProductionSystemEngine()
        rules = create_all_syntactic_realization_rules()
        engine.add_rules(rules)

        # Execute
        final_state = engine.generate(state)

        # Validierungen
        # 1. Satzanfang kapitalisiert
        assert final_state.text.completed_sentences[0][0].isupper()

        # 2. Punkt am Ende
        assert final_state.text.completed_sentences[0].endswith(".")

        # 3. Syntax abgeschlossen
        assert final_state.constraints.get("syntax_finished") is True

    def test_complete_pipeline_phase2_to_phase4(self):
        """Test: Vollständige Pipeline von Content Selection bis Syntax."""
        from component_54_production_system import create_complete_production_system

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "frucht", "confidence": 0.95},
                {"relation_type": "HAS_PROPERTY", "object": "rot", "confidence": 0.90},
            ],
        )

        engine = ProductionSystemEngine()
        rules = create_complete_production_system()
        engine.add_rules(rules)

        # Execute
        state.max_cycles = 200  # Genug Zyklen für alle Phasen
        final_state = engine.generate(state)

        # Validierungen
        # 1. Content Selection durchgeführt
        assert final_state.constraints.get("content_selection_finished") is True

        # 2. Lexicalization durchgeführt
        assert final_state.constraints.get("lexicalization_finished") is True

        # 3. Syntax durchgeführt
        assert final_state.constraints.get("syntax_finished") is True

        # 4. Sätze erzeugt und grammatisch korrekt
        assert len(final_state.text.completed_sentences) >= 1

        # 5. Alle Sätze beginnen mit Großbuchstaben
        for sentence in final_state.text.completed_sentences:
            if sentence:
                assert sentence[0].isupper()

        # 6. Alle Sätze enden mit Punkt
        for sentence in final_state.text.completed_sentences:
            if sentence:
                assert sentence.endswith((".", "!", "?"))

    def test_phase4_with_multiple_sentences(self):
        """Test: Syntactic Realization mit mehreren Sätzen."""
        from component_54_production_system import (
            create_all_syntactic_realization_rules,
        )

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="apfel",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            constraints={"lexicalization_finished": True},
        )

        # Füge mehrere unvollständige Sätze hinzu
        state.add_sentence("apfel ist rot")
        state.add_sentence("äpfel wachsen an bäumen")
        state.add_sentence("sie schmecken süß")

        engine = ProductionSystemEngine()
        rules = create_all_syntactic_realization_rules()
        engine.add_rules(rules)

        # Execute
        final_state = engine.generate(state)

        # Validierungen
        assert len(final_state.text.completed_sentences) == 3

        # Alle Sätze sollten grammatisch korrigiert sein
        for sentence in final_state.text.completed_sentences:
            assert sentence[0].isupper()  # Kapitalisierung
            assert sentence.endswith(".")  # Punkt


# ============================================================================
# Tests: PHASE 4 Convenience Functions
# ============================================================================


class TestPhase4ConvenienceFunctions:
    """Tests für PHASE 4 Convenience-Funktionen."""

    def test_create_all_syntactic_realization_rules(self):
        """Test: create_all_syntactic_realization_rules gibt 12 Regeln zurück."""
        from component_54_production_system import (
            create_all_syntactic_realization_rules,
        )

        rules = create_all_syntactic_realization_rules()

        assert len(rules) == 12
        assert all(r.category == RuleCategory.SYNTAX for r in rules)

    def test_create_all_phase4_rules(self):
        """Test: create_all_phase4_rules gibt 12 Regeln zurück."""
        from component_54_production_system import create_all_phase4_rules

        rules = create_all_phase4_rules()

        assert len(rules) == 12

    def test_complete_system_includes_syntax(self):
        """Test: Vollständiges System enthält alle Syntax-Regeln."""
        from component_54_production_system import create_complete_production_system

        rules = create_complete_production_system()

        # Zähle Syntax-Regeln
        syntax_rules = [r for r in rules if r.category == RuleCategory.SYNTAX]

        assert len(syntax_rules) == 12
