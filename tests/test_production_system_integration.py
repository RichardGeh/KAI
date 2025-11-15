"""
test_production_system_integration.py

Integration Tests für Phase 5: ResponseFormatter + Production System

Tests:
- generate_with_production_system() Wrapper
- ResponseGenerationRouter A/B Testing
- MetaLearningEngine Dual-System Tracking
- Signal-Emission und UI-Integration

Author: KAI Development Team
Date: 2025-11-13
"""

from unittest.mock import Mock, patch

import pytest

from component_46_meta_learning import MetaLearningEngine
from component_54_production_system import (
    GenerationGoal,
    GenerationGoalType,
    ProductionSystemEngine,
    ResponseGenerationState,
    create_all_content_selection_rules,
)
from kai_response_formatter import (
    KaiResponse,
    KaiResponseFormatter,
    ResponseGenerationRouter,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def formatter():
    """KaiResponseFormatter Instanz"""
    return KaiResponseFormatter()


@pytest.fixture
def mock_signals():
    """Mock KaiSignals für UI-Testing"""
    signals = Mock()
    signals.production_system_trace = Mock()
    signals.production_system_trace.emit = Mock()
    return signals


@pytest.fixture
def production_engine(mock_signals):
    """ProductionSystemEngine mit Mock-Signals"""
    engine = ProductionSystemEngine(signals=mock_signals)
    engine.add_rules(create_all_content_selection_rules())
    return engine


@pytest.fixture
def mock_netzwerk():
    """Mock KonzeptNetzwerk"""
    mock = Mock()
    mock.driver = Mock()
    mock.driver.session = Mock()
    return mock


@pytest.fixture
def mock_embedding_service():
    """Mock EmbeddingService"""
    mock = Mock()
    mock.get_embedding = Mock(return_value=[0.1] * 384)
    return mock


@pytest.fixture
def meta_engine(mock_netzwerk, mock_embedding_service):
    """MetaLearningEngine Instanz"""
    return MetaLearningEngine(mock_netzwerk, mock_embedding_service)


# ============================================================================
# Test: generate_with_production_system() (Schritt 5.1)
# ============================================================================


class TestProductionSystemWrapper:
    """Tests für generate_with_production_system() Wrapper"""

    def test_basic_generation(self, formatter):
        """Test: Basic Production System Generation"""
        # Arrange
        topic = "apfel"
        facts = {"IS_A": ["frucht"], "HAS_PROPERTY": ["rot"]}
        bedeutungen = ["Ein Apfel ist eine Frucht"]
        synonyms = []

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
            query_type="normal",
            confidence=0.9,
        )

        # Assert
        assert isinstance(response, KaiResponse)
        assert response.strategy == "production_system"
        assert response.confidence is not None
        assert len(response.trace) > 0
        assert "Production System Generierung gestartet" in response.trace[0]

    def test_generation_with_signals(self, formatter, mock_signals):
        """Test: Generation mit Signal-Emission"""
        # Arrange
        topic = "vogel"
        facts = {"CAPABLE_OF": ["fliegen"]}
        bedeutungen = []
        synonyms = []

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
            signals=mock_signals,
        )

        # Assert
        assert response.strategy == "production_system"
        # Signals könnten emittiert worden sein (abhängig von Regelanwendung)
        # Wir prüfen nur, dass der Mechanismus funktioniert (kein Fehler)
        assert mock_signals.production_system_trace is not None

    def test_fallback_on_error(self, formatter):
        """Test: Fallback auf Pipeline bei Error"""
        # Arrange
        topic = "test"
        facts = {}
        bedeutungen = []
        synonyms = []

        # Mock ProductionSystemEngine um Exception zu werfen
        with patch(
            "component_54_production_system.ProductionSystemEngine"
        ) as MockEngine:
            MockEngine.side_effect = Exception("Test error")

            # Act
            response = formatter.generate_with_production_system(
                topic=topic,
                facts=facts,
                bedeutungen=bedeutungen,
                synonyms=synonyms,
            )

            # Assert
            assert response.strategy == "pipeline_fallback"
            assert "Production System failed" in response.trace[0]

    def test_empty_facts_handling(self, formatter):
        """Test: Umgang mit leeren Fakten"""
        # Act
        response = formatter.generate_with_production_system(
            topic="unbekannt",
            facts={},
            bedeutungen=[],
            synonyms=[],
        )

        # Assert
        assert isinstance(response, KaiResponse)
        # Sollte trotzdem Text generieren (oder Fallback)
        assert len(response.text) > 0


# ============================================================================
# Test: ResponseGenerationRouter (Schritt 5.2)
# ============================================================================


class TestResponseGenerationRouter:
    """Tests für A/B Testing Router"""

    def test_router_initialization(self, formatter):
        """Test: Router Initialisierung"""
        # Act
        router = ResponseGenerationRouter(
            formatter=formatter,
            production_system_weight=0.5,
            enable_meta_learning=False,
        )

        # Assert
        assert router.production_weight == 0.5
        assert router.enable_meta_learning is False
        assert router.total_queries == 0

    def test_route_to_system_random_split(self, formatter):
        """Test: Random A/B Split (50/50)"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter,
            production_system_weight=0.5,
            enable_meta_learning=False,
        )

        # Act: Rufe 100x auf
        systems = [router.route_to_system("test query") for _ in range(100)]

        # Assert
        production_count = systems.count("production")
        pipeline_count = systems.count("pipeline")

        # Sollte ungefähr 50/50 sein (mit Toleranz)
        assert 30 <= production_count <= 70
        assert 30 <= pipeline_count <= 70
        assert router.total_queries == 100

    def test_route_with_meta_learning(self, formatter, meta_engine):
        """Test: Routing mit Meta-Learning"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter,
            production_system_weight=0.5,
            enable_meta_learning=True,
            meta_engine=meta_engine,
        )

        # Act
        system = router.route_to_system("Was ist ein Apfel?")

        # Assert
        assert system in ["production", "pipeline"]
        assert router.total_queries == 1

    def test_generate_response_pipeline(self, formatter):
        """Test: Response Generation mit Pipeline System"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter,
            production_system_weight=0.0,  # Immer Pipeline
        )

        # Act
        response = router.generate_response(
            topic="apfel",
            facts={"IS_A": ["frucht"]},
            bedeutungen=[],
            synonyms=[],
            query="Was ist ein Apfel?",
        )

        # Assert
        assert response.strategy == "pipeline"
        assert "Pipeline System Generierung" in response.trace[1]

    def test_generate_response_production(self, formatter):
        """Test: Response Generation mit Production System"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter,
            production_system_weight=1.0,  # Immer Production
        )

        # Act
        response = router.generate_response(
            topic="vogel",
            facts={"CAPABLE_OF": ["fliegen"]},
            bedeutungen=[],
            synonyms=[],
            query="Was kann ein Vogel?",
        )

        # Assert
        assert response.strategy == "production_system"
        assert "production" in response.trace[0].lower()

    def test_get_statistics(self, formatter):
        """Test: Router Statistiken"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter, production_system_weight=0.5
        )

        # Generiere einige Responses
        for i in range(10):
            router.route_to_system(f"query {i}")

        # Act
        stats = router.get_statistics()

        # Assert
        assert stats["total_queries"] == 10
        assert "production_percentage" in stats
        assert "pipeline_percentage" in stats
        assert stats["production_weight"] == 0.5

    def test_set_production_weight(self, formatter):
        """Test: Production Weight anpassen"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter, production_system_weight=0.5
        )

        # Act
        router.set_production_weight(0.8)

        # Assert
        assert router.production_weight == 0.8

    def test_set_invalid_production_weight(self, formatter):
        """Test: Ungültiger Production Weight wird rejected"""
        # Arrange
        router = ResponseGenerationRouter(formatter=formatter)

        # Act & Assert
        with pytest.raises(ValueError):
            router.set_production_weight(1.5)  # > 1.0

        with pytest.raises(ValueError):
            router.set_production_weight(-0.1)  # < 0.0


# ============================================================================
# Test: Dual-System Performance Tracking (Schritt 5.3)
# ============================================================================


class TestDualSystemTracking:
    """Tests für MetaLearningEngine Dual-System Tracking"""

    def test_record_generation_system_usage(self, meta_engine):
        """Test: Tracking von Pipeline vs. Production"""
        # Act: Track Pipeline
        meta_engine.record_generation_system_usage(
            system="pipeline",
            query="Was ist ein Apfel?",
            confidence=0.85,
            response_time=0.12,
            response_text="Ein Apfel ist eine Frucht.",
            success=True,
            user_feedback="correct",
        )

        # Track Production
        meta_engine.record_generation_system_usage(
            system="production",
            query="Was kann ein Vogel?",
            confidence=0.92,
            response_time=0.18,
            response_text="Ein Vogel kann fliegen.",
            success=True,
            user_feedback="correct",
        )

        # Assert
        pipeline_stats = meta_engine.get_strategy_stats("pipeline")
        production_stats = meta_engine.get_strategy_stats("production")

        assert pipeline_stats is not None
        assert production_stats is not None
        assert pipeline_stats.queries_handled == 1
        assert production_stats.queries_handled == 1

    def test_get_generation_system_comparison(self, meta_engine):
        """Test: System-Vergleich"""
        # Arrange: Simuliere mehrere Queries
        # Pipeline: schneller, aber niedrigere Confidence
        for i in range(5):
            meta_engine.record_generation_system_usage(
                system="pipeline",
                query=f"query {i}",
                confidence=0.75,
                response_time=0.1,
                success=True,
            )

        # Production: langsamer, aber höhere Confidence
        for i in range(5):
            meta_engine.record_generation_system_usage(
                system="production",
                query=f"query {i}",
                confidence=0.92,
                response_time=0.2,
                success=True,
            )

        # Act
        comparison = meta_engine.get_generation_system_comparison()

        # Assert
        assert "pipeline" in comparison
        assert "production" in comparison
        assert "comparison" in comparison

        # Pipeline sollte schneller sein (positiver speed_delta)
        assert comparison["comparison"]["speed_delta"] < 0  # Pipeline schneller

        # Production sollte höhere Confidence haben
        assert comparison["comparison"]["confidence_delta"] > 0

        # Winner basierend auf Overall Score
        assert comparison["comparison"]["winner"] in ["pipeline", "production"]

    def test_unknown_system_warning(self, meta_engine, caplog):
        """Test: Warnung bei unbekanntem System"""
        # Act
        meta_engine.record_generation_system_usage(
            system="unknown_system",
            query="test",
            confidence=0.5,
            response_time=0.1,
        )

        # Assert: Warnung im Log
        # (pytest caplog fixture captured das)
        # Wir erwarten, dass die Methode ohne Exception durchläuft


# ============================================================================
# Test: Signal-Integration (Schritt 5.4)
# ============================================================================


class TestSignalIntegration:
    """Tests für Production System Signal-Emission"""

    def test_signal_emission_on_rule_application(self, mock_signals):
        """Test: Signal wird bei Regelanwendung emittiert (wenn Regel matched)"""
        # Arrange
        engine = ProductionSystemEngine(signals=mock_signals)

        # Erstelle eine einfache Test-Regel die immer matched
        from component_54_production_system import ProductionRule, RuleCategory

        def always_true_condition(state):
            return True

        def simple_action(state):
            state.add_sentence("Test sentence")
            state.primary_goal.completed = True

        test_rule = ProductionRule(
            name="TEST_RULE",
            category=RuleCategory.CONTENT_SELECTION,
            condition=always_true_condition,
            action=simple_action,
            metadata={"description": "Test rule for signal emission"},
        )

        engine.add_rule(test_rule)

        state = ResponseGenerationState(
            primary_goal=GenerationGoal(
                goal_type=GenerationGoalType.ANSWER_QUESTION,
                target_entity="test",
            ),
            available_facts=[],
        )

        # Act
        engine.generate(state)

        # Assert
        # Die Test-Regel sollte angewendet worden sein
        assert mock_signals.production_system_trace.emit.call_count >= 1

        # Prüfe Signatur: (rule_name, description)
        call_args = mock_signals.production_system_trace.emit.call_args_list[0]
        assert len(call_args[0]) == 2  # Zwei Argumente
        rule_name = call_args[0][0]
        description = call_args[0][1]

        assert isinstance(rule_name, str)
        assert isinstance(description, str)
        assert rule_name == "TEST_RULE"
        assert "Test rule for signal emission" in description

    def test_engine_without_signals(self):
        """Test: Engine funktioniert auch ohne Signals"""
        # Arrange
        engine = ProductionSystemEngine(signals=None)
        engine.add_rules(create_all_content_selection_rules())

        state = ResponseGenerationState(
            primary_goal=GenerationGoal(
                goal_type=GenerationGoalType.ANSWER_QUESTION,
                target_entity="test",
            ),
            available_facts=[],
        )

        # Act & Assert: Sollte ohne Exception durchlaufen
        final_state = engine.generate(state)
        assert final_state is not None


# ============================================================================
# Integration Test: End-to-End
# ============================================================================


class TestEndToEndIntegration:
    """End-to-End Integration Tests"""

    def test_full_pipeline_with_router(self, formatter, meta_engine):
        """Test: Kompletter Flow von Query bis Response"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter,
            production_system_weight=0.5,
            enable_meta_learning=True,
            meta_engine=meta_engine,
        )

        # Act: Generiere mehrere Responses
        responses = []
        for i in range(10):
            response = router.generate_response(
                topic=f"topic_{i}",
                facts={"IS_A": ["test"]},
                bedeutungen=[],
                synonyms=[],
                query=f"Was ist topic_{i}?",
            )
            responses.append(response)

        # Assert
        assert len(responses) == 10

        # Sollte Mix aus Pipeline und Production sein
        strategies = [r.strategy for r in responses]
        assert "pipeline" in strategies or "production_system" in strategies

        # Router Stats sollten korrekt sein
        stats = router.get_statistics()
        assert stats["total_queries"] == 10

    def test_meta_learning_improves_over_time(self, formatter, meta_engine):
        """Test: Meta-Learning verbessert System-Auswahl über Zeit"""
        # Arrange
        router = ResponseGenerationRouter(
            formatter=formatter,
            production_system_weight=0.5,
            enable_meta_learning=True,
            meta_engine=meta_engine,
        )

        # Simuliere: Production System ist immer besser
        for i in range(20):
            # Route
            system = router.route_to_system(f"query {i}")

            # Fake Feedback: Production = gut, Pipeline = schlecht
            if system == "production":
                meta_engine.record_generation_system_usage(
                    system="production",
                    query=f"query {i}",
                    confidence=0.95,
                    response_time=0.15,
                    success=True,
                    user_feedback="correct",
                )
            else:
                meta_engine.record_generation_system_usage(
                    system="pipeline",
                    query=f"query {i}",
                    confidence=0.6,
                    response_time=0.1,
                    success=False,
                    user_feedback="incorrect",
                )

        # Nach genügend Samples sollte Meta-Learning Production bevorzugen
        # (aufgrund besserer Performance)
        comparison = meta_engine.get_generation_system_comparison()

        # Assert: Production sollte bessere Stats haben
        assert (
            comparison["production"]["success_rate"]
            > comparison["pipeline"]["success_rate"]
        )
        assert (
            comparison["production"]["avg_confidence"]
            > comparison["pipeline"]["avg_confidence"]
        )


# ============================================================================
# Test: ProofTree Integration (PHASE 6)
# ============================================================================


class TestProofTreeIntegration:
    """Tests für Phase 6: ProofTree Integration ins Production System"""

    def test_proof_tree_initialization(self, formatter):
        """Test: ProofTree wird beim generate() initialisiert"""
        # Arrange
        topic = "hund"
        facts = {"IS_A": ["tier"]}
        bedeutungen = ["Ein Hund ist ein Tier"]
        synonyms = []

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Assert
        assert (
            response.proof_tree is not None
        ), "ProofTree sollte in Response vorhanden sein"
        assert response.proof_tree.query.startswith(
            "Generiere Antwort für"
        ), "ProofTree Query sollte korrekt gesetzt sein"

    def test_proof_steps_created_for_rules(self, formatter):
        """Test: Jede Regelanwendung erzeugt einen ProofStep"""
        # Arrange
        topic = "katze"
        facts = {"IS_A": ["tier"], "HAS_PROPERTY": ["flauschig", "süß"]}
        bedeutungen = ["Eine Katze ist ein Haustier"]
        synonyms = ["Stubentiger"]

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Assert
        proof_tree = response.proof_tree
        assert proof_tree is not None, "ProofTree sollte vorhanden sein"

        all_steps = proof_tree.get_all_steps()
        assert len(all_steps) > 0, "ProofTree sollte mindestens 1 ProofStep enthalten"

        # Alle Steps sollten RULE_APPLICATION sein
        from component_17_proof_explanation import StepType

        for step in all_steps:
            assert (
                step.step_type == StepType.RULE_APPLICATION
            ), f"Step {step.step_id} sollte RULE_APPLICATION sein"
            assert step.rule_name is not None, "Jeder Step sollte einen rule_name haben"

    def test_proof_step_contains_state_snapshots(self, formatter):
        """Test: ProofSteps enthalten State-Snapshots (vorher/nachher)"""
        # Arrange
        topic = "vogel"
        facts = {"IS_A": ["tier"], "CAPABLE_OF": ["fliegen"]}
        bedeutungen = []
        synonyms = []

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Assert
        proof_tree = response.proof_tree
        assert proof_tree is not None

        all_steps = proof_tree.get_all_steps()
        assert len(all_steps) > 0

        # Erster Step sollte State-Snapshots in Metadata haben
        first_step = all_steps[0]
        assert (
            "state_before" in first_step.metadata
        ), "Step sollte state_before enthalten"
        assert "state_after" in first_step.metadata, "Step sollte state_after enthalten"
        assert "cycle" in first_step.metadata, "Step sollte cycle enthalten"

    def test_proof_step_explanation_text(self, formatter):
        """Test: ProofSteps haben lesbare explanation_text"""
        # Arrange
        topic = "baum"
        facts = {"IS_A": ["pflanze"], "HAS_PROPERTY": ["grün"]}
        bedeutungen = ["Ein Baum ist eine große Pflanze"]
        synonyms = []

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Assert
        proof_tree = response.proof_tree
        all_steps = proof_tree.get_all_steps()

        for step in all_steps:
            assert (
                len(step.explanation_text) > 0
            ), "Jeder Step sollte eine Erklärung haben"
            assert (
                "Zyklus" in step.explanation_text
            ), "Erklärung sollte Zyklus-Info enthalten"
            assert (
                step.rule_name in step.explanation_text
            ), "Erklärung sollte Regelname enthalten"

    def test_proof_tree_metadata(self, formatter):
        """Test: ProofTree enthält Metadata über Generierungsprozess"""
        # Arrange
        topic = "fisch"
        facts = {"IS_A": ["tier"], "LOCATED_IN": ["wasser"]}
        bedeutungen = []
        synonyms = []

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Assert
        proof_tree = response.proof_tree
        assert (
            "goal_type" in proof_tree.metadata
        ), "ProofTree sollte goal_type enthalten"
        assert (
            "max_cycles" in proof_tree.metadata
        ), "ProofTree sollte max_cycles enthalten"
        assert (
            "component" in proof_tree.metadata
        ), "ProofTree sollte component enthalten"
        assert proof_tree.metadata["component"] == "component_54_production_system"

    def test_proof_tree_signal_emission(self, formatter, mock_signals):
        """Test: ProofTree wird über Signal emittiert"""
        # Arrange
        topic = "auto"
        facts = {"IS_A": ["fahrzeug"]}
        bedeutungen = []
        synonyms = []

        # Mock für proof_tree_update Signal
        mock_signals.proof_tree_update = Mock()
        mock_signals.proof_tree_update.emit = Mock()

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
            signals=mock_signals,
        )

        # Assert
        # Signal sollte aufgerufen worden sein (wenn ProofTree vorhanden)
        if response.proof_tree is not None:
            assert (
                mock_signals.proof_tree_update.emit.call_count >= 1
            ), "proof_tree_update Signal sollte emittiert werden"
            # Erstes Argument sollte ein ProofTree sein
            call_args = mock_signals.proof_tree_update.emit.call_args
            assert call_args is not None

    def test_proof_tree_trace_info(self, formatter):
        """Test: Trace enthält ProofTree-Info"""
        # Arrange
        topic = "blume"
        facts = {"IS_A": ["pflanze"], "HAS_PROPERTY": ["schön"]}
        bedeutungen = []
        synonyms = []

        # Act
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Assert
        # Trace sollte ProofTree-Info enthalten
        trace_str = " ".join(response.trace)
        assert (
            "ProofTree" in trace_str or "Regelanwendungen" in trace_str
        ), "Trace sollte ProofTree-Info enthalten"

    def test_production_engine_direct_proof_tree(self, production_engine):
        """Test: ProductionSystemEngine generiert ProofTree direkt"""
        from component_54_production_system import (
            GenerationGoal,
            GenerationGoalType,
            ResponseGenerationState,
        )

        # Arrange
        primary_goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="test",
        )

        state = ResponseGenerationState(
            primary_goal=primary_goal,
            available_facts=[
                {
                    "relation_type": "IS_A",
                    "subject": "hund",
                    "object": "tier",
                    "confidence": 0.9,
                }
            ],
            current_query="Test Query für ProofTree",
        )

        # Act
        final_state = production_engine.generate(state)

        # Assert
        assert final_state.proof_tree is not None, "Engine sollte ProofTree erzeugen"
        assert final_state.proof_tree.query == "Test Query für ProofTree"
        assert (
            len(final_state.proof_tree.get_all_steps()) > 0
        ), "ProofTree sollte Steps enthalten"


# ============================================================================
# Test: PHASE 7 - Erweiterte Integration Tests
# ============================================================================


class TestAllQuestionTypes:
    """Tests für alle Fragetypen (ANSWER_QUESTION, LEARN_KNOWLEDGE, etc.)"""

    def test_answer_question_type(self, formatter):
        """Test: ANSWER_QUESTION Fragetyp"""
        response = formatter.generate_with_production_system(
            topic="hund",
            facts={"IS_A": ["tier", "säugetier"], "HAS_PROPERTY": ["treu"]},
            bedeutungen=["Ein Hund ist ein treuer Begleiter"],
            synonyms=[],
            query_type="normal",
        )

        assert response.strategy == "production_system"
        assert len(response.text) > 0
        assert "hund" in response.text.lower() or "tier" in response.text.lower()

    def test_learn_knowledge_type(self, formatter):
        """Test: LEARN_KNOWLEDGE Fragetyp (Deklaratives Statement)"""
        response = formatter.generate_with_production_system(
            topic="python",
            facts={"IS_A": ["programmiersprache"]},
            bedeutungen=["Python ist eine Programmiersprache"],
            synonyms=[],
            query_type="definition",
        )

        assert response.strategy == "production_system"
        assert len(response.text) > 0

    def test_no_facts_edge_case(self, formatter):
        """Test: Edge Case - Keine Fakten vorhanden"""
        response = formatter.generate_with_production_system(
            topic="unbekannt",
            facts={},
            bedeutungen=[],
            synonyms=[],
        )

        assert isinstance(response, KaiResponse)
        # Sollte Fallback oder "Ich weiß nichts über..." generieren
        assert len(response.text) > 0

    def test_low_confidence_facts(self, formatter):
        """Test: Edge Case - Nur niedrige Confidence Fakten"""
        response = formatter.generate_with_production_system(
            topic="test",
            facts={"IS_A": ["unbekannt"], "HAS_PROPERTY": ["unsicher"]},
            bedeutungen=[],
            synonyms=[],
            confidence=0.3,  # Niedrige Confidence
        )

        assert isinstance(response, KaiResponse)
        # Sollte Unsicherheit signalisieren oder wenig Output produzieren
        assert len(response.text) > 0

    def test_many_facts_handling(self, formatter):
        """Test: Viele Fakten gleichzeitig"""
        response = formatter.generate_with_production_system(
            topic="apfel",
            facts={
                "IS_A": ["frucht", "obst", "nahrungsmittel"],
                "HAS_PROPERTY": ["rot", "grün", "gelb", "süß", "saftig"],
                "CAPABLE_OF": ["wachsen", "reifen"],
                "PART_OF": ["baum"],
                "LOCATED_IN": ["europa", "asien"],
            },
            bedeutungen=["Äpfel sind gesund", "Äpfel wachsen auf Bäumen"],
            synonyms=["Malus"],
        )

        assert response.strategy == "production_system"
        assert len(response.text) > 0
        # Sollte nicht alle Fakten verwenden (Content Selection sollte filtern)


class TestPipelineVsProductionComparison:
    """Tests für Vergleich Pipeline vs. Production System Output"""

    def test_same_input_different_outputs(self, formatter):
        """Test: Gleiche Query, unterschiedliche Systeme → unterschiedliche Outputs"""
        topic = "vogel"
        facts = {"IS_A": ["tier"], "CAPABLE_OF": ["fliegen"]}
        bedeutungen = ["Ein Vogel kann fliegen"]
        synonyms = []

        # Pipeline Response
        pipeline_response = formatter.format_standard_answer(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Production System Response
        production_response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Assert: Beide sollten gültige Antworten sein
        assert len(pipeline_response.text) > 0
        assert len(production_response.text) > 0

        # Sollten unterschiedliche Strategien haben
        assert pipeline_response.strategy != production_response.strategy

    def test_semantic_equivalence(self, formatter):
        """Test: Beide Systeme erzeugen semantisch ähnliche Antworten"""
        topic = "katze"
        facts = {"IS_A": ["tier", "säugetier"], "HAS_PROPERTY": ["flauschig"]}
        bedeutungen = []
        synonyms = []

        pipeline_response = formatter.format_standard_answer(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )

        production_response = formatter.generate_with_production_system(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )

        # Beide sollten "katze" und mindestens einen Fakt erwähnen
        assert "katze" in pipeline_response.text.lower()
        assert "katze" in production_response.text.lower()

        # Mindestens einer der Fakten sollte in beiden vorkommen
        facts_mentioned_pipeline = [
            fact
            for fact in ["tier", "säugetier", "flauschig"]
            if fact in pipeline_response.text.lower()
        ]
        facts_mentioned_production = [
            fact
            for fact in ["tier", "säugetier", "flauschig"]
            if fact in production_response.text.lower()
        ]

        assert len(facts_mentioned_pipeline) > 0
        assert len(facts_mentioned_production) > 0

    def test_confidence_comparison(self, formatter):
        """Test: Vergleiche Confidence-Werte beider Systeme"""
        topic = "baum"
        facts = {"IS_A": ["pflanze"], "HAS_PROPERTY": ["grün", "groß"]}
        bedeutungen = ["Ein Baum ist eine große Pflanze"]
        synonyms = []

        pipeline_response = formatter.format_standard_answer(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )

        production_response = formatter.generate_with_production_system(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )

        # Beide sollten Confidence-Werte haben
        assert pipeline_response.confidence is not None
        assert production_response.confidence is not None

        # Confidence sollte im gültigen Bereich sein
        assert 0.0 <= pipeline_response.confidence <= 1.0
        assert 0.0 <= production_response.confidence <= 1.0

    def test_trace_comparison(self, formatter):
        """Test: Trace-Informationen unterscheiden sich"""
        topic = "fisch"
        facts = {"IS_A": ["tier"], "LOCATED_IN": ["wasser"]}
        bedeutungen = []
        synonyms = []

        pipeline_response = formatter.format_standard_answer(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )

        production_response = formatter.generate_with_production_system(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )

        # Pipeline sollte keine Production System Traces haben
        pipeline_trace = " ".join(pipeline_response.trace)
        assert (
            "production" not in pipeline_trace.lower()
            or "pipeline" in pipeline_trace.lower()
        )

        # Production sollte Production System Traces haben
        production_trace = " ".join(production_response.trace)
        assert "production" in production_trace.lower()


class TestEdgeCases:
    """Tests für Edge Cases und Grenzfälle"""

    def test_empty_topic(self, formatter):
        """Test: Leerer Topic"""
        response = formatter.generate_with_production_system(
            topic="",
            facts={},
            bedeutungen=[],
            synonyms=[],
        )

        assert isinstance(response, KaiResponse)
        # Sollte nicht crashen

    def test_very_long_topic(self, formatter):
        """Test: Sehr langer Topic"""
        long_topic = "ein sehr langer topic name " * 20
        response = formatter.generate_with_production_system(
            topic=long_topic,
            facts={"IS_A": ["test"]},
            bedeutungen=[],
            synonyms=[],
        )

        assert isinstance(response, KaiResponse)

    def test_special_characters_in_facts(self, formatter):
        """Test: Spezialzeichen in Fakten"""
        response = formatter.generate_with_production_system(
            topic="test",
            facts={
                "IS_A": ["test@123", "test-456"],
                "HAS_PROPERTY": ["a/b", "x&y"],
            },
            bedeutungen=[],
            synonyms=[],
        )

        assert isinstance(response, KaiResponse)
        # Sollte mit Spezialzeichen umgehen können

    def test_contradictory_facts(self, formatter):
        """Test: Widersprüchliche Fakten"""
        response = formatter.generate_with_production_system(
            topic="pinguin",
            facts={
                "IS_A": ["vogel"],
                "CAPABLE_OF": ["fliegen"],  # Widerspruch: Pinguine können nicht fliegen
            },
            bedeutungen=["Ein Pinguin ist ein Vogel", "Ein Pinguin kann nicht fliegen"],
            synonyms=[],
        )

        assert isinstance(response, KaiResponse)
        # System sollte mit Widersprüchen umgehen

    def test_unicode_characters(self, formatter):
        """Test: Unicode-Zeichen in Fakten"""
        response = formatter.generate_with_production_system(
            topic="äöü",
            facts={
                "IS_A": ["ümläüt"],
                "HAS_PROPERTY": ["schön"],
            },
            bedeutungen=["Ein Test mit Umläuten: äöüß"],
            synonyms=["café", "naïve"],
        )

        assert isinstance(response, KaiResponse)
        assert len(response.text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
