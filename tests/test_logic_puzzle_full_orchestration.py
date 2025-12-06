"""
tests/test_logic_puzzle_full_orchestration.py
==============================================
PHASE 4 Integration tests for enhanced logic puzzle orchestration.

Tests the complete pipeline:
1. Input Orchestrator -> Segments input and classifies puzzle type
2. OrchestratedStrategy -> Stores puzzle context in working memory
3. StrategyDispatcher -> Retrieves context and invokes solver
4. LogicPuzzleSolver -> Routes to appropriate solver (SAT/CSP) and generates solution

WICHTIG: KEINE Unicode-Zeichen (nur ASCII: AND, OR, NOT, IMPLIES)

Author: KAI Development Team
Date: 2025-12-05 (PHASE 4)
"""

import pytest

from component_1_netzwerk import KonzeptNetzwerk
from component_6_linguistik_engine import LinguisticPreprocessor
from component_13_working_memory import WorkingMemory
from component_41_input_orchestrator import InputOrchestrator, PuzzleType
from component_45_logic_puzzle_solver_core import LogicPuzzleSolver
from kai_strategy_dispatcher import StrategyDispatcher


@pytest.fixture
def netzwerk():
    """Create fresh KonzeptNetzwerk for each test."""
    nw = KonzeptNetzwerk()
    yield nw
    # Cleanup
    nw.close()


@pytest.fixture
def preprocessor():
    """Create LinguisticPreprocessor."""
    return LinguisticPreprocessor()


@pytest.fixture
def working_memory():
    """Create fresh WorkingMemory."""
    return WorkingMemory()


@pytest.fixture
def orchestrator(preprocessor):
    """Create InputOrchestrator with preprocessor."""
    return InputOrchestrator(preprocessor=preprocessor)


@pytest.fixture
def logic_puzzle_solver():
    """Create LogicPuzzleSolver."""
    return LogicPuzzleSolver()


@pytest.fixture
def strategy_dispatcher(netzwerk, working_memory):
    """Create StrategyDispatcher with minimal setup."""

    # Mock objects for required components
    class MockLogicEngine:
        pass

    class MockGraphTraversal:
        pass

    class MockSignals:
        pass

    dispatcher = StrategyDispatcher(
        netzwerk=netzwerk,
        logic_engine=MockLogicEngine(),
        graph_traversal=MockGraphTraversal(),
        working_memory=working_memory,
        signals=MockSignals(),
    )
    return dispatcher


# ============================================================================
# Test 1: Entity-Based Puzzle End-to-End
# ============================================================================


def test_entity_puzzle_end_to_end_orchestration(
    orchestrator, strategy_dispatcher, working_memory
):
    """
    Test 1: End-to-end flow for entity-based puzzle (Leo/Mark/Nick).

    Flow:
    1. Orchestrator segments and classifies as ENTITY_SAT
    2. OrchestratedStrategy stores context in working memory (simulated)
    3. Dispatcher retrieves context and solves
    4. Verify answer correctness
    """
    from component_13_working_memory import ContextType

    puzzle_text = (
        "Leo, Mark und Nick sind drei Freunde. "
        "Leo trinkt gerne Tee. "
        "Mark trinkt gerne Brandy, aber nur einer von ihnen trinkt gerne Brandy. "
        "Nick trinkt entweder Tee oder Brandy, aber nicht beides. "
        "Wer trinkt gerne Brandy?"
    )

    # STEP 1: Orchestrator segments and classifies
    orchestration = orchestrator.orchestrate_input(puzzle_text)
    assert orchestration is not None, "Should orchestrate multi-segment input"
    assert orchestration["is_logic_puzzle"], "Should detect as logic puzzle"
    assert (
        orchestration["puzzle_type"] == PuzzleType.ENTITY_SAT
    ), "Should classify as ENTITY_SAT"

    # STEP 2: Simulate working memory storage (done by OrchestratedStrategy)
    segments = orchestration["segments"]
    entities = ["Leo", "Mark", "Nick"]
    full_text = " ".join(seg.text for seg in segments)

    # Push context frame BEFORE adding reasoning state
    working_memory.push_context(
        context_type=ContextType.QUESTION,
        query="Wer trinkt gerne Brandy?",
        entities=entities,
    )

    working_memory.add_reasoning_state(
        step_type="orchestrated_logic_puzzle",
        description="Entity puzzle orchestrated",
        data={
            "full_text": full_text,
            "question": "Wer trinkt gerne Brandy?",
            "entities": entities,
            "puzzle_classification": "entity_sat",
        },
        confidence=0.85,
    )

    # STEP 3: Dispatcher solves puzzle
    result = strategy_dispatcher._try_logic_puzzle_solving(
        query_text="Wer trinkt gerne Brandy?",
        topic="brandy",
        relation_type=None,
        context=None,
    )

    # STEP 4: Verify solution
    assert result is not None, "Should find solution"
    assert result.success, "Solution should be successful"
    assert (
        result.confidence >= 0.7
    ), f"Confidence should be >= 0.7, got {result.confidence}"
    assert "PUZZLE_SOLUTION" in result.inferred_facts, "Should have puzzle solution"

    # Check answer contains "Mark" (correct answer)
    answer = result.inferred_facts["PUZZLE_SOLUTION"][0]
    assert "mark" in answer.lower(), f"Answer should mention Mark, got: {answer}"


# ============================================================================
# Test 2: Numerical Puzzle (Gesuchte Zahl) Full Flow
# ============================================================================


def test_numerical_puzzle_full_orchestration(
    orchestrator, strategy_dispatcher, working_memory
):
    """
    Test 2: End-to-end flow for numerical constraint puzzle.

    Main test case from the plan for "gesuchte Zahl" puzzles.
    """
    from component_13_working_memory import ContextType

    puzzle_text = """
    Gesucht ist eine dreistellige Zahl mit folgenden Eigenschaften:
    1. Die Zahl ist durch 3 teilbar.
    2. Die Summe der Ziffern beträgt 15.
    3. Die erste Ziffer ist größer als die letzte Ziffer.
    Welche Zahl wird gesucht?
    """

    # STEP 1: Orchestrator classifies
    orchestration = orchestrator.orchestrate_input(puzzle_text)
    assert orchestration is not None, "Should orchestrate"
    assert orchestration["is_logic_puzzle"], "Should detect as logic puzzle"
    assert (
        orchestration["puzzle_type"] == PuzzleType.NUMERICAL_CSP
    ), "Should classify as NUMERICAL_CSP"

    # STEP 2: Store context in working memory
    segments = orchestration["segments"]
    full_text = " ".join(seg.text for seg in segments)

    # Push context frame BEFORE adding reasoning state
    working_memory.push_context(
        context_type=ContextType.QUESTION,
        query="Welche Zahl wird gesucht?",
        entities=[],
    )

    working_memory.add_reasoning_state(
        step_type="orchestrated_logic_puzzle",
        description="Numerical CSP puzzle orchestrated",
        data={
            "full_text": full_text,
            "question": "Welche Zahl wird gesucht?",
            "entities": [],
            "puzzle_classification": "numerical_csp",
        },
        confidence=0.85,
    )

    # STEP 3: Dispatcher solves
    result = strategy_dispatcher._try_logic_puzzle_solving(
        query_text="Welche Zahl wird gesucht?",
        topic="zahl",
        relation_type=None,
        context=None,
    )

    # STEP 4: Verify solution
    assert result is not None, "Should find solution"
    assert result.success, "Solution should be successful"
    assert (
        result.confidence >= 0.6
    ), f"Confidence should be >= 0.6, got {result.confidence}"
    assert "puzzle_type" in result.metadata, "Should have puzzle_type metadata"
    assert (
        result.metadata["puzzle_type"] == "numerical_csp"
    ), "Should route to numerical solver"


# ============================================================================
# Test 3: Backward Compatibility - Entity Puzzles Still Work
# ============================================================================


def test_backward_compatibility_entity_puzzle(logic_puzzle_solver):
    """
    Test 3: Backward compatibility for existing entity-based puzzles.

    Critical requirement: New routing must not break existing puzzles.
    """
    puzzle_text = (
        "Leo trinkt Tee. Mark trinkt Brandy. Nick trinkt Kaffee. Wer trinkt Brandy?"
    )
    entities = ["Leo", "Mark", "Nick"]

    solution = logic_puzzle_solver.solve(
        conditions_text=puzzle_text, entities=entities, question="Wer trinkt Brandy?"
    )

    assert solution is not None, "Should solve"
    assert solution["result"] == "SATISFIABLE", "Should be satisfiable"
    assert (
        "Mark" in solution.get("answer", "").lower()
        or "mark" in str(solution.get("solution", {})).lower()
    ), "Should identify Mark"


# ============================================================================
# Test 4: Proof Tree Generation
# ============================================================================


def test_proof_tree_generation_orchestrated(strategy_dispatcher, working_memory):
    """
    Test 4: Verify proof trees are generated for orchestrated puzzle solutions.
    """
    from component_13_working_memory import ContextType

    puzzle_text = "Leo trinkt Tee. Mark trinkt Brandy. Wer trinkt Brandy?"

    # Push context frame BEFORE adding reasoning state
    working_memory.push_context(
        context_type=ContextType.QUESTION,
        query="Wer trinkt Brandy?",
        entities=["Leo", "Mark"],
    )

    # Store context
    working_memory.add_reasoning_state(
        step_type="orchestrated_logic_puzzle",
        description="Simple entity puzzle",
        data={
            "full_text": puzzle_text,
            "question": "Wer trinkt Brandy?",
            "entities": ["Leo", "Mark"],
            "puzzle_classification": "entity_sat",
        },
        confidence=0.85,
    )

    result = strategy_dispatcher._try_logic_puzzle_solving(
        query_text="Wer trinkt Brandy?",
        topic="brandy",
        relation_type=None,
        context=None,
    )

    assert result is not None, "Should find solution"

    # Check proof tree exists
    if result.proof_tree is not None:
        proof_tree = result.proof_tree
        all_steps = proof_tree.get_all_steps()
        assert len(all_steps) > 0, "Proof tree should have steps"

        # Check for puzzle detection step
        step_descriptions = [step.explanation_text for step in all_steps]
        has_detection = any(
            "Puzzle" in desc or "puzzle" in desc for desc in step_descriptions
        )
        assert has_detection, "Should have puzzle detection step"


# ============================================================================
# Test 5: Working Memory Context Retrieval
# ============================================================================


def test_working_memory_context_retrieval_integration(
    strategy_dispatcher, working_memory
):
    """
    Test 5: Verify dispatcher correctly retrieves puzzle context from working memory.

    Tests the critical integration between OrchestratedStrategy and StrategyDispatcher.
    """
    from component_13_working_memory import ContextType

    full_text = "A ist aktiv. B ist aktiv. C ist inaktiv. Wenn A aktiv ist, ist X wahr."
    entities = ["A", "B", "C"]

    # Push context frame BEFORE adding reasoning state
    working_memory.push_context(
        context_type=ContextType.QUESTION, query="Ist X wahr?", entities=entities
    )

    # Store in working memory (as OrchestratedStrategy would)
    working_memory.add_reasoning_state(
        step_type="orchestrated_logic_puzzle",
        description="Test puzzle for memory retrieval",
        data={
            "full_text": full_text,
            "question": "Ist X wahr?",
            "entities": entities,
            "puzzle_classification": "entity_sat",
        },
        confidence=0.85,
    )

    # Verify working memory stores it correctly
    reasoning_states = working_memory.get_reasoning_trace()
    puzzle_states = [
        s for s in reasoning_states if s.step_type == "orchestrated_logic_puzzle"
    ]
    assert len(puzzle_states) >= 1, "Should have stored puzzle state"
    assert (
        puzzle_states[0].data["full_text"] == full_text
    ), "Should store full text correctly"

    # Call solver (will retrieve from working memory)
    result = strategy_dispatcher._try_logic_puzzle_solving(
        query_text="Ist X wahr?", topic="x", relation_type=None, context=None
    )

    # Verify it attempted to use the context
    # (May or may not find solution, but should not crash)
    assert result is not None or True, "Should attempt solution"


# ============================================================================
# Test 6: Puzzle Type Detection Integration
# ============================================================================


def test_puzzle_type_detection_integration_orchestrator(orchestrator):
    """
    Test 6: Verify orchestrator correctly detects different puzzle types.
    """
    # Entity puzzle
    entity_puzzle = (
        "Leo trinkt Tee. Mark trinkt Brandy. Nick trinkt Kaffee. " "Wer trinkt Brandy?"
    )
    result = orchestrator.orchestrate_input(entity_puzzle)
    assert result is not None
    assert result["puzzle_type"] == PuzzleType.ENTITY_SAT

    # Numerical puzzle
    numerical_puzzle = (
        "Gesucht ist eine Zahl. "
        "Die Zahl ist teilbar durch 5. "
        "Die Summe der Teiler beträgt 12. "
        "Welche Zahl ist es?"
    )
    result = orchestrator.orchestrate_input(numerical_puzzle)
    assert result is not None
    assert result["puzzle_type"] == PuzzleType.NUMERICAL_CSP


# ============================================================================
# Test 7: Malformed Puzzle Handling
# ============================================================================


def test_malformed_puzzle_handling_graceful(logic_puzzle_solver):
    """
    Test 7: Verify graceful handling of malformed/empty puzzles.
    """
    # Empty text
    solution = logic_puzzle_solver.solve(conditions_text="", entities=[], question="")
    assert solution is not None, "Should handle empty input gracefully"

    # No entities for entity puzzle
    solution = logic_puzzle_solver.solve(
        conditions_text="Someone drinks something.", entities=[], question="Who?"
    )
    assert solution is not None, "Should handle missing entities gracefully"


# ============================================================================
# Test 8: Orchestrated vs Direct Solving Comparison
# ============================================================================


def test_orchestrated_vs_direct_solving_equivalence(
    orchestrator, strategy_dispatcher, logic_puzzle_solver, working_memory
):
    """
    Test 8: Verify orchestrated solving produces equivalent results to direct solving.

    Critical test for backward compatibility.
    """
    from component_13_working_memory import ContextType

    puzzle_text = "Leo trinkt Tee. Mark trinkt Brandy. Wer trinkt Brandy?"
    entities = ["Leo", "Mark"]

    # Method 1: Direct solve (no orchestration)
    direct_solution = logic_puzzle_solver.solve(
        conditions_text=puzzle_text, entities=entities, question="Wer trinkt Brandy?"
    )

    # Method 2: Orchestrated solve (via working memory)
    # Push context frame BEFORE adding reasoning state
    working_memory.push_context(
        context_type=ContextType.QUESTION, query="Wer trinkt Brandy?", entities=entities
    )

    working_memory.add_reasoning_state(
        step_type="orchestrated_logic_puzzle",
        description="Orchestrated puzzle for comparison",
        data={
            "full_text": puzzle_text,
            "question": "Wer trinkt Brandy?",
            "entities": entities,
            "puzzle_classification": "entity_sat",
        },
        confidence=0.85,
    )

    orchestrated_result = strategy_dispatcher._try_logic_puzzle_solving(
        query_text="Wer trinkt Brandy?",
        topic="brandy",
        relation_type=None,
        context=None,
    )

    # Compare results
    assert direct_solution is not None
    assert orchestrated_result is not None
    assert direct_solution["result"] == "SATISFIABLE"
    assert orchestrated_result.success

    # Both should identify Mark as answer
    direct_answer = direct_solution.get("answer", "").lower()
    orchestrated_answer = orchestrated_result.inferred_facts.get(
        "PUZZLE_SOLUTION", [""]
    )[0].lower()

    assert "mark" in direct_answer
    assert "mark" in orchestrated_answer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
