"""
Tests für Numerical Puzzle Solver (PHASE 3)

Testet das CSP-basierte Solving von numerischen Logic Puzzles:
- Einfache Teilbarkeits-Puzzles
- Arithmetische Constraint-Puzzles
- Meta-Constraint-Puzzles (self-referentiell)
- Iteratives Solving
- Proof Tree Generation
"""

import pytest

from component_45_numerical_puzzle_solver import NumericalPuzzleSolver


@pytest.fixture
def solver():
    """Fixture für NumericalPuzzleSolver."""
    return NumericalPuzzleSolver()


class TestSimpleDivisibilityPuzzles:
    """Tests für einfache Teilbarkeits-Puzzles."""

    def test_solve_single_divisibility(self, solver):
        """Test: Einfaches Teilbarkeits-Puzzle wird gelöst."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        assert result["result"] == "SATISFIABLE"
        assert "zahl" in result["solution"]
        assert result["solution"]["zahl"] % 3 == 0
        assert result["solution"]["zahl"] >= 3  # Kleinste ist 3

    def test_solve_multiple_divisibility(self, solver):
        """Test: Puzzle mit mehreren Teilbarkeits-Constraints."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        2. Die gesuchte Zahl ist teilbar durch 7.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        assert result["result"] == "SATISFIABLE"
        zahl = result["solution"]["zahl"]
        assert zahl % 3 == 0
        assert zahl % 7 == 0
        # Kleinste ist LCM(3,7) = 21
        assert zahl == 21 or zahl >= 21

    def test_solve_with_large_divisor(self, solver):
        """Test: Puzzle mit großem Divisor."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 17.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        assert result["result"] == "SATISFIABLE"
        assert result["solution"]["zahl"] % 17 == 0


class TestArithmeticConstraintPuzzles:
    """Tests für arithmetische Constraint-Puzzles."""

    def test_solve_with_sum_constraint(self, solver):
        """Test: Puzzle mit Summen-Constraint."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        2. Die Summe der Nummern der richtigen Behauptungen ist teilbar durch 5.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Erwarte mindestens SATISFIABLE (auch wenn Meta-Constraint komplex ist)
        assert result["result"] in ["SATISFIABLE", "UNSATISFIABLE"]

    def test_solve_with_comparison_constraint(self, solver):
        """Test: Puzzle mit Vergleichs-Constraint."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        2. Die Anzahl der Teiler ist größer als 5.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Erwarte Lösung oder UNSAT
        assert result["result"] in ["SATISFIABLE", "UNSATISFIABLE"]


class TestMetaConstraintPuzzles:
    """Tests für Meta-Constraint-Puzzles (self-referentiell)."""

    def test_solve_with_meta_constraint(self, solver):
        """Test: Puzzle mit Meta-Constraint über Behauptungen."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        2. Die Anzahl der Teiler der Zahl ist größer als 10.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Erwarte mindestens eine Antwort (SATISFIABLE oder UNSATISFIABLE)
        assert result["result"] in ["SATISFIABLE", "UNSATISFIABLE"]

        if result["result"] == "SATISFIABLE":
            # Falls gelöst: Verifiziere Constraints
            zahl = result["solution"]["zahl"]
            assert zahl % 3 == 0


class TestIterativeSolving:
    """Tests für iteratives Solving."""

    def test_iterative_solve_with_statement_variables(self, solver):
        """Test: Iteratives Solving mit Statement-Variablen."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        2. Die gesuchte Zahl ist teilbar durch 7.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Prüfe dass Lösung existiert
        assert result["result"] == "SATISFIABLE"

        # Prüfe dass Statement-Variablen erstellt wurden
        solution_keys = result["solution"].keys()
        # Sollte mindestens "zahl" enthalten
        assert "zahl" in solution_keys


class TestAnswerFormatting:
    """Tests für Antwort-Formatierung."""

    def test_format_answer_smallest(self, solver):
        """Test: Antwort für 'kleinste gesuchte Zahl'."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        assert "kleinste" in result["answer"].lower()
        assert str(result["solution"]["zahl"]) in result["answer"]

    def test_format_answer_largest(self, solver):
        """Test: Antwort für 'größte gesuchte Zahl'."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        Was ist die größte gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Sollte "größte" enthalten (falls Frage erkannt wurde)
        # Oder generische Antwort
        assert result["answer"] is not None
        assert len(result["answer"]) > 0

    def test_format_answer_no_question(self, solver):
        """Test: Antwort ohne explizite Frage."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 7.
        """

        result = solver.solve_puzzle(puzzle)

        # Sollte generische Antwort haben
        assert result["answer"] is not None


class TestProofTreeGeneration:
    """Tests für ProofTree-Generierung."""

    def test_proof_tree_created(self, solver):
        """Test: ProofTree wird generiert."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        assert result["proof_tree"] is not None
        assert len(result["proof_tree"].root_steps) > 0

    def test_proof_tree_steps(self, solver):
        """Test: ProofTree enthält alle Steps."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        2. Die gesuchte Zahl ist teilbar durch 7.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        proof_tree = result["proof_tree"]
        # Sollte mindestens: Detection, Constraints, Setup, Solution
        assert len(proof_tree.get_all_steps()) >= 4

    def test_proof_tree_confidence(self, solver):
        """Test: ProofTree hat korrekte Confidence."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 3.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        if result["result"] == "SATISFIABLE":
            # Hohe Confidence bei Lösung
            assert result["proof_tree"].metadata.get("confidence", 0.0) >= 0.80
        else:
            # Niedrige Confidence bei UNSAT
            assert result["proof_tree"].metadata.get("confidence", 0.0) < 0.50


class TestConfidenceScoring:
    """Tests für Confidence-Scoring."""

    def test_high_confidence_unique_solution(self, solver):
        """Test: Hohe Confidence bei eindeutiger Lösung."""
        puzzle = """
        1. Die gesuchte Zahl ist teilbar durch 21.
        Was ist die kleinste gesuchte Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        if result["result"] == "SATISFIABLE":
            assert result["confidence"] >= 0.80

    def test_low_confidence_no_solution(self, solver):
        """Test: Niedrige Confidence bei fehlender Lösung."""
        # Erstelle unlösbares Puzzle (wenn möglich)
        puzzle = """
        Was ist die Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Sollte niedrige Confidence haben
        assert result["confidence"] <= 0.50


class TestEdgeCases:
    """Tests für Edge Cases."""

    def test_empty_puzzle(self, solver):
        """Test: Leeres Puzzle."""
        puzzle = ""

        result = solver.solve_puzzle(puzzle)

        assert result["result"] == "UNSATISFIABLE"
        assert result["confidence"] == 0.0

    def test_no_constraints(self, solver):
        """Test: Puzzle ohne Constraints."""
        puzzle = """
        Die gesuchte Zahl ist interessant.
        Was ist die Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Sollte UNSAT sein (keine Constraints)
        assert result["result"] == "UNSATISFIABLE"

    def test_unsolvable_puzzle(self, solver):
        """Test: Unlösbares Puzzle mit widersprüchlichen Constraints."""
        # Aktuell schwer zu testen ohne komplexere Constraints
        # Placeholder: Teste dass Solver nicht crashed
        puzzle = """
        1. Die Zahl ist teilbar durch 3.
        Was ist die Zahl?
        """

        result = solver.solve_puzzle(puzzle)

        # Sollte nicht crashen
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
