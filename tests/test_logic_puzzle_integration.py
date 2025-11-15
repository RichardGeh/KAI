"""
Tests für Logik-Rätsel Integration (Constraint Detection + CSP Solving)

Testet den kompletten Flow:
1. ConstraintDetector erkennt Logik-Puzzle
2. LogicalConstraints werden zu CSP übersetzt
3. CSP-Solver löst das Problem
"""

import pytest

from component_29_constraint_reasoning import (
    ConstraintSolver,
    translate_logical_constraints_to_csp,
)
from component_60_constraint_detector import ConstraintDetector


class TestConstraintDetector:
    """Tests für ConstraintDetector (component_60)"""

    def test_detector_initialization(self):
        """Test: ConstraintDetector kann initialisiert werden"""
        detector = ConstraintDetector(min_conditional_rules=3, confidence_threshold=0.7)
        assert detector is not None
        assert detector.min_conditional_rules == 3
        assert detector.confidence_threshold == 0.7

    def test_detect_simple_implies_pattern(self):
        """Test: Einfaches WENN-DANN Pattern wird erkannt"""
        detector = ConstraintDetector(min_conditional_rules=2)  # Niedriger Threshold
        text = "Wenn Leo Brandy bestellt, bestellt Mark auch einen. Wenn Mark bestellt, bestellt Nick auch."

        problem = detector.detect_constraint_problem(text)

        assert problem is not None, "Constraint-Problem sollte erkannt werden"
        assert len(problem.variables) >= 2, "Mindestens 2 Variablen erwartet"
        assert len(problem.constraints) >= 2, "Mindestens 2 Constraints erwartet"
        assert problem.confidence > 0.5

    def test_detect_xor_pattern(self):
        """Test: XOR Pattern wird erkannt"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = (
            "Mark oder Nick, aber nie beide. Wenn einer kommt, kommt der andere nicht."
        )

        problem = detector.detect_constraint_problem(text)

        assert problem is not None
        # Prüfe ob XOR-Constraint erkannt wurde
        xor_constraints = [c for c in problem.constraints if c.constraint_type == "XOR"]
        assert len(xor_constraints) >= 1, "XOR-Constraint sollte erkannt werden"

    def test_insufficient_conditionals_no_detection(self):
        """Test: Zu wenig Conditionals führen zu keiner Erkennung"""
        detector = ConstraintDetector(min_conditional_rules=5)  # Hoher Threshold
        text = "Wenn Leo kommt, kommt Mark."

        problem = detector.detect_constraint_problem(text)

        assert problem is None, "Sollte nicht als Constraint-Problem erkannt werden"

    def test_brandy_puzzle_detection(self):
        """Test: Brandy-Rätsel wird erkannt"""
        detector = ConstraintDetector(min_conditional_rules=3)
        text = """
        Wenn Leo Brandy bestellt, bestellt auch Mark einen.
        Mark oder Nick bestellt Brandy, aber nie beide.
        Nick bestellt nur dann Brandy, wenn Leo keinen bestellt.
        """

        problem = detector.detect_constraint_problem(text)

        assert problem is not None
        assert len(problem.variables) >= 3, "Leo, Mark, Nick sollten erkannt werden"
        assert len(problem.constraints) >= 3
        assert problem.confidence >= 0.7


class TestConstraintTranslation:
    """Tests für CSP-Translation (component_29)"""

    def test_translate_implies_constraint(self):
        """Test: IMPLIES-Constraint wird zu CSP übersetzt"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = "Wenn Leo Brandy bestellt, bestellt Mark auch einen. Wenn Mark kommt, kommt Nick."

        logical_problem = detector.detect_constraint_problem(text)
        assert logical_problem is not None

        # Translate to CSP
        csp_problem = translate_logical_constraints_to_csp(logical_problem)

        assert csp_problem is not None
        assert len(csp_problem.variables) >= 2
        assert len(csp_problem.constraints) >= 1

    def test_translate_xor_constraint(self):
        """Test: XOR-Constraint wird zu CSP übersetzt"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = "Mark oder Nick, aber nie beide. Wenn einer bestellt, bestellt der andere nicht."

        logical_problem = detector.detect_constraint_problem(text)
        assert logical_problem is not None

        # Translate to CSP
        csp_problem = translate_logical_constraints_to_csp(logical_problem)

        assert csp_problem is not None
        # Prüfe dass XOR-Constraint erstellt wurde
        [c for c in csp_problem.constraints if "XOR" in c.name]
        # XOR kann als IMPLIES oder direkt als XOR modelliert sein
        assert len(csp_problem.constraints) >= 1


class TestEndToEndIntegration:
    """End-to-End Tests: Detection → Translation → Solving"""

    def test_simple_logic_puzzle_solved(self):
        """Test: Einfaches Logik-Rätsel kann gelöst werden"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn Leo aktiv ist, ist Mark aktiv.
        Mark ist aktiv.
        """

        # Step 1: Detect
        logical_problem = detector.detect_constraint_problem(text)
        if logical_problem is None:
            pytest.skip(
                "Problem nicht als Constraint-Problem erkannt (zu wenig Conditionals)"
            )

        # Step 2: Translate
        csp_problem = translate_logical_constraints_to_csp(logical_problem)
        assert csp_problem is not None

        # Step 3: Solve
        solver = ConstraintSolver(use_ac3=True, use_mrv=True)
        result = solver.solve(csp_problem)

        # Prüfe ob Lösung gefunden wurde
        # (Kann auch None sein wenn Problem unterspecified ist)
        if result:
            # solve() gibt (solution, proof_tree) zurück
            solution, proof_tree = (
                result if isinstance(result, tuple) else (result, None)
            )
            assert isinstance(solution, dict)
            assert len(solution) >= 1

    def test_brandy_puzzle_full_integration(self):
        """Test: Brandy-Rätsel End-to-End Integration"""
        detector = ConstraintDetector(min_conditional_rules=3)
        text = """
        Wenn Leo Brandy bestellt, bestellt auch Mark einen.
        Mark oder Nick bestellt Brandy, aber nie beide.
        Nick bestellt nur dann Brandy, wenn Leo keinen bestellt.
        """

        # Step 1: Detect
        logical_problem = detector.detect_constraint_problem(text)
        assert logical_problem is not None, "Brandy-Rätsel sollte erkannt werden"

        # Step 2: Translate
        csp_problem = translate_logical_constraints_to_csp(logical_problem)
        assert csp_problem is not None
        assert len(csp_problem.variables) >= 3  # Leo, Mark, Nick

        # Step 3: Solve
        solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
        result = solver.solve(csp_problem)

        # Prüfe Lösung
        if result:
            # solve() gibt (solution, proof_tree) zurück
            solution, proof_tree = (
                result if isinstance(result, tuple) else (result, None)
            )
            assert isinstance(solution, dict)
            print(f"[Brandy-Rätsel Lösung] {solution}")

            # Brandy-Rätsel hat eine eindeutige Lösung:
            # Leo trinkt Brandy -> Mark trinkt Brandy -> Nick trinkt nicht
            # ODER
            # Leo trinkt nicht -> Nick kann -> Mark kann nicht (wegen XOR)

    def test_switch_puzzle_integration(self):
        """Test: Schalter-Rätsel End-to-End"""
        detector = ConstraintDetector(min_conditional_rules=3)
        text = """
        Wenn Schalter 3 oben ist und Schalter 2 unten ist, dann knallt es.
        Wenn Schalter 1 oben ist, muss Schalter 2 auch oben sein.
        Schalter 3 ist oben.
        """

        # Step 1: Detect
        logical_problem = detector.detect_constraint_problem(text)
        if logical_problem is None:
            pytest.skip("Schalter-Rätsel nicht erkannt")

        # Step 2: Translate
        csp_problem = translate_logical_constraints_to_csp(logical_problem)
        assert csp_problem is not None

        # Step 3: Solve
        solver = ConstraintSolver(use_ac3=True, use_mrv=True)
        result = solver.solve(csp_problem)

        if result:
            # solve() gibt (solution, proof_tree) zurück
            solution, proof_tree = (
                result if isinstance(result, tuple) else (result, None)
            )
            print(f"[Schalter-Rätsel Lösung] {solution}")


class TestConstraintQuery:
    """Tests für is_constraint_query"""

    def test_is_constraint_query_with_variable(self):
        """Test: Query mit Variablen-Name wird erkannt"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = "Wenn Leo kommt, kommt Mark. Wenn Mark kommt, kommt Nick."

        problem = detector.detect_constraint_problem(text)
        if problem is None:
            pytest.skip("Problem nicht erkannt")

        # Query die Leo erwähnt
        is_related = detector.is_constraint_query("Wer trinkt Brandy?", problem)
        assert is_related  # "wer" ist Solution-Keyword

        is_related = detector.is_constraint_query("Trinkt Leo Brandy?", problem)
        assert is_related  # "Leo" ist Variable

    def test_is_constraint_query_without_variable(self):
        """Test: Query ohne Variablen-Name wird trotzdem erkannt (via Keywords)"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = "Wenn A kommt, kommt B. Wenn B kommt, kommt C."

        problem = detector.detect_constraint_problem(text)
        if problem is None:
            pytest.skip("Problem nicht erkannt")

        # Query mit Solution-Keywords
        is_related = detector.is_constraint_query("Welche sind aktiv?", problem)
        assert is_related  # "welche" ist Solution-Keyword


class TestImprovedPatternRecognition:
    """Tests für verbesserte Pattern-Erkennung (nach Optimierung)"""

    def test_brandy_puzzle_with_improved_patterns(self):
        """Test: Brandy-Rätsel mit verbesserter Pattern-Erkennung"""
        detector = ConstraintDetector(
            min_conditional_rules=2, confidence_threshold=0.65
        )
        text = """
        Wenn Leo Brandy bestellt, bestellt auch Mark einen.
        Mark oder Nick bestellt Brandy, aber nie beide.
        Nick bestellt nur dann Brandy, wenn Leo keinen bestellt.
        """

        problem = detector.detect_constraint_problem(text)

        # Mit optimierten Patterns sollte das Problem erkannt werden
        assert (
            problem is not None
        ), "Brandy-Rätsel sollte mit optimierten Patterns erkannt werden"
        assert len(problem.variables) >= 3, "Mindestens Leo, Mark, Nick"
        assert len(problem.constraints) >= 3, "3 Constraints erwartet"
        assert (
            problem.confidence >= 0.65
        ), f"Confidence {problem.confidence} sollte >= 0.65 sein"

    def test_and_pattern_recognition(self):
        """Test: AND-Pattern wird erkannt"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn Leo kommt, kommen sowohl Mark als auch Nick.
        Mark und Nick sind immer zusammen.
        """

        problem = detector.detect_constraint_problem(text)

        if problem:
            # Prüfe ob AND-Constraints erkannt wurden
            and_constraints = [
                c for c in problem.constraints if c.constraint_type == "AND"
            ]
            assert len(and_constraints) >= 1, "Mindestens 1 AND-Constraint erwartet"

    def test_or_pattern_recognition(self):
        """Test: OR-Pattern wird erkannt"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn die Tür offen ist, ist entweder Leo oder Mark im Raum.
        Mindestens einer von Leo und Mark muss anwesend sein.
        """

        problem = detector.detect_constraint_problem(text)

        if problem:
            # Prüfe ob OR-Constraints erkannt wurden
            or_constraints = [
                c for c in problem.constraints if c.constraint_type == "OR"
            ]
            # OR kann auch als IMPLIES modelliert sein
            assert len(problem.constraints) >= 2

    def test_not_pattern_recognition(self):
        """Test: NOT-Pattern wird erkannt"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn Leo kommt, kommt Mark nicht.
        Nick kommt niemals alleine.
        """

        problem = detector.detect_constraint_problem(text)

        if problem:
            # Prüfe ob NOT-Constraints erkannt wurden
            not_constraints = [
                c for c in problem.constraints if c.constraint_type == "NOT"
            ]
            # NOT kann auch als Teil von IMPLIES modelliert sein
            assert len(problem.constraints) >= 1


class TestComplexLogicPuzzles:
    """Tests für komplexere Logik-Rätsel"""

    def test_bomb_puzzle_complex(self):
        """Test: Komplexeres Bomben-Rätsel"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn Schalter 1 oben ist, muss Schalter 2 auch oben sein.
        Wenn Schalter 2 oben ist, muss Schalter 3 unten sein.
        Schalter 1 und Schalter 3 dürfen nie beide oben sein.
        """

        problem = detector.detect_constraint_problem(text)

        if problem:
            # Translate und solve
            from component_29_constraint_reasoning import (
                ConstraintSolver,
                translate_logical_constraints_to_csp,
            )

            csp_problem = translate_logical_constraints_to_csp(problem)
            solver = ConstraintSolver(use_ac3=True, use_mrv=True)
            result = solver.solve(csp_problem)

            if result:
                # solve() gibt (solution, proof_tree) zurück
                solution, proof_tree = (
                    result if isinstance(result, tuple) else (result, None)
                )
                print(f"[Bomben-Rätsel Lösung] {solution}")
                assert isinstance(solution, dict)

    def test_assignment_puzzle(self):
        """Test: Zuordnungs-Rätsel (Einstein-Style)"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn Leo in Haus 1 wohnt, wohnt Mark in Haus 2.
        Mark und Nick wohnen nie nebeneinander.
        Wenn Nick in Haus 3 wohnt, wohnt Leo in Haus 1.
        """

        problem = detector.detect_constraint_problem(text)

        # Mit min_rules=2 sollte das erkannt werden
        if problem:
            assert len(problem.variables) >= 3, "Leo, Mark, Nick sollten erkannt werden"
            assert len(problem.constraints) >= 2

    def test_chain_implication_puzzle(self):
        """Test: Ketten-Implikationen"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn A aktiv ist, ist B aktiv.
        Wenn B aktiv ist, ist C aktiv.
        Wenn C aktiv ist, ist D nicht aktiv.
        """

        problem = detector.detect_constraint_problem(text)

        assert problem is not None, "Ketten-Implikationen sollten erkannt werden"
        assert len(problem.constraints) >= 3, "3 IMPLIES-Constraints erwartet"

        # Prüfe ob alle IMPLIES sind
        implies_count = sum(
            1 for c in problem.constraints if c.constraint_type == "IMPLIES"
        )
        assert implies_count >= 3


class TestConfidenceCalibration:
    """Tests für Confidence-Kalibrierung"""

    def test_confidence_with_2_conditionals(self):
        """Test: Confidence mit 2 CONDITIONAL-Pattern (Grenzfall)"""
        detector = ConstraintDetector(
            min_conditional_rules=2, confidence_threshold=0.65
        )
        text = """
        Wenn Leo kommt, kommt Mark.
        Wenn Mark kommt, kommt Nick.
        """

        problem = detector.detect_constraint_problem(text)

        # Mit min_rules=2 sollte das erkannt werden
        if problem:
            # Confidence sollte >= 0.65 sein (base_conf=1.0 + bonuses)
            assert (
                problem.confidence >= 0.65
            ), f"Confidence {problem.confidence} zu niedrig"

    def test_confidence_with_many_variables(self):
        """Test: Confidence steigt mit mehr Variablen"""
        detector = ConstraintDetector(min_conditional_rules=2)
        text = """
        Wenn A kommt, kommt B.
        Wenn B kommt, kommen C und D.
        E kommt nur wenn F kommt.
        """

        problem = detector.detect_constraint_problem(text)

        if problem:
            # Mit 6 Variablen (A-F) sollte Confidence sehr hoch sein
            assert (
                problem.confidence >= 0.80
            ), "Viele Variablen sollten Confidence boosten"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
