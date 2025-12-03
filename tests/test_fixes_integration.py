"""
Integrations-Test für kritische Fixes.

Testet die tatsächliche Funktionalität der geänderten Module.
"""

import sys
import traceback


def test_resonance_engine_unicode_output():
    """Test 1: Resonance Engine generiert ASCII-Output"""
    print("Test 1: Resonance Engine Unicode-Compliance...")
    try:
        from component_44_resonance_engine import (
            ActivationMap,
            ResonancePoint,
        )

        # Erstelle Mock ActivationMap mit Resonanz-Punkten
        activation_map = ActivationMap(
            activations={"test_konzept": 0.8, "resonanz_punkt": 0.9},
            resonance_points=[
                ResonancePoint(
                    concept="resonanz_punkt",
                    resonance_boost=0.5,
                    wave_depth=2,
                    num_paths=3,
                )
            ],
        )

        # Generiere Summary (enthält Resonanz-Marker)
        from component_44_resonance_engine import ResonanceEngine

        # Mock netzwerk
        class MockNetzwerk:
            driver = None

        engine = ResonanceEngine(MockNetzwerk())
        summary = engine.get_activation_summary(activation_map)

        # Prüfe auf Unicode-Verletzungen
        if "⭐" in summary:
            print("  [FEHLER] Unicode-Symbol ⭐ im Output gefunden!")
            print(f"  Output: {summary[:200]}...")
            return False
        elif "[R]" not in summary:
            print("  [FEHLER] Ersatz-Symbol [R] nicht im Output!")
            return False
        else:
            print("  [OK] Output enthält [R] statt ⭐")
            return True

    except Exception as e:
        print(f"  [FEHLER] Exception: {e}")
        traceback.print_exc()
        return False


def test_logic_puzzle_solver_spacy_error():
    """Test 2: Logic Puzzle Solver handelt spaCy-Fehler korrekt"""
    print("\nTest 2: Logic Puzzle Solver spaCy Error Handling...")
    try:
        from component_45_logic_puzzle_solver import (
            _get_nlp_model,
        )

        # Teste Lazy Loading
        print("  [INFO] Lade spaCy-Modell (lazy)...")
        try:
            nlp = _get_nlp_model()
            print(f"  [OK] spaCy-Modell geladen: {type(nlp).__name__}")
            return True
        except Exception as e:
            # SpaCyModelError ist erwartbar wenn Modell nicht installiert
            if "SpaCyModelError" in str(type(e).__name__):
                print(f"  [OK] SpaCyModelError korrekt geworfen: {e.message}")
                return True
            else:
                print(f"  [FEHLER] Falsche Exception: {type(e).__name__}")
                return False

    except Exception as e:
        print(f"  [FEHLER] Unerwartete Exception: {e}")
        traceback.print_exc()
        return False


def test_logic_puzzle_solver_exception_hierarchy():
    """Test 3: Logic Puzzle Solver nutzt KAI Exception-Hierarchie"""
    print("\nTest 3: Logic Puzzle Solver Exception-Hierarchie...")
    try:
        from component_45_logic_puzzle_solver import LogicPuzzleSolver

        solver = LogicPuzzleSolver()

        # Teste mit leerem Input (sollte leere conditions zurückgeben, nicht crashen)
        result = solver.solve("", ["TestEntity"])
        if result["answer"] == "Keine logischen Bedingungen gefunden.":
            print("  [OK] Leere Eingabe wird korrekt behandelt")
            return True
        else:
            print(f"  [FEHLER] Unerwartete Antwort: {result['answer']}")
            return False

    except Exception as e:
        print(f"  [FEHLER] Exception: {e}")
        traceback.print_exc()
        return False


def test_spatial_grid_widget_imports():
    """Test 4: Spatial Grid Widget hat FileSystemError"""
    print("\nTest 4: Spatial Grid Widget FileSystemError...")
    try:
        from component_43_spatial_grid_widget import FileSystemError
        from kai_exceptions import KAIException

        # Prüfe, ob FileSystemError von KAIException erbt
        if issubclass(FileSystemError, KAIException):
            print("  [OK] FileSystemError erbt von KAIException")
            return True
        else:
            print("  [FEHLER] FileSystemError erbt nicht von KAIException")
            return False

    except ImportError as e:
        print(f"  [FEHLER] Import fehlgeschlagen: {e}")
        return False
    except Exception as e:
        print(f"  [FEHLER] Exception: {e}")
        traceback.print_exc()
        return False


def test_resonance_engine_neo4j_graceful_degradation():
    """Test 5: Resonance Engine fällt zurück wenn Neo4j fehlt"""
    print("\nTest 5: Resonance Engine Neo4j Graceful Degradation...")
    try:
        from component_44_resonance_engine import ResonanceEngine

        # Mock netzwerk ohne driver
        class MockNetzwerk:
            driver = None

        engine = ResonanceEngine(MockNetzwerk())

        # _get_semantic_neighbors sollte [] zurückgeben (nicht crashen)
        neighbors = engine._get_semantic_neighbors("test", {}, 1.0, None)

        if neighbors == []:
            print("  [OK] Graceful degradation: [] bei fehlendem Neo4j")
            return True
        else:
            print(f"  [FEHLER] Unerwartetes Ergebnis: {neighbors}")
            return False

    except Exception as e:
        print(f"  [FEHLER] Exception: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("INTEGRATION TESTS FÜR KRITISCHE FIXES")
    print("=" * 70)

    results = {
        "Resonance Engine Unicode-Output": test_resonance_engine_unicode_output(),
        "Logic Puzzle spaCy Error Handling": test_logic_puzzle_solver_spacy_error(),
        "Logic Puzzle Exception-Hierarchie": test_logic_puzzle_solver_exception_hierarchy(),
        "Spatial Grid FileSystemError": test_spatial_grid_widget_imports(),
        "Resonance Engine Graceful Degradation": test_resonance_engine_neo4j_graceful_degradation(),
    }

    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    for test_name, result in results.items():
        status = "[OK]" if result else "[FEHLER]"
        print(f"{status} {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)

    if all_passed:
        print("[OK] Alle Integration-Tests bestanden!")
        return 0
    else:
        failed_count = sum(1 for v in results.values() if not v)
        print(f"[FEHLER] {failed_count} von {len(results)} Tests fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(main())
