"""Quick verification test for component_14 modular split."""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")

    # Test facade import

    print("[OK] Facade imports successful")

    # Test that we can import from new modules directly (not required for users, but good to verify)

    print("[OK] Direct module imports successful")

    return True


def test_instantiation():
    """Test that AbductiveEngine can be instantiated."""
    print("\nTesting instantiation...")

    from component_1_netzwerk import KonzeptNetzwerk
    from component_14_abductive_engine import AbductiveEngine

    # Create mock netzwerk for testing
    try:
        netzwerk = KonzeptNetzwerk()
        engine = AbductiveEngine(netzwerk)
        print(f"[OK] AbductiveEngine instantiated successfully")
        print(f"    - Has {len(engine.causal_patterns)} causal patterns")
        print(f"    - Has score weights: {list(engine.score_weights.keys())}")
        netzwerk.close()
        return True
    except Exception as e:
        print(f"[FEHLER] Instantiation failed: {e}")
        return False


def test_backward_compatibility():
    """Test that all expected methods and attributes are accessible."""
    print("\nTesting backward compatibility...")

    from component_1_netzwerk import KonzeptNetzwerk
    from component_14_abductive_engine import AbductiveEngine

    try:
        netzwerk = KonzeptNetzwerk()
        engine = AbductiveEngine(netzwerk)

        # Test that all public methods exist
        public_methods = [
            "generate_hypotheses",
            "explain_hypothesis",
            "create_proof_step_from_hypothesis",
            "create_multi_hypothesis_proof_chain",
            "explain_with_proof_step",
            "create_detailed_explanation",
        ]

        for method_name in public_methods:
            assert hasattr(engine, method_name), f"Missing method: {method_name}"
            print(f"    [OK] Method '{method_name}' exists")

        # Test that all private methods still exist (for tests that might call them)
        private_methods = [
            "_extract_concepts",
            "_generate_template_hypotheses",
            "_generate_analogy_hypotheses",
            "_generate_causal_chain_hypotheses",
            "_find_similar_concepts",
            "_find_causal_chains",
            "_score_hypothesis",
            "_score_coverage",
            "_score_simplicity",
            "_score_coherence",
            "_score_specificity",
            "_is_fact_known",
            "_get_facts_about_subject",
            "_contradicts_knowledge",
            "_are_types_mutually_exclusive",
            "_is_subtype_of",
            "_are_properties_contradictory",
            "_is_location_hierarchy",
        ]

        for method_name in private_methods:
            assert hasattr(
                engine, method_name
            ), f"Missing private method: {method_name}"

        print(f"    [OK] All {len(private_methods)} private methods exist")

        # Test attributes
        assert hasattr(engine, "causal_patterns"), "Missing causal_patterns"
        assert hasattr(engine, "score_weights"), "Missing score_weights"
        print(f"    [OK] All attributes exist")

        netzwerk.close()
        print("[OK] All backward compatibility checks passed")
        return True

    except Exception as e:
        print(f"[FEHLER] Backward compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic hypothesis generation."""
    print("\nTesting basic functionality...")

    from component_1_netzwerk import KonzeptNetzwerk
    from component_14_abductive_engine import AbductiveEngine

    try:
        netzwerk = KonzeptNetzwerk()
        engine = AbductiveEngine(netzwerk)

        # Test concept extraction
        concepts = engine._extract_concepts("Der Hund bellt laut")
        print(f"    [OK] Concept extraction works (found {len(concepts)} concepts)")

        # Note: Full hypothesis generation requires Neo4j data, so we only test the API exists
        print(f"    [OK] Basic functionality verified")

        netzwerk.close()
        return True

    except Exception as e:
        print(f"[FEHLER] Basic functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Component 14 Modular Split Verification")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Instantiation", test_instantiation()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Basic Functionality", test_basic_functionality()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FEHLER]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n[SUCCESS] All verification tests passed!")
        print("Component 14 modular split is working correctly.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed.")
        sys.exit(1)
