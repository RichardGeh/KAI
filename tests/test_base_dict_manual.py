"""
Manual test script for base dictionary edge cases and performance.
"""

import time

from component_1_netzwerk import KonzeptNetzwerk
from component_24_pattern_orchestrator import PatternOrchestrator


def test_edge_cases():
    """Test edge cases for base dictionary."""
    print("=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)

    netz = KonzeptNetzwerk()
    orch = PatternOrchestrator(netz)

    # Test 1: Empty input
    print("\n[TEST 1] Empty input")
    result = orch.process_input("")
    print(f"  Result: {len(result['typo_corrections'])} typo corrections")
    assert len(result["typo_corrections"]) == 0, "Empty input should have no typos"
    print("  [OK] PASSED")

    # Test 2: Very long input (1000+ words)
    print("\n[TEST 2] Very long input (1000 words)")
    long_text = " ".join(["Ein Mann trinkt Bier und eine Frau trinkt Wein"] * 100)
    start = time.time()
    result = orch.process_input(long_text)
    duration = time.time() - start
    print(f"  Words: {len(long_text.split())}")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Typo corrections: {len(result['typo_corrections'])}")
    assert duration < 2.0, f"Should complete in <2s, took {duration:.3f}s"
    print("  [OK] PASSED")

    # Test 3: Mixed German/English text
    print("\n[TEST 3] Mixed German/English text")
    mixed_text = "Ein Mann trinkt Coffee und eine Frau trinkt Tea"
    result = orch.process_input(mixed_text)
    flagged = [c["original"] for c in result["typo_corrections"]]
    print(f"  Flagged words: {flagged}")
    assert "Mann" not in flagged, "German word should not be flagged"
    assert "trinkt" not in flagged, "German word should not be flagged"
    print("  [OK] PASSED")

    # Test 4: Capitalized words
    print("\n[TEST 4] Capitalized words")
    cap_text = "Leo Mark Nick bestellen Brandy Whiskey Vodka"
    result = orch.process_input(cap_text)
    flagged = [c["original"] for c in result["typo_corrections"]]
    print(f"  Flagged words: {flagged}")
    # Capitalized words (names, foreign words) should NOT be flagged
    assert "Leo" not in flagged
    assert "Mark" not in flagged
    assert "Nick" not in flagged
    print("  [OK] PASSED")

    # Test 5: Punctuation handling
    print("\n[TEST 5] Punctuation handling")
    punct_text = "Trinkt, eine? Person! Bier."
    result = orch.process_input(punct_text)
    flagged = [c["original"] for c in result["typo_corrections"]]
    print(f"  Flagged words: {flagged}")
    assert (
        len(flagged) == 0
    ), "Valid German words with punctuation should not be flagged"
    print("  [OK] PASSED")

    # Test 6: Unicode characters
    print("\n[TEST 6] Unicode characters (umlauts)")
    unicode_text = "Ein Mädchen trinkt Äpfelsaft und isst Brötchen"
    result = orch.process_input(unicode_text)
    flagged = [c["original"] for c in result["typo_corrections"]]
    print(f"  Flagged words: {flagged}")
    # These are valid German words, should be in spaCy vocabulary
    assert "Mädchen" not in flagged or "mädchen" not in [f.lower() for f in flagged]
    print("  [OK] PASSED")

    # Test 7: Brandy puzzle text
    print("\n[TEST 7] Brandy puzzle text (integration)")
    brandy_text = """
    Wenn Leo einen Brandy bestellt, bestellt auch Mark einen.
    Mark oder Nick bestellt Brandy, aber nie beide.
    Trinkt eine Person Whiskey und bestellt eine andere Person Vodka?
    """
    result = orch.process_input(brandy_text)
    flagged = [c["original"] for c in result["typo_corrections"]]
    print(f"  Flagged words: {flagged}")

    # Common German words should NOT be flagged
    common_words = [
        "wenn",
        "einen",
        "bestellt",
        "auch",
        "oder",
        "aber",
        "nie",
        "beide",
        "trinkt",
        "eine",
        "person",
        "und",
        "andere",
    ]
    for word in common_words:
        assert word not in [
            f.lower() for f in flagged
        ], f"Common word '{word}' should not be flagged"

    # Should have minimal typo warnings (foreign words like Brandy/Whiskey/Vodka are OK)
    print(
        f"  Total typo corrections: {len(result['typo_corrections'])} (expected <=3 for foreign words)"
    )
    assert len(result["typo_corrections"]) <= 3, "Should have minimal typo warnings"
    print("  [OK] PASSED")

    netz.close()
    print("\n" + "=" * 80)
    print("ALL EDGE CASE TESTS PASSED")
    print("=" * 80)


def test_performance():
    """Test performance metrics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTS")
    print("=" * 80)

    netz = KonzeptNetzwerk()

    # Test 1: Dictionary loading time
    print("\n[TEST 1] Dictionary loading time")
    start = time.time()
    orch = PatternOrchestrator(netz)
    load_time = time.time() - start
    print(f"  Loading time: {load_time:.3f}s")
    print(f"  Dictionary size: {len(orch.base_dictionary)} words")
    assert load_time < 2.0, f"Should load in <2s, took {load_time:.3f}s"
    assert len(orch.base_dictionary) > 100000, "Should have >100k words"
    print("  [OK] PASSED")

    # Test 2: Lookup performance (should be O(1))
    print("\n[TEST 2] Lookup performance (1000 lookups)")
    test_words = ["trinken", "bestellen", "person", "brandy"] * 250
    start = time.time()
    for word in test_words:
        _ = word.lower() in orch.base_dictionary
    lookup_time = time.time() - start
    print(f"  Lookup time for {len(test_words)} words: {lookup_time:.3f}s")
    print(f"  Average per lookup: {lookup_time/len(test_words)*1000:.3f}ms")
    assert lookup_time < 0.1, f"Should complete in <0.1s, took {lookup_time:.3f}s"
    print("  [OK] PASSED")

    # Test 3: Processing performance with base dictionary
    print("\n[TEST 3] Processing performance (150 words)")
    text = " ".join(["Ein Mann trinkt Bier und eine Frau trinkt Wein"] * 15)
    start = time.time()
    result = orch.process_input(text)
    proc_time = time.time() - start
    print(f"  Processing time: {proc_time:.3f}s")
    print(f"  Words processed: {len(text.split())}")
    print(f"  Typo corrections: {len(result['typo_corrections'])}")
    assert proc_time < 1.0, f"Should complete in <1s, took {proc_time:.3f}s"
    print("  [OK] PASSED")

    netz.close()
    print("\n" + "=" * 80)
    print("ALL PERFORMANCE TESTS PASSED")
    print("=" * 80)


def test_memory():
    """Test memory usage."""
    print("\n" + "=" * 80)
    print("MEMORY TESTS")
    print("=" * 80)

    netz = KonzeptNetzwerk()
    orch = PatternOrchestrator(netz)

    # Estimate dictionary memory usage
    dict_size = len(orch.base_dictionary)
    # Rough estimate: each string ~20 bytes + overhead
    estimated_mb = dict_size * 20 / (1024 * 1024)

    print(f"\n[TEST 1] Dictionary memory usage")
    print(f"  Dictionary size: {dict_size} words")
    print(f"  Estimated memory: ~{estimated_mb:.1f} MB")

    # This is acceptable overhead for a base dictionary
    assert (
        estimated_mb < 50
    ), f"Memory usage should be <50 MB, estimated {estimated_mb:.1f} MB"
    print("  [OK] PASSED - Memory usage is acceptable")

    netz.close()
    print("\n" + "=" * 80)
    print("ALL MEMORY TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_edge_cases()
    test_performance()
    test_memory()

    print("\n" + "=" * 80)
    print("ALL MANUAL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
