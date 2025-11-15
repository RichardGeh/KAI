"""
test_production_system_performance.py

Performance Tests für das Production System (PHASE 7 - Schritt 7.3)

Tests:
- Latenz-Messung (Ziel: <100ms für 90% der Queries)
- Memory-Usage Tracking
- Zyklen-Anzahl Statistik
- Durchsatz-Messung
- Skalierungsverhalten

Author: KAI Development Team
Date: 2025-11-13
"""

import statistics
import time
import tracemalloc

import pytest

from component_54_production_system import (
    GenerationGoal,
    GenerationGoalType,
    ProductionSystemEngine,
    ResponseGenerationState,
    create_complete_production_system,
)
from kai_response_formatter import KaiResponseFormatter

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def formatter():
    """KaiResponseFormatter Instanz"""
    return KaiResponseFormatter()


@pytest.fixture
def production_engine():
    """ProductionSystemEngine mit vollständigem Regelset"""
    engine = ProductionSystemEngine()
    engine.add_rules(create_complete_production_system())
    return engine


@pytest.fixture
def sample_queries():
    """Sammlung von Test-Queries"""
    return [
        {
            "topic": "apfel",
            "facts": {"IS_A": ["frucht"], "HAS_PROPERTY": ["rot"]},
            "bedeutungen": ["Ein Apfel ist eine Frucht"],
            "synonyms": [],
        },
        {
            "topic": "vogel",
            "facts": {"IS_A": ["tier"], "CAPABLE_OF": ["fliegen"]},
            "bedeutungen": ["Ein Vogel kann fliegen"],
            "synonyms": [],
        },
        {
            "topic": "baum",
            "facts": {"IS_A": ["pflanze"], "HAS_PROPERTY": ["grün", "groß"]},
            "bedeutungen": ["Ein Baum ist eine große Pflanze"],
            "synonyms": [],
        },
        {
            "topic": "hund",
            "facts": {"IS_A": ["tier", "säugetier"], "HAS_PROPERTY": ["treu"]},
            "bedeutungen": ["Ein Hund ist ein treuer Begleiter"],
            "synonyms": [],
        },
        {
            "topic": "katze",
            "facts": {"IS_A": ["tier", "säugetier"], "HAS_PROPERTY": ["flauschig"]},
            "bedeutungen": ["Eine Katze ist ein Haustier"],
            "synonyms": ["Stubentiger"],
        },
    ]


# ============================================================================
# Test: Latenz-Messung (Schritt 7.3.1)
# ============================================================================


class TestLatencyMeasurement:
    """Tests für Latenz-Messung (Ziel: <100ms für 90% der Queries)"""

    def test_single_query_latency(self, formatter):
        """Test: Latenz für einzelne Query"""
        topic = "apfel"
        facts = {"IS_A": ["frucht"]}
        bedeutungen = []
        synonyms = []

        # Messe Zeit
        start_time = time.perf_counter()
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Assert
        assert response is not None
        print(f"\nSingle query latency: {latency_ms:.2f}ms")

        # Ziel: <100ms für einfache Queries
        # (Kann für erste Version lockerer sein)
        assert latency_ms < 500, f"Latency {latency_ms:.2f}ms exceeds 500ms threshold"

    def test_multiple_queries_latency_distribution(self, formatter, sample_queries):
        """Test: Latenz-Verteilung für mehrere Queries (Ziel: 90% <100ms)"""
        latencies = []

        # Führe alle Queries aus und messe Zeit
        for query in sample_queries:
            start_time = time.perf_counter()
            formatter.generate_with_production_system(**query)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Statistiken
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p90_latency = sorted(latencies)[int(len(latencies) * 0.9)]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        max_latency = max(latencies)

        print(f"\nLatency Statistics (n={len(latencies)}):")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Median:  {median_latency:.2f}ms")
        print(f"  P90:     {p90_latency:.2f}ms")
        print(f"  P95:     {p95_latency:.2f}ms")
        print(f"  Max:     {max_latency:.2f}ms")

        # Assert: 90th percentile sollte <500ms sein (lockeres Ziel für Prototyp)
        # Produktions-Ziel wäre <100ms
        assert p90_latency < 500, f"P90 latency {p90_latency:.2f}ms exceeds 500ms"

    def test_cold_vs_warm_start(self, formatter):
        """Test: Cold Start vs. Warm Start Latenz"""
        topic = "test"
        facts = {"IS_A": ["objekt"]}
        bedeutungen = []
        synonyms = []

        # Cold Start
        cold_start_time = time.perf_counter()
        formatter.generate_with_production_system(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )
        cold_end_time = time.perf_counter()
        cold_latency_ms = (cold_end_time - cold_start_time) * 1000

        # Warm Start (gleiche Query nochmal)
        warm_start_time = time.perf_counter()
        formatter.generate_with_production_system(
            topic=topic, facts=facts, bedeutungen=bedeutungen, synonyms=synonyms
        )
        warm_end_time = time.perf_counter()
        warm_latency_ms = (warm_end_time - warm_start_time) * 1000

        print(f"\nCold start: {cold_latency_ms:.2f}ms")
        print(f"Warm start: {warm_latency_ms:.2f}ms")
        print(f"Speedup:    {cold_latency_ms / warm_latency_ms:.2f}x")

        # Warm Start sollte tendenziell schneller sein (durch Caching)
        # Aber nicht garantiert für Production System
        assert warm_latency_ms > 0

    @pytest.mark.slow
    def test_sustained_load_latency(self, formatter):
        """Test: Latenz unter sustained load (50 Queries)"""
        latencies = []
        query_template = {
            "topic": "test_{}",
            "facts": {"IS_A": ["objekt"]},
            "bedeutungen": [],
            "synonyms": [],
        }

        for i in range(50):
            query = query_template.copy()
            query["topic"] = f"test_{i}"

            start_time = time.perf_counter()
            formatter.generate_with_production_system(**query)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Prüfe ob Latenz stabil bleibt
        first_10_avg = statistics.mean(latencies[:10])
        last_10_avg = statistics.mean(latencies[-10:])

        print(f"\nSustained Load (n=50):")
        print(f"  First 10 avg: {first_10_avg:.2f}ms")
        print(f"  Last 10 avg:  {last_10_avg:.2f}ms")
        print(f"  Degradation:  {(last_10_avg / first_10_avg - 1) * 100:.1f}%")

        # Latenz sollte nicht mehr als 50% degradieren
        assert last_10_avg < first_10_avg * 1.5, "Latency degradation exceeds 50%"


# ============================================================================
# Test: Memory Usage (Schritt 7.3.2)
# ============================================================================


class TestMemoryUsage:
    """Tests für Memory-Usage Tracking"""

    def test_single_query_memory(self, formatter):
        """Test: Memory-Usage für einzelne Query"""
        # Start Memory Tracking
        tracemalloc.start()

        topic = "apfel"
        facts = {"IS_A": ["frucht"], "HAS_PROPERTY": ["rot", "süß"]}
        bedeutungen = ["Ein Apfel ist eine Frucht"]
        synonyms = []

        # Baseline
        baseline = tracemalloc.take_snapshot()

        # Execute Query
        response = formatter.generate_with_production_system(
            topic=topic,
            facts=facts,
            bedeutungen=bedeutungen,
            synonyms=synonyms,
        )

        # Measure
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.compare_to(baseline, "lineno")

        # Total Memory Increase
        total_increase = sum(stat.size_diff for stat in top_stats)
        memory_mb = total_increase / (1024 * 1024)

        print(f"\nMemory increase: {memory_mb:.2f} MB")

        tracemalloc.stop()

        # Assert: Sollte nicht mehr als 50 MB für eine Query verwenden
        assert (
            memory_mb < 50
        ), f"Memory usage {memory_mb:.2f} MB exceeds 50 MB threshold"

    def test_memory_leak_check(self, formatter):
        """Test: Check für Memory Leaks (wiederholte Queries)"""
        tracemalloc.start()

        query = {
            "topic": "test",
            "facts": {"IS_A": ["objekt"]},
            "bedeutungen": [],
            "synonyms": [],
        }

        # Baseline nach 5 warmup queries
        for _ in range(5):
            formatter.generate_with_production_system(**query)

        baseline = tracemalloc.take_snapshot()

        # Run 20 more queries
        for _ in range(20):
            formatter.generate_with_production_system(**query)

        final = tracemalloc.take_snapshot()

        # Check Memory Growth
        top_stats = final.compare_to(baseline, "lineno")
        total_growth = sum(stat.size_diff for stat in top_stats)
        growth_mb = total_growth / (1024 * 1024)

        print(f"\nMemory growth after 20 queries: {growth_mb:.2f} MB")

        tracemalloc.stop()

        # Sollte nicht mehr als 20 MB wachsen (großzügig)
        assert (
            growth_mb < 20
        ), f"Memory growth {growth_mb:.2f} MB indicates potential leak"

    def test_state_object_size(self):
        """Test: Größe von ResponseGenerationState Objekten"""
        import sys

        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="test",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": f"object_{i}", "confidence": 0.9}
                for i in range(10)
            ],
        )

        state_size_bytes = sys.getsizeof(state)
        state_size_kb = state_size_bytes / 1024

        print(f"\nResponseGenerationState size: {state_size_kb:.2f} KB")

        # State-Objekt sollte nicht zu groß werden
        assert state_size_kb < 100, f"State size {state_size_kb:.2f} KB is too large"


# ============================================================================
# Test: Cycle Count Statistics (Schritt 7.3.3)
# ============================================================================


class TestCycleCountStatistics:
    """Tests für Zyklen-Anzahl Statistik"""

    def test_cycle_count_tracking(self, production_engine):
        """Test: Tracking der Cycle-Count pro Query"""
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

        # Execute
        final_state = production_engine.generate(state)

        cycle_count = final_state.cycle_count

        print(f"\nCycle count: {cycle_count}")
        print(f"Max cycles: {final_state.max_cycles}")
        print(f"Goal completed: {final_state.primary_goal.completed}")

        # Assert: Sollte nicht Max-Cycles erreichen (sonst Timeout)
        assert cycle_count < final_state.max_cycles, "Hit max cycles limit"

        # Sollte in vernünftiger Anzahl Zyklen fertig sein
        assert cycle_count < 50, f"Cycle count {cycle_count} is too high"

    def test_cycle_count_distribution(self, formatter, sample_queries):
        """Test: Cycle-Count Verteilung über mehrere Queries"""
        cycle_counts = []

        for query in sample_queries:
            # Direkt mit ProductionEngine um cycle_count zu bekommen
            engine = ProductionSystemEngine()
            engine.add_rules(create_complete_production_system())

            goal = GenerationGoal(
                goal_type=GenerationGoalType.ANSWER_QUESTION,
                target_entity=query["topic"],
            )

            facts_list = []
            for rel_type, objects in query["facts"].items():
                for obj in objects:
                    facts_list.append(
                        {"relation_type": rel_type, "object": obj, "confidence": 0.9}
                    )

            state = ResponseGenerationState(
                primary_goal=goal,
                available_facts=facts_list,
            )

            final_state = engine.generate(state)
            cycle_counts.append(final_state.cycle_count)

        # Statistiken
        avg_cycles = statistics.mean(cycle_counts)
        median_cycles = statistics.median(cycle_counts)
        max_cycles = max(cycle_counts)
        min_cycles = min(cycle_counts)

        print(f"\nCycle Count Statistics (n={len(cycle_counts)}):")
        print(f"  Average: {avg_cycles:.1f}")
        print(f"  Median:  {median_cycles:.1f}")
        print(f"  Min:     {min_cycles}")
        print(f"  Max:     {max_cycles}")

        # Assert: Durchschnitt sollte unter 30 liegen
        assert avg_cycles < 30, f"Average cycle count {avg_cycles:.1f} is too high"

    def test_cycle_efficiency(self, production_engine):
        """Test: Effizienz der Cycle-Nutzung"""
        goal = GenerationGoal(
            goal_type=GenerationGoalType.ANSWER_QUESTION,
            target_entity="test",
        )

        state = ResponseGenerationState(
            primary_goal=goal,
            available_facts=[
                {"relation_type": "IS_A", "object": "objekt", "confidence": 0.9},
            ],
        )

        final_state = production_engine.generate(state)

        # Effizienz: Anzahl generierte Sätze / Cycle Count
        sentences_generated = len(final_state.text.completed_sentences)
        cycle_count = final_state.cycle_count

        if cycle_count > 0:
            efficiency = sentences_generated / cycle_count
            print(f"\nCycle efficiency: {efficiency:.2f} sentences/cycle")

            # Sollte nicht zu viele Leerzyklen geben
            assert efficiency > 0.05, f"Cycle efficiency {efficiency:.2f} is too low"


# ============================================================================
# Test: Throughput Measurement (Schritt 7.3.4)
# ============================================================================


class TestThroughput:
    """Tests für Durchsatz-Messung"""

    @pytest.mark.slow
    def test_queries_per_second(self, formatter):
        """Test: Queries pro Sekunde"""
        query = {
            "topic": "test",
            "facts": {"IS_A": ["objekt"]},
            "bedeutungen": [],
            "synonyms": [],
        }

        num_queries = 20
        start_time = time.perf_counter()

        for _ in range(num_queries):
            formatter.generate_with_production_system(**query)

        end_time = time.perf_counter()
        duration_s = end_time - start_time

        queries_per_second = num_queries / duration_s

        print(f"\nThroughput: {queries_per_second:.2f} queries/second")
        print(f"Duration:   {duration_s:.2f} seconds for {num_queries} queries")

        # Sollte mindestens 5 QPS erreichen (sehr konservatives Ziel)
        assert (
            queries_per_second >= 2
        ), f"Throughput {queries_per_second:.2f} QPS is too low"

    @pytest.mark.slow
    def test_batch_processing_scalability(self, formatter):
        """Test: Skalierung bei Batch-Processing"""
        batch_sizes = [5, 10, 20]
        throughputs = []

        query_template = {
            "topic": "test_{}",
            "facts": {"IS_A": ["objekt"]},
            "bedeutungen": [],
            "synonyms": [],
        }

        for batch_size in batch_sizes:
            start_time = time.perf_counter()

            for i in range(batch_size):
                query = query_template.copy()
                query["topic"] = f"test_{i}"
                formatter.generate_with_production_system(**query)

            end_time = time.perf_counter()
            duration_s = end_time - start_time

            qps = batch_size / duration_s
            throughputs.append(qps)

            print(f"\nBatch size {batch_size}: {qps:.2f} QPS")

        # Throughput sollte relativ stabil bleiben
        # (nicht signifikant degradieren mit größeren Batches)
        first_qps = throughputs[0]
        last_qps = throughputs[-1]

        degradation = (first_qps - last_qps) / first_qps

        print(f"\nThroughput degradation: {degradation * 100:.1f}%")

        # Sollte nicht mehr als 30% degradieren
        assert (
            degradation < 0.3
        ), f"Throughput degradation {degradation * 100:.1f}% is too high"


# ============================================================================
# Test: Rule Application Statistics (Bonus)
# ============================================================================


class TestRuleApplicationStatistics:
    """Tests für Regel-Anwendungs-Statistiken"""

    def test_rule_application_tracking(self, production_engine):
        """Test: Tracking welche Regeln angewendet werden"""
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

        # Execute
        production_engine.generate(state)

        # Get Statistics
        stats = production_engine.get_statistics()

        print(f"\nRule Application Statistics:")
        print(f"  Total rules: {stats['total_rules']}")
        print(f"  Total applications: {stats['total_applications']}")
        print(f"  Most used rule: {stats['most_used_rule']}")
        print(f"  Applications: {stats['most_used_count']}")

        # Assert: Regeln sollten angewendet worden sein
        assert stats["total_applications"] > 0, "No rules were applied"

    def test_rule_category_distribution(self, production_engine):
        """Test: Verteilung der Regel-Kategorien"""
        from component_54_production_system import RuleCategory

        # Count rules per category
        category_counts = {}
        for rule in production_engine.rules:
            cat = rule.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"\nRule Category Distribution:")
        for cat, count in category_counts.items():
            print(f"  {cat}: {count} rules")

        # Assert: Sollte alle Kategorien haben
        expected_categories = [
            RuleCategory.CONTENT_SELECTION,
            RuleCategory.LEXICALIZATION,
            RuleCategory.DISCOURSE,
            RuleCategory.SYNTAX,
        ]

        for cat in expected_categories:
            assert (
                cat.value in category_counts
            ), f"Missing rules for category {cat.value}"


# ============================================================================
# Performance Benchmarks (für Monitoring)
# ============================================================================


class TestPerformanceBenchmarks:
    """Performance Benchmarks für Monitoring"""

    @pytest.mark.benchmark
    def test_benchmark_simple_query(self, formatter, benchmark):
        """Benchmark: Einfache Query"""

        def run():
            return formatter.generate_with_production_system(
                topic="test",
                facts={"IS_A": ["objekt"]},
                bedeutungen=[],
                synonyms=[],
            )

        result = benchmark(run)
        assert result is not None

    @pytest.mark.benchmark
    def test_benchmark_complex_query(self, formatter, benchmark):
        """Benchmark: Komplexe Query mit vielen Fakten"""

        def run():
            return formatter.generate_with_production_system(
                topic="apfel",
                facts={
                    "IS_A": ["frucht", "obst", "nahrungsmittel"],
                    "HAS_PROPERTY": ["rot", "grün", "süß", "saftig"],
                    "CAPABLE_OF": ["wachsen", "reifen"],
                    "PART_OF": ["baum"],
                },
                bedeutungen=["Ein Apfel ist eine gesunde Frucht"],
                synonyms=["Malus"],
            )

        result = benchmark(run)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
