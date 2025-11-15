#!/usr/bin/env python3
"""
Production System Evaluation Script

Sammelt Metriken über 1000+ Queries für:
- Performance-Vergleich (Pipeline vs. Production System)
- Bottleneck-Identifikation
- Utility-Tuning Empfehlungen
- Rollout-Strategie

Usage:
    python evaluate_production_system.py --queries 1000 --output evaluation_results.json
"""

import argparse
import json
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

# KAI Imports
from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from component_46_meta_learning import MetaLearningEngine
from component_54_production_system import ProductionSystemEngine
from kai_response_formatter import ResponseGenerationRouter
from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Test Query Sets
# ═══════════════════════════════════════════════════════════════

TEST_QUERIES = {
    "taxonomy": [
        "Was ist ein Hund?",
        "Was ist eine Katze?",
        "Was ist ein Vogel?",
        "Was ist ein Auto?",
        "Was ist ein Computer?",
        "Was ist eine Blume?",
        "Was ist ein Tisch?",
        "Was ist Wasser?",
        "Was ist ein Baum?",
        "Was ist eine Tür?",
    ],
    "properties": [
        "Welche Eigenschaften hat ein Hund?",
        "Welche Farbe hat ein Apfel?",
        "Wie groß ist ein Elefant?",
        "Ist ein Auto schnell?",
        "Ist Wasser flüssig?",
        "Ist ein Diamant hart?",
        "Welche Form hat ein Ball?",
        "Wie schmeckt Schokolade?",
        "Welche Temperatur hat Eis?",
        "Ist Gold wertvoll?",
    ],
    "capabilities": [
        "Was kann ein Vogel?",
        "Was kann ein Computer?",
        "Was kann eine Katze?",
        "Was kann ein Flugzeug?",
        "Was kann ein Mensch?",
        "Was kann ein Roboter?",
        "Was kann ein Telefon?",
        "Was kann Wasser?",
        "Was kann ein Motor?",
        "Was kann ein Mikroskop?",
    ],
    "multi_hop": [
        "Ist ein Hund ein Lebewesen?",
        "Kann ein Vogel ein Tier sein?",
        "Ist eine Rose eine Pflanze?",
        "Gehört ein Auto zu Fahrzeugen?",
        "Ist ein Apfel gesund?",
        "Kann ein Computer denken?",
        "Ist ein Diamant ein Mineral?",
        "Gehört Wasser zur Natur?",
        "Ist ein Tisch ein Möbel?",
        "Kann ein Flugzeug fliegen?",
    ],
    "complex": [
        "Was ist der Unterschied zwischen einem Hund und einer Katze?",
        "Warum können Vögel fliegen?",
        "Wie funktioniert ein Computer?",
        "Was macht einen Diamant wertvoll?",
        "Welche Tiere sind Säugetiere?",
        "Was sind die Haupteigenschaften von Wasser?",
        "Wie unterscheiden sich Bäume und Blumen?",
        "Was sind die wichtigsten Teile eines Autos?",
        "Welche Materialien sind hart?",
        "Was können Menschen, das Tiere nicht können?",
    ]
}


@dataclass
class QueryResult:
    """Ergebnis einer einzelnen Query"""
    query: str
    query_type: str
    system_used: str  # "pipeline" oder "production"
    response_text: str
    confidence: float
    response_time: float
    rules_applied: List[str]
    cycles_count: int
    success: bool
    error: str = ""


@dataclass
class SystemMetrics:
    """Metriken für ein System (Pipeline oder Production)"""
    system_name: str
    queries_handled: int
    avg_confidence: float
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    success_rate: float
    avg_cycles: float  # Nur für Production System
    total_rules_applied: int  # Nur für Production System
    rule_usage_distribution: Dict[str, int]


@dataclass
class EvaluationReport:
    """Gesamter Evaluationsbericht"""
    timestamp: str
    total_queries: int
    pipeline_metrics: SystemMetrics
    production_metrics: SystemMetrics
    winner: str
    confidence_diff: float
    speed_diff: float
    bottlenecks: List[str]
    tuning_recommendations: List[str]
    rollout_recommendation: str


# ═══════════════════════════════════════════════════════════════
# Evaluation Engine
# ═══════════════════════════════════════════════════════════════

class ProductionSystemEvaluator:
    """Evaluiert das Production System gegen die Pipeline"""

    def __init__(self, netzwerk: KonzeptNetzwerk):
        self.netzwerk = netzwerk
        self.embedding_service = EmbeddingService()
        self.meta_learning = MetaLearningEngine(netzwerk, self.embedding_service)
        self.production_engine = ProductionSystemEngine(netzwerk)
        self.router = ResponseGenerationRouter(
            netzwerk=netzwerk,
            production_system_engine=self.production_engine,
            meta_learning=self.meta_learning,
            production_weight=0.5  # 50/50 Split
        )

        self.results: List[QueryResult] = []

    def run_evaluation(self, num_queries: int = 1000) -> EvaluationReport:
        """
        Führt Evaluation über num_queries Queries aus.

        Args:
            num_queries: Anzahl der Test-Queries

        Returns:
            EvaluationReport mit allen Metriken
        """
        logger.info(f"Starting evaluation with {num_queries} queries")

        # Generiere Test-Queries
        test_queries = self._generate_test_queries(num_queries)

        # Führe Queries aus
        for i, (query, query_type) in enumerate(test_queries, 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{num_queries} queries processed")

            result = self._execute_query(query, query_type)
            self.results.append(result)

        # Analysiere Ergebnisse
        report = self._analyze_results()

        logger.info("Evaluation complete")
        return report

    def _generate_test_queries(self, num_queries: int) -> List[Tuple[str, str]]:
        """
        Generiert Test-Queries aus den definierten Sets.

        Args:
            num_queries: Anzahl gewünschter Queries

        Returns:
            List von (query, query_type) Tuples
        """
        queries = []
        query_types = list(TEST_QUERIES.keys())

        # Round-Robin durch Query-Types
        while len(queries) < num_queries:
            for query_type in query_types:
                if len(queries) >= num_queries:
                    break

                # Wähle zyklisch aus dem Query-Set
                query_set = TEST_QUERIES[query_type]
                query = query_set[len(queries) % len(query_set)]
                queries.append((query, query_type))

        return queries[:num_queries]

    def _execute_query(self, query: str, query_type: str) -> QueryResult:
        """
        Führt eine einzelne Query aus und misst Metriken.

        Args:
            query: Die Test-Query
            query_type: Typ der Query (taxonomy, properties, etc.)

        Returns:
            QueryResult mit allen Metriken
        """
        start_time = time.time()

        try:
            # Simuliere Reasoning-Ergebnis (vereinfacht)
            # In realer Evaluation würde hier kai_worker.process_query() aufgerufen
            content_elements = self._simulate_reasoning(query, query_type)

            # Route durch Router (50/50 Split)
            response = self.router.route_and_generate(
                answer_text="",
                confidence=0.85,
                query=query,
                content_elements=content_elements,
                discourse_context={},
                use_production_system_override=None  # Zufällig
            )

            response_time = time.time() - start_time

            # Extrahiere Metriken
            system_used = "production" if response.generation_metadata.get("system_used") == "production" else "pipeline"
            rules_applied = response.generation_metadata.get("rules_applied", [])
            cycles_count = response.generation_metadata.get("cycles_count", 0)

            return QueryResult(
                query=query,
                query_type=query_type,
                system_used=system_used,
                response_text=response.answer_text,
                confidence=response.confidence,
                response_time=response_time,
                rules_applied=rules_applied,
                cycles_count=cycles_count,
                success=True
            )

        except Exception as e:
            logger.error(f"Query execution failed: {query}", extra={"error": str(e)})
            return QueryResult(
                query=query,
                query_type=query_type,
                system_used="unknown",
                response_text="",
                confidence=0.0,
                response_time=time.time() - start_time,
                rules_applied=[],
                cycles_count=0,
                success=False,
                error=str(e)
            )

    def _simulate_reasoning(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        Simuliert Reasoning-Ergebnis für Test-Zwecke.

        In realer Evaluation würde hier echtes Reasoning verwendet.
        """
        # Einfache Simulation basierend auf Query-Type
        facts = []

        if query_type == "taxonomy":
            # Simuliere IS_A Fakten
            facts = [
                {"subject": "hund", "relation": "IS_A", "object": "tier", "confidence": 0.95},
                {"subject": "hund", "relation": "IS_A", "object": "säugetier", "confidence": 0.90}
            ]
        elif query_type == "properties":
            facts = [
                {"subject": "apfel", "relation": "HAS_PROPERTY", "object": "rot", "confidence": 0.88},
                {"subject": "apfel", "relation": "HAS_PROPERTY", "object": "rund", "confidence": 0.85}
            ]
        elif query_type == "capabilities":
            facts = [
                {"subject": "vogel", "relation": "CAPABLE_OF", "object": "fliegen", "confidence": 0.92}
            ]

        return {"facts": facts}

    def _analyze_results(self) -> EvaluationReport:
        """
        Analysiert gesammelte Ergebnisse und erstellt Report.

        Returns:
            EvaluationReport mit allen Analysen
        """
        # Separiere Ergebnisse nach System
        pipeline_results = [r for r in self.results if r.system_used == "pipeline"]
        production_results = [r for r in self.results if r.system_used == "production"]

        # Berechne Metriken für beide Systeme
        pipeline_metrics = self._compute_system_metrics("Pipeline", pipeline_results)
        production_metrics = self._compute_system_metrics("Production System", production_results)

        # Bestimme Winner
        winner, confidence_diff, speed_diff = self._determine_winner(
            pipeline_metrics, production_metrics
        )

        # Identifiziere Bottlenecks
        bottlenecks = self._identify_bottlenecks(production_results)

        # Generiere Tuning-Empfehlungen
        tuning_recommendations = self._generate_tuning_recommendations(production_results)

        # Rollout-Empfehlung
        rollout_recommendation = self._generate_rollout_recommendation(
            pipeline_metrics, production_metrics, winner
        )

        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_queries=len(self.results),
            pipeline_metrics=pipeline_metrics,
            production_metrics=production_metrics,
            winner=winner,
            confidence_diff=confidence_diff,
            speed_diff=speed_diff,
            bottlenecks=bottlenecks,
            tuning_recommendations=tuning_recommendations,
            rollout_recommendation=rollout_recommendation
        )

    def _compute_system_metrics(self, system_name: str, results: List[QueryResult]) -> SystemMetrics:
        """Berechnet Metriken für ein System"""
        if not results:
            return SystemMetrics(
                system_name=system_name,
                queries_handled=0,
                avg_confidence=0.0,
                avg_response_time=0.0,
                p50_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                success_rate=0.0,
                avg_cycles=0.0,
                total_rules_applied=0,
                rule_usage_distribution={}
            )

        confidences = [r.confidence for r in results]
        response_times = [r.response_time for r in results]
        successes = [r.success for r in results]
        cycles = [r.cycles_count for r in results if r.cycles_count > 0]

        # Rule Usage (nur für Production System relevant)
        rule_usage = Counter()
        for r in results:
            for rule in r.rules_applied:
                rule_usage[rule] += 1

        return SystemMetrics(
            system_name=system_name,
            queries_handled=len(results),
            avg_confidence=statistics.mean(confidences) if confidences else 0.0,
            avg_response_time=statistics.mean(response_times) if response_times else 0.0,
            p50_response_time=statistics.median(response_times) if response_times else 0.0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0.0,
            p99_response_time=self._percentile(response_times, 99) if response_times else 0.0,
            success_rate=sum(successes) / len(successes) if successes else 0.0,
            avg_cycles=statistics.mean(cycles) if cycles else 0.0,
            total_rules_applied=sum(len(r.rules_applied) for r in results),
            rule_usage_distribution=dict(rule_usage)
        )

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Berechnet Perzentil"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _determine_winner(
        self,
        pipeline: SystemMetrics,
        production: SystemMetrics
    ) -> Tuple[str, float, float]:
        """
        Bestimmt Winner basierend auf gewichteten Metriken.

        Returns:
            (winner_name, confidence_diff, speed_diff)
        """
        # Scoring-Funktion (gewichtet)
        def score(metrics: SystemMetrics) -> float:
            return (
                metrics.avg_confidence * 0.4 +        # 40% Confidence
                (1.0 - metrics.avg_response_time) * 0.3 +  # 30% Speed (invertiert)
                metrics.success_rate * 0.3            # 30% Success Rate
            )

        pipeline_score = score(pipeline)
        production_score = score(production)

        if production_score > pipeline_score:
            winner = "Production System"
        elif pipeline_score > production_score:
            winner = "Pipeline"
        else:
            winner = "Tie"

        confidence_diff = production.avg_confidence - pipeline.avg_confidence
        speed_diff = pipeline.avg_response_time - production.avg_response_time  # Positiv = Production schneller

        return winner, confidence_diff, speed_diff

    def _identify_bottlenecks(self, results: List[QueryResult]) -> List[str]:
        """Identifiziert Performance-Bottlenecks"""
        bottlenecks = []

        # Bottleneck 1: Langsame Queries (>500ms)
        slow_queries = [r for r in results if r.response_time > 0.5]
        if len(slow_queries) > len(results) * 0.1:  # >10% langsam
            bottlenecks.append(
                f"PERFORMANCE: {len(slow_queries)} queries (${len(slow_queries)/len(results)*100:.1f}%) took >500ms"
            )

        # Bottleneck 2: Viele Zyklen
        high_cycle_queries = [r for r in results if r.cycles_count > 10]
        if len(high_cycle_queries) > len(results) * 0.05:  # >5% mit >10 Zyklen
            bottlenecks.append(
                f"CYCLES: {len(high_cycle_queries)} queries required >10 cycles (avg: {statistics.mean([r.cycles_count for r in high_cycle_queries]):.1f})"
            )

        # Bottleneck 3: Niedrige Confidence
        low_conf_queries = [r for r in results if r.confidence < 0.6]
        if len(low_conf_queries) > len(results) * 0.15:  # >15% niedrig
            bottlenecks.append(
                f"CONFIDENCE: {len(low_conf_queries)} queries had low confidence (<0.6)"
            )

        # Bottleneck 4: Fehlerrate
        failed_queries = [r for r in results if not r.success]
        if len(failed_queries) > 0:
            bottlenecks.append(
                f"ERRORS: {len(failed_queries)} queries failed ({len(failed_queries)/len(results)*100:.1f}%)"
            )

        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]

    def _generate_tuning_recommendations(self, results: List[QueryResult]) -> List[str]:
        """Generiert Tuning-Empfehlungen basierend auf Regel-Usage"""
        recommendations = []

        if not results:
            return ["Insufficient data for recommendations"]

        # Analysiere Regel-Usage
        rule_usage = Counter()
        for r in results:
            for rule in r.rules_applied:
                rule_usage[rule] += 1

        # Top 5 und Bottom 5 Regeln
        most_used = rule_usage.most_common(5)
        least_used = rule_usage.most_common()[-5:]

        # Empfehlung 1: Utility erhöhen für häufig genutzte Regeln
        if most_used:
            top_rule, top_count = most_used[0]
            if top_count > len(results) * 0.5:  # >50% Usage
                recommendations.append(
                    f"UTILITY UP: Rule '{top_rule}' used in {top_count/len(results)*100:.1f}% of queries. "
                    f"Consider increasing utility to prioritize further."
                )

        # Empfehlung 2: Utility senken oder entfernen für selten genutzte Regeln
        if least_used:
            bottom_rule, bottom_count = least_used[0]
            if bottom_count < len(results) * 0.01:  # <1% Usage
                recommendations.append(
                    f"UTILITY DOWN: Rule '{bottom_rule}' rarely used ({bottom_count} times). "
                    f"Consider lowering utility or removing if not critical."
                )

        # Empfehlung 3: Conflict Resolution Optimierung
        avg_rules_per_query = statistics.mean([len(r.rules_applied) for r in results if r.rules_applied])
        if avg_rules_per_query > 8:
            recommendations.append(
                f"CONFLICT RESOLUTION: Average {avg_rules_per_query:.1f} rules per query. "
                f"Consider increasing specificity to reduce conflict set size."
            )

        # Empfehlung 4: Zyklus-Optimierung
        avg_cycles = statistics.mean([r.cycles_count for r in results if r.cycles_count > 0])
        if avg_cycles > 6:
            recommendations.append(
                f"CYCLES: Average {avg_cycles:.1f} cycles per query. "
                f"Consider adding terminal rules or reducing max_cycles."
            )

        return recommendations if recommendations else ["No specific tuning recommendations - system performing well"]

    def _generate_rollout_recommendation(
        self,
        pipeline: SystemMetrics,
        production: SystemMetrics,
        winner: str
    ) -> str:
        """Generiert Rollout-Empfehlung"""
        if winner == "Production System":
            # Production System ist besser - empfehle schrittweisen Rollout
            confidence_improvement = (production.avg_confidence - pipeline.avg_confidence) * 100
            speed_improvement = (pipeline.avg_response_time - production.avg_response_time) / pipeline.avg_response_time * 100

            if confidence_improvement > 5 and speed_improvement > 10:
                return (
                    "ROLLOUT: RECOMMENDED (High Priority)\n"
                    f"- Confidence improvement: +{confidence_improvement:.1f}%\n"
                    f"- Speed improvement: +{speed_improvement:.1f}%\n"
                    f"- Success rate: {production.success_rate*100:.1f}%\n\n"
                    "Recommended schedule:\n"
                    "  Week 1: 50% Production Weight (A/B testing, current)\n"
                    "  Week 2: 75% Production Weight (if no issues)\n"
                    "  Week 3: 100% Production Weight (full rollout)\n"
                    "  Week 4+: Monitor, keep Pipeline as fallback"
                )
            else:
                return (
                    "ROLLOUT: RECOMMENDED (Medium Priority)\n"
                    f"- Confidence improvement: +{confidence_improvement:.1f}%\n"
                    f"- Speed improvement: +{speed_improvement:.1f}%\n\n"
                    "Recommended schedule:\n"
                    "  Week 1-2: 50% Production Weight (extended A/B testing)\n"
                    "  Week 3: 75% Production Weight\n"
                    "  Week 4+: 100% Production Weight if metrics stable"
                )

        elif winner == "Pipeline":
            # Pipeline ist besser - NICHT ausrollen
            confidence_decline = (pipeline.avg_confidence - production.avg_confidence) * 100
            return (
                "ROLLOUT: NOT RECOMMENDED\n"
                f"- Pipeline outperforms Production System\n"
                f"- Confidence decline: -{confidence_decline:.1f}%\n\n"
                "Actions:\n"
                "  1. Investigate why Production System underperforms\n"
                "  2. Review rule utilities and specificities\n"
                "  3. Re-evaluate after tuning\n"
                "  4. Keep Production Weight at 50% or lower"
            )

        else:
            # Tie - vorsichtig vorgehen
            return (
                "ROLLOUT: CAUTIOUS APPROACH\n"
                "- Both systems perform similarly\n"
                "- No clear winner\n\n"
                "Recommended schedule:\n"
                "  Week 1-3: 50% Production Weight (monitor closely)\n"
                "  Week 4: Decide based on user feedback and edge case analysis"
            )


# ═══════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════

def generate_report_text(report: EvaluationReport) -> str:
    """Generiert formatierten Text-Report"""
    lines = [
        "═" * 80,
        "PRODUCTION SYSTEM EVALUATION REPORT",
        "═" * 80,
        f"Timestamp: {report.timestamp}",
        f"Total Queries: {report.total_queries}",
        "",
        "─" * 80,
        "PIPELINE METRICS",
        "─" * 80,
        f"Queries Handled: {report.pipeline_metrics.queries_handled}",
        f"Avg Confidence: {report.pipeline_metrics.avg_confidence:.3f}",
        f"Avg Response Time: {report.pipeline_metrics.avg_response_time:.3f}s",
        f"P50 Response Time: {report.pipeline_metrics.p50_response_time:.3f}s",
        f"P95 Response Time: {report.pipeline_metrics.p95_response_time:.3f}s",
        f"P99 Response Time: {report.pipeline_metrics.p99_response_time:.3f}s",
        f"Success Rate: {report.pipeline_metrics.success_rate*100:.1f}%",
        "",
        "─" * 80,
        "PRODUCTION SYSTEM METRICS",
        "─" * 80,
        f"Queries Handled: {report.production_metrics.queries_handled}",
        f"Avg Confidence: {report.production_metrics.avg_confidence:.3f}",
        f"Avg Response Time: {report.production_metrics.avg_response_time:.3f}s",
        f"P50 Response Time: {report.production_metrics.p50_response_time:.3f}s",
        f"P95 Response Time: {report.production_metrics.p95_response_time:.3f}s",
        f"P99 Response Time: {report.production_metrics.p99_response_time:.3f}s",
        f"Success Rate: {report.production_metrics.success_rate*100:.1f}%",
        f"Avg Cycles: {report.production_metrics.avg_cycles:.1f}",
        f"Total Rules Applied: {report.production_metrics.total_rules_applied}",
        "",
        "Top 5 Most Used Rules:",
    ]

    # Top 5 Regeln
    rule_usage = sorted(
        report.production_metrics.rule_usage_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for i, (rule, count) in enumerate(rule_usage, 1):
        lines.append(f"  {i}. {rule}: {count} times ({count/report.production_metrics.queries_handled*100:.1f}%)")

    lines.extend([
        "",
        "─" * 80,
        "COMPARISON",
        "─" * 80,
        f"Winner: {report.winner}",
        f"Confidence Difference: {report.confidence_diff:+.3f} ({'Production' if report.confidence_diff > 0 else 'Pipeline'} better)",
        f"Speed Difference: {report.speed_diff:+.3f}s ({'Production' if report.speed_diff > 0 else 'Pipeline'} faster)",
        "",
        "─" * 80,
        "BOTTLENECKS",
        "─" * 80,
    ])

    for bottleneck in report.bottlenecks:
        lines.append(f"• {bottleneck}")

    lines.extend([
        "",
        "─" * 80,
        "TUNING RECOMMENDATIONS",
        "─" * 80,
    ])

    for rec in report.tuning_recommendations:
        lines.append(f"• {rec}")

    lines.extend([
        "",
        "─" * 80,
        "ROLLOUT RECOMMENDATION",
        "─" * 80,
        report.rollout_recommendation,
        "",
        "═" * 80,
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate Production System")
    parser.add_argument("--queries", type=int, default=1000, help="Number of test queries (default: 1000)")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--report", type=str, default="evaluation_report.txt", help="Output text report")
    args = parser.parse_args()

    # Initialisiere System
    print("Initializing KAI components...")
    netzwerk = KonzeptNetzwerk()
    evaluator = ProductionSystemEvaluator(netzwerk)

    # Führe Evaluation aus
    print(f"Running evaluation with {args.queries} queries...")
    report = evaluator.run_evaluation(num_queries=args.queries)

    # Speichere JSON-Ergebnisse
    print(f"Saving results to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # Generiere und speichere Text-Report
    print(f"Generating report to {args.report}...")
    report_text = generate_report_text(report)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Zeige Report auf Console
    print("\n" + report_text)

    print(f"\n✓ Evaluation complete!")
    print(f"  - Results: {args.output}")
    print(f"  - Report: {args.report}")


if __name__ == "__main__":
    main()
