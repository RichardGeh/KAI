# test_introspection_manual.py
"""
Manueller Test für Introspektions-Queries (PHASE 9).

Testet die IntrospectionStrategy mit echten Queries.
"""

from component_1_netzwerk import KonzeptNetzwerk
from component_54_production_system import (
    ProductionSystemEngine,
    create_all_content_selection_rules,
)

def test_introspection_queries():
    """Test IntrospectionStrategy mit verschiedenen Queries."""
    print("=" * 80)
    print("PHASE 9: Introspektions-Queries Test")
    print("=" * 80)

    # Setup
    netzwerk = KonzeptNetzwerk()

    # Erstelle Production System Engine und speichere Regeln zu Neo4j
    engine = ProductionSystemEngine(neo4j_repository=netzwerk._production_rules)
    rules = create_all_content_selection_rules()
    engine.add_rules(rules)

    # Initial Save
    saved_count = engine.save_rules_to_neo4j()
    print(f"\n✓ {saved_count} Regeln zu Neo4j gespeichert\n")

    # Simuliere Regelanwendungen (für Stats)
    for i, rule in enumerate(rules[:5]):
        rule.application_count = (i + 1) * 10
        rule.success_count = (i + 1) * 8

    # Sync Stats
    engine.sync_rule_stats_to_neo4j(force=True)
    print("✓ Stats zu Neo4j synchronisiert\n")

    # Test verschiedene Introspektions-Queries
    test_queries = [
        ("Alle Regeln", "query_production_rules", {}),
        ("Content-Selection", "query_production_rules", {"category": "content_selection"}),
        ("Top 5 meistverwendet", "query_production_rules", {"min_application_count": 1, "order_by": "usage", "limit": 5}),
        ("Niedrige Utility", "query_production_rules", {"max_utility": 0.5}),
        ("Statistiken", "get_production_rule_statistics", {}),
    ]

    for query_name, method_name, kwargs in test_queries:
        print(f"--- {query_name} ---")
        method = getattr(netzwerk, method_name)
        result = method(**kwargs)

        if method_name == "get_production_rule_statistics":
            print(f"Total Rules: {result.get('total_rules', 0)}")
            print(f"Categories: {len(result.get('by_category', {}))}")
            print(f"Most Used: {len(result.get('most_used', []))}")
        else:
            print(f"Ergebnisse: {len(result)} Regeln")
            if result:
                for rule in result[:3]:
                    print(f"  - {rule['name']} (util={rule['utility']:.2f}, apps={rule['application_count']})")
        print()

    # Cleanup
    netzwerk.close()
    print("✓ Test abgeschlossen\n")
    print("=" * 80)

if __name__ == "__main__":
    test_introspection_queries()
