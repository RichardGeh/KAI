# component_1_netzwerk_production_rules.py
"""
Production Rule Repository for Neo4j persistence.

This module handles:
- Production rule creation and persistence
- Rule retrieval (single, all, filtered queries)
- Statistics updates (application_count, success_count, last_applied)
- Introspection queries (most used rules, low utility rules, etc.)

PHASE 9: Neo4j Rule Repository (Woche 14)
"""

import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger
from infrastructure.cache_manager import cache_manager

logger = get_logger(__name__)


def _deserialize_rule_record(record) -> Dict[str, Any]:
    """
    Hilfsfunktion: Deserialisiert einen Neo4j Record zu Rule-Dict.

    Args:
        record: Neo4j Record mit Rule-Daten

    Returns:
        Dict mit deserialisierten Rule-Daten
    """
    # Deserialisiere metadata_json
    metadata_json = record.get("metadata_json") or "{}"
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        logger.warning(
            f"_deserialize_rule_record: Ungültiges JSON in metadata für '{record.get('name')}'"
        )
        metadata = {}

    return {
        "name": record["name"],
        "category": record["category"],
        "condition_code": record["condition_code"],
        "action_code": record["action_code"],
        "utility": record["utility"],
        "specificity": record["specificity"],
        "metadata": metadata,
        "application_count": record["application_count"],
        "success_count": record["success_count"],
        "created_at": record["created_at"],
        "last_applied": record["last_applied"],
    }


class KonzeptNetzwerkProductionRules:
    """
    Production Rule Repository für Neo4j-basierte Persistierung.

    Performance-Optimierung:
    - TTL-Cache für häufig abgefragte Regeln (10 Minuten TTL)
    - Batch-Updates für Statistics (alle 10 Queries)
    """

    def __init__(self, driver: Driver):
        """
        Initialize with an existing Neo4j driver.

        Args:
            driver: Neo4j driver instance from KonzeptNetzwerkCore
        """
        self.driver = driver

        # Cache für Production Rules (10 Minuten TTL, da sich diese selten ändern) via CacheManager
        cache_manager.register_cache("production_rules", maxsize=100, ttl=600)

        # Pending updates (für Batch-Processing)
        self._pending_stats_updates: Dict[str, Dict[str, Any]] = {}
        self._update_counter = 0

        # FIX: Thread-Lock für Race Condition Protection (Code Review 2025-11-21, Concern 11)
        self._update_lock = threading.Lock()

    def create_production_rule(
        self,
        name: str,
        category: str,
        condition_code: str,
        action_code: str,
        utility: float = 1.0,
        specificity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Erstellt oder aktualisiert eine Production Rule in Neo4j.

        Args:
            name: Eindeutiger Name der Regel
            category: Kategorie (content_selection, lexicalization, discourse, syntax)
            condition_code: Serialisierter Code für Condition (z.B. via pickle/dill)
            action_code: Serialisierter Code für Action
            utility: Statische Utility (Präferenz)
            specificity: Spezifität der Regel
            metadata: Zusätzliche Metadaten (Tags, Beschreibung)

        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        if not self.driver:
            logger.error("create_production_rule: Kein DB-Driver verfügbar")
            return False

        metadata = metadata or {}

        # Serialisiere metadata als JSON (Neo4j unterstützt keine verschachtelten Maps)
        metadata_json = json.dumps(metadata)

        try:
            with self.driver.session(database="neo4j") as session:
                # Erstelle/Aktualisiere Rule Node mit expliziter Transaktion
                def create_or_update_rule(tx):
                    result = tx.run(
                        """
                        MERGE (pr:ProductionRule {name: $name})
                        ON CREATE SET
                            pr.category = $category,
                            pr.condition_code = $condition_code,
                            pr.action_code = $action_code,
                            pr.utility = $utility,
                            pr.specificity = $specificity,
                            pr.metadata_json = $metadata_json,
                            pr.application_count = 0,
                            pr.success_count = 0,
                            pr.created_at = timestamp(),
                            pr.last_applied = null
                        ON MATCH SET
                            pr.category = $category,
                            pr.condition_code = $condition_code,
                            pr.action_code = $action_code,
                            pr.utility = $utility,
                            pr.specificity = $specificity,
                            pr.metadata_json = $metadata_json
                        RETURN pr.name AS name
                        """,
                        name=name,
                        category=category,
                        condition_code=condition_code,
                        action_code=action_code,
                        utility=utility,
                        specificity=specificity,
                        metadata_json=metadata_json,
                    )
                    return result.single()

                record = session.execute_write(create_or_update_rule)

                if not record:
                    logger.error(
                        f"create_production_rule: Keine Rückgabe für Regel '{name}'"
                    )
                    return False

                # Cache invalidieren
                cache_manager.invalidate("production_rules", name)

                logger.info(
                    f"Production Rule '{name}' erfolgreich erstellt/aktualisiert",
                    extra={"category": category, "utility": utility},
                )
                return True

        except Exception as e:
            logger.error(
                f"create_production_rule: Fehler beim Erstellen von Regel '{name}'",
                extra={"error": str(e)},
                exc_info=True,
            )
            return False

    def get_production_rule(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Lädt eine Production Rule aus Neo4j.

        Args:
            name: Name der Regel

        Returns:
            Dict mit Rule-Daten oder None wenn nicht gefunden
        """
        if not self.driver:
            logger.error("get_production_rule: Kein DB-Driver verfügbar")
            return None

        # Prüfe Cache
        cached_rule = cache_manager.get("production_rules", name)
        if cached_rule is not None:
            logger.debug(f"get_production_rule: Cache Hit für '{name}'")
            return cached_rule

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (pr:ProductionRule {name: $name})
                    RETURN pr.name AS name,
                           pr.category AS category,
                           pr.condition_code AS condition_code,
                           pr.action_code AS action_code,
                           pr.utility AS utility,
                           pr.specificity AS specificity,
                           pr.metadata_json AS metadata_json,
                           pr.application_count AS application_count,
                           pr.success_count AS success_count,
                           pr.created_at AS created_at,
                           pr.last_applied AS last_applied
                    """,
                    name=name,
                )

                record = result.single()

                if not record:
                    logger.debug(f"get_production_rule: Regel '{name}' nicht gefunden")
                    return None

                rule_data = _deserialize_rule_record(record)

                # Cache update
                cache_manager.set("production_rules", name, rule_data)

                logger.debug(f"get_production_rule: Regel '{name}' geladen")
                return rule_data

        except Exception as e:
            logger.error(
                f"get_production_rule: Fehler beim Laden von Regel '{name}'",
                extra={"error": str(e)},
                exc_info=True,
            )
            return None

    def get_all_production_rules(self) -> List[Dict[str, Any]]:
        """
        Lädt alle Production Rules aus Neo4j.

        Returns:
            Liste von Rule-Dictionaries
        """
        if not self.driver:
            logger.error("get_all_production_rules: Kein DB-Driver verfügbar")
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (pr:ProductionRule)
                    RETURN pr.name AS name,
                           pr.category AS category,
                           pr.condition_code AS condition_code,
                           pr.action_code AS action_code,
                           pr.utility AS utility,
                           pr.specificity AS specificity,
                           pr.metadata_json AS metadata_json,
                           pr.application_count AS application_count,
                           pr.success_count AS success_count,
                           pr.created_at AS created_at,
                           pr.last_applied AS last_applied
                    ORDER BY pr.category, pr.name
                    """
                )

                rules = []
                for record in result:
                    rule_data = _deserialize_rule_record(record)
                    rules.append(rule_data)

                logger.info(f"get_all_production_rules: {len(rules)} Regeln geladen")
                return rules

        except Exception as e:
            logger.error(
                "get_all_production_rules: Fehler beim Laden",
                extra={"error": str(e)},
                exc_info=True,
            )
            return []

    def update_production_rule_stats(
        self,
        name: str,
        application_count: Optional[int] = None,
        success_count: Optional[int] = None,
        last_applied: Optional[datetime] = None,
        force_sync: bool = False,
    ) -> bool:
        """
        Aktualisiert Statistiken einer Production Rule.

        Verwendet Batch-Processing: Updates werden gesammelt und alle 10 Aufrufe
        synchronisiert, außer force_sync=True.

        Args:
            name: Name der Regel
            application_count: Neue application_count (oder None für Inkrement um 1)
            success_count: Neue success_count (oder None für keine Änderung)
            last_applied: Timestamp der letzten Anwendung
            force_sync: Sofort synchronisieren (überspringt Batching)

        Returns:
            bool: True wenn erfolgreich (oder gebatched), False bei Fehler
        """
        if not self.driver:
            logger.error("update_production_rule_stats: Kein DB-Driver verfügbar")
            return False

        # FIX: Thread-safe Update mit Lock (Code Review 2025-11-21, Concern 11)
        with self._update_lock:
            # Sammle Update
            if name not in self._pending_stats_updates:
                self._pending_stats_updates[name] = {}

            if application_count is not None:
                self._pending_stats_updates[name][
                    "application_count"
                ] = application_count
            if success_count is not None:
                self._pending_stats_updates[name]["success_count"] = success_count
            if last_applied is not None:
                self._pending_stats_updates[name]["last_applied"] = last_applied

            self._update_counter += 1

            # Synchronisiere wenn Threshold erreicht oder force_sync
            if force_sync or self._update_counter >= 10:
                return self._flush_pending_stats()

            logger.debug(
                f"update_production_rule_stats: Update für '{name}' gebatched "
                f"({self._update_counter}/10)"
            )
            return True

    def _flush_pending_stats(self) -> bool:
        """
        Synchronisiert alle pending Stats zu Neo4j.

        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        if not self._pending_stats_updates:
            self._update_counter = 0
            return True

        try:
            with self.driver.session(database="neo4j") as session:
                for rule_name, updates in self._pending_stats_updates.items():
                    # Baue SET-Clause dynamisch
                    set_clauses = []
                    params = {"name": rule_name}

                    if "application_count" in updates:
                        set_clauses.append("pr.application_count = $app_count")
                        params["app_count"] = updates["application_count"]

                    if "success_count" in updates:
                        set_clauses.append("pr.success_count = $succ_count")
                        params["succ_count"] = updates["success_count"]

                    if "last_applied" in updates:
                        # Konvertiere datetime zu Timestamp (milliseconds since epoch)
                        timestamp = int(updates["last_applied"].timestamp() * 1000)
                        set_clauses.append("pr.last_applied = $last_app")
                        params["last_app"] = timestamp

                    if not set_clauses:
                        continue

                    query = f"""
                    MATCH (pr:ProductionRule {{name: $name}})
                    SET {', '.join(set_clauses)}
                    RETURN pr.name AS name
                    """

                    result = session.run(query, **params)
                    record = result.single()

                    if not record:
                        logger.warning(
                            f"_flush_pending_stats: Regel '{rule_name}' nicht gefunden"
                        )

                    # Invalidiere Cache
                    cache_manager.invalidate("production_rules", rule_name)

                logger.info(
                    f"_flush_pending_stats: {len(self._pending_stats_updates)} Regeln aktualisiert"
                )

                # Reset
                self._pending_stats_updates.clear()
                self._update_counter = 0
                return True

        except Exception as e:
            logger.error(
                "_flush_pending_stats: Fehler beim Synchronisieren",
                extra={"error": str(e)},
                exc_info=True,
            )
            return False

    def query_production_rules(
        self,
        category: Optional[str] = None,
        min_utility: Optional[float] = None,
        max_utility: Optional[float] = None,
        min_application_count: Optional[int] = None,
        order_by: str = "priority",  # 'priority', 'usage', 'name'
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Introspektions-Query für Production Rules mit Filterung.

        Args:
            category: Filter nach Kategorie (z.B. 'content_selection')
            min_utility: Minimale Utility
            max_utility: Maximale Utility
            min_application_count: Minimale Anwendungsanzahl
            order_by: Sortierung ('priority', 'usage', 'name')
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Liste von gefilterten/sortierten Rule-Dictionaries
        """
        if not self.driver:
            logger.error("query_production_rules: Kein DB-Driver verfügbar")
            return []

        try:
            # Baue WHERE-Clauses dynamisch
            where_clauses = []
            params = {}

            if category is not None:
                where_clauses.append("pr.category = $category")
                params["category"] = category

            if min_utility is not None:
                where_clauses.append("pr.utility >= $min_utility")
                params["min_utility"] = min_utility

            if max_utility is not None:
                where_clauses.append("pr.utility <= $max_utility")
                params["max_utility"] = max_utility

            if min_application_count is not None:
                where_clauses.append("pr.application_count >= $min_app_count")
                params["min_app_count"] = min_application_count

            where_clause = (
                f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            )

            # Baue ORDER BY dynamisch
            if order_by == "priority":
                order_clause = "ORDER BY (pr.utility * pr.specificity) DESC"
            elif order_by == "usage":
                order_clause = "ORDER BY pr.application_count DESC"
            elif order_by == "name":
                order_clause = "ORDER BY pr.name ASC"
            else:
                order_clause = "ORDER BY pr.category, pr.name"

            # Baue LIMIT dynamisch
            limit_clause = f"LIMIT {limit}" if limit is not None else ""

            query = f"""
            MATCH (pr:ProductionRule)
            {where_clause}
            RETURN pr.name AS name,
                   pr.category AS category,
                   pr.condition_code AS condition_code,
                   pr.action_code AS action_code,
                   pr.utility AS utility,
                   pr.specificity AS specificity,
                   pr.metadata_json AS metadata_json,
                   pr.application_count AS application_count,
                   pr.success_count AS success_count,
                   pr.created_at AS created_at,
                   pr.last_applied AS last_applied,
                   (pr.utility * pr.specificity) AS priority
            {order_clause}
            {limit_clause}
            """

            with self.driver.session(database="neo4j") as session:
                result = session.run(query, **params)

                rules = []
                for record in result:
                    rule_data = _deserialize_rule_record(record)
                    rule_data["priority"] = record["priority"]  # Add priority field
                    rules.append(rule_data)

                logger.info(
                    f"query_production_rules: {len(rules)} Regeln gefunden",
                    extra={"filters": params, "order_by": order_by},
                )
                return rules

        except Exception as e:
            logger.error(
                "query_production_rules: Fehler beim Query",
                extra={"error": str(e)},
                exc_info=True,
            )
            return []

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Gibt aggregierte Statistiken über alle Production Rules zurück.

        Returns:
            Dict mit Statistiken (total_rules, by_category, most_used, etc.)
        """
        if not self.driver:
            logger.error("get_rule_statistics: Kein DB-Driver verfügbar")
            return {}

        try:
            with self.driver.session(database="neo4j") as session:
                # Gesamtzahl und Kategorie-Verteilung
                result = session.run(
                    """
                    MATCH (pr:ProductionRule)
                    RETURN count(pr) AS total_rules,
                           pr.category AS category,
                           count(*) AS count_per_category,
                           avg(pr.utility) AS avg_utility,
                           avg(pr.application_count) AS avg_applications
                    ORDER BY category
                    """
                )

                stats = {
                    "total_rules": 0,
                    "by_category": {},
                    "avg_utility": 0.0,
                    "avg_applications": 0.0,
                }

                for record in result:
                    stats["total_rules"] = record["total_rules"]
                    category = record["category"]
                    stats["by_category"][category] = {
                        "count": record["count_per_category"],
                        "avg_utility": round(record["avg_utility"], 2),
                        "avg_applications": round(record["avg_applications"], 2),
                    }

                # Top 5 meistverwendete Regeln
                result = session.run(
                    """
                    MATCH (pr:ProductionRule)
                    WHERE pr.application_count > 0
                    RETURN pr.name AS name,
                           pr.category AS category,
                           pr.application_count AS count
                    ORDER BY pr.application_count DESC
                    LIMIT 5
                    """
                )

                stats["most_used"] = [
                    {
                        "name": record["name"],
                        "category": record["category"],
                        "count": record["count"],
                    }
                    for record in result
                ]

                # Regeln mit niedriger Utility
                result = session.run(
                    """
                    MATCH (pr:ProductionRule)
                    WHERE pr.utility < 0.5
                    RETURN pr.name AS name,
                           pr.category AS category,
                           pr.utility AS utility
                    ORDER BY pr.utility ASC
                    LIMIT 5
                    """
                )

                stats["low_utility"] = [
                    {
                        "name": record["name"],
                        "category": record["category"],
                        "utility": record["utility"],
                    }
                    for record in result
                ]

                logger.info(
                    f"get_rule_statistics: Statistiken für {stats['total_rules']} Regeln erstellt"
                )
                return stats

        except Exception as e:
            logger.error(
                "get_rule_statistics: Fehler beim Erstellen der Statistiken",
                extra={"error": str(e)},
                exc_info=True,
            )
            return {}

    def clear_cache(self):
        """Invalidiert den Rule-Cache."""
        cache_manager.invalidate("production_rules")
        logger.debug("Production Rule Cache geleert")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Gibt Cache-Statistiken zurück."""
        cache_stats = cache_manager.get_stats("production_rules")

        return {
            "rule_cache": {
                "size": cache_stats["size"],
                "maxsize": cache_stats["maxsize"],
                "ttl": cache_stats["ttl"],
                "hits": cache_stats["hits"],
                "misses": cache_stats["misses"],
                "hit_rate": cache_stats["hit_rate"],
            },
            "pending_updates": len(self._pending_stats_updates),
            "update_counter": self._update_counter,
        }
