# component_1_netzwerk_word_usage.py
"""
Word usage tracking and context management.

This module provides functionality for storing and querying word usage patterns:
- UsageContext Nodes: Authentic text fragments showing how words are used
- CONNECTION Edges: N-gram statistics for direct word connections
- Similarity-based counter updates for flexible pattern learning
"""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import PerformanceLogger, get_logger
from kai_config import get_config

logger = get_logger(__name__)
config = get_config()


class KonzeptNetzwerkWordUsage:
    """
    Word usage tracking functionality for Neo4j knowledge graph.

    Provides two layers of word connection tracking:
    1. CONNECTION Edges: Direct N-gram statistics (word1 -> word2, distance, count)
    2. UsageContext Nodes: Authentic text fragments with word position
    """

    def __init__(self, driver: Optional[Driver] = None):
        """
        Args:
            driver: Neo4j driver instance (shared with KonzeptNetzwerkCore)
        """
        self.driver = driver
        logger.debug("KonzeptNetzwerkWordUsage initialisiert")

    def _create_constraints(self):
        """Erstellt Neo4j Constraints für UsageContext Nodes"""
        if not self.driver:
            return

        try:
            with self.driver.session(database="neo4j") as session:
                # UsageContext braucht unique ID
                constraint_query = """
                CREATE CONSTRAINT UsageContextId IF NOT EXISTS
                FOR (uc:UsageContext) REQUIRE uc.id IS UNIQUE
                """
                session.run(constraint_query)
                logger.debug("Constraint 'UsageContextId' erstellt/verifiziert")

        except Exception as e:
            logger.warning(
                "Fehler beim Erstellen von UsageContext Constraints",
                extra={"error": str(e)},
            )

    # ========================================================================
    # CONNECTION EDGES (Layer 1: N-Gram Statistics)
    # ========================================================================

    def add_word_connection(
        self,
        word1_lemma: str,
        word2_lemma: str,
        distance: int = 1,
        direction: str = "before",
    ) -> bool:
        """
        Erstellt oder aktualisiert CONNECTION Edge zwischen zwei Wörtern.

        Args:
            word1_lemma: Erstes Wort (normalisiert)
            word2_lemma: Zweites Wort (normalisiert)
            distance: Abstand zwischen Wörtern (1-3)
            direction: "before" (word1 kommt VOR word2) oder "after"

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.driver:
            logger.error("add_word_connection: Kein DB-Driver verfügbar")
            return False

        word1_lemma = word1_lemma.lower()
        word2_lemma = word2_lemma.lower()

        try:
            with PerformanceLogger(
                logger.logger,
                "add_word_connection",
                word1=word1_lemma,
                word2=word2_lemma,
                distance=distance,
            ):
                with self.driver.session(database="neo4j") as session:
                    # Cypher: MERGE Edge mit Properties, erhöhe count bei Duplikat
                    query = """
                    MATCH (w1:Wort {lemma: $word1})
                    MATCH (w2:Wort {lemma: $word2})
                    MERGE (w1)-[c:CONNECTION {
                        direction: $direction,
                        distance: $distance
                    }]->(w2)
                    ON CREATE SET c.count = 1, c.created_at = datetime()
                    ON MATCH SET c.count = c.count + 1
                    SET c.last_seen = datetime()
                    RETURN c.count AS count
                    """

                    result = session.run(
                        query,
                        word1=word1_lemma,
                        word2=word2_lemma,
                        direction=direction,
                        distance=distance,
                    )

                    record = result.single()
                    if record:
                        count = record["count"]
                        logger.debug(
                            "CONNECTION Edge aktualisiert",
                            extra={
                                "word1": word1_lemma,
                                "word2": word2_lemma,
                                "distance": distance,
                                "direction": direction,
                                "count": count,
                            },
                        )
                        return True
                    else:
                        logger.warning(
                            "CONNECTION Edge konnte nicht erstellt werden (Wörter fehlen?)"
                        )
                        return False

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler beim Erstellen von CONNECTION Edge",
                word1=word1_lemma,
                word2=word2_lemma,
            )
            return False

    def get_word_connections(
        self, word_lemma: str, direction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Gibt alle CONNECTION Edges für ein Wort zurück.

        Args:
            word_lemma: Wort (normalisiert)
            direction: Optional filter ("before" oder "after")

        Returns:
            Liste von Dicts mit {connected_word, distance, direction, count, confidence}
        """
        if not self.driver:
            logger.error("get_word_connections: Kein DB-Driver verfügbar")
            return []

        word_lemma = word_lemma.lower()

        try:
            with self.driver.session(database="neo4j") as session:
                # Query abhängig von direction
                if direction:
                    query = """
                    MATCH (w:Wort {lemma: $word})-[c:CONNECTION {direction: $direction}]->(w2:Wort)
                    RETURN w2.lemma AS connected_word, c.distance AS distance,
                           c.direction AS direction, c.count AS count
                    ORDER BY c.count DESC
                    """
                    result = session.run(query, word=word_lemma, direction=direction)
                else:
                    query = """
                    MATCH (w:Wort {lemma: $word})-[c:CONNECTION]->(w2:Wort)
                    RETURN w2.lemma AS connected_word, c.distance AS distance,
                           c.direction AS direction, c.count AS count
                    ORDER BY c.count DESC
                    """
                    result = session.run(query, word=word_lemma)

                connections = []
                total_count = 0

                # Sammle alle Connections
                for record in result:
                    count = record["count"]
                    total_count += count
                    connections.append(
                        {
                            "connected_word": record["connected_word"],
                            "distance": record["distance"],
                            "direction": record["direction"],
                            "count": count,
                        }
                    )

                # Berechne Confidence (Anteil an Gesamt-Vorkommen)
                for conn in connections:
                    conn["confidence"] = conn["count"] / max(1, total_count)

                logger.debug(
                    "Word connections abgerufen",
                    extra={"word": word_lemma, "connection_count": len(connections)},
                )

                return connections

        except Exception as e:
            logger.log_exception(
                e, "Fehler beim Abrufen von Word Connections", word=word_lemma
            )
            return []

    # ========================================================================
    # USAGE CONTEXT NODES (Layer 2: Authentic Text Fragments)
    # ========================================================================

    def add_usage_context(
        self,
        word_lemma: str,
        fragment: str,
        word_position: int,
        fragment_type: str = "window",
    ) -> bool:
        """
        Erstellt oder aktualisiert UsageContext Node für ein Wort.

        Bei exakter Übereinstimmung wird nur der Counter erhöht.
        Bei ähnlichen Fragmenten (basierend auf Config) werden beide Counter erhöht.

        Args:
            word_lemma: Wort (normalisiert)
            fragment: Kontext-Fragment (z.B. "im großen Haus")
            word_position: Index des Wortes im Fragment (0-basiert)
            fragment_type: "window" (±N Wörter) oder "comma_delimited" (bis Komma)

        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.driver:
            logger.error("add_usage_context: Kein DB-Driver verfügbar")
            return False

        word_lemma = word_lemma.lower()
        fragment = fragment.strip()

        # Fragment-Normalisierung für Matching (aber original speichern)
        self._normalize_fragment(fragment)

        try:
            with PerformanceLogger(
                logger.logger,
                "add_usage_context",
                word=word_lemma,
                fragment_preview=fragment[:30],
            ):
                with self.driver.session(database="neo4j") as session:
                    # 1. Prüfe auf exakte Übereinstimmung
                    exact_match_query = """
                    MATCH (w:Wort {lemma: $word})-[o:OCCURS_IN]->(uc:UsageContext {fragment: $fragment})
                    SET o.count = o.count + 1, o.last_seen = datetime()
                    SET uc.last_seen = datetime()
                    RETURN uc.id AS id, o.count AS count
                    """

                    result = session.run(
                        exact_match_query, word=word_lemma, fragment=fragment
                    )

                    record = result.single()
                    if record:
                        # Exakte Übereinstimmung gefunden
                        count = record["count"]
                        logger.debug(
                            "UsageContext exakte Übereinstimmung (Counter erhöht)",
                            extra={
                                "word": word_lemma,
                                "fragment": fragment,
                                "count": count,
                            },
                        )
                        return True

                    # 2. Prüfe auf ähnliche Fragmente (basierend auf Schwellenwert)
                    similarity_threshold = (
                        config.usage_similarity_threshold / 100.0
                    )  # Prozent -> Dezimal

                    # Hole alle existierenden Contexts für dieses Wort
                    existing_contexts_query = """
                    MATCH (w:Wort {lemma: $word})-[o:OCCURS_IN]->(uc:UsageContext)
                    RETURN uc.fragment AS fragment, uc.id AS id, o.count AS count
                    """

                    existing_results = session.run(
                        existing_contexts_query, word=word_lemma
                    )

                    for existing_record in existing_results:
                        existing_fragment = existing_record["fragment"]
                        existing_id = existing_record["id"]

                        # Berechne Ähnlichkeit
                        similarity = self.calculate_fragment_similarity(
                            fragment, existing_fragment
                        )

                        # Wenn ähnlich genug: Erhöhe Counter des existierenden UND erstelle neuen
                        if similarity >= similarity_threshold and similarity < 1.0:
                            # Erhöhe Counter des existierenden Context
                            update_query = """
                            MATCH (w:Wort {lemma: $word})-[o:OCCURS_IN]->(uc:UsageContext {id: $existing_id})
                            SET o.count = o.count + 1, o.last_seen = datetime()
                            SET uc.last_seen = datetime()
                            RETURN o.count AS count
                            """

                            update_result = session.run(
                                update_query, word=word_lemma, existing_id=existing_id
                            )

                            update_record = update_result.single()
                            if update_record:
                                logger.info(
                                    "Ähnlicher UsageContext gefunden - beide Counter erhöht",
                                    extra={
                                        "word": word_lemma,
                                        "new_fragment": fragment,
                                        "existing_fragment": existing_fragment,
                                        "similarity": f"{similarity:.2%}",
                                        "existing_count": update_record["count"],
                                    },
                                )
                                # Breche nicht ab - erstelle trotzdem neuen Context unten

                    # 3. Erstelle neuen Context (entweder kein ähnlicher gefunden, oder trotz Ähnlichkeit)
                    context_id = str(uuid.uuid4())
                    total_words = len(fragment.split())
                    timestamp = datetime.now().isoformat()

                    create_query = """
                    MATCH (w:Wort {lemma: $word})
                    CREATE (uc:UsageContext {
                        id: $context_id,
                        fragment: $fragment,
                        word_position: $word_position,
                        fragment_type: $fragment_type,
                        total_words: $total_words,
                        created_at: $timestamp,
                        last_seen: $timestamp
                    })
                    CREATE (w)-[o:OCCURS_IN {
                        count: 1,
                        created_at: $timestamp,
                        last_seen: $timestamp
                    }]->(uc)
                    RETURN uc.id AS id
                    """

                    result = session.run(
                        create_query,
                        word=word_lemma,
                        context_id=context_id,
                        fragment=fragment,
                        word_position=word_position,
                        fragment_type=fragment_type,
                        total_words=total_words,
                        timestamp=timestamp,
                    )

                    record = result.single()
                    if record:
                        logger.info(
                            "Neuer UsageContext erstellt",
                            extra={
                                "word": word_lemma,
                                "fragment": fragment,
                                "context_id": context_id,
                            },
                        )
                        return True
                    else:
                        logger.warning(
                            "UsageContext konnte nicht erstellt werden (Wort fehlt?)"
                        )
                        return False

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler beim Erstellen von UsageContext",
                word=word_lemma,
                fragment=fragment,
            )
            return False

    def get_usage_contexts(
        self, word_lemma: str, min_count: int = 1, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Gibt alle UsageContext Nodes für ein Wort zurück.

        Args:
            word_lemma: Wort (normalisiert)
            min_count: Minimale Anzahl an Vorkommen (Filter)
            limit: Maximale Anzahl an Ergebnissen (Top-N)

        Returns:
            Liste von Dicts mit {fragment, word_position, count, fragment_type, ...}
        """
        if not self.driver:
            logger.error("get_usage_contexts: Kein DB-Driver verfügbar")
            return []

        word_lemma = word_lemma.lower()

        try:
            with self.driver.session(database="neo4j") as session:
                query = """
                MATCH (w:Wort {lemma: $word})-[o:OCCURS_IN]->(uc:UsageContext)
                WHERE o.count >= $min_count
                RETURN uc.fragment AS fragment,
                       uc.word_position AS word_position,
                       uc.fragment_type AS fragment_type,
                       uc.total_words AS total_words,
                       o.count AS count,
                       uc.created_at AS created_at,
                       uc.last_seen AS last_seen
                ORDER BY o.count DESC
                LIMIT $limit
                """

                result = session.run(
                    query, word=word_lemma, min_count=min_count, limit=limit
                )

                contexts = []
                for record in result:
                    contexts.append(
                        {
                            "fragment": record["fragment"],
                            "word_position": record["word_position"],
                            "fragment_type": record["fragment_type"],
                            "total_words": record["total_words"],
                            "count": record["count"],
                            "created_at": record["created_at"],
                            "last_seen": record["last_seen"],
                        }
                    )

                logger.debug(
                    "Usage contexts abgerufen",
                    extra={"word": word_lemma, "context_count": len(contexts)},
                )

                return contexts

        except Exception as e:
            logger.log_exception(
                e, "Fehler beim Abrufen von Usage Contexts", word=word_lemma
            )
            return []

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _normalize_fragment(self, fragment: str) -> str:
        """
        Normalisiert Fragment für Ähnlichkeitsvergleich.

        - Lowercase
        - Mehrfach-Leerzeichen entfernen
        - Satzzeichen entfernen (außer Bindestriche)

        Returns:
            Normalisiertes Fragment
        """
        # Lowercase
        normalized = fragment.lower()

        # Satzzeichen entfernen (außer Bindestriche und Leerzeichen)
        normalized = re.sub(r"[^\w\s-]", "", normalized)

        # Mehrfach-Leerzeichen zu einfachem
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def calculate_fragment_similarity(self, fragment1: str, fragment2: str) -> float:
        """
        Berechnet Ähnlichkeit zwischen zwei Fragmenten (0.0 - 1.0).

        Verwendet einfache Levenshtein-basierte Methode.
        Für bessere Genauigkeit könnte später Cosine Similarity auf Embeddings verwendet werden.

        Args:
            fragment1: Erstes Fragment
            fragment2: Zweites Fragment

        Returns:
            Ähnlichkeit als Float (0.0 = komplett unterschiedlich, 1.0 = identisch)
        """
        # Normalisiere beide Fragmente
        f1 = self._normalize_fragment(fragment1)
        f2 = self._normalize_fragment(fragment2)

        if f1 == f2:
            return 1.0

        # Levenshtein Distance (vereinfachte Implementierung)
        distance = self._levenshtein_distance(f1, f2)
        max_len = max(len(f1), len(f2))

        if max_len == 0:
            return 1.0

        # Similarity = 1 - (distance / max_length)
        similarity = 1.0 - (distance / max_len)

        return max(0.0, similarity)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Berechnet Levenshtein Distance zwischen zwei Strings.

        Simple dynamic programming implementation.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer than s2
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


# ============================================================================
# CONVENIENCE FUNCTIONS (für einfachen Import)
# ============================================================================


def calculate_similarity(fragment1: str, fragment2: str) -> float:
    """
    Standalone-Funktion für Fragment-Ähnlichkeit.
    Kann ohne DB-Driver verwendet werden.
    """
    wu = KonzeptNetzwerkWordUsage(driver=None)
    return wu.calculate_fragment_similarity(fragment1, fragment2)


if __name__ == "__main__":
    # Test-Code für Ähnlichkeits-Berechnung
    print("=== Fragment Similarity Tests ===\n")

    test_pairs = [
        ("im großen Haus", "im großen Haus"),  # Identisch -> 1.0
        ("im großen Haus", "im großen alten Haus"),  # Ähnlich -> ~0.75
        ("im Haus", "am Haus"),  # Geringe Änderung -> ~0.8
        ("Katze", "Hund"),  # Komplett unterschiedlich -> ~0.0
    ]

    for f1, f2 in test_pairs:
        sim = calculate_similarity(f1, f2)
        print(f"'{f1}' vs '{f2}'")
        print(f"  Similarity: {sim:.2%}\n")
