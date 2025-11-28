# component_48_enhanced_schema.py
"""
Enhanced Graph Schema Management für KAI - Phase 1.2: Cognitive Resonance

Erweitert das Knowledge-Graph-Schema mit reichhaltigen Properties für:
- Wort/Konzept Nodes: Semantische Felder, Abstraktionslevels, Kontexte, Usage-Tracking
- Relations: Erweiterte Metadaten, Kontext-Abhängigkeit, Bidirektionalität

Design-Prinzipien:
- Non-Breaking: Alle neuen Properties sind optional
- Backwards Compatible: Alte Queries funktionieren weiterhin
- Incremental Population: Properties werden schrittweise aus bestehenden Daten befüllt
- Performance-Optimized: Batch-Updates für große Datenmengen
"""

import json
import re
import threading
from typing import Optional, Set

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ==================== CONSTANTS ====================

# Whitelist for allowed relation types (injection prevention)
ALLOWED_RELATION_TYPES: Set[str] = {
    "IS_A",
    "HAS_PROPERTY",
    "CAPABLE_OF",
    "PART_OF",
    "LOCATED_IN",
    "SYNONYM_OF",
    "SIMILAR_TO",
    "RELATED_TO",
    "EQUIVALENT_TO",
    "OPPOSITE_OF",
    "TRIGGERS",
    "CONNECTION",
}

# Entity name validation pattern
ENTITY_NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")


# ==================== SCHEMA DEFINITIONS ====================


class EnhancedNodeProperties:
    """
    Definition der erweiterten Properties für Wort/Konzept Nodes.

    Properties:
        lemma: str - Bereits vorhanden (unique constraint)
        pos: str - Part-of-Speech Tag (spaCy format)
        definitions: List[str] - Gesammelte Definitionen aus Lerne-Befehlen
        semantic_field: str - Semantisches Feld ("Politik", "Natur", etc.)
        abstraction_level: int - 1=konkret, 5=abstrakt
        contexts: List[str] - Aggregierte Kontexte aus Episodes
        typical_relations: Dict[str, int] - Häufigkeitsverteilung von Relationen
        usage_frequency: int - Wie oft in Queries verwendet
        first_seen: datetime - Wann erstmals angelegt
        last_used: datetime - Wann zuletzt in Query verwendet
    """

    # Defaults für neue Nodes
    DEFAULTS = {
        "pos": None,
        "definitions": [],
        "semantic_field": None,
        "abstraction_level": 3,  # Neutral default
        "contexts": [],
        "typical_relations": {},
        "usage_frequency": 0,
        "first_seen": None,  # Wird automatisch bei Erstellung gesetzt
        "last_used": None,
    }

    @staticmethod
    def validate_abstraction_level(level: int) -> bool:
        """Validiert Abstraktionslevel (1-5)."""
        return 1 <= level <= 5

    @staticmethod
    def infer_abstraction_level(lemma: str, pos: Optional[str] = None) -> int:
        """
        Inferiert Abstraktionslevel basierend auf Heuristiken.

        Heuristiken:
        - Eigennamen (PROPN): 1 (sehr konkret)
        - Nomen (NOUN): 2-3 (je nach Kontext)
        - Verben (VERB): 3-4 (abstrakter)
        - Adjektive (ADJ): 4 (abstrakt)
        - Abstrakte Konzepte (erkannt via Keywords): 5
        """
        # Abstrakte Keywords
        abstract_keywords = {
            "konzept",
            "idee",
            "theorie",
            "prinzip",
            "wert",
            "ethik",
            "moral",
            "gerechtigkeit",
            "freiheit",
            "wahrheit",
            "liebe",
        }

        if lemma.lower() in abstract_keywords:
            return 5

        if pos:
            pos_mapping = {
                "PROPN": 1,  # Eigennamen sehr konkret
                "NOUN": 2,  # Nomen eher konkret
                "VERB": 3,  # Verben neutral
                "ADJ": 4,  # Adjektive abstrakt
            }
            return pos_mapping.get(pos, 3)

        return 3  # Default: neutral


class EnhancedRelationProperties:
    """
    Definition der erweiterten Properties für Relations.

    Properties:
        confidence: float - Base confidence (statisch, 0.0-1.0)
        source_text: str - Original-Text aus dem die Relation extrahiert wurde
        asserted_at: datetime - Zeitstempel der Erstellung
        timestamp: datetime - UTC datetime für Decay-Berechnung
        context: List[str] - Kontexte in denen die Relation gilt
        bidirectional: bool - True für symmetrische Relationen
        inference_rule: str - Welche Regel hat Relation erzeugt?
        usage_count: int - Für Reinforcement Learning (via Episodes)
        last_reinforced: datetime - Wann zuletzt genutzt
    """

    # Defaults für neue Relations
    DEFAULTS = {
        "confidence": 0.85,
        "source_text": None,
        "asserted_at": None,  # Wird automatisch gesetzt
        "timestamp": None,  # Wird automatisch gesetzt
        "context": [],
        "bidirectional": False,
        "inference_rule": None,
        "usage_count": 0,
        "last_reinforced": None,
    }

    # Symmetrische Relationstypen (bidirectional)
    BIDIRECTIONAL_RELATIONS = {
        "SYNONYM_OF",
        "SIMILAR_TO",
        "RELATED_TO",
        "EQUIVALENT_TO",
        "OPPOSITE_OF",  # Auch Antonyme sind symmetrisch
    }

    @staticmethod
    def is_bidirectional(relation_type: str) -> bool:
        """Prüft ob Relationstyp bidirektional ist."""
        return (
            relation_type.upper() in EnhancedRelationProperties.BIDIRECTIONAL_RELATIONS
        )


# ==================== SCHEMA MANAGER ====================


class EnhancedSchemaManager:
    """
    Manager für erweiterte Schema-Operationen.

    Verantwortlichkeiten:
    - Initialisierung neuer Properties bei Node/Relation-Erstellung
    - Update-Methoden für Property-Werte
    - Aggregation von Metadaten (z.B. typical_relations aus bestehenden Relations)
    - Migration bestehender Daten
    """

    def __init__(self, netzwerk):
        """
        Initialisiert den EnhancedSchemaManager.

        Args:
            netzwerk: KonzeptNetzwerk-Instanz für DB-Zugriff
        """
        self.netzwerk = netzwerk

        # Thread safety for Neo4j access
        self._neo4j_lock = threading.RLock()

        logger.info("EnhancedSchemaManager initialisiert")

    # ==================== VALIDATION METHODS ====================

    def _validate_entity_name(self, name: str) -> None:
        """
        Validates entity name against whitelist pattern.

        Args:
            name: Entity name to validate

        Raises:
            ValueError: If name doesn't match pattern
        """
        if not ENTITY_NAME_PATTERN.match(name.lower()):
            raise ValueError(
                f"Invalid entity name: '{name}'. "
                f"Must match pattern: lowercase alphanumeric + underscore, max 64 chars"
            )

    def _validate_relation_type(self, relation_type: str) -> str:
        """
        Validates relation type against whitelist.

        Args:
            relation_type: Relation type to validate

        Returns:
            Uppercase relation type

        Raises:
            ValueError: If relation type not in whitelist
        """
        safe_relation = relation_type.upper()
        if safe_relation not in ALLOWED_RELATION_TYPES:
            raise ValueError(
                f"Invalid relation type: {relation_type}. "
                f"Allowed types: {', '.join(sorted(ALLOWED_RELATION_TYPES))}"
            )
        return safe_relation

    # ==================== NODE PROPERTY METHODS ====================

    def init_node_properties(
        self,
        lemma: str,
        pos: Optional[str] = None,
        semantic_field: Optional[str] = None,
    ) -> bool:
        """
        Initialisiert erweiterte Properties für einen Wort/Konzept Node.

        Wird automatisch beim ersten Anlegen eines Nodes aufgerufen.
        Setzt intelligente Defaults basierend auf verfügbaren Informationen.

        Args:
            lemma: Das Wort/Konzept
            pos: Part-of-Speech Tag (optional)
            semantic_field: Semantisches Feld (optional)

        Returns:
            True bei Erfolg, False bei Fehler
        """
        # Validate entity name
        try:
            self._validate_entity_name(lemma)
        except ValueError as e:
            logger.error(f"Invalid entity name: {e}")
            return False

        if not self.netzwerk or not self.netzwerk.driver:
            logger.warning("Kein Netzwerk-Driver verfügbar")
            return False

        try:
            # Inferiere Abstraktionslevel
            abstraction_level = EnhancedNodeProperties.infer_abstraction_level(
                lemma, pos
            )

            with self._neo4j_lock:
                with self.netzwerk.driver.session(database="neo4j") as session:
                    session.run(
                        """
                    MATCH (k:Konzept {name: $lemma})
                    SET k.pos = $pos,
                        k.definitions = [],
                        k.semantic_field = $semantic_field,
                        k.abstraction_level = $abstraction_level,
                        k.contexts = [],
                        k.typical_relations = '{}',
                        k.usage_frequency = 0,
                        k.first_seen = CASE
                            WHEN k.first_seen IS NULL THEN datetime({timezone: 'UTC'})
                            ELSE k.first_seen
                        END,
                        k.last_used = NULL
                    """,
                        lemma=lemma.lower(),
                        pos=pos,
                        semantic_field=semantic_field,
                        abstraction_level=abstraction_level,
                    )

            logger.debug(
                f"Node properties initialisiert: {lemma} "
                f"(pos={pos}, abstraction={abstraction_level})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Fehler beim Initialisieren von Node Properties: {e}",
                exc_info=True,
                extra={"lemma": lemma},
            )
            return False

    def add_definition(self, lemma: str, definition: str) -> bool:
        """
        Fügt eine Definition zu einem Konzept hinzu.

        Args:
            lemma: Das Wort/Konzept
            definition: Die Definition (aus Lerne-Befehl)

        Returns:
            True bei Erfolg
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return False

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (k:Konzept {name: $lemma})
                    SET k.definitions = CASE
                        WHEN k.definitions IS NULL THEN [$definition]
                        WHEN NOT $definition IN k.definitions THEN k.definitions + $definition
                        ELSE k.definitions
                    END
                    """,
                    lemma=lemma.lower(),
                    definition=definition,
                )

            logger.debug(f"Definition hinzugefügt: {lemma} -> {definition[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Definition: {e}", exc_info=True)
            return False

    def add_context(self, lemma: str, context: str) -> bool:
        """
        Fügt einen Kontext zu einem Konzept hinzu.

        Args:
            lemma: Das Wort/Konzept
            context: Der Kontext (aus Episode)

        Returns:
            True bei Erfolg
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return False

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (k:Konzept {name: $lemma})
                    SET k.contexts = CASE
                        WHEN k.contexts IS NULL THEN [$context]
                        WHEN NOT $context IN k.contexts THEN k.contexts + $context
                        ELSE k.contexts
                    END
                    """,
                    lemma=lemma.lower(),
                    context=context,
                )

            logger.debug(f"Kontext hinzugefügt: {lemma} -> {context[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Kontext: {e}", exc_info=True)
            return False

    def update_usage(self, lemma: str) -> bool:
        """
        Aktualisiert Usage-Statistiken für ein Konzept.

        Inkrementiert usage_frequency und setzt last_used.
        Wird bei jeder Query aufgerufen, die dieses Konzept verwendet.

        Args:
            lemma: Das Wort/Konzept

        Returns:
            True bei Erfolg
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return False

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                session.run(
                    """
                    MATCH (k:Konzept {name: $lemma})
                    SET k.usage_frequency = COALESCE(k.usage_frequency, 0) + 1,
                        k.last_used = datetime({timezone: 'UTC'})
                    """,
                    lemma=lemma.lower(),
                )

            logger.debug(f"Usage aktualisiert: {lemma}")
            return True

        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren von Usage: {e}", exc_info=True)
            return False

    def update_typical_relations(self, lemma: str) -> bool:
        """
        Berechnet typical_relations Häufigkeitsverteilung aus bestehenden Relations.

        Zählt alle ausgehenden Relationen und speichert die Verteilung.

        Args:
            lemma: Das Wort/Konzept

        Returns:
            True bei Erfolg
        """
        if not self.netzwerk or not self.netzwerk.driver:
            return False

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                # Zähle Relationstypen
                result = session.run(
                    """
                    MATCH (k:Konzept {name: $lemma})-[r]->(o)
                    RETURN type(r) AS relation_type, count(r) AS count
                    """,
                    lemma=lemma.lower(),
                )

                # Erstelle Distribution
                distribution = {}
                for record in result:
                    relation_type = record["relation_type"]
                    count = record["count"]
                    distribution[relation_type] = count

                # Speichere als JSON
                session.run(
                    """
                    MATCH (k:Konzept {name: $lemma})
                    SET k.typical_relations = $distribution_json
                    """,
                    lemma=lemma.lower(),
                    distribution_json=json.dumps(distribution),
                )

            logger.debug(f"Typical relations aktualisiert: {lemma} -> {distribution}")
            return True

        except Exception as e:
            logger.error(
                f"Fehler beim Aktualisieren von typical_relations: {e}", exc_info=True
            )
            return False

    # ==================== RELATION PROPERTY METHODS ====================

    def init_relation_properties(
        self,
        subject: str,
        relation_type: str,
        object_: str,
        source_text: Optional[str] = None,
        inference_rule: Optional[str] = None,
    ) -> bool:
        """
        Initialisiert erweiterte Properties für eine Relation.

        Wird automatisch beim Erstellen einer Relation aufgerufen.

        Args:
            subject: Subject der Relation
            relation_type: Typ der Relation
            object_: Object der Relation
            source_text: Original-Text (optional)
            inference_rule: Regel die Relation erzeugt hat (optional)

        Returns:
            True bei Erfolg
        """
        # Validate inputs
        try:
            self._validate_entity_name(subject)
            self._validate_entity_name(object_)
            safe_relation = self._validate_relation_type(relation_type)
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return False

        if not self.netzwerk or not self.netzwerk.driver:
            return False

        try:
            # Prüfe ob bidirektional
            is_bidirectional = EnhancedRelationProperties.is_bidirectional(
                relation_type
            )

            with self._neo4j_lock:
                with self.netzwerk.driver.session(database="neo4j") as session:
                    # Use WHERE clause with type() function instead of f-string
                    session.run(
                        """
                        MATCH (s:Konzept {name: $subject})-[r]->(o:Konzept {name: $object})
                        WHERE type(r) = $relation_type
                        SET r.context = COALESCE(r.context, []),
                            r.bidirectional = $bidirectional,
                            r.inference_rule = $inference_rule,
                            r.usage_count = COALESCE(r.usage_count, 0),
                            r.last_reinforced = NULL
                        """,
                        subject=subject.lower(),
                        object=object_.lower(),
                        relation_type=safe_relation,
                        bidirectional=is_bidirectional,
                        inference_rule=inference_rule,
                    )

            logger.debug(
                f"Relation properties initialisiert: {subject} -{relation_type}-> {object_}"
            )
            return True

        except ValueError as e:
            logger.error(
                "Validation error beim Initialisieren von Relation Properties",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation_type": relation_type,
                    "object": object_,
                    "error_type": "ValueError",
                },
            )
            return False
        except Exception as e:
            logger.error(
                "Fehler beim Initialisieren von Relation Properties",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation_type": relation_type,
                    "object": object_,
                    "error_type": type(e).__name__,
                },
            )
            return False

    def add_relation_context(
        self,
        subject: str,
        relation_type: str,
        object_: str,
        context: str,
    ) -> bool:
        """
        Fügt einen Kontext zu einer Relation hinzu.

        Args:
            subject: Subject der Relation
            relation_type: Typ der Relation
            object_: Object der Relation
            context: Der Kontext

        Returns:
            True bei Erfolg
        """
        # Validate inputs
        try:
            self._validate_entity_name(subject)
            self._validate_entity_name(object_)
            safe_relation = self._validate_relation_type(relation_type)
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return False

        if not self.netzwerk or not self.netzwerk.driver:
            return False

        try:
            with self._neo4j_lock:
                with self.netzwerk.driver.session(database="neo4j") as session:
                    # Use WHERE clause with type() function
                    session.run(
                        """
                        MATCH (s:Konzept {name: $subject})-[r]->(o:Konzept {name: $object})
                        WHERE type(r) = $relation_type
                        SET r.context = CASE
                            WHEN r.context IS NULL THEN [$context]
                            WHEN NOT $context IN r.context THEN r.context + $context
                            ELSE r.context
                        END
                        """,
                        subject=subject.lower(),
                        object=object_.lower(),
                        relation_type=safe_relation,
                        context=context,
                    )

            logger.debug(
                f"Kontext zu Relation hinzugefügt: {subject} -{relation_type}-> {object_}"
            )
            return True

        except ValueError as e:
            logger.error(
                "Validation error beim Hinzufügen von Relation Context",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation_type": relation_type,
                    "object": object_,
                },
            )
            return False
        except Exception as e:
            logger.error(
                "Fehler beim Hinzufügen von Relation Context",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation_type": relation_type,
                    "object": object_,
                    "error_type": type(e).__name__,
                },
            )
            return False

    def reinforce_relation(
        self,
        subject: str,
        relation_type: str,
        object_: str,
    ) -> bool:
        """
        Reinforced eine Relation (inkrementiert usage_count).

        Wird automatisch aufgerufen wenn Relation in Reasoning verwendet wird.

        Args:
            subject: Subject der Relation
            relation_type: Typ der Relation
            object_: Object der Relation

        Returns:
            True bei Erfolg
        """
        # Validate inputs
        try:
            self._validate_entity_name(subject)
            self._validate_entity_name(object_)
            safe_relation = self._validate_relation_type(relation_type)
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return False

        if not self.netzwerk or not self.netzwerk.driver:
            return False

        try:
            with self._neo4j_lock:
                with self.netzwerk.driver.session(database="neo4j") as session:
                    # Use WHERE clause with type() function
                    session.run(
                        """
                        MATCH (s:Konzept {name: $subject})-[r]->(o:Konzept {name: $object})
                        WHERE type(r) = $relation_type
                        SET r.usage_count = COALESCE(r.usage_count, 0) + 1,
                            r.last_reinforced = datetime({timezone: 'UTC'})
                        """,
                        subject=subject.lower(),
                        object=object_.lower(),
                        relation_type=safe_relation,
                    )

            logger.debug(f"Relation reinforced: {subject} -{relation_type}-> {object_}")
            return True

        except ValueError as e:
            logger.error(
                "Validation error beim Reinforcen von Relation",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation_type": relation_type,
                    "object": object_,
                },
            )
            return False
        except Exception as e:
            logger.error(
                "Fehler beim Reinforcen von Relation",
                exc_info=True,
                extra={
                    "subject": subject,
                    "relation_type": relation_type,
                    "object": object_,
                    "error_type": type(e).__name__,
                },
            )
            return False


# ==================== GLOBAL INSTANCE ====================

_global_schema_manager: Optional[EnhancedSchemaManager] = None
_schema_manager_lock = threading.Lock()


def get_schema_manager(netzwerk=None) -> EnhancedSchemaManager:
    """
    Gibt die globale EnhancedSchemaManager-Instanz zurück.

    Thread-safe mit double-check locking pattern.

    Args:
        netzwerk: KonzeptNetzwerk-Instanz (nur beim ersten Aufruf erforderlich)

    Returns:
        Globale EnhancedSchemaManager-Instanz

    Raises:
        ValueError: Wenn beim ersten Aufruf kein netzwerk übergeben wurde
    """
    global _global_schema_manager

    # Double-check locking pattern for thread safety
    if _global_schema_manager is None:
        with _schema_manager_lock:
            if _global_schema_manager is None:
                if netzwerk is None:
                    raise ValueError(
                        "Beim ersten Aufruf muss netzwerk-Parameter übergeben werden"
                    )

                _global_schema_manager = EnhancedSchemaManager(netzwerk)
                logger.info("Globale EnhancedSchemaManager-Instanz erstellt")

    return _global_schema_manager


# ==================== BEISPIEL-USAGE ====================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Mock Netzwerk für Demo
    class MockNetzwerk:
        driver = None

    netzwerk = MockNetzwerk()
    manager = EnhancedSchemaManager(netzwerk)

    print("=== Enhanced Schema Manager Demo ===\n")

    # Beispiel 1: Abstraktionslevel-Inferenz
    print("Beispiel 1: Abstraktionslevel-Inferenz")
    for word, pos in [
        ("berlin", "PROPN"),
        ("hund", "NOUN"),
        ("laufen", "VERB"),
        ("schön", "ADJ"),
        ("freiheit", None),
    ]:
        level = EnhancedNodeProperties.infer_abstraction_level(word, pos)
        print(f"  {word} ({pos}): Level {level}")

    # Beispiel 2: Bidirectional Relations
    print("\nBeispiel 2: Bidirectional Relations")
    for rel_type in ["IS_A", "SYNONYM_OF", "HAS_PROPERTY", "SIMILAR_TO"]:
        is_bi = EnhancedRelationProperties.is_bidirectional(rel_type)
        print(f"  {rel_type}: {'bidirectional' if is_bi else 'unidirectional'}")
