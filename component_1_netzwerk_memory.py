# component_1_netzwerk_memory.py
"""
Episodic memory and hypothesis storage.

This module handles:
- Learning episodes (create, query, delete)
- Inference episodes and proof step tracking
- Hypothesis storage and retrieval (abductive reasoning)
- Explanation generation for reasoning processes
"""

import re
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class KonzeptNetzwerkMemory:
    """Episodic memory and hypothesis management."""

    def __init__(self, driver: Driver):
        """
        Initialize with an existing Neo4j driver.

        Args:
            driver: Neo4j driver instance from KonzeptNetzwerkCore
        """
        self.driver = driver

    # --- PHASE 3: EPISODISCHES GEDÄCHTNIS MIT ZEITSTEMPELN ---

    def create_episode(
        self, episode_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Erstellt einen Episode-Knoten im Graphen zur Nachverfolgung von Lernereignissen.

        PHASE 3 (Episodisches Gedächtnis): Ermöglicht das Tracking WANN und WIE
        Wissen erworben wurde. Jede Episode repräsentiert ein Lernereignis
        (z.B. Text-Ingestion, manuelle Definition, Pattern-Learning).

        Args:
            episode_type: Art der Episode (z.B. "ingestion", "definition", "pattern_learning")
            content: Der ursprüngliche Text/Inhalt der Episode
            metadata: Zusätzliche Metadaten (z.B. {"query": "...", "user_action": "..."})

        Returns:
            Episode-ID (UUID) wenn erfolgreich, None bei Fehler

        Beispiel:
            episode_id = netzwerk.create_episode(
                episode_type="ingestion",
                content="Ein Hund ist ein Tier.",
                metadata={"source": "user_input"}
            )
        """
        if not self.driver:
            logger.error("create_episode: Kein DB-Driver verfügbar")
            return None

        try:
            import json

            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    CREATE (e:Episode {
                        id: randomUUID(),
                        type: $type,
                        content: $content,
                        timestamp: timestamp(),
                        metadata: $metadata
                    })
                    RETURN e.id AS episode_id
                    """,
                    type=episode_type,
                    content=content,
                    metadata=json.dumps(metadata or {}),  # Serialisiere zu JSON-String
                )

                record = result.single()
                episode_id = str(record["episode_id"]) if record else None

                if episode_id:
                    logger.info(
                        f"Episode erstellt: {episode_type}",
                        extra={
                            "episode_id": episode_id,
                            "content_preview": content[:50],
                        },
                    )

                return episode_id

        except Exception as e:
            logger.log_exception(
                e, "Fehler in create_episode", episode_type=episode_type
            )
            return None

    def link_fact_to_episode(
        self, subject: str, relation: str, object: str, episode_id: str
    ) -> bool:
        """
        Verknüpft eine Relation mit einer Episode, um nachzuvollziehen,
        aus welchem Lernereignis ein Fakt stammt.

        PHASE 3: Dies ermöglicht Transparenz über Wissensherkunft und
        spätere Fehlerkorrektur ("Lösche alles aus Episode X").

        Args:
            subject: Subject der Relation
            relation: Relationstyp (z.B. "IS_A")
            object: Object der Relation
            episode_id: Die Episode-ID, aus der dieser Fakt stammt

        Returns:
            True wenn erfolgreich verknüpft, False bei Fehler

        Beispiel:
            # Nach assert_relation():
            netzwerk.link_fact_to_episode("hund", "IS_A", "tier", episode_id)
        """
        if not self.driver:
            logger.error("link_fact_to_episode: Kein DB-Driver verfügbar")
            return False

        safe_relation = re.sub(r"[^a-zA-Z0-9_]", "", relation.upper())
        if not safe_relation:
            logger.error("Ungültiger Relationstyp", extra={"relation": relation})
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                # KORREKTUR: Erstelle einen Fact-Node statt direkt auf Relationship zu verlinken
                # Cypher erlaubt keine Beziehungen zu Relationships, nur zu Nodes
                result = session.run(
                    f"""
                    MATCH (s:Konzept {{name: $subject}})-[rel:{safe_relation}]->(o:Konzept {{name: $object}})
                    MATCH (e:Episode {{id: $episode_id}})
                    MERGE (f:Fact {{
                        subject: $subject,
                        relation: $relation,
                        object: $object
                    }})
                    MERGE (e)-[learned:LEARNED_FACT]->(f)
                    ON CREATE SET learned.linked_at = timestamp()
                    RETURN learned IS NOT NULL AS success
                    """,
                    subject=subject.lower(),
                    object=object.lower(),
                    relation=safe_relation,
                    episode_id=episode_id,
                )

                record = result.single()
                success = record["success"] if record else False

                if success:
                    logger.debug(
                        "Fakt mit Episode verknüpft",
                        extra={
                            "subject": subject,
                            "relation": safe_relation,
                            "object": object,
                            "episode_id": episode_id,
                        },
                    )

                return success

        except Exception as e:
            logger.log_exception(
                e,
                "Fehler in link_fact_to_episode",
                subject=subject,
                relation=relation,
                object=object,
                episode_id=episode_id,
            )
            return False

    def query_episodes_about(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Findet alle Episoden, in denen über ein bestimmtes Thema gelernt wurde.

        PHASE 3: Beantwortet die Frage "Wann habe ich über X gelernt?"

        Args:
            topic: Das Thema, über das gesucht wird
            limit: Maximale Anzahl Episoden (neueste zuerst)

        Returns:
            Liste von Episode-Dictionaries mit:
            - episode_id: UUID der Episode
            - type: Episode-Typ
            - content: Original-Inhalt
            - timestamp: Zeitstempel (Neo4j timestamp)
            - learned_facts: Liste der gelernten Fakten aus dieser Episode

        Beispiel:
            episodes = netzwerk.query_episodes_about("hund")
            for ep in episodes:
                print(f"Am {ep['timestamp']} gelernt: {ep['content']}")
        """
        if not self.driver:
            logger.error("query_episodes_about: Kein DB-Driver verfügbar")
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (e:Episode)-[:LEARNED_FACT]->(f:Fact)
                    WHERE f.subject = $topic OR f.object = $topic
                    WITH e, collect({
                        subject: f.subject,
                        relation: f.relation,
                        object: f.object
                    }) AS facts
                    RETURN e.id AS episode_id,
                           e.type AS type,
                           e.content AS content,
                           e.timestamp AS timestamp,
                           e.metadata AS metadata,
                           facts AS learned_facts
                    ORDER BY e.timestamp DESC
                    LIMIT $limit
                    """,
                    topic=topic.lower(),
                    limit=limit,
                )

                episodes = [record.data() for record in result]

                logger.debug(
                    f"query_episodes_about: '{topic}' -> {len(episodes)} Episoden gefunden"
                )

                return episodes

        except Exception as e:
            logger.log_exception(e, "Fehler in query_episodes_about", topic=topic)
            return []

    def query_all_episodes(
        self, episode_type: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Holt alle Episoden aus dem Graphen, optional gefiltert nach Typ.

        PHASE 3: Ermöglicht einen Überblick über alle Lernereignisse.

        Args:
            episode_type: Optional - filtert nach Episode-Typ
            limit: Maximale Anzahl Episoden (neueste zuerst)

        Returns:
            Liste von Episode-Dictionaries (siehe query_episodes_about)

        Beispiel:
            # Alle Text-Ingestion-Episoden
            episodes = netzwerk.query_all_episodes(episode_type="ingestion")
        """
        if not self.driver:
            logger.error("query_all_episodes: Kein DB-Driver verfügbar")
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                if episode_type:
                    result = session.run(
                        """
                        MATCH (e:Episode {type: $type})
                        OPTIONAL MATCH (e)-[:LEARNED_FACT]->(f:Fact)
                        WITH e, collect(CASE WHEN f IS NOT NULL THEN {
                            subject: f.subject,
                            relation: f.relation,
                            object: f.object
                        } END) AS facts
                        RETURN e.id AS episode_id,
                               e.type AS type,
                               e.content AS content,
                               e.timestamp AS timestamp,
                               e.metadata AS metadata,
                               facts AS learned_facts
                        ORDER BY e.timestamp DESC
                        LIMIT $limit
                        """,
                        type=episode_type,
                        limit=limit,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (e:Episode)
                        OPTIONAL MATCH (e)-[:LEARNED_FACT]->(f:Fact)
                        WITH e, collect(CASE WHEN f IS NOT NULL THEN {
                            subject: f.subject,
                            relation: f.relation,
                            object: f.object
                        } END) AS facts
                        RETURN e.id AS episode_id,
                               e.type AS type,
                               e.content AS content,
                               e.timestamp AS timestamp,
                               e.metadata AS metadata,
                               facts AS learned_facts
                        ORDER BY e.timestamp DESC
                        LIMIT $limit
                        """,
                        limit=limit,
                    )

                episodes = [record.data() for record in result]

                logger.debug(
                    f"query_all_episodes: {len(episodes)} Episoden gefunden"
                    + (f" (Typ: {episode_type})" if episode_type else "")
                )

                return episodes

        except Exception as e:
            logger.log_exception(
                e, "Fehler in query_all_episodes", episode_type=episode_type
            )
            return []

    def delete_episode(self, episode_id: str, cascade: bool = False) -> bool:
        """
        Löscht eine Episode aus dem Graphen.

        PHASE 3: Ermöglicht Fehlerkorrektur durch Löschen fehlerhafter
        Lernereignisse.

        Args:
            episode_id: Die ID der zu löschenden Episode
            cascade: Wenn True, werden auch die gelernten Fakten gelöscht
                     (WARNUNG: Dies löscht die tatsächlichen Relationen!)

        Returns:
            True wenn erfolgreich gelöscht, False bei Fehler

        Beispiel:
            # Lösche nur Episode-Knoten, behalte Fakten
            netzwerk.delete_episode(episode_id)

            # Lösche Episode UND alle gelernten Fakten
            netzwerk.delete_episode(episode_id, cascade=True)
        """
        if not self.driver:
            logger.error("delete_episode: Kein DB-Driver verfügbar")
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                if cascade:
                    # KORREKTUR: Lösche Episode, Fact-Nodes UND die tatsächlichen Relationen zwischen Konzepten
                    result = session.run(
                        """
                        MATCH (e:Episode {id: $episode_id})-[:LEARNED_FACT]->(f:Fact)
                        WITH e, collect(f) AS facts
                        UNWIND facts AS fact
                        // Lösche die tatsächliche Relation zwischen den Konzepten
                        CALL {
                            WITH fact
                            MATCH (s:Konzept {name: fact.subject})-[r]->(o:Konzept {name: fact.object})
                            WHERE type(r) = fact.relation
                            DELETE r
                        }
                        // Lösche den Fact-Node
                        DETACH DELETE fact
                        // Lösche die Episode
                        WITH e
                        DETACH DELETE e
                        RETURN count(e) AS deleted_count
                        """,
                        episode_id=episode_id,
                    )
                else:
                    # Lösche nur Episode-Knoten, behalte Fakten
                    result = session.run(
                        """
                        MATCH (e:Episode {id: $episode_id})
                        DETACH DELETE e
                        RETURN count(e) AS deleted_count
                        """,
                        episode_id=episode_id,
                    )

                record = result.single()
                deleted_count = record["deleted_count"] if record else 0

                if deleted_count > 0:
                    logger.info(
                        f"Episode gelöscht: {episode_id}", extra={"cascade": cascade}
                    )
                    return True
                else:
                    logger.warning(f"Episode nicht gefunden: {episode_id}")
                    return False

        except Exception as e:
            logger.log_exception(e, "Fehler in delete_episode", episode_id=episode_id)
            return False

    # --- EPISODIC MEMORY FOR REASONING: INFERENCE TRACKING ---

    def create_inference_episode(
        self, inference_type: str, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Erstellt einen InferenceEpisode-Knoten zur Nachverfolgung von Schlussfolgerungen.

        Episodic Memory for Reasoning: Ermöglicht das Tracking WANN, WIE und WARUM
        Inferenzen durchgeführt wurden. Jede InferenceEpisode repräsentiert einen
        vollständigen Reasoning-Durchlauf (z.B. Backward-Chaining, Multi-Hop-Reasoning).

        Args:
            inference_type: Art der Inferenz ("forward_chaining", "backward_chaining",
                           "graph_traversal", "abductive", "hybrid")
            query: Die ursprüngliche Frage/Goal
            metadata: Zusätzliche Metadaten (z.B. {"topic": "...", "confidence": 0.8})

        Returns:
            InferenceEpisode-ID (UUID) wenn erfolgreich, None bei Fehler

        Beispiel:
            episode_id = netzwerk.create_inference_episode(
                inference_type="backward_chaining",
                query="Was ist ein Hund?",
                metadata={"topic": "hund", "max_depth": 5}
            )
        """
        if not self.driver:
            logger.error("create_inference_episode: Kein DB-Driver verfügbar")
            return None

        try:
            import json

            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    CREATE (ie:InferenceEpisode {
                        id: randomUUID(),
                        inference_type: $type,
                        query: $user_query,
                        timestamp: timestamp(),
                        metadata: $metadata
                    })
                    RETURN ie.id AS episode_id
                    """,
                    type=inference_type,
                    user_query=query,
                    metadata=json.dumps(metadata or {}),
                )

                record = result.single()
                episode_id = str(record["episode_id"]) if record else None

                if episode_id:
                    logger.info(
                        f"InferenceEpisode erstellt: {inference_type}",
                        extra={"episode_id": episode_id, "query": query[:50]},
                    )

                return episode_id

        except Exception as e:
            logger.log_exception(
                e, "Fehler in create_inference_episode", inference_type=inference_type
            )
            return None

    def create_proof_step(
        self,
        goal: str,
        method: str,
        confidence: float,
        depth: int,
        bindings: Optional[Dict[str, Any]] = None,
        parent_step_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Erstellt einen ProofStep-Knoten zur Repräsentation eines Beweisschritts.

        Args:
            goal: Das zu beweisende Goal (z.B. "IS_A(hund, ?x)")
            method: Beweismethode ("fact", "rule", "graph_traversal", "hypothesis")
            confidence: Konfidenz des Beweisschritts (0.0 bis 1.0)
            depth: Tiefe im Beweisbaum
            bindings: Variable-Bindungen (z.B. {"?x": "tier"})
            parent_step_id: ID des übergeordneten ProofSteps (für Hierarchie)

        Returns:
            ProofStep-ID (UUID) wenn erfolgreich, None bei Fehler
        """
        if not self.driver:
            logger.error("create_proof_step: Kein DB-Driver verfügbar")
            return None

        try:
            import json

            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    CREATE (ps:ProofStep {
                        id: randomUUID(),
                        goal: $goal,
                        method: $method,
                        confidence: $confidence,
                        depth: $depth,
                        bindings: $bindings,
                        timestamp: timestamp()
                    })
                    RETURN ps.id AS step_id
                    """,
                    goal=goal,
                    method=method,
                    confidence=confidence,
                    depth=depth,
                    bindings=json.dumps(bindings or {}),
                )

                record = result.single()
                step_id = str(record["step_id"]) if record else None

                # Verknüpfe mit Parent-Step falls vorhanden
                if step_id and parent_step_id:
                    session.run(
                        """
                        MATCH (parent:ProofStep {id: $parent_id})
                        MATCH (child:ProofStep {id: $child_id})
                        MERGE (parent)-[:CHILD_PROOF]->(child)
                        """,
                        parent_id=parent_step_id,
                        child_id=step_id,
                    )

                if step_id:
                    logger.debug(
                        f"ProofStep erstellt: {method}",
                        extra={"step_id": step_id, "goal": goal[:50], "depth": depth},
                    )

                return step_id

        except Exception as e:
            logger.log_exception(e, "Fehler in create_proof_step", goal=goal)
            return None

    def link_inference_to_proof(
        self, inference_episode_id: str, proof_step_id: str
    ) -> bool:
        """
        Verknüpft eine InferenceEpisode mit dem Root-ProofStep.

        Args:
            inference_episode_id: ID der InferenceEpisode
            proof_step_id: ID des Root-ProofSteps

        Returns:
            True wenn erfolgreich verknüpft, False bei Fehler
        """
        if not self.driver:
            logger.error("link_inference_to_proof: Kein DB-Driver verfügbar")
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (ie:InferenceEpisode {id: $episode_id})
                    MATCH (ps:ProofStep {id: $step_id})
                    MERGE (ie)-[r:PRODUCED]->(ps)
                    ON CREATE SET r.linked_at = timestamp()
                    RETURN r IS NOT NULL AS success
                    """,
                    episode_id=inference_episode_id,
                    step_id=proof_step_id,
                )

                record = result.single()
                success = record["success"] if record else False

                if success:
                    logger.debug(
                        "InferenceEpisode mit ProofStep verknüpft",
                        extra={
                            "episode_id": inference_episode_id[:8],
                            "step_id": proof_step_id[:8],
                        },
                    )

                return success

        except Exception as e:
            logger.log_exception(e, "Fehler in link_inference_to_proof")
            return False

    def link_inference_to_facts(
        self, inference_episode_id: str, fact_ids: List[str]
    ) -> bool:
        """
        Verknüpft eine InferenceEpisode mit allen verwendeten Fakten.

        Args:
            inference_episode_id: ID der InferenceEpisode
            fact_ids: Liste von Fact-IDs (aus component_9_logik_engine.py)

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if not self.driver or not fact_ids:
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                for fact_id in fact_ids:
                    # Erstelle Fact-Node falls noch nicht vorhanden
                    # (fact_id könnte aus der Engine kommen)
                    session.run(
                        """
                        MATCH (ie:InferenceEpisode {id: $episode_id})
                        MERGE (f:InferenceFact {fact_id: $fact_id})
                        MERGE (ie)-[r:USED_FACT]->(f)
                        ON CREATE SET r.linked_at = timestamp()
                        """,
                        episode_id=inference_episode_id,
                        fact_id=fact_id,
                    )

                logger.debug(
                    f"InferenceEpisode mit {len(fact_ids)} Fakten verknüpft",
                    extra={"episode_id": inference_episode_id[:8]},
                )

                return True

        except Exception as e:
            logger.log_exception(e, "Fehler in link_inference_to_facts")
            return False

    def link_inference_to_rules(
        self, inference_episode_id: str, rule_ids: List[str]
    ) -> bool:
        """
        Verknüpft eine InferenceEpisode mit allen angewendeten Regeln.

        Args:
            inference_episode_id: ID der InferenceEpisode
            rule_ids: Liste von Regel-IDs (aus component_9_logik_engine.py)

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if not self.driver or not rule_ids:
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                for rule_id in rule_ids:
                    session.run(
                        """
                        MATCH (ie:InferenceEpisode {id: $episode_id})
                        MATCH (r:Regel {id: $rule_id})
                        MERGE (ie)-[rel:APPLIED_RULE]->(r)
                        ON CREATE SET rel.applied_at = timestamp()
                        """,
                        episode_id=inference_episode_id,
                        rule_id=rule_id,
                    )

                logger.debug(
                    f"InferenceEpisode mit {len(rule_ids)} Regeln verknüpft",
                    extra={"episode_id": inference_episode_id[:8]},
                )

                return True

        except Exception as e:
            logger.log_exception(e, "Fehler in link_inference_to_rules")
            return False

    def query_inference_history(
        self,
        topic: Optional[str] = None,
        inference_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Findet alle InferenceEpisodes, optional gefiltert nach Thema oder Typ.

        Beantwortet Meta-Fragen wie:
        - "Wann habe ich über X nachgedacht?"
        - "Welche Backward-Chaining-Inferenzen habe ich durchgeführt?"
        - "Wie oft musste ich Multi-Hop-Reasoning verwenden?"

        Args:
            topic: Optional - filtert nach Thema (durchsucht Query und Metadata)
            inference_type: Optional - filtert nach Inferenz-Typ
            limit: Maximale Anzahl Episoden (neueste zuerst)

        Returns:
            Liste von InferenceEpisode-Dictionaries mit:
            - episode_id: UUID der Episode
            - inference_type: Typ der Inferenz
            - query: Ursprüngliche Frage
            - timestamp: Zeitstempel
            - metadata: Zusätzliche Metadaten
            - used_facts_count: Anzahl verwendeter Fakten
            - applied_rules_count: Anzahl angewendeter Regeln

        Beispiel:
            episodes = netzwerk.query_inference_history(topic="hund")
            for ep in episodes:
                print(f"{ep['inference_type']}: {ep['query']}")
        """
        if not self.driver:
            logger.error("query_inference_history: Kein DB-Driver verfügbar")
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                # Baue Query dynamisch auf
                where_clauses = []
                params: Dict[str, Any] = {"limit": limit}

                if inference_type:
                    where_clauses.append("ie.inference_type = $inference_type")
                    params["inference_type"] = inference_type

                if topic:
                    # Suche in Query und Metadata
                    where_clauses.append(
                        "(ie.query CONTAINS $topic OR ie.metadata CONTAINS $topic)"
                    )
                    params["topic"] = topic.lower()

                where_clause = (
                    "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                )

                query = f"""
                    MATCH (ie:InferenceEpisode)
                    {where_clause}
                    OPTIONAL MATCH (ie)-[:USED_FACT]->(f:InferenceFact)
                    WITH ie, count(f) AS facts_count
                    OPTIONAL MATCH (ie)-[:APPLIED_RULE]->(r:Regel)
                    WITH ie, facts_count, count(r) AS rules_count
                    OPTIONAL MATCH (ie)-[:PRODUCED]->(ps:ProofStep)
                    RETURN ie.id AS episode_id,
                           ie.inference_type AS inference_type,
                           ie.query AS query,
                           ie.timestamp AS timestamp,
                           ie.metadata AS metadata,
                           facts_count AS used_facts_count,
                           rules_count AS applied_rules_count,
                           collect(ps.id) AS proof_step_ids
                    ORDER BY ie.timestamp DESC
                    LIMIT $limit
                """

                result = session.run(query, **params)
                episodes = [record.data() for record in result]

                logger.debug(
                    f"query_inference_history: {len(episodes)} Episoden gefunden"
                    + (f" (Thema: {topic})" if topic else "")
                    + (f" (Typ: {inference_type})" if inference_type else "")
                )

                return episodes

        except Exception as e:
            logger.log_exception(e, "Fehler in query_inference_history")
            return []

    def get_proof_tree(self, root_step_id: str) -> Optional[Dict[str, Any]]:
        """
        Rekonstruiert den vollständigen Beweisbaum ab einem Root-ProofStep.

        Args:
            root_step_id: ID des Root-ProofSteps

        Returns:
            Hierarchisches Dictionary mit:
            - step_id: UUID des Steps
            - goal: Das Goal
            - method: Beweismethode
            - confidence: Konfidenz
            - depth: Tiefe
            - bindings: Variable-Bindungen
            - children: Liste von Kind-Steps (rekursiv)

        Beispiel:
            tree = netzwerk.get_proof_tree(root_step_id)
            # tree = {
            #     "goal": "IS_A(hund, ?x)",
            #     "method": "rule",
            #     "children": [
            #         {"goal": "IS_A(hund, säugetier)", "method": "fact", ...},
            #         {"goal": "IS_A(säugetier, tier)", "method": "fact", ...}
            #     ]
            # }
        """
        if not self.driver:
            logger.error("get_proof_tree: Kein DB-Driver verfügbar")
            return None

        try:
            import json

            with self.driver.session(database="neo4j") as session:
                # Hole den gesamten Baum mit Cypher Path-Query
                result = session.run(
                    """
                    MATCH path = (root:ProofStep {id: $root_id})-[:CHILD_PROOF*0..]->(step:ProofStep)
                    WITH root, collect(DISTINCT step) AS all_steps
                    RETURN root.id AS root_id,
                           root.goal AS root_goal,
                           root.method AS root_method,
                           root.confidence AS root_confidence,
                           root.depth AS root_depth,
                           root.bindings AS root_bindings,
                           all_steps
                    """,
                    root_id=root_step_id,
                )

                record = result.single()
                if not record:
                    logger.warning(f"ProofStep nicht gefunden: {root_step_id}")
                    return None

                # Baue Hierarchie rekursiv auf
                def build_tree(step_id: str) -> Dict[str, Any]:
                    """Rekursive Hilfsfunktion zum Aufbau des Baums"""
                    step_result = session.run(
                        """
                        MATCH (step:ProofStep {id: $step_id})
                        OPTIONAL MATCH (step)-[:CHILD_PROOF]->(child:ProofStep)
                        RETURN step.id AS id,
                               step.goal AS goal,
                               step.method AS method,
                               step.confidence AS confidence,
                               step.depth AS depth,
                               step.bindings AS bindings,
                               collect(child.id) AS child_ids
                        """,
                        step_id=step_id,
                    )

                    step_record = step_result.single()
                    if not step_record:
                        return {}

                    node = {
                        "step_id": step_record["id"],
                        "goal": step_record["goal"],
                        "method": step_record["method"],
                        "confidence": step_record["confidence"],
                        "depth": step_record["depth"],
                        "bindings": (
                            json.loads(step_record["bindings"])
                            if step_record["bindings"]
                            else {}
                        ),
                        "children": [],
                    }

                    # Rekursiv Kinder aufbauen
                    for child_id in step_record["child_ids"]:
                        if child_id:  # Null-Check
                            child_tree = build_tree(child_id)
                            if child_tree:
                                node["children"].append(child_tree)

                    return node

                tree = build_tree(root_step_id)

                logger.debug(
                    f"Proof-Tree rekonstruiert: {root_step_id[:8]}",
                    extra={"num_steps": len(record["all_steps"])},
                )

                return tree

        except Exception as e:
            logger.log_exception(e, "Fehler in get_proof_tree")
            return None

    def explain_inference(self, episode_id: str) -> str:
        """
        Generiert eine natürlichsprachliche Erklärung einer Inferenz-Episode.

        Args:
            episode_id: ID der InferenceEpisode

        Returns:
            Lesbare Erklärung des Reasoning-Prozesses

        Beispiel:
            explanation = netzwerk.explain_inference(episode_id)
            print(explanation)
            # Output:
            # "Um die Frage 'Was ist ein Hund?' zu beantworten, habe ich:
            #  1. Backward-Chaining verwendet
            #  2. Die Fakten 'hund IS_A säugetier' und 'säugetier IS_A tier' gefunden
            #  3. Mit einer Konfidenz von 100% geschlossen: Ein Hund ist ein Tier."
        """
        if not self.driver:
            logger.error("explain_inference: Kein DB-Driver verfügbar")
            return "Erklärung nicht verfügbar."

        try:
            import json

            with self.driver.session(database="neo4j") as session:
                # Hole Episode-Details
                result = session.run(
                    """
                    MATCH (ie:InferenceEpisode {id: $episode_id})
                    OPTIONAL MATCH (ie)-[:USED_FACT]->(f:InferenceFact)
                    WITH ie, collect(f.fact_id) AS fact_ids
                    OPTIONAL MATCH (ie)-[:APPLIED_RULE]->(r:Regel)
                    WITH ie, fact_ids, collect(r.id) AS rule_ids
                    OPTIONAL MATCH (ie)-[:PRODUCED]->(ps:ProofStep)
                    RETURN ie.inference_type AS inference_type,
                           ie.query AS query,
                           ie.metadata AS metadata,
                           fact_ids,
                           rule_ids,
                           ps.id AS root_step_id,
                           ps.confidence AS final_confidence
                    """,
                    episode_id=episode_id,
                )

                record = result.single()
                if not record:
                    return f"InferenceEpisode {episode_id[:8]} nicht gefunden."

                # Extrahiere Metadaten
                json.loads(record["metadata"]) if record["metadata"] else {}
                inference_type = record["inference_type"]
                query = record["query"]
                final_confidence = record["final_confidence"] or 0.0

                # Generiere Erklärung
                explanation_parts = []
                explanation_parts.append(
                    f"Um die Frage '{query}' zu beantworten, habe ich:"
                )

                # Methode
                method_desc = {
                    "forward_chaining": "vorwärts-verkettete Schlussfolgerung (Forward-Chaining)",
                    "backward_chaining": "rückwärts-verkettete Schlussfolgerung (Backward-Chaining)",
                    "graph_traversal": "Graph-Traversal (Multi-Hop-Reasoning)",
                    "abductive": "abduktive Schlussfolgerung (Hypothesen-Generierung)",
                    "hybrid": "hybride Reasoning-Strategie",
                }.get(inference_type, inference_type)

                explanation_parts.append(f"1. {method_desc} verwendet")

                # Fakten
                num_facts = len(record["fact_ids"])
                if num_facts > 0:
                    explanation_parts.append(
                        f"2. {num_facts} Fakt{'en' if num_facts > 1 else ''} analysiert"
                    )

                # Regeln
                num_rules = len(record["rule_ids"])
                if num_rules > 0:
                    explanation_parts.append(
                        f"3. {num_rules} Regel{'n' if num_rules > 1 else ''} angewendet"
                    )

                # Konfidenz
                confidence_percent = int(final_confidence * 100)
                explanation_parts.append(
                    f"4. Mit einer Konfidenz von {confidence_percent}% eine Antwort gefunden"
                )

                explanation = "\n".join(explanation_parts) + "."

                logger.debug(
                    f"Erklärung generiert für InferenceEpisode {episode_id[:8]}"
                )

                return explanation

        except Exception as e:
            logger.log_exception(e, "Fehler in explain_inference")
            return f"Fehler bei der Erklärungsgenerierung: {e}"

    # --- ABDUCTIVE REASONING: HYPOTHESIS STORAGE ---

    def store_hypothesis(
        self,
        hypothesis_id: str,
        explanation: str,
        observations: List[str],
        strategy: str,
        confidence: float,
        scores: Dict[str, float],
        abduced_facts: List[Dict[str, Any]],
        sources: Optional[List[str]] = None,
        reasoning_trace: str = "",
    ) -> bool:
        """
        Speichert eine Hypothese aus dem Abductive Reasoning im Graphen.

        Args:
            hypothesis_id: UUID der Hypothese
            explanation: Natürlichsprachliche Erklärung
            observations: Liste beobachteter Phänomene
            strategy: Strategie ("template", "analogy", "causal_chain")
            confidence: Gesamtkonfidenz (0.0 bis 1.0)
            scores: Multi-Kriterien-Scores (coverage, simplicity, coherence, specificity)
            abduced_facts: Liste von abduzierten Fakten (als Dicts)
            sources: Liste von Quellen/Wissensbasen
            reasoning_trace: Detaillierter Reasoning-Trace

        Returns:
            True wenn erfolgreich gespeichert, False bei Fehler
        """
        if not self.driver:
            logger.error("store_hypothesis: Kein DB-Driver verfügbar")
            return False

        try:
            import json

            with self.driver.session(database="neo4j") as session:
                # Erstelle Hypothesis-Node
                result = session.run(
                    """
                    CREATE (h:Hypothesis {
                        id: $id,
                        explanation: $explanation,
                        observations: $observations,
                        strategy: $strategy,
                        confidence: $confidence,
                        scores: $scores,
                        abduced_facts: $abduced_facts,
                        sources: $sources,
                        reasoning_trace: $reasoning_trace,
                        timestamp: timestamp()
                    })
                    RETURN h.id AS hypothesis_id
                    """,
                    id=hypothesis_id,
                    explanation=explanation,
                    observations=observations,
                    strategy=strategy,
                    confidence=confidence,
                    scores=json.dumps(scores),
                    abduced_facts=json.dumps(abduced_facts),
                    sources=sources or [],
                    reasoning_trace=reasoning_trace,
                )

                record = result.single()
                success = record is not None

                if success:
                    logger.info(
                        f"Hypothese gespeichert: {strategy}",
                        extra={
                            "hypothesis_id": hypothesis_id,
                            "confidence": confidence,
                        },
                    )

                return success

        except Exception as e:
            logger.log_exception(
                e, "Fehler in store_hypothesis", hypothesis_id=hypothesis_id
            )
            return False

    def link_hypothesis_to_observations(
        self, hypothesis_id: str, observations: List[str]
    ) -> bool:
        """
        Verknüpft eine Hypothese mit den Beobachtungen, die sie erklärt.

        Args:
            hypothesis_id: ID der Hypothese
            observations: Liste von Beobachtungen (Textstrings)

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if not self.driver or not observations:
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                for obs in observations:
                    session.run(
                        """
                        MATCH (h:Hypothesis {id: $hyp_id})
                        MERGE (o:Observation {text: $obs})
                        MERGE (h)-[r:EXPLAINS]->(o)
                        ON CREATE SET r.linked_at = timestamp()
                        """,
                        hyp_id=hypothesis_id,
                        obs=obs,
                    )

                logger.debug(
                    f"Hypothese mit {len(observations)} Beobachtungen verknüpft",
                    extra={"hypothesis_id": hypothesis_id[:8]},
                )

                return True

        except Exception as e:
            logger.log_exception(e, "Fehler in link_hypothesis_to_observations")
            return False

    def link_hypothesis_to_concepts(
        self, hypothesis_id: str, concepts: List[str], ensure_wort_callback
    ) -> bool:
        """
        Verknüpft eine Hypothese mit den Konzepten, die sie betrifft.

        Args:
            hypothesis_id: ID der Hypothese
            concepts: Liste von Konzept-Namen
            ensure_wort_callback: Callback-Funktion um ensure_wort_und_konzept aufzurufen

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if not self.driver or not concepts:
            return False

        try:
            with self.driver.session(database="neo4j") as session:
                for concept in concepts:
                    ensure_wort_callback(concept)

                    session.run(
                        """
                        MATCH (h:Hypothesis {id: $hyp_id})
                        MATCH (k:Konzept {name: $concept})
                        MERGE (h)-[r:ABOUT]->(k)
                        ON CREATE SET r.linked_at = timestamp()
                        """,
                        hyp_id=hypothesis_id,
                        concept=concept.lower(),
                    )

                logger.debug(
                    f"Hypothese mit {len(concepts)} Konzepten verknüpft",
                    extra={"hypothesis_id": hypothesis_id[:8]},
                )

                return True

        except Exception as e:
            logger.log_exception(e, "Fehler in link_hypothesis_to_concepts")
            return False

    def query_hypotheses_about(
        self,
        topic: str,
        strategy: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Findet alle Hypothesen über ein bestimmtes Thema.

        Args:
            topic: Das Thema
            strategy: Optional - Filter nach Strategie
            min_confidence: Minimale Konfidenz
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Liste von Hypothese-Dictionaries
        """
        if not self.driver:
            logger.error("query_hypotheses_about: Kein DB-Driver verfügbar")
            return []

        try:
            with self.driver.session(database="neo4j") as session:
                # Baue Query dynamisch
                where_clauses = ["h.confidence >= $min_confidence"]
                params = {
                    "topic": topic.lower(),
                    "min_confidence": min_confidence,
                    "limit": limit,
                }

                if strategy:
                    where_clauses.append("h.strategy = $strategy")
                    params["strategy"] = strategy

                where_clause = " AND ".join(where_clauses)

                query = f"""
                    MATCH (h:Hypothesis)-[:ABOUT]->(k:Konzept {{name: $topic}})
                    WHERE {where_clause}
                    RETURN h.id AS hypothesis_id,
                           h.explanation AS explanation,
                           h.observations AS observations,
                           h.strategy AS strategy,
                           h.confidence AS confidence,
                           h.scores AS scores,
                           h.abduced_facts AS abduced_facts,
                           h.sources AS sources,
                           h.reasoning_trace AS reasoning_trace,
                           h.timestamp AS timestamp
                    ORDER BY h.confidence DESC, h.timestamp DESC
                    LIMIT $limit
                """

                result = session.run(query, **params)  # type: ignore[arg-type]
                hypotheses = [record.data() for record in result]

                logger.debug(
                    f"query_hypotheses_about: '{topic}' -> {len(hypotheses)} Hypothesen gefunden"
                )

                return hypotheses

        except Exception as e:
            logger.log_exception(e, "Fehler in query_hypotheses_about")
            return []

    def get_best_hypothesis_for(
        self, topic: str, strategy: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Holt die beste Hypothese für ein Thema (höchste Konfidenz).

        Args:
            topic: Das Thema
            strategy: Optional - Filter nach Strategie

        Returns:
            Hypothese-Dictionary oder None
        """
        hypotheses = self.query_hypotheses_about(
            topic=topic, strategy=strategy, limit=1
        )

        return hypotheses[0] if hypotheses else None

    def explain_hypothesis(self, hypothesis_id: str) -> str:
        """
        Generiert eine natürlichsprachliche Erklärung einer Hypothese.

        Args:
            hypothesis_id: ID der Hypothese

        Returns:
            Lesbare Erklärung der Hypothese
        """
        if not self.driver:
            logger.error("explain_hypothesis: Kein DB-Driver verfügbar")
            return "Erklärung nicht verfügbar."

        try:
            import json

            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (h:Hypothesis {id: $hyp_id})
                    RETURN h.explanation AS explanation,
                           h.strategy AS strategy,
                           h.confidence AS confidence,
                           h.scores AS scores,
                           h.abduced_facts AS abduced_facts,
                           h.reasoning_trace AS reasoning_trace
                    """,
                    hyp_id=hypothesis_id,
                )

                record = result.single()
                if not record:
                    return f"Hypothese {hypothesis_id[:8]} nicht gefunden."

                explanation_parts = []
                explanation_parts.append(f"**Hypothese (ID: {hypothesis_id[:8]})**")
                explanation_parts.append(f"Erklärung: {record['explanation']}")
                explanation_parts.append(f"Strategie: {record['strategy']}")
                explanation_parts.append(f"Konfidenz: {record['confidence']:.2f}")
                explanation_parts.append("")

                # Scores
                scores = json.loads(record["scores"]) if record["scores"] else {}
                if scores:
                    explanation_parts.append("**Bewertung:**")
                    for criterion, score in scores.items():
                        explanation_parts.append(f"  - {criterion}: {score:.2f}")
                    explanation_parts.append("")

                # Abduced Facts
                abduced_facts = (
                    json.loads(record["abduced_facts"])
                    if record["abduced_facts"]
                    else []
                )
                if abduced_facts:
                    explanation_parts.append("**Abgeleitete Fakten:**")
                    for fact in abduced_facts:
                        explanation_parts.append(f"  - {fact}")
                    explanation_parts.append("")

                # Reasoning Trace
                if record["reasoning_trace"]:
                    explanation_parts.append("**Reasoning Trace:**")
                    explanation_parts.append(f"  {record['reasoning_trace']}")

                return "\n".join(explanation_parts)

        except Exception as e:
            logger.log_exception(e, "Fehler in explain_hypothesis")
            return f"Fehler bei der Erklärungsgenerierung: {e}"
