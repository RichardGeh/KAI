"""
component_1_inference_memory.py

Inference tracking and proof step management for KAI reasoning engines.

This module handles:
- InferenceEpisode creation (reasoning event tracking)
- ProofStep creation and hierarchical proof trees
- Inference-to-fact/rule linking (provenance)
- Inference history queries
- Natural language explanation generation

Episodic Memory for Reasoning:
    Tracks WHEN, HOW, and WHY inferences were performed. Each InferenceEpisode
    represents a complete reasoning run (e.g., backward chaining, multi-hop reasoning).
    ProofSteps form hierarchical trees showing the reasoning process step-by-step.

Architecture:
    - Uses Neo4jSessionMixin for thread-safe database access
    - Neo4j Schema:
        * InferenceEpisode nodes (reasoning events)
        * ProofStep nodes (proof tree nodes)
        * PRODUCED: InferenceEpisode -> ProofStep (root)
        * CHILD_PROOF: ProofStep -> ProofStep (hierarchy)
        * USED_FACT/APPLIED_RULE: Tracking applied knowledge
    - Supports meta-reasoning queries ("What inferences have I performed?")

Thread Safety:
    All database operations are thread-safe via Neo4jSessionMixin._safe_run()
    with RLock synchronization.

Dependencies:
    - infrastructure/neo4j_session_mixin.py: Session management
    - component_15_logging_config.py: Structured logging
    - kai_exceptions.py: Exception hierarchy

Usage:
    from neo4j import Driver
    from component_1_inference_memory import InferenceMemory

    driver = Driver("bolt://localhost:7687", auth=("neo4j", "password"))
    memory = InferenceMemory(driver)

    # Track inference
    episode_id = memory.create_inference_episode(
        inference_type="backward_chaining",
        query="Was ist ein Hund?",
        metadata={"max_depth": 5}
    )

    # Create proof steps
    step_id = memory.create_proof_step(
        goal="IS_A(hund, ?x)",
        method="fact",
        confidence=1.0,
        depth=0,
        bindings={"?x": "tier"}
    )

    # Link inference to proof
    memory.link_inference_to_proof(episode_id, step_id)

    # Query history
    history = memory.query_inference_history(topic="hund")

    # Generate explanation
    explanation = memory.explain_inference(episode_id)

Note: Follows CLAUDE.md standards - NO cp1252-unsafe Unicode, structured logging,
      comprehensive error handling, thread safety.
"""

import json
from typing import Any, Dict, List, Optional

from neo4j import Driver

from component_15_logging_config import get_logger
from infrastructure.neo4j_session_mixin import Neo4jSessionMixin

logger = get_logger(__name__)


class InferenceMemory(Neo4jSessionMixin):
    """
    Inference tracking and proof step management.

    Provides storage and retrieval of reasoning episodes with hierarchical
    proof trees. Enables transparent meta-reasoning and explanation generation.

    Attributes:
        driver: Neo4j driver instance (inherited from Neo4jSessionMixin)

    Thread Safety:
        All methods are thread-safe via Neo4jSessionMixin._safe_run()

    Example:
        memory = InferenceMemory(driver)

        # Track inference
        ep_id = memory.create_inference_episode("backward_chaining", "Query?")
        step_id = memory.create_proof_step("goal", "fact", 1.0, 0)
        memory.link_inference_to_proof(ep_id, step_id)

        # Query and explain
        history = memory.query_inference_history(topic="concept")
        explanation = memory.explain_inference(ep_id)
    """

    def __init__(self, driver: Driver):
        """
        Initialize inference memory with Neo4j driver.

        Args:
            driver: Neo4j driver instance

        Raises:
            Neo4jConnectionError: If driver is None or connection fails
        """
        super().__init__(driver, enable_cache=False)
        logger.debug("InferenceMemory initialized")

    def create_inference_episode(
        self, inference_type: str, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create an InferenceEpisode node to track reasoning processes.

        Episodic Memory for Reasoning: Tracks WHEN, HOW, and WHY inferences
        were performed. Each InferenceEpisode represents a complete reasoning
        run (e.g., backward chaining, multi-hop reasoning).

        Args:
            inference_type: Type of inference ("forward_chaining", "backward_chaining",
                           "graph_traversal", "abductive", "hybrid")
            query: Original question/goal
            metadata: Additional metadata (e.g., {"topic": "...", "confidence": 0.8})

        Returns:
            InferenceEpisode ID (UUID) if successful, None on error

        Example:
            episode_id = memory.create_inference_episode(
                inference_type="backward_chaining",
                query="Was ist ein Hund?",
                metadata={"topic": "hund", "max_depth": 5}
            )
        """
        try:
            results = self._safe_run(
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
                operation="create_inference_episode",
                type=inference_type,
                user_query=query,
                metadata=json.dumps(metadata or {}),
            )

            episode_id = results[0]["episode_id"] if results else None

            if episode_id:
                logger.info(
                    "InferenceEpisode created: %s",
                    inference_type,
                    extra={"episode_id": episode_id, "query": query[:50]},
                )

            return episode_id

        except Exception as e:
            logger.error(
                "Error creating inference episode: %s",
                str(e)[:100],
                extra={"inference_type": inference_type},
                exc_info=True,
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
        Create a ProofStep node representing a reasoning step.

        Args:
            goal: Goal being proven (e.g., "IS_A(hund, ?x)")
            method: Proof method ("fact", "rule", "graph_traversal", "hypothesis")
            confidence: Confidence of this proof step (0.0 to 1.0)
            depth: Depth in proof tree
            bindings: Variable bindings (e.g., {"?x": "tier"})
            parent_step_id: ID of parent ProofStep (for hierarchy)

        Returns:
            ProofStep ID (UUID) if successful, None on error

        Example:
            step_id = memory.create_proof_step(
                goal="IS_A(hund, ?x)",
                method="fact",
                confidence=1.0,
                depth=0,
                bindings={"?x": "tier"}
            )
        """
        try:
            results = self._safe_run(
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
                operation="create_proof_step",
                goal=goal,
                method=method,
                confidence=confidence,
                depth=depth,
                bindings=json.dumps(bindings or {}),
            )

            step_id = results[0]["step_id"] if results else None

            # Link to parent if provided
            if step_id and parent_step_id:
                self._safe_run(
                    """
                    MATCH (parent:ProofStep {id: $parent_id})
                    MATCH (child:ProofStep {id: $child_id})
                    MERGE (parent)-[:CHILD_PROOF]->(child)
                    """,
                    operation="link_proof_steps",
                    parent_id=parent_step_id,
                    child_id=step_id,
                )

            if step_id:
                logger.debug(
                    "ProofStep created: %s",
                    method,
                    extra={"step_id": step_id, "goal": goal[:50], "depth": depth},
                )

            return step_id

        except Exception as e:
            logger.error(
                "Error creating proof step: %s",
                str(e)[:100],
                extra={"goal": goal[:50]},
                exc_info=True,
            )
            return None

    def link_inference_to_proof(
        self, inference_episode_id: str, proof_step_id: str
    ) -> bool:
        """
        Link an InferenceEpisode to its root ProofStep.

        Args:
            inference_episode_id: ID of InferenceEpisode
            proof_step_id: ID of root ProofStep

        Returns:
            True if successfully linked, False on error
        """
        try:
            results = self._safe_run(
                """
                MATCH (ie:InferenceEpisode {id: $episode_id})
                MATCH (ps:ProofStep {id: $step_id})
                MERGE (ie)-[r:PRODUCED]->(ps)
                ON CREATE SET r.linked_at = timestamp()
                RETURN r IS NOT NULL AS success
                """,
                operation="link_inference_to_proof",
                episode_id=inference_episode_id,
                step_id=proof_step_id,
            )

            success = results[0]["success"] if results else False

            if success:
                logger.debug(
                    "InferenceEpisode linked to ProofStep",
                    extra={
                        "episode_id": inference_episode_id[:8],
                        "step_id": proof_step_id[:8],
                    },
                )

            return success

        except Exception as e:
            logger.error(
                "Error linking inference to proof: %s", str(e)[:100], exc_info=True
            )
            return False

    def link_inference_to_facts(
        self, inference_episode_id: str, fact_ids: List[str]
    ) -> bool:
        """
        Link an InferenceEpisode to all facts used in reasoning.

        Args:
            inference_episode_id: ID of InferenceEpisode
            fact_ids: List of Fact IDs (from reasoning engine)

        Returns:
            True if successful, False on error
        """
        if not fact_ids:
            return False

        try:
            for fact_id in fact_ids:
                self._safe_run(
                    """
                    MATCH (ie:InferenceEpisode {id: $episode_id})
                    MERGE (f:InferenceFact {fact_id: $fact_id})
                    MERGE (ie)-[r:USED_FACT]->(f)
                    ON CREATE SET r.linked_at = timestamp()
                    """,
                    operation="link_inference_to_fact",
                    episode_id=inference_episode_id,
                    fact_id=fact_id,
                )

            logger.debug(
                "InferenceEpisode linked to %d facts",
                len(fact_ids),
                extra={"episode_id": inference_episode_id[:8]},
            )

            return True

        except Exception as e:
            logger.error(
                "Error linking inference to facts: %s", str(e)[:100], exc_info=True
            )
            return False

    def link_inference_to_rules(
        self, inference_episode_id: str, rule_ids: List[str]
    ) -> bool:
        """
        Link an InferenceEpisode to all rules applied in reasoning.

        Args:
            inference_episode_id: ID of InferenceEpisode
            rule_ids: List of Rule IDs (from reasoning engine)

        Returns:
            True if successful, False on error
        """
        if not rule_ids:
            return False

        try:
            for rule_id in rule_ids:
                self._safe_run(
                    """
                    MATCH (ie:InferenceEpisode {id: $episode_id})
                    MATCH (r:Regel {id: $rule_id})
                    MERGE (ie)-[rel:APPLIED_RULE]->(r)
                    ON CREATE SET rel.applied_at = timestamp()
                    """,
                    operation="link_inference_to_rule",
                    episode_id=inference_episode_id,
                    rule_id=rule_id,
                )

            logger.debug(
                "InferenceEpisode linked to %d rules",
                len(rule_ids),
                extra={"episode_id": inference_episode_id[:8]},
            )

            return True

        except Exception as e:
            logger.error(
                "Error linking inference to rules: %s", str(e)[:100], exc_info=True
            )
            return False

    def query_inference_history(
        self,
        topic: Optional[str] = None,
        inference_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find all InferenceEpisodes, optionally filtered by topic or type.

        Answers meta-questions like:
        - "When did I reason about X?"
        - "Which backward-chaining inferences have I performed?"
        - "How often did I use multi-hop reasoning?"

        Args:
            topic: Optional - filter by topic (searches query and metadata)
            inference_type: Optional - filter by inference type
            limit: Maximum number of episodes (newest first)

        Returns:
            List of InferenceEpisode dictionaries with:
            - episode_id: UUID of episode
            - inference_type: Type of inference
            - query: Original question
            - timestamp: Timestamp
            - metadata: Additional metadata
            - used_facts_count: Number of facts used
            - applied_rules_count: Number of rules applied

        Example:
            episodes = memory.query_inference_history(topic="hund")
            for ep in episodes:
                print(f"{ep['inference_type']}: {ep['query']}")
        """
        try:
            # Build query dynamically
            where_clauses = []
            params: Dict[str, Any] = {"limit": limit}

            if inference_type:
                where_clauses.append("ie.inference_type = $inference_type")
                params["inference_type"] = inference_type

            if topic:
                # Search in query and metadata
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

            results = self._safe_run(
                query, operation="query_inference_history", **params
            )

            logger.debug(
                "Query inference history: %d episodes found%s%s",
                len(results),
                f" (Topic: {topic})" if topic else "",
                f" (Type: {inference_type})" if inference_type else "",
            )

            return results

        except Exception as e:
            logger.error(
                "Error querying inference history: %s", str(e)[:100], exc_info=True
            )
            return []

    def get_proof_tree(self, root_step_id: str) -> Optional[Dict[str, Any]]:
        """
        Reconstruct the complete proof tree from a root ProofStep.

        Args:
            root_step_id: ID of root ProofStep

        Returns:
            Hierarchical dictionary with:
            - step_id: UUID of step
            - goal: Goal
            - method: Proof method
            - confidence: Confidence
            - depth: Depth
            - bindings: Variable bindings
            - children: List of child steps (recursive)

        Example:
            tree = memory.get_proof_tree(root_step_id)
            # tree = {
            #     "goal": "IS_A(hund, ?x)",
            #     "method": "rule",
            #     "children": [
            #         {"goal": "IS_A(hund, saeugetier)", "method": "fact", ...},
            #         {"goal": "IS_A(saeugetier, tier)", "method": "fact", ...}
            #     ]
            # }
        """
        try:
            # Recursive helper to build tree
            def build_tree(step_id: str) -> Dict[str, Any]:
                """Recursively build proof tree"""
                step_results = self._safe_run(
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
                    operation="get_proof_step",
                    step_id=step_id,
                )

                if not step_results:
                    return {}

                record = step_results[0]

                node = {
                    "step_id": record["id"],
                    "goal": record["goal"],
                    "method": record["method"],
                    "confidence": record["confidence"],
                    "depth": record["depth"],
                    "bindings": (
                        json.loads(record["bindings"]) if record["bindings"] else {}
                    ),
                    "children": [],
                }

                # Recursively build children
                for child_id in record["child_ids"]:
                    if child_id:  # Null-check
                        child_tree = build_tree(child_id)
                        if child_tree:
                            node["children"].append(child_tree)

                return node

            tree = build_tree(root_step_id)

            if tree:
                logger.debug("Proof tree reconstructed: %s", root_step_id[:8])

            return tree if tree else None

        except Exception as e:
            logger.error("Error getting proof tree: %s", str(e)[:100], exc_info=True)
            return None

    def explain_inference(self, episode_id: str) -> str:
        """
        Generate natural language explanation of an inference episode.

        Args:
            episode_id: ID of InferenceEpisode

        Returns:
            Human-readable explanation of the reasoning process

        Example:
            explanation = memory.explain_inference(episode_id)
            print(explanation)
            # Output:
            # "To answer 'Was ist ein Hund?', I:
            #  1. Used backward chaining
            #  2. Analyzed 3 facts
            #  3. Applied 2 rules
            #  4. Found an answer with 100% confidence."
        """
        try:
            results = self._safe_run(
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
                operation="explain_inference",
                episode_id=episode_id,
            )

            if not results:
                return f"InferenceEpisode {episode_id[:8]} not found."

            record = results[0]
            inference_type = record["inference_type"]
            query = record["query"]
            final_confidence = record["final_confidence"] or 0.0

            # Generate explanation
            explanation_parts = []
            explanation_parts.append(f"To answer '{query}', I:")

            # Method
            method_desc = {
                "forward_chaining": "forward chaining (rule-based reasoning)",
                "backward_chaining": "backward chaining (goal-driven reasoning)",
                "graph_traversal": "graph traversal (multi-hop reasoning)",
                "abductive": "abductive reasoning (hypothesis generation)",
                "hybrid": "hybrid reasoning strategy",
            }.get(inference_type, inference_type)

            explanation_parts.append(f"1. Used {method_desc}")

            # Facts
            num_facts = len(record["fact_ids"])
            if num_facts > 0:
                explanation_parts.append(
                    f"2. Analyzed {num_facts} fact{'s' if num_facts > 1 else ''}"
                )

            # Rules
            num_rules = len(record["rule_ids"])
            if num_rules > 0:
                explanation_parts.append(
                    f"3. Applied {num_rules} rule{'s' if num_rules > 1 else ''}"
                )

            # Confidence
            confidence_percent = int(final_confidence * 100)
            explanation_parts.append(
                f"4. Found an answer with {confidence_percent}% confidence"
            )

            explanation = "\n".join(explanation_parts) + "."

            logger.debug(
                "Explanation generated for InferenceEpisode %s", episode_id[:8]
            )

            return explanation

        except Exception as e:
            logger.error("Error explaining inference: %s", str(e)[:100], exc_info=True)
            return f"Error generating explanation: {str(e)[:100]}"
