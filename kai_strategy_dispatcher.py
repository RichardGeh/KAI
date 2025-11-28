# kai_strategy_dispatcher.py
"""
Strategy Dispatcher for KAI Reasoning Orchestrator

Handles strategy selection, execution, parallel dispatch, and error handling
for hybrid reasoning across multiple reasoning engines.

Responsibilities:
- Select appropriate reasoning strategies based on query type
- Execute strategies sequentially or in parallel
- Handle strategy failures with fallback mechanisms
- Cost-based routing for optimal strategy selection
- Meta-learning integration for strategy selection

Architecture:
    StrategyDispatcher coordinates all strategy execution, delegating to
    specialized strategy execution methods (_try_*) that interact with
    individual reasoning engines.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from component_1_netzwerk import KonzeptNetzwerk

logger = logging.getLogger(__name__)


class StrategyDispatcher:
    """
    Dispatches queries to appropriate reasoning engines.

    Responsible for strategy selection, parallel execution, error handling,
    and cost-based routing across all available reasoning engines.
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        logic_engine,
        graph_traversal,
        working_memory,
        signals,
        probabilistic_engine=None,
        abductive_engine=None,
        combinatorial_reasoner=None,
        spatial_reasoner=None,
        resonance_engine=None,
        meta_learning_engine=None,
        self_evaluator=None,
        enable_parallel_execution: bool = False,
        min_confidence_threshold: float = 0.4,
    ):
        """
        Initialize Strategy Dispatcher.

        Args:
            netzwerk: KonzeptNetzwerk instance
            logic_engine: Logic Engine instance
            graph_traversal: GraphTraversal instance
            working_memory: WorkingMemory instance
            signals: KaiSignals for UI updates
            probabilistic_engine: ProbabilisticEngine (optional)
            abductive_engine: AbductiveEngine (optional)
            combinatorial_reasoner: CombinatorialReasoner (optional)
            spatial_reasoner: SpatialReasoner (optional)
            resonance_engine: ResonanceEngine (optional)
            meta_learning_engine: MetaLearningEngine (optional)
            self_evaluator: SelfEvaluator (optional)
            enable_parallel_execution: Enable parallel strategy execution
            min_confidence_threshold: Minimum confidence for success
        """
        self.netzwerk = netzwerk
        self.logic_engine = logic_engine
        self.graph_traversal = graph_traversal
        self.working_memory = working_memory
        self.signals = signals
        self.probabilistic_engine = probabilistic_engine
        self.abductive_engine = abductive_engine
        self.combinatorial_reasoner = combinatorial_reasoner
        self.spatial_reasoner = spatial_reasoner
        self.resonance_engine = resonance_engine
        self.meta_learning_engine = meta_learning_engine
        self.self_evaluator = self_evaluator

        self.enable_parallel_execution = enable_parallel_execution
        self.min_confidence_threshold = min_confidence_threshold

        # Import ReasoningResult and ReasoningStrategy from orchestrator
        from kai_reasoning_orchestrator import ReasoningResult, ReasoningStrategy

        self.ReasoningResult = ReasoningResult
        self.ReasoningStrategy = ReasoningStrategy

        # Import unified proof system
        try:
            from component_17_proof_explanation import ProofStep as UnifiedProofStep
            from component_17_proof_explanation import (
                ProofTree,
                StepType,
                create_proof_tree_from_logic_engine,
            )

            self.ProofTree = ProofTree
            self.UnifiedProofStep = UnifiedProofStep
            self.StepType = StepType
            self.create_proof_tree_from_logic_engine = (
                create_proof_tree_from_logic_engine
            )
            self.PROOF_SYSTEM_AVAILABLE = True
        except ImportError:
            self.PROOF_SYSTEM_AVAILABLE = False

        logger.info(
            f"StrategyDispatcher initialized: "
            f"parallel={self.enable_parallel_execution}, "
            f"min_confidence={self.min_confidence_threshold}"
        )

    def execute_strategies(
        self,
        topic: str,
        relation_type: str,
        strategies: Optional[List],
        probabilistic_enhancement: bool = True,
    ) -> List:
        """
        Execute multiple reasoning strategies.

        Args:
            topic: The topic to reason about
            relation_type: Type of relation to find
            strategies: List of ReasoningStrategy enums to execute
            probabilistic_enhancement: Whether to enhance with probabilistic reasoning

        Returns:
            List of ReasoningResult objects
        """
        logger.info(
            f"[Strategy Dispatcher] Executing {len(strategies)} strategies for: {topic}"
        )

        results = []

        # Stage 1: Direct Fact Lookup (Fast Path)
        if self.ReasoningStrategy.DIRECT_FACT in strategies:
            direct_result = self._try_direct_fact_lookup(topic, relation_type)
            if direct_result and direct_result.success:
                if direct_result.confidence >= 0.95:
                    # High confidence direct fact - return immediately
                    logger.info(
                        "[Strategy Dispatcher] [OK] High-confidence direct fact found"
                    )
                    return [direct_result]
                results.append(direct_result)

        # Stage 2: Deterministic Reasoning (Graph + Logic + Spatial + Constraint + Resonance)
        deterministic_results = self._execute_deterministic_strategies(
            topic, relation_type, strategies
        )
        results.extend(deterministic_results)

        # Stage 3: Probabilistic Enhancement
        if (
            self.ReasoningStrategy.PROBABILISTIC in strategies
            and probabilistic_enhancement
        ):
            if deterministic_results:
                # Enhance deterministic results with probabilistic reasoning
                prob_result = self._enhance_with_probabilistic(
                    topic, relation_type, deterministic_results
                )
                if prob_result and prob_result.success:
                    results.append(prob_result)
            else:
                # Try standalone probabilistic reasoning
                prob_result = self._try_probabilistic(topic, relation_type)
                if prob_result and prob_result.success:
                    results.append(prob_result)

        # Stage 4: Abductive Fallback (Hypothesis Generation)
        if self.ReasoningStrategy.ABDUCTIVE in strategies:
            if not results or all(
                r.confidence < self.min_confidence_threshold for r in results
            ):
                logger.info("[Strategy Dispatcher] Falling back to Abductive Reasoning")
                abd_result = self._try_abductive(topic, relation_type)
                if abd_result and abd_result.success:
                    results.append(abd_result)

        logger.info(
            f"[Strategy Dispatcher] Completed: {len(results)} successful strategies"
        )
        return results

    def _execute_deterministic_strategies(
        self, topic: str, relation_type: str, strategies: List
    ) -> List:
        """
        Execute deterministic reasoning strategies (Graph, Logic, Spatial, Constraint, Resonance).

        Uses parallel execution if enabled for Graph + Logic.

        Args:
            topic: Topic to reason about
            relation_type: Relation type
            strategies: List of strategies to execute

        Returns:
            List of ReasoningResult objects
        """
        results = []

        # Determine which deterministic strategies to run
        deterministic_strategies = [
            s
            for s in strategies
            if s
            in [
                self.ReasoningStrategy.GRAPH_TRAVERSAL,
                self.ReasoningStrategy.LOGIC_ENGINE,
            ]
        ]

        # Parallel execution for Graph + Logic if enabled
        if self.enable_parallel_execution and len(deterministic_strategies) > 1:
            logger.debug("[Strategy Dispatcher] Running Graph + Logic in parallel")

            futures = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                if self.ReasoningStrategy.GRAPH_TRAVERSAL in strategies:
                    futures[
                        executor.submit(self._try_graph_traversal, topic, relation_type)
                    ] = "graph"
                if self.ReasoningStrategy.LOGIC_ENGINE in strategies:
                    futures[
                        executor.submit(self._try_logic_engine, topic, relation_type)
                    ] = "logic"

                for future in as_completed(futures):
                    strategy_name = futures[future]
                    try:
                        result = future.result()
                        if result and result.success:
                            results.append(result)
                            logger.debug(
                                f"[Parallel Execution] {strategy_name} completed successfully"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[Parallel Execution] {strategy_name} failed: {e}"
                        )
        else:
            # Sequential execution (default)
            if self.ReasoningStrategy.GRAPH_TRAVERSAL in strategies:
                graph_result = self._try_graph_traversal(topic, relation_type)
                if graph_result and graph_result.success:
                    results.append(graph_result)

            if self.ReasoningStrategy.LOGIC_ENGINE in strategies:
                logic_result = self._try_logic_engine(topic, relation_type)
                if logic_result and logic_result.success:
                    results.append(logic_result)

        # Sequential for other deterministic strategies
        if self.ReasoningStrategy.SPATIAL in strategies:
            spatial_result = self._try_spatial_reasoning(topic, relation_type)
            if spatial_result and spatial_result.success:
                results.append(spatial_result)

        if self.ReasoningStrategy.CONSTRAINT in strategies:
            constraint_result = self._try_constraint_solving(topic, relation_type)
            if constraint_result and constraint_result.success:
                results.append(constraint_result)

        if self.ReasoningStrategy.RESONANCE in strategies:
            resonance_result = self._try_resonance(topic, relation_type)
            if resonance_result and resonance_result.success:
                results.append(resonance_result)

        return results

    def execute_single_strategy(
        self, topic: str, relation_type: str, strategy
    ) -> Optional:
        """
        Execute a single reasoning strategy.

        Args:
            topic: Topic to reason about
            relation_type: Relation type
            strategy: ReasoningStrategy enum to execute

        Returns:
            ReasoningResult or None
        """
        try:
            result = None

            if strategy == self.ReasoningStrategy.DIRECT_FACT:
                result = self._try_direct_fact_lookup(topic, relation_type)
            elif strategy == self.ReasoningStrategy.GRAPH_TRAVERSAL:
                result = self._try_graph_traversal(topic, relation_type)
            elif strategy == self.ReasoningStrategy.LOGIC_ENGINE:
                result = self._try_logic_engine(topic, relation_type)
            elif strategy == self.ReasoningStrategy.PROBABILISTIC:
                result = self._try_probabilistic(topic, relation_type)
            elif strategy == self.ReasoningStrategy.ABDUCTIVE:
                result = self._try_abductive(topic, relation_type)
            elif strategy == self.ReasoningStrategy.SPATIAL:
                result = self._try_spatial_reasoning(topic, relation_type)
            elif strategy == self.ReasoningStrategy.RESONANCE:
                result = self._try_resonance(topic, relation_type)

            return result

        except Exception as e:
            logger.error(f"Error executing strategy {strategy}: {e}", exc_info=True)
            return None

    def execute_resonance_strategy(
        self,
        topic: str,
        relation_type: str,
        context: Dict[str, Any],
        query_text: str,
    ) -> Optional:
        """
        Execute Resonance Strategy with Adaptive Hyperparameters.

        Special handling for Resonance:
        1. Use AdaptiveResonanceEngine with auto-tuning
        2. Self-evaluation of activation map
        3. Enhanced proof tree generation

        Args:
            topic: Topic to reason about
            relation_type: Relation type
            context: Context dict
            query_text: Query text

        Returns:
            ReasoningResult or None
        """
        if not self.resonance_engine:
            logger.warning("[Resonance] ResonanceEngine not available")
            return None

        logger.info("[Resonance] Executing with adaptive hyperparameters")

        # 1. Auto-tune hyperparameters (if AdaptiveResonanceEngine)
        try:
            from component_44_resonance_engine import AdaptiveResonanceEngine

            if isinstance(self.resonance_engine, AdaptiveResonanceEngine):
                try:
                    self.resonance_engine.auto_tune_hyperparameters()
                    logger.debug("[Resonance] Auto-tuning completed")
                except Exception as e:
                    logger.warning(f"[Resonance] Auto-tuning failed: {e}")
        except ImportError:
            pass

        # 2. Execute resonance activation
        try:
            allowed_relations = context.get("allowed_relations", [relation_type])

            activation_map = self.resonance_engine.activate_concept(
                start_word=topic,
                query_context=context,
                allowed_relations=allowed_relations,
            )

            if not activation_map or activation_map.concepts_activated <= 1:
                logger.info("[Resonance] No significant activation")
                return None

            # 3. Extract inferred facts from activation map
            inferred_facts = {}
            top_concepts = activation_map.get_top_concepts(n=20)

            for concept, activation in top_concepts:
                if concept == topic:
                    continue  # Skip start concept

                # Find paths to determine relation type
                paths = activation_map.get_paths_to(concept)
                if paths:
                    best_path = max(paths, key=lambda p: p.confidence_product)
                    if best_path.relations:
                        rel_type = best_path.relations[0]
                        if rel_type not in inferred_facts:
                            inferred_facts[rel_type] = []
                        if concept not in inferred_facts[rel_type]:
                            inferred_facts[rel_type].append(concept)

            # Calculate confidence
            confidence = min(activation_map.max_activation, 1.0)

            # Boost for resonance points
            if activation_map.resonance_points:
                resonance_boost = len(activation_map.resonance_points) * 0.05
                confidence = min(confidence + resonance_boost, 1.0)

            # 4. Create ReasoningResult
            result = self.ReasoningResult(
                strategy=self.ReasoningStrategy.RESONANCE,
                success=True,
                confidence=confidence,
                inferred_facts=inferred_facts,
                proof_tree=None,  # Will be created below
                proof_trace=self.resonance_engine.get_activation_summary(
                    activation_map
                ),
                metadata={
                    "concepts_activated": activation_map.concepts_activated,
                    "waves_executed": activation_map.waves_executed,
                    "resonance_points": len(activation_map.resonance_points),
                    "max_activation": activation_map.max_activation,
                },
            )

            # 5. Create Proof Tree
            if self.PROOF_SYSTEM_AVAILABLE:
                proof_tree = self.ProofTree(query=query_text)

                # Root step: Activation summary
                summary = self.resonance_engine.get_activation_summary(activation_map)
                root_step = self.UnifiedProofStep(
                    step_id="resonance_activation",
                    step_type=self.StepType.INFERENCE,
                    inputs=[topic],
                    output=f"{activation_map.concepts_activated} Konzepte aktiviert",
                    confidence=confidence,
                    explanation_text=summary,
                    source_component="adaptive_resonance_engine",
                    metadata={
                        "waves": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                    },
                )
                proof_tree.add_root_step(root_step)

                result.proof_tree = proof_tree

            return result

        except Exception as e:
            logger.error(f"[Resonance] Execution failed: {e}", exc_info=True)
            return None

    # ========================================================================
    # Strategy Execution Methods (Internal)
    # ========================================================================

    def _try_direct_fact_lookup(self, topic: str, relation_type: str) -> Optional:
        """
        Fast path: Direct fact lookup in knowledge graph.
        """
        logger.debug(f"[Direct Fact] Querying: {topic}")

        try:
            facts = self.netzwerk.query_graph_for_facts(topic)

            if relation_type in facts and facts[relation_type]:
                # Found direct facts
                inferred_facts = {relation_type: facts[relation_type]}

                # Create simple proof tree
                proof_tree = None
                if self.PROOF_SYSTEM_AVAILABLE:
                    proof_tree = self.ProofTree(query=f"Was ist ein {topic}?")

                    for obj in facts[relation_type][:3]:  # Limit to 3
                        step = self.UnifiedProofStep(
                            step_id=f"direct_{topic}_{obj}",
                            step_type=self.StepType.FACT_MATCH,
                            inputs=[topic],
                            output=f"{topic} {relation_type} {obj}",
                            confidence=1.0,
                            explanation_text=f"Direkter Fakt in Wissensbasis: {topic} -> {obj}",
                            source_component="direct_fact_lookup",
                        )
                        proof_tree.add_root_step(step)

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.DIRECT_FACT,
                    success=True,
                    confidence=1.0,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=f"Direkte Fakten gefunden: {len(facts[relation_type])} Eintraege",
                    metadata={"num_facts": len(facts[relation_type])},
                )

            return None

        except Exception as e:
            logger.warning(f"[Direct Fact] Fehler: {e}")
            return None

    def _try_graph_traversal(self, topic: str, relation_type: str) -> Optional:
        """
        Graph-based multi-hop reasoning.
        """
        logger.debug(f"[Graph Traversal] Topic: {topic}")

        try:
            paths = self.graph_traversal.find_transitive_relations(
                topic, relation_type, max_depth=5
            )

            if paths:
                # Extract inferred facts
                inferred_facts = {relation_type: []}
                for path in paths:
                    target = path.nodes[-1]
                    if target not in inferred_facts[relation_type]:
                        inferred_facts[relation_type].append(target)

                # Create proof tree
                proof_tree = None
                if self.PROOF_SYSTEM_AVAILABLE:
                    proof_tree = self.ProofTree(query=f"Was ist ein {topic}?")
                    for path in paths[:5]:
                        proof_step = self.graph_traversal.create_proof_step_from_path(
                            path, query=f"{topic} {relation_type}"
                        )
                        if proof_step:
                            proof_tree.add_root_step(proof_step)

                # Best path confidence
                best_confidence = paths[0].confidence if paths else 0.0

                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="graph_traversal_orchestrator",
                    description=f"Graph-Traversal fuer '{topic}' via Orchestrator",
                    data={
                        "num_paths": len(paths),
                        "inferred_facts": inferred_facts,
                        "relation_type": relation_type,
                    },
                    confidence=best_confidence,
                )

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.GRAPH_TRAVERSAL,
                    success=True,
                    confidence=best_confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=f"Graph-Traversal: {len(paths)} Pfade gefunden",
                    metadata={
                        "num_paths": len(paths),
                        "avg_hops": sum(len(p.relations) for p in paths) / len(paths),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Graph Traversal] Fehler: {e}")
            return None

    def _try_logic_engine(self, topic: str, relation_type: str) -> Optional:
        """
        Rule-based backward chaining.
        """
        logger.debug(f"[Logic Engine] Topic: {topic}")

        try:
            from component_9_logik_engine import Goal

            # Create goal
            goal = Goal(
                pred=relation_type, args={"subject": topic.lower(), "object": None}
            )

            # Load facts
            all_facts = self._load_facts_from_graph(topic)
            for fact in all_facts:
                self.logic_engine.add_fact(fact)

            # Run with tracking
            query_text = f"Was ist ein {topic}?"
            proof = self.logic_engine.run_with_tracking(
                goal=goal,
                inference_type="backward_chaining",
                query=query_text,
                max_depth=5,
            )

            if proof:
                # Extract facts
                inferred_facts = self._extract_facts_from_proof(proof)

                # Create proof tree
                proof_tree = None
                if self.PROOF_SYSTEM_AVAILABLE:
                    proof_tree = self.create_proof_tree_from_logic_engine(
                        proof, query=query_text
                    )

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.LOGIC_ENGINE,
                    success=True,
                    confidence=proof.confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=self.logic_engine.format_proof_trace(proof),
                    metadata={"method": proof.method, "depth": proof.goal.depth},
                )

            return None

        except Exception as e:
            logger.warning(f"[Logic Engine] Fehler: {e}")
            return None

    def _try_probabilistic(self, topic: str, relation_type: str) -> Optional:
        """
        Standalone probabilistic reasoning.
        """
        if not self.probabilistic_engine:
            return None

        logger.debug(f"[Probabilistic] Topic: {topic}")

        try:
            # Query probabilistic engine
            goal_sig = f"{relation_type}(subject={topic.lower()},object=?)"
            prob, conf = self.probabilistic_engine.query(goal_sig)

            if prob > 0.3:  # Minimum threshold
                # Create result
                explanation = self.probabilistic_engine.generate_response(
                    goal_sig, threshold_high=0.8, threshold_low=0.2
                )

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.PROBABILISTIC,
                    success=True,
                    confidence=prob,
                    inferred_facts={},  # Probabilistic gibt keine konkreten Fakten
                    proof_tree=None,
                    proof_trace=explanation,
                    metadata={"probability": prob, "confidence": conf},
                )

            return None

        except Exception as e:
            logger.warning(f"[Probabilistic] Fehler: {e}")
            return None

    def _enhance_with_probabilistic(
        self,
        topic: str,
        relation_type: str,
        deterministic_results: List,
    ) -> Optional:
        """
        Enhance deterministic results with probabilistic uncertainty quantification.
        """
        if not self.probabilistic_engine:
            return None

        logger.debug(
            f"[Probabilistic Enhancement] Enhancing {len(deterministic_results)} results"
        )

        try:
            from component_16_probabilistic_engine import ProbabilisticFact

            # Add deterministic facts to probabilistic engine
            for result in deterministic_results:
                for rel_type, objects in result.inferred_facts.items():
                    for obj in objects:
                        # FIX: strategy kann String oder Enum sein
                        strategy_value = (
                            result.strategy.value
                            if hasattr(result.strategy, "value")
                            else str(result.strategy)
                        )
                        fact = ProbabilisticFact(
                            pred=rel_type,
                            args={"subject": topic.lower(), "object": obj.lower()},
                            probability=result.confidence,
                            source=f"deterministic_{strategy_value}",
                        )
                        self.probabilistic_engine.add_fact(fact)

            # Run probabilistic inference
            derived_facts = self.probabilistic_engine.infer(max_iterations=3)

            if derived_facts:
                # Calculate enhanced confidence
                goal_sig = f"{relation_type}(subject={topic.lower()},object=?)"
                enhanced_prob, enhanced_conf = self.probabilistic_engine.query(goal_sig)

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.PROBABILISTIC,
                    success=True,
                    confidence=enhanced_conf,
                    inferred_facts={},
                    proof_tree=None,
                    proof_trace=f"Probabilistische Verbesserung: P={enhanced_prob:.2f}, Conf={enhanced_conf:.2f}",
                    metadata={
                        "enhanced": True,
                        "base_results": len(deterministic_results),
                        "derived_facts": len(derived_facts),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Probabilistic Enhancement] Fehler: {e}")
            return None

    def _try_abductive(self, topic: str, relation_type: str) -> Optional:
        """
        Abductive hypothesis generation.
        """
        if not self.abductive_engine:
            return None

        logger.debug(f"[Abductive] Topic: {topic}")

        try:
            # Load context facts
            all_facts = self._load_facts_from_graph(topic)

            # Generate hypotheses
            observation = f"Es wurde nach '{topic}' gefragt"
            hypotheses = self.abductive_engine.generate_hypotheses(
                observation=observation,
                context_facts=all_facts,
                strategies=["template", "analogy", "causal_chain"],
                max_hypotheses=5,
            )

            if hypotheses:
                best_hypothesis = hypotheses[0]

                # Extract inferred facts
                inferred_facts = {}
                for fact in best_hypothesis.abduced_facts:
                    rel = fact.pred
                    obj = fact.args.get("object", "")
                    if rel not in inferred_facts:
                        inferred_facts[rel] = []
                    if obj and obj not in inferred_facts[rel]:
                        inferred_facts[rel].append(obj)

                # Create proof tree
                proof_tree = None
                if self.PROOF_SYSTEM_AVAILABLE:
                    proof_tree = self.ProofTree(query=observation)
                    hypothesis_steps = (
                        self.abductive_engine.create_multi_hypothesis_proof_chain(
                            hypotheses[:3], query=observation
                        )
                    )
                    for step in hypothesis_steps:
                        proof_tree.add_root_step(step)

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.ABDUCTIVE,
                    success=True,
                    confidence=best_hypothesis.confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=best_hypothesis.explanation,
                    metadata={
                        "strategy": best_hypothesis.strategy,
                        "num_hypotheses": len(hypotheses),
                        "scores": best_hypothesis.scores,
                    },
                    is_hypothesis=True,
                )

            return None

        except Exception as e:
            logger.warning(f"[Abductive] Fehler: {e}")
            return None

    def _try_spatial_reasoning(self, topic: str, relation_type: str = None) -> Optional:
        """
        Spatial reasoning using SpatialReasoner.
        """
        if not self.spatial_reasoner:
            return None

        logger.debug(f"[Spatial Reasoning] Topic: {topic}")

        try:
            # Use spatial reasoner to infer spatial relations
            result = self.spatial_reasoner.infer_spatial_relations(
                subject=topic, relation_type=relation_type
            )

            if result and result.success:
                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="spatial_reasoning",
                    description=f"Raeumliches Reasoning fuer '{topic}'",
                    data={
                        "relations": result.relations,
                        "confidence": result.confidence,
                    },
                    confidence=result.confidence,
                )

                # Create proof tree if available
                proof_tree = None
                if self.PROOF_SYSTEM_AVAILABLE:
                    proof_tree = self.ProofTree(
                        query=f"Raeumliche Relationen fuer {topic}"
                    )

                    # Add spatial reasoning steps
                    for rel_type, targets in result.relations.items():
                        for target in targets:
                            step = self.UnifiedProofStep(
                                step_type=self.StepType.QUERY,
                                description=f"{topic} {rel_type} {target}",
                                confidence=result.confidence,
                                metadata={
                                    "source": "spatial_reasoning",
                                    "relation": rel_type,
                                },
                            )
                            proof_tree.add_root_step(step)

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.SPATIAL,
                    success=True,
                    confidence=result.confidence,
                    inferred_facts=result.relations,
                    proof_tree=proof_tree,
                    proof_trace=f"Spatial Reasoning: {len(result.relations)} Relationen gefunden",
                    metadata={
                        "relation_types": list(result.relations.keys()),
                        "total_relations": sum(
                            len(v) for v in result.relations.values()
                        ),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Spatial Reasoning] Fehler: {e}")
            return None

    def _try_constraint_solving(
        self, topic: str, relation_type: str = None
    ) -> Optional:
        """
        Constraint Satisfaction Problem (CSP) solving for logic puzzles.
        """
        logger.debug(f"[Constraint Solving] Topic: {topic}")

        try:
            # Check if there is a constraint problem in working memory
            states = self.working_memory.get_reasoning_trace()
            constraint_problem = None

            for state in states:
                if state.step_type == "constraint_problem_detected":
                    constraint_problem = state.data.get("problem")
                    break

            if not constraint_problem:
                logger.debug("[Constraint Solving] Kein Constraint-Problem im Kontext")
                return None

            # Import CSP solver and translator
            from component_29_constraint_reasoning import (
                ConstraintSolver,
                translate_logical_constraints_to_csp,
            )

            # Translate logical constraints to CSP
            csp_problem = translate_logical_constraints_to_csp(constraint_problem)

            # Solve CSP
            solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
            solution = solver.solve(csp_problem)

            if solution:
                # Format solution as inferred facts
                inferred_facts = {}
                for var_name, value in solution.items():
                    # Create a "HAS_VALUE" relation for each variable
                    if "HAS_VALUE" not in inferred_facts:
                        inferred_facts["HAS_VALUE"] = []
                    inferred_facts["HAS_VALUE"].append(f"{var_name}={value}")

                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="constraint_solving",
                    description=f"CSP-Loesung fuer {len(solution)} Variablen gefunden",
                    data={
                        "solution": solution,
                        "confidence": constraint_problem.confidence,
                    },
                    confidence=constraint_problem.confidence,
                )

                # Create proof tree if available
                proof_tree = None
                if self.PROOF_SYSTEM_AVAILABLE:
                    proof_tree = self.ProofTree(
                        query=f"Constraint-Loesung fuer {topic}"
                    )

                    # Add solution steps
                    for var_name, value in solution.items():
                        step = self.UnifiedProofStep(
                            step_type=self.StepType.QUERY,
                            description=f"{var_name} = {value}",
                            confidence=constraint_problem.confidence,
                            metadata={
                                "source": "constraint_solving",
                                "csp_variables": len(solution),
                            },
                        )
                        proof_tree.add_root_step(step)

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.CONSTRAINT,
                    success=True,
                    confidence=constraint_problem.confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=f"CSP-Loesung: {len(solution)} Variablen geloest",
                    metadata={
                        "solution": solution,
                        "csp_variables": len(solution),
                        "csp_constraints": len(csp_problem.constraints),
                    },
                )

            logger.debug("[Constraint Solving] Keine Loesung gefunden")
            return None

        except Exception as e:
            logger.warning(f"[Constraint Solving] Fehler: {e}")
            return None

    def _try_resonance(self, topic: str, relation_type: str = None) -> Optional:
        """
        Spreading activation with resonance amplification.
        """
        if not self.resonance_engine:
            return None

        logger.debug(f"[Resonance] Topic: {topic}")

        try:
            # Prepare query context
            query_context = {}
            allowed_relations = []
            if relation_type:
                allowed_relations = [relation_type]

            # Activate concept with spreading activation
            activation_map = self.resonance_engine.activate_concept(
                start_word=topic,
                query_context=query_context,
                allowed_relations=allowed_relations,
            )

            if activation_map.concepts_activated > 1:  # More than just start concept
                # Extract inferred facts from activated concepts
                inferred_facts = {}

                # Get top activated concepts
                top_concepts = activation_map.get_top_concepts(n=20)

                for concept, activation in top_concepts:
                    if concept == topic:
                        continue  # Skip start concept

                    # Find paths to this concept to determine relation type
                    paths = activation_map.get_paths_to(concept)
                    if paths:
                        # Use relation from strongest path
                        best_path = max(paths, key=lambda p: p.confidence_product)
                        if best_path.relations:
                            rel_type = best_path.relations[0]
                            if rel_type not in inferred_facts:
                                inferred_facts[rel_type] = []
                            if concept not in inferred_facts[rel_type]:
                                inferred_facts[rel_type].append(concept)

                # Calculate confidence based on activation map
                confidence = min(activation_map.max_activation, 1.0)

                # Boost confidence if resonance points found
                if activation_map.resonance_points:
                    resonance_boost = len(activation_map.resonance_points) * 0.05
                    confidence = min(confidence + resonance_boost, 1.0)

                # Create proof tree
                proof_tree = None
                if self.PROOF_SYSTEM_AVAILABLE:
                    proof_tree = self.ProofTree(
                        query=f"Resonanz-Aktivierung fuer {topic}"
                    )

                    # Add activation explanation as root step
                    summary = self.resonance_engine.get_activation_summary(
                        activation_map
                    )

                    root_step = self.UnifiedProofStep(
                        step_id="resonance_activation",
                        step_type=self.StepType.INFERENCE,
                        inputs=[topic],
                        output=f"{activation_map.concepts_activated} Konzepte aktiviert",
                        confidence=confidence,
                        explanation_text=summary,
                        source_component="resonance_engine",
                        metadata={
                            "waves": activation_map.waves_executed,
                            "resonance_points": len(activation_map.resonance_points),
                            "max_activation": activation_map.max_activation,
                        },
                    )
                    proof_tree.add_root_step(root_step)

                    # Add top resonance points as substeps
                    for rp in sorted(
                        activation_map.resonance_points,
                        key=lambda x: x.resonance_boost,
                        reverse=True,
                    )[:5]:
                        explanation = self.resonance_engine.explain_activation(
                            rp.concept, activation_map, max_paths=3
                        )
                        resonance_step = self.UnifiedProofStep(
                            step_id=f"resonance_{rp.concept}",
                            step_type=self.StepType.FACT_MATCH,
                            inputs=[topic],
                            output=f"{rp.concept} (Resonanz: {rp.num_paths} Pfade)",
                            confidence=min(rp.resonance_boost, 1.0),
                            explanation_text=explanation,
                            source_component="resonance_engine",
                            metadata={"resonance_boost": rp.resonance_boost},
                        )
                        proof_tree.add_child_step(
                            "resonance_activation", resonance_step
                        )

                # Track in working memory
                self.working_memory.add_reasoning_state(
                    step_type="resonance_activation",
                    description=f"Spreading Activation von '{topic}'",
                    data={
                        "concepts_activated": activation_map.concepts_activated,
                        "waves": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                        "inferred_facts": inferred_facts,
                    },
                    confidence=confidence,
                )

                # Generate proof trace
                proof_trace = self.resonance_engine.get_activation_summary(
                    activation_map
                )

                return self.ReasoningResult(
                    strategy=self.ReasoningStrategy.RESONANCE,
                    success=True,
                    confidence=confidence,
                    inferred_facts=inferred_facts,
                    proof_tree=proof_tree,
                    proof_trace=proof_trace,
                    metadata={
                        "concepts_activated": activation_map.concepts_activated,
                        "waves_executed": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                        "max_activation": activation_map.max_activation,
                        "total_paths": len(activation_map.reasoning_paths),
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"[Resonance] Fehler: {e}")
            return None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _load_facts_from_graph(self, topic: str):
        """Load facts from knowledge graph (helper method)."""
        from component_9_logik_engine import Fact

        facts = []
        fact_data = self.netzwerk.query_graph_for_facts(topic)

        for relation_type, objects in fact_data.items():
            for obj in objects:
                fact = Fact(
                    pred=relation_type,
                    args={"subject": topic.lower(), "object": obj.lower()},
                    confidence=1.0,
                    source="graph",
                )
                facts.append(fact)

        return facts

    def _extract_facts_from_proof(self, proof) -> Dict[str, List[str]]:
        """Extract facts from Logic Engine proof (helper method)."""
        facts = {}

        if proof.supporting_facts:
            for fact in proof.supporting_facts:
                relation = fact.pred
                obj = fact.args.get("object", "")

                if relation not in facts:
                    facts[relation] = []

                if obj and obj not in facts[relation]:
                    facts[relation].append(obj)

        # Recursive through subgoals
        for subproof in proof.subgoals:
            subfacts = self._extract_facts_from_proof(subproof)
            for relation, objects in subfacts.items():
                if relation not in facts:
                    facts[relation] = []
                facts[relation].extend([o for o in objects if o not in facts[relation]])

        return facts

    def get_available_strategy_names(
        self, exclude: Optional[List[str]] = None
    ) -> List[str]:
        """
        Returns list of available strategy names (string format for Meta-Learning).

        Args:
            exclude: Optional list of strategy names to exclude

        Returns:
            List of available strategy names
        """
        exclude = exclude or []
        strategies = []

        # Map available engines to strategy names
        strategy_availability = {
            "direct_fact": True,  # Always available
            "logic_engine": self.logic_engine is not None,
            "graph_traversal": self.graph_traversal is not None,
            "probabilistic": self.probabilistic_engine is not None,
            "abductive": self.abductive_engine is not None,
            "combinatorial": self.combinatorial_reasoner is not None,
            "spatial": self.spatial_reasoner is not None,
            "resonance": self.resonance_engine is not None,
        }

        for name, available in strategy_availability.items():
            if available and name not in exclude:
                strategies.append(name)

        return strategies
