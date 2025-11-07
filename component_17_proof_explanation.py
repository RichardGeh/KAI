"""
component_17_proof_explanation.py

Unified Proof Explanation System for KAI

Provides a unified data structure and formatting system for proof explanations
across all reasoning engines (Logic, Graph Traversal, Abductive, Probabilistic).

Functions:
- ProofStep: Unified data structure for all reasoning steps
- ProofTree: Hierarchical proof structure with dependencies
- Explanation generation and formatting
- Conversion utilities for existing ProofStep structures
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class StepType(Enum):
    """Types of reasoning steps"""

    FACT_MATCH = "fact_match"  # Direct fact lookup
    RULE_APPLICATION = "rule_application"  # Logical rule applied
    INFERENCE = "inference"  # General inference
    HYPOTHESIS = "hypothesis"  # Abductive hypothesis generation
    GRAPH_TRAVERSAL = "graph_traversal"  # Multi-hop reasoning via graph
    PROBABILISTIC = "probabilistic"  # Bayesian inference
    DECOMPOSITION = "decomposition"  # Goal decomposition into subgoals
    UNIFICATION = "unification"  # Variable binding/unification
    PREMISE = "premise"  # Initial assumption/premise (constraint reasoning)
    ASSUMPTION = "assumption"  # Trial assumption (constraint search)
    CONCLUSION = "conclusion"  # Final conclusion/solution
    CONTRADICTION = "contradiction"  # Contradiction detected (backtrack)

    # Spatial Reasoning Steps (Phase 9)
    SPATIAL_MODEL_CREATION = (
        "spatial_model_creation"  # Grid, Positions, Relations model created
    )
    SPATIAL_CONSTRAINT_SOLVING = (
        "spatial_constraint_solving"  # CSP solving for spatial constraints
    )
    SPATIAL_PLANNING = "spatial_planning"  # State-space planning (path finding)
    SPATIAL_RELATION_CHECK = (
        "spatial_relation_check"  # Check spatial relation (e.g., "A north of B?")
    )
    SPATIAL_TRANSITIVE_INFERENCE = "spatial_transitive_inference"  # Transitive spatial reasoning (A north of B, B north of C => A north of C)


@dataclass
class ProofStep:
    """
    Unified proof step structure for all reasoning engines.

    Represents a single step in a reasoning process with full traceability.

    Attributes:
        step_id: Unique identifier for this step
        step_type: Type of reasoning performed (StepType enum)
        inputs: List of input facts/premises (as strings or IDs)
        rule_name: Name/ID of rule applied (if applicable)
        output: The conclusion/result of this step
        confidence: Confidence score (0.0 - 1.0)
        explanation_text: Natural language explanation
        parent_steps: List of step_ids this step depends on
        bindings: Variable bindings (e.g., {"?x": "hund", "?y": "tier"})
        metadata: Additional data (component-specific)
        timestamp: When this step was created
        source_component: Which component generated this step
        subgoals: Child ProofSteps for decomposition
    """

    step_id: str
    step_type: StepType
    inputs: List[str] = field(default_factory=list)
    rule_name: Optional[str] = None
    output: str = ""
    confidence: float = 1.0
    explanation_text: str = ""
    parent_steps: List[str] = field(default_factory=list)
    bindings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source_component: str = "unknown"
    subgoals: List["ProofStep"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "inputs": self.inputs,
            "rule_name": self.rule_name,
            "output": self.output,
            "confidence": self.confidence,
            "explanation_text": self.explanation_text,
            "parent_steps": self.parent_steps,
            "bindings": self.bindings,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "source_component": self.source_component,
            "subgoals": [sg.to_dict() for sg in self.subgoals],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofStep":
        """Create ProofStep from dictionary"""
        data_copy = data.copy()
        data_copy["step_type"] = StepType(data_copy["step_type"])
        data_copy["timestamp"] = datetime.fromisoformat(data_copy["timestamp"])
        data_copy["subgoals"] = [
            cls.from_dict(sg) for sg in data_copy.get("subgoals", [])
        ]
        return cls(**data_copy)

    def add_subgoal(self, subgoal: "ProofStep") -> None:
        """Add a subgoal/child step"""
        self.subgoals.append(subgoal)

    def get_all_dependencies(self) -> Set[str]:
        """Get all dependent step IDs recursively"""
        deps = set(self.parent_steps)
        for subgoal in self.subgoals:
            deps.update(subgoal.get_all_dependencies())
        return deps


@dataclass
class ProofTreeNode:
    """
    UI-specific tree node wrapper for ProofStep visualization.

    Wraps a ProofStep with UI-specific metadata like expansion state,
    position, and parent references for interactive tree rendering.

    Attributes:
        step: The actual ProofStep data
        children: Child nodes (converted from step.subgoals)
        parent: Parent node reference (for path tracing)
        expanded: Whether this node is expanded in UI
        position: (x, y) coordinates for rendering
        ui_metadata: Additional UI-specific data
    """

    step: ProofStep
    children: List["ProofTreeNode"] = field(default_factory=list)
    parent: Optional["ProofTreeNode"] = None
    expanded: bool = True  # Default to expanded
    position: tuple = field(default_factory=lambda: (0, 0))
    ui_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "ProofTreeNode") -> None:
        """Add a child node and set parent reference"""
        child.parent = self
        self.children.append(child)

    def get_depth(self) -> int:
        """Get depth of this node in the tree (0 = root)"""
        depth = 0
        node = self.parent
        while node:
            depth += 1
            node = node.parent
        return depth

    def get_path_to_root(self) -> List["ProofTreeNode"]:
        """Get path from this node to root"""
        path = [self]
        node = self.parent
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_all_descendants(self) -> List["ProofTreeNode"]:
        """Get all descendant nodes (recursive)"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def collapse(self) -> None:
        """Collapse this node (hide children)"""
        self.expanded = False

    def expand(self) -> None:
        """Expand this node (show children)"""
        self.expanded = True

    def toggle_expansion(self) -> None:
        """Toggle expansion state"""
        self.expanded = not self.expanded

    @classmethod
    def from_proof_step(
        cls, step: ProofStep, parent: Optional["ProofTreeNode"] = None
    ) -> "ProofTreeNode":
        """
        Convert a ProofStep (with subgoals) into a ProofTreeNode tree.

        Args:
            step: The ProofStep to convert
            parent: Parent node (for recursive construction)

        Returns:
            ProofTreeNode with children converted from step.subgoals
        """
        node = cls(step=step, parent=parent)

        # Recursively convert subgoals to children
        for subgoal in step.subgoals:
            child_node = cls.from_proof_step(subgoal, parent=node)
            node.children.append(child_node)

        return node


@dataclass
class ProofTree:
    """
    Hierarchical proof structure with multiple reasoning chains.

    Represents the complete reasoning process for answering a query.
    """

    query: str
    root_steps: List[ProofStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_root_step(self, step: ProofStep) -> None:
        """Add a top-level proof step"""
        self.root_steps.append(step)

    def get_all_steps(self) -> List[ProofStep]:
        """Get all proof steps (flattened)"""
        all_steps = []

        def collect_steps(step: ProofStep) -> None:
            all_steps.append(step)
            for subgoal in step.subgoals:
                collect_steps(subgoal)

        for root in self.root_steps:
            collect_steps(root)

        return all_steps

    def get_step_by_id(self, step_id: str) -> Optional[ProofStep]:
        """Find a step by its ID"""
        for step in self.get_all_steps():
            if step.step_id == step_id:
                return step
        return None

    def to_tree_nodes(self) -> List[ProofTreeNode]:
        """
        Convert ProofTree to list of ProofTreeNode roots.

        Returns:
            List of root ProofTreeNode objects (one per root_step)
        """
        return [ProofTreeNode.from_proof_step(step) for step in self.root_steps]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "root_steps": [step.to_dict() for step in self.root_steps],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


# ==================== Explanation Generators ====================


def format_proof_step(
    step: ProofStep, indent: int = 0, show_details: bool = True
) -> str:
    """
    Format a single proof step as human-readable text.

    Args:
        step: The ProofStep to format
        indent: Indentation level
        show_details: Whether to show full details

    Returns:
        Formatted string
    """
    prefix = "  " * indent
    lines = []

    # Step header with icon
    icon = _get_step_icon(step.step_type)
    lines.append(f"{prefix}{icon} [{step.step_type.value}]")

    # Explanation
    if step.explanation_text:
        lines.append(f"{prefix}   {step.explanation_text}")

    # Output
    if step.output:
        lines.append(f"{prefix}   -> {step.output}")

    # Confidence
    conf_str = f"{step.confidence:.2f}"
    conf_bar = _confidence_bar(step.confidence)
    lines.append(f"{prefix}   Konfidenz: {conf_str} {conf_bar}")

    # Details (optional)
    if show_details:
        if step.inputs:
            lines.append(
                f"{prefix}   Eingaben: {', '.join(step.inputs[:3])}"
                + (f" ... (+{len(step.inputs)-3})" if len(step.inputs) > 3 else "")
            )

        if step.rule_name:
            lines.append(f"{prefix}   Regel: {step.rule_name}")

        if step.bindings:
            bindings_str = ", ".join(f"{k}={v}" for k, v in step.bindings.items())
            lines.append(f"{prefix}   Bindings: {bindings_str}")

    # Subgoals (recursive)
    if step.subgoals:
        lines.append(f"{prefix}   Unterbeweise:")
        for subgoal in step.subgoals:
            lines.append(format_proof_step(subgoal, indent + 2, show_details))

    return "\n".join(lines)


def format_proof_tree(tree: ProofTree, show_details: bool = True) -> str:
    """
    Format an entire proof tree as human-readable text.

    Args:
        tree: The ProofTree to format
        show_details: Whether to show full details

    Returns:
        Formatted string for UI display
    """
    lines = ["=" * 60, f"Beweisbaum für: {tree.query}", "=" * 60, ""]

    if not tree.root_steps:
        lines.append("Keine Beweisschritte vorhanden.")
        return "\n".join(lines)

    for i, step in enumerate(tree.root_steps, 1):
        lines.append(f"Beweiskette {i}:")
        lines.append(format_proof_step(step, indent=1, show_details=show_details))
        lines.append("")

    # Summary
    all_steps = tree.get_all_steps()
    lines.append(f"Gesamt: {len(all_steps)} Schritte")

    return "\n".join(lines)


def format_proof_chain(steps: List[ProofStep]) -> str:
    """
    Format a linear chain of proof steps (simplified view).

    Args:
        steps: Ordered list of proof steps

    Returns:
        Formatted chain string
    """
    lines = []
    for i, step in enumerate(steps, 1):
        icon = _get_step_icon(step.step_type)
        lines.append(f"Schritt {i}: {icon} {step.explanation_text}")
        if step.output:
            lines.append(f"   -> {step.output}")
    return "\n".join(lines)


def generate_explanation_text(
    step_type: StepType,
    inputs: List[str],
    output: str,
    rule_name: Optional[str] = None,
    bindings: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate natural language explanation for a proof step.

    Args:
        step_type: Type of reasoning step
        inputs: Input facts/premises
        output: Conclusion
        rule_name: Rule applied (if applicable)
        bindings: Variable bindings
        metadata: Additional context

    Returns:
        Natural language explanation
    """
    if step_type == StepType.FACT_MATCH:
        return f"Fand Fakt direkt in der Wissensbasis: '{output}'"

    elif step_type == StepType.RULE_APPLICATION:
        rule_str = f" '{rule_name}'" if rule_name else ""
        input_str = f" mit {len(inputs)} Prämissen" if inputs else ""
        return f"Wendete Regel{rule_str}{input_str} an -> '{output}'"

    elif step_type == StepType.GRAPH_TRAVERSAL:
        hops = metadata.get("hops", "?") if metadata else "?"
        path = metadata.get("path", []) if metadata else []
        if path:
            path_str = " -> ".join(path)
            return f"Fand Pfad über {hops} Schritte: {path_str}"
        return f"Fand Verbindung über {hops} Schritte im Graphen"

    elif step_type == StepType.HYPOTHESIS:
        strategy = metadata.get("strategy", "unbekannt") if metadata else "unbekannt"
        score = metadata.get("score", 0.0) if metadata else 0.0
        return f"Generierte Hypothese (Strategie: {strategy}, Score: {score:.2f}): '{output}'"

    elif step_type == StepType.PROBABILISTIC:
        prob = metadata.get("probability", 0.0) if metadata else 0.0
        return f"Bayesianische Inferenz ergab Wahrscheinlichkeit {prob:.2f}: '{output}'"

    elif step_type == StepType.INFERENCE:
        return f"Leitete ab: '{output}'"

    elif step_type == StepType.DECOMPOSITION:
        num_subgoals = len(metadata.get("subgoals", [])) if metadata else 0
        return f"Zerlegte Ziel in {num_subgoals} Unterziele"

    else:  # step_type == StepType.UNIFICATION
        if bindings:
            bindings_str = ", ".join(f"{k}={v}" for k, v in bindings.items())
            return f"Unifizierte Variablen: {bindings_str}"
        return "Unifizierte Variablen"


# ==================== Conversion Utilities ====================


def convert_logic_engine_proof(logic_proof: Any) -> ProofStep:
    """
    Convert component_9 ProofStep to unified ProofStep.

    Args:
        logic_proof: ProofStep from component_9_logik_engine

    Returns:
        Unified ProofStep
    """
    # Map method to StepType
    method_mapping = {
        "fact": StepType.FACT_MATCH,
        "rule": StepType.RULE_APPLICATION,
        "graph_traversal": StepType.GRAPH_TRAVERSAL,
        "probabilistic": StepType.PROBABILISTIC,
        "decomposition": StepType.DECOMPOSITION,
    }

    step_type = method_mapping.get(logic_proof.method, StepType.INFERENCE)

    # Extract inputs
    inputs = [f"{f.pred}({f.args})" for f in logic_proof.supporting_facts]

    # Generate output
    goal_args_str = ", ".join(f"{k}={v}" for k, v in logic_proof.goal.args.items())
    output = f"{logic_proof.goal.pred}({goal_args_str})"

    # Generate explanation
    explanation = generate_explanation_text(
        step_type=step_type,
        inputs=inputs,
        output=output,
        rule_name=logic_proof.rule_id,
        bindings=logic_proof.bindings,
        metadata={"method": logic_proof.method, "goal_depth": logic_proof.goal.depth},
    )

    # Create unified ProofStep
    unified_step = ProofStep(
        step_id=logic_proof.goal.id,
        step_type=step_type,
        inputs=inputs,
        rule_name=logic_proof.rule_id,
        output=output,
        confidence=logic_proof.confidence,
        explanation_text=explanation,
        parent_steps=[],  # Populated from subgoals
        bindings=logic_proof.bindings,
        metadata={
            "original_method": logic_proof.method,
            "goal_depth": logic_proof.goal.depth,
        },
        source_component="component_9_logik_engine",
    )

    # Convert subgoals recursively
    for subproof in logic_proof.subgoals:
        subgoal_step = convert_logic_engine_proof(subproof)
        unified_step.add_subgoal(subgoal_step)
        unified_step.parent_steps.append(subgoal_step.step_id)

    return unified_step


def convert_reasoning_state(reasoning_state: Any) -> ProofStep:
    """
    Convert component_13 ReasoningState to unified ProofStep.

    Args:
        reasoning_state: ReasoningState from component_13_working_memory

    Returns:
        Unified ProofStep
    """
    # Map step_type string to StepType enum
    type_mapping = {
        "fact_retrieval": StepType.FACT_MATCH,
        "pattern_match": StepType.FACT_MATCH,
        "inference": StepType.INFERENCE,
        "rule_application": StepType.RULE_APPLICATION,
        "graph_query": StepType.GRAPH_TRAVERSAL,
        "hypothesis": StepType.HYPOTHESIS,
    }

    step_type = type_mapping.get(reasoning_state.step_type, StepType.INFERENCE)

    # Extract data
    inputs = reasoning_state.data.get("inputs", [])
    output = reasoning_state.data.get("output", "")
    rule_name = reasoning_state.data.get("rule_name")
    bindings = reasoning_state.data.get("bindings", {})

    return ProofStep(
        step_id=reasoning_state.step_id,
        step_type=step_type,
        inputs=inputs if isinstance(inputs, list) else [str(inputs)],
        rule_name=rule_name,
        output=output,
        confidence=reasoning_state.confidence,
        explanation_text=reasoning_state.description,
        parent_steps=[],
        bindings=bindings,
        metadata=reasoning_state.data,
        timestamp=reasoning_state.timestamp,
        source_component="component_13_working_memory",
    )


# ==================== Helper Functions ====================


def _get_step_icon(step_type: StepType) -> str:
    """Get icon for step type"""
    icons = {
        StepType.FACT_MATCH: "[INFO]",
        StepType.RULE_APPLICATION: "⚙️",
        StepType.INFERENCE: "💡",
        StepType.HYPOTHESIS: "🔬",
        StepType.GRAPH_TRAVERSAL: "🗺️",
        StepType.PROBABILISTIC: "🎲",
        StepType.DECOMPOSITION: "🔀",
        StepType.UNIFICATION: "🔗",
    }
    return icons.get(step_type, "*")


def _confidence_bar(confidence: float, width: int = 10) -> str:
    """Generate visual confidence bar"""
    filled = int(confidence * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


# ==================== Hybrid Reasoning Integration ====================


def create_hybrid_proof_step(
    results: List[Any], query: str, aggregation_method: str = "noisy_or"
) -> ProofStep:
    """
    Create a unified ProofStep from multiple reasoning results.

    Combines evidence from different reasoning engines (Logic, Graph, Probabilistic, Abductive)
    into a single proof step with subgoals representing each strategy.

    Args:
        results: List of ReasoningResult objects from different strategies
        query: The original query
        aggregation_method: How to combine confidences ("noisy_or", "weighted_avg", "min")

    Returns:
        Unified ProofStep with all strategies as subgoals
    """
    import uuid

    # Calculate combined confidence
    if aggregation_method == "noisy_or":
        confidences = [getattr(r, "confidence", 0.0) for r in results]
        combined_conf = _noisy_or_combination(confidences)
    elif aggregation_method == "weighted_avg":
        # Weight by strategy reliability
        weights = {
            "logic_engine": 0.35,
            "graph_traversal": 0.30,
            "probabilistic": 0.25,
            "abductive": 0.10,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for r in results:
            strategy_name = getattr(r, "strategy", None)
            if strategy_name:
                weight = weights.get(str(strategy_name).split(".")[-1].lower(), 0.2)
                weighted_sum += getattr(r, "confidence", 0.0) * weight
                total_weight += weight
        combined_conf = weighted_sum / total_weight if total_weight > 0 else 0.0
    else:  # min
        combined_conf = min(getattr(r, "confidence", 0.0) for r in results)

    # Collect all inferred facts
    all_facts = []
    for r in results:
        facts = getattr(r, "inferred_facts", {})
        for rel_type, objects in facts.items():
            for obj in objects:
                fact_str = f"{rel_type}: {obj}"
                if fact_str not in all_facts:
                    all_facts.append(fact_str)

    # Create main hybrid proof step
    step_id = f"hybrid_{uuid.uuid4().hex[:8]}"

    explanation = (
        f"Hybrid Reasoning kombinierte {len(results)} Strategien: "
        f"{', '.join(str(getattr(r, 'strategy', 'unknown')).split('.')[-1] for r in results)}. "
        f"Kombinierte Konfidenz: {combined_conf:.2f}"
    )

    main_step = ProofStep(
        step_id=step_id,
        step_type=StepType.INFERENCE,
        inputs=[query],
        output=f"Gefunden: {len(all_facts)} Fakten",
        confidence=combined_conf,
        explanation_text=explanation,
        metadata={
            "hybrid": True,
            "num_strategies": len(results),
            "aggregation_method": aggregation_method,
            "all_facts": all_facts,
        },
        source_component="hybrid_reasoning_orchestrator",
    )

    # Add each strategy result as subgoal
    for i, result in enumerate(results):
        strategy_name = str(getattr(result, "strategy", f"strategy_{i}")).split(".")[-1]

        # Get proof tree from result if available
        proof_tree = getattr(result, "proof_tree", None)

        if proof_tree and hasattr(proof_tree, "root_steps"):
            # Add all root steps from this strategy as subgoals
            for root_step in proof_tree.root_steps:
                main_step.add_subgoal(root_step)
        else:
            # Create a simple subgoal step
            subgoal_step = ProofStep(
                step_id=f"{step_id}_sub_{i}",
                step_type=StepType.INFERENCE,
                inputs=[query],
                output=str(getattr(result, "proof_trace", "No trace available")),
                confidence=getattr(result, "confidence", 0.0),
                explanation_text=f"Strategie: {strategy_name}",
                metadata={"strategy": strategy_name, "original_result": True},
                source_component=f"strategy_{strategy_name}",
            )
            main_step.add_subgoal(subgoal_step)

    return main_step


def _noisy_or_combination(probabilities: List[float]) -> float:
    """
    Noisy-OR combination for redundant evidence.

    P(E | C1, C2, ..., Cn) = 1 - ∏(1 - P(E | Ci))
    """
    if not probabilities:
        return 0.0

    product = 1.0
    for p in probabilities:
        product *= 1.0 - p

    return 1.0 - product


def create_aggregated_proof_tree(
    individual_trees: List[ProofTree], query: str, aggregation_method: str = "noisy_or"
) -> ProofTree:
    """
    Create an aggregated ProofTree from multiple reasoning strategy trees.

    Creates a meta-level proof tree that shows how different strategies
    contributed to the final answer.

    Args:
        individual_trees: List of ProofTree objects from different strategies
        query: The original query
        aggregation_method: How to aggregate ("noisy_or", "weighted_avg", "hierarchical")

    Returns:
        Aggregated ProofTree with strategy-level organization
    """
    aggregated_tree = ProofTree(query=query)

    if aggregation_method == "hierarchical":
        # Create a hierarchical structure with meta-step
        meta_step = ProofStep(
            step_id=f"aggregation_{uuid.uuid4().hex[:8]}",
            step_type=StepType.DECOMPOSITION,
            inputs=[query],
            output="Kombinierte Ergebnisse aus mehreren Strategien",
            confidence=1.0,
            explanation_text=f"Zerlegte Anfrage in {len(individual_trees)} Reasoning-Strategien",
            metadata={
                "num_trees": len(individual_trees),
                "aggregation": "hierarchical",
            },
            source_component="proof_aggregator",
        )

        # Add each tree's root steps as subgoals
        for tree in individual_trees:
            for root_step in tree.root_steps:
                meta_step.add_subgoal(root_step)

        aggregated_tree.add_root_step(meta_step)

    else:
        # Flat aggregation - merge all root steps
        for tree in individual_trees:
            for root_step in tree.root_steps:
                aggregated_tree.add_root_step(root_step)

    # Add metadata
    aggregated_tree.metadata["aggregated"] = True
    aggregated_tree.metadata["num_source_trees"] = len(individual_trees)
    aggregated_tree.metadata["aggregation_method"] = aggregation_method

    return aggregated_tree


# ==================== Integration Functions ====================


def create_proof_tree_from_logic_engine(logic_proof: Any, query: str) -> ProofTree:
    """
    Create ProofTree from Logic Engine proof.

    Args:
        logic_proof: ProofStep from component_9
        query: The original query

    Returns:
        Complete ProofTree
    """
    tree = ProofTree(query=query)

    if logic_proof:
        unified_step = convert_logic_engine_proof(logic_proof)
        tree.add_root_step(unified_step)

    return tree


def create_proof_tree_from_working_memory(memory: Any, query: str) -> ProofTree:
    """
    Create ProofTree from WorkingMemory ReasoningStates.

    Args:
        memory: WorkingMemory instance from component_13
        query: The original query

    Returns:
        Complete ProofTree
    """
    tree = ProofTree(query=query)

    reasoning_states = memory.get_full_reasoning_trace()

    for state in reasoning_states:
        unified_step = convert_reasoning_state(state)
        tree.add_root_step(unified_step)

    return tree


def merge_proof_trees(trees: List[ProofTree], query: str) -> ProofTree:
    """
    Merge multiple proof trees into one.

    Useful when combining results from multiple reasoning engines.

    Args:
        trees: List of ProofTree instances
        query: The query (should be same for all)

    Returns:
        Merged ProofTree
    """
    merged = ProofTree(query=query)

    for tree in trees:
        for root_step in tree.root_steps:
            merged.add_root_step(root_step)

    return merged


# ==================== Export Functions ====================


def export_proof_to_json(tree: ProofTree, filepath: str) -> None:
    """Export proof tree to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)


def import_proof_from_json(filepath: str) -> ProofTree:
    """Import proof tree from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    tree = ProofTree(
        query=data["query"],
        metadata=data.get("metadata", {}),
        created_at=datetime.fromisoformat(data["created_at"]),
    )

    for root_data in data["root_steps"]:
        tree.add_root_step(ProofStep.from_dict(root_data))

    return tree
