# kai_proof_merger.py
"""
Proof Tree Merger for KAI Reasoning Orchestrator

Merges proof trees from different reasoning engines into unified proof trees.
Handles proof simplification, validation, and metadata consolidation.

Responsibilities:
- Merge proof trees from multiple sources
- Unified proof tree generation
- Proof simplification (remove redundant steps)
- Proof validation (check consistency)
- Metadata consolidation

Architecture:
    ProofMerger uses component_17's merge_proof_trees() as the core merging
    function, adding additional validation and simplification layers.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ProofMerger:
    """
    Merges proof trees from different reasoning engines.

    Delegates to component_17's merge_proof_trees() for core merging,
    with additional validation and simplification layers.
    """

    def __init__(self, signals):
        """
        Initialize Proof Merger.

        Args:
            signals: KaiSignals for UI updates
        """
        self.signals = signals

        # Import unified proof system
        try:
            from component_17_proof_explanation import (
                ProofTree,
                merge_proof_trees,
            )

            self.ProofTree = ProofTree
            self.merge_proof_trees = merge_proof_trees
            self.PROOF_SYSTEM_AVAILABLE = True
        except ImportError:
            self.PROOF_SYSTEM_AVAILABLE = False

        logger.info(
            f"ProofMerger initialized: proof_system={self.PROOF_SYSTEM_AVAILABLE}"
        )

    def merge(
        self,
        proof_trees: List,
        query: str,
        emit_signal: bool = True,
    ) -> Optional:
        """
        Merge multiple proof trees into one unified tree.

        Args:
            proof_trees: List of ProofTree objects to merge
            query: Query text for merged tree
            emit_signal: Whether to emit proof_tree_update signal

        Returns:
            Merged ProofTree or None if merging fails
        """
        if not self.PROOF_SYSTEM_AVAILABLE:
            logger.debug("[Proof Merger] Proof system not available")
            return None

        if not proof_trees:
            logger.debug("[Proof Merger] No proof trees to merge")
            return None

        # Filter out None values
        valid_trees = [tree for tree in proof_trees if tree is not None]

        if not valid_trees:
            logger.debug("[Proof Merger] No valid proof trees after filtering")
            return None

        if len(valid_trees) == 1:
            # Only one tree, no merging needed
            logger.debug("[Proof Merger] Only one proof tree, returning as-is")
            if emit_signal and self.signals:
                self.signals.proof_tree_update.emit(valid_trees[0])
            return valid_trees[0]

        try:
            logger.debug(f"[Proof Merger] Merging {len(valid_trees)} proof trees")

            # Use component_17's merge function
            merged = self.merge_proof_trees(valid_trees, query)

            if not merged:
                logger.warning("[Proof Merger] Merge returned None, using first tree")
                merged = valid_trees[0]

            # Emit signal if requested
            if emit_signal and self.signals:
                self.signals.proof_tree_update.emit(merged)

            logger.info(f"[Proof Merger] Successfully merged {len(valid_trees)} trees")

            return merged

        except Exception as e:
            logger.error(f"[Proof Merger] Merge failed: {e}", exc_info=True)
            # Fallback to first proof tree
            logger.warning("[Proof Merger] Falling back to first proof tree")
            fallback = valid_trees[0]
            if emit_signal and self.signals:
                self.signals.proof_tree_update.emit(fallback)
            return fallback

    def merge_from_results(
        self,
        results: List,
        query: str,
        emit_signal: bool = True,
    ) -> Optional:
        """
        Extract and merge proof trees from ReasoningResult objects.

        Args:
            results: List of ReasoningResult objects
            query: Query text for merged tree
            emit_signal: Whether to emit proof_tree_update signal

        Returns:
            Merged ProofTree or None
        """
        # Extract proof trees from results
        proof_trees = [
            r.proof_tree for r in results if hasattr(r, "proof_tree") and r.proof_tree
        ]

        return self.merge(proof_trees, query, emit_signal)

    def simplify_proof(self, proof_tree) -> Optional:
        """
        Simplify proof tree by removing redundant steps.

        This is a placeholder for future proof simplification logic.
        Currently returns the proof tree unchanged.

        Args:
            proof_tree: ProofTree to simplify

        Returns:
            Simplified ProofTree
        """
        if not self.PROOF_SYSTEM_AVAILABLE or not proof_tree:
            return proof_tree

        # TODO: Implement proof simplification
        # - Remove duplicate steps
        # - Merge equivalent paths
        # - Remove trivial steps (identity transformations)

        logger.debug("[Proof Merger] Simplification not yet implemented")
        return proof_tree

    def validate_proof(self, proof_tree) -> bool:
        """
        Validate proof tree for consistency.

        This is a placeholder for future proof validation logic.
        Currently always returns True.

        Args:
            proof_tree: ProofTree to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.PROOF_SYSTEM_AVAILABLE or not proof_tree:
            return True

        # TODO: Implement proof validation
        # - Check step dependencies are met
        # - Verify confidence scores are valid (0.0-1.0)
        # - Check for circular dependencies
        # - Validate step type transitions

        logger.debug("[Proof Merger] Validation not yet implemented")
        return True

    def consolidate_metadata(self, proof_tree) -> None:
        """
        Consolidate metadata across proof tree steps.

        This is a placeholder for future metadata consolidation.
        Currently does nothing.

        Args:
            proof_tree: ProofTree to process
        """
        if not self.PROOF_SYSTEM_AVAILABLE or not proof_tree:
            return

        # TODO: Implement metadata consolidation
        # - Extract common metadata to tree level
        # - Remove duplicate metadata from steps
        # - Normalize metadata keys/values

        logger.debug("[Proof Merger] Metadata consolidation not yet implemented")
