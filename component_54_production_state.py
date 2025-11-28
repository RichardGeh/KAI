"""
Component 54: Production System - State Management

ResponseGenerationState class - Working Memory for response generation.

Author: KAI Development Team
Date: 2025-11-14
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# ProofTree Integration (PHASE 6)
from component_17_proof_explanation import ProofTree
from component_54_production_types import (
    DiscourseState,
    GenerationGoal,
    PartialTextStructure,
)


@dataclass
class ResponseGenerationState:
    """
    Working Memory State für Response Generation.

    Enthält den vollständigen Zustand während der schrittweisen
    Antwortgenerierung durch das Produktionssystem.
    """

    # Generierungsziele
    primary_goal: GenerationGoal
    sub_goals: List[GenerationGoal] = field(default_factory=list)

    # Diskurs-Zustand
    discourse: DiscourseState = field(default_factory=DiscourseState)

    # Text-Struktur
    text: PartialTextStructure = field(default_factory=PartialTextStructure)

    # Verfügbare Fakten aus Knowledge Graph
    available_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Constraints
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Metadaten
    cycle_count: int = 0
    max_cycles: int = 50
    generation_start: datetime = field(default_factory=datetime.now)

    # PHASE 6: ProofTree Integration
    proof_tree: Optional[ProofTree] = None
    current_query: str = ""  # Original query für ProofTree

    def is_goal_completed(self) -> bool:
        """Prüft, ob das Hauptziel erreicht wurde."""
        return self.primary_goal.completed and all(g.completed for g in self.sub_goals)

    def add_sentence(self, sentence: str) -> None:
        """Fügt einen fertigen Satz hinzu."""
        self.text.completed_sentences.append(sentence)
        self.discourse.sentence_count += 1

    def mention_entity(self, entity: str) -> None:
        """Markiert eine Entität als erwähnt."""
        self.discourse.mentioned_entities.add(entity)

    def get_full_text(self) -> str:
        """Gibt den vollständigen generierten Text zurück."""
        return " ".join(self.text.completed_sentences)

    def get_facts_by_relation(
        self, relation_type: str, source: str = "available"
    ) -> List[Dict[str, Any]]:
        """
        Get facts filtered by relation type.

        Args:
            relation_type: Type of relation to filter by (IS_A, HAS_PROPERTY, etc.)
            source: Source list - "available" or "pending"

        Returns:
            List of matching facts
        """
        facts_list = (
            self.available_facts
            if source == "available"
            else self.discourse.pending_facts
        )
        return [f for f in facts_list if f.get("relation_type") == relation_type]

    def is_phase_complete(self, phase: str) -> bool:
        """
        Check if a processing phase is complete.

        Args:
            phase: Phase name (e.g., "content_selection", "lexicalization")

        Returns:
            True if phase is marked as finished
        """
        return self.constraints.get(f"{phase}_finished", False)

    def to_serializable_snapshot(self) -> Dict[str, Any]:
        """
        Erstellt einen serialisierbaren Snapshot des aktuellen State.

        PHASE 6: Für ProofStep Inputs/Outputs
        Enthält nur serialisierbare Daten (keine Callables, keine komplexen Objekte).

        Returns:
            Dictionary mit State-Snapshot
        """
        return {
            "cycle": self.cycle_count,
            "goal_type": self.primary_goal.goal_type.value,
            "goal_completed": self.primary_goal.completed,
            "num_sub_goals": len(self.sub_goals),
            "num_available_facts": len(self.available_facts),
            "num_pending_facts": len(self.discourse.pending_facts),
            "num_sentences": len(self.text.completed_sentences),
            "num_fragments": len(self.text.sentence_fragments),
            "mentioned_entities": list(self.discourse.mentioned_entities),
            "current_focus": self.discourse.current_focus,
            "text_preview": (
                self.get_full_text()[:100] + "..."
                if len(self.get_full_text()) > 100
                else self.get_full_text()
            ),
        }
