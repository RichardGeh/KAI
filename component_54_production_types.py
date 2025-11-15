"""
Component 54: Production System - Type Definitions

Enums and basic data structures for the production system.

Author: KAI Development Team
Date: 2025-11-14
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class RuleCategory(Enum):
    """Kategorien von Produktionsregeln."""

    CONTENT_SELECTION = "content_selection"  # Was sagen?
    LEXICALIZATION = "lexicalization"  # Wie formulieren?
    DISCOURSE = "discourse"  # Diskursstruktur
    SYNTAX = "syntax"  # Syntaktische Konstruktion


class GenerationGoalType(Enum):
    """Typen von Generierungszielen."""

    ANSWER_QUESTION = "answer_question"
    EXPLAIN_CONCEPT = "explain_concept"
    DESCRIBE_RELATION = "describe_relation"
    ENUMERATE_FACTS = "enumerate_facts"
    PROVIDE_EVIDENCE = "provide_evidence"
    CLARIFY_AMBIGUITY = "clarify_ambiguity"


@dataclass
class GenerationGoal:
    """Ein Generierungsziel mit Constraints."""

    goal_type: GenerationGoalType
    target_entity: Optional[str] = None
    relation_type: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    completed: bool = False


@dataclass
class DiscourseState:
    """Diskurs-Zustand während der Generierung."""

    current_focus: Optional[str] = None  # Aktuelles Fokus-Konzept
    mentioned_entities: Set[str] = field(default_factory=set)  # Bereits erwähnt
    pending_facts: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Noch zu erwähnen
    discourse_markers_used: List[str] = field(
        default_factory=list
    )  # z.B. "außerdem", "jedoch"
    sentence_count: int = 0
    current_paragraph: int = 0


@dataclass
class PartialTextStructure:
    """Partielle Textstruktur während der Generierung."""

    sentence_fragments: List[str] = field(default_factory=list)  # Satz-Fragmente
    semantic_chunks: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Semantische Einheiten
    completed_sentences: List[str] = field(default_factory=list)  # Fertige Sätze
    current_fragment: str = ""  # Aktuelles Fragment
