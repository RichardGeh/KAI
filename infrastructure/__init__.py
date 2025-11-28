"""
infrastructure package

Shared infrastructure components for the KAI system.
Provides base interfaces, session management, cache management, and common utilities.

Modules:
    - interfaces: Base interfaces for reasoning engines
    - neo4j_session_mixin: Shared Neo4j session management
    - cache_manager: Centralized cache management system
"""

from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult
from infrastructure.neo4j_session_mixin import Neo4jSessionMixin
from infrastructure.cache_manager import CacheManager, get_cache_manager

__all__ = [
    "BaseReasoningEngine",
    "ReasoningResult",
    "Neo4jSessionMixin",
    "CacheManager",
    "get_cache_manager",
]
