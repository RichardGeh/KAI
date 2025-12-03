"""
component_46_meta_learning.py

Meta-Learning Layer: Strategy Performance Tracking & Selection

FACADE: Delegates to modular components for backward compatibility.

Split into 4 modules (Phase 4 Architectural Refactoring):
- component_46_performance_tracker.py: Performance tracking, pattern matching
- component_46_ab_testing_manager.py: A/B testing for generation systems
- component_46_meta_learning_engine.py: Core engine, strategy selection, Neo4j persistence
- component_46_meta_learning.py: This facade (maintains backward compatibility)

Author: KAI Development Team
Last Updated: 2025-11-28 (Modular Refactoring)
"""

# Re-export all classes and functions for backward compatibility
from component_46_ab_testing_manager import ABTestingManager
from component_46_meta_learning_engine import (
    MetaLearningConfig,
    MetaLearningEngine,
    MetaLearningException,
    PersistenceException,
    StrategySelectionException,
)
from component_46_performance_tracker import (
    STRATEGY_NAME_PATTERN,
    PerformanceTracker,
    QueryPattern,
    StrategyPerformance,
    StrategyUsageEpisode,
)

__all__ = [
    # Data Structures
    "StrategyPerformance",
    "QueryPattern",
    "StrategyUsageEpisode",
    # Configuration
    "MetaLearningConfig",
    # Main Engine
    "MetaLearningEngine",
    # Sub-Components
    "PerformanceTracker",
    "ABTestingManager",
    # Exceptions
    "MetaLearningException",
    "StrategySelectionException",
    "PersistenceException",
    # Constants
    "STRATEGY_NAME_PATTERN",
]
