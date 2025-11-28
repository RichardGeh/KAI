"""
Centralized constants for the KAI (Konzeptueller AI Prototyp) project.

This module provides a single source of truth for all magic numbers, thresholds,
and configuration values used throughout the KAI codebase. Centralizing these
values improves maintainability, makes tuning easier, and provides clear
documentation for architectural decisions.

Organization:
    - Embedding/Similarity Thresholds: Vector similarity and pattern matching
    - File Size Limits: Code organization and modularity constraints
    - Cache Configuration: TTL and size limits for various cache types
    - Confidence Thresholds: Autonomous learning decision boundaries
    - Adaptive Pattern Recognition: Thresholds for typo/sequence detection
    - Working Memory: Context management limits
    - Production System: Response generation configuration

Usage:
    from common.constants import EMBEDDING_DISTANCE_THRESHOLD, CONFIDENCE_AUTO_SAVE

Note:
    These constants define default values. Some components may override these
    via configuration objects or constructor parameters. See individual component
    documentation for override mechanisms.

Last Updated: 2025-11-27
"""

# =============================================================================
# Embedding/Similarity Thresholds
# =============================================================================

EMBEDDING_DISTANCE_THRESHOLD: float = 15.0
"""
Euclidean distance threshold for prototype matching in component_8.

- Distance < 15.0: Update existing prototype (similar enough)
- Distance >= 15.0: Create new prototype (too different)

Rationale:
    Empirically determined through testing with 384-dimensional embeddings
    from sentence-transformers. This value balances generalization (avoiding
    too many near-duplicate prototypes) with specificity (preserving meaningful
    distinctions between patterns).

Used by:
    - component_8_prototype_matcher.py: Pattern clustering and recognition
    - component_11_embedding_service.py: Similarity calculations

Tuning:
    - Lower values (e.g., 10.0): More prototypes, finer distinctions
    - Higher values (e.g., 20.0): Fewer prototypes, more generalization
"""

SIMILARITY_THRESHOLD: float = 0.85
"""
Minimum confidence threshold for automatic knowledge storage.

- Confidence >= 0.85: Auto-save to knowledge graph (high confidence)
- Confidence 0.70-0.84: Request user confirmation (medium confidence)
- Confidence < 0.70: Clarify intent before saving (low confidence)

Rationale:
    Set high enough to avoid learning incorrect information autonomously,
    while low enough to enable bootstrapping from well-formed declarative
    statements. Validated through testing with definition detection patterns.

Used by:
    - component_4_goal_planner.py: Decision gates for autonomous learning
    - component_7_meaning_extractor.py: Definition confidence scoring

Related:
    See CONFIDENCE_* constants below for complete decision boundary set.
"""

# =============================================================================
# File Size Limits
# =============================================================================

MAX_FILE_LINES: int = 800
"""
Maximum recommended lines per source file for maintainability.

Files exceeding this threshold should be split into multiple modules with
clear separation of concerns. This is a guideline, not a hard limit.

Rationale:
    - Improves code navigability and comprehension
    - Encourages modular design with clean interfaces
    - Reduces cognitive load when reviewing or debugging
    - Facilitates parallel development without merge conflicts

Exceptions:
    - Large test files with many similar test cases
    - Auto-generated code (e.g., rule factories)
    - UI files with extensive layout definitions

Used by:
    - Code review processes
    - Refactoring decisions
    - Architecture planning (see .claude/doc/*-plan.md files)
"""

# =============================================================================
# Cache Configuration
# =============================================================================
# TTL (Time-To-Live) values in seconds for various cache types.
# Longer TTLs reduce database load but may serve stale data.
# Size limits prevent unbounded memory growth.

CACHE_TTL_FACTS: int = 300
"""
TTL for fact query cache (5 minutes).

Caches results from query_graph_for_facts() and similar operations.

Rationale:
    Facts change relatively frequently as KAI learns new information.
    5-minute TTL balances performance (86x speedup observed) with freshness.
    Most conversational contexts don't require immediate cache invalidation.

Used by:
    - component_1_netzwerk_core.py: Graph query caching
"""

CACHE_TTL_WORDS: int = 600
"""
TTL for word/concept cache (10 minutes).

Caches Wort node lookups and concept relationship traversals.

Rationale:
    Word nodes are relatively stable (only change when learning new vocabulary).
    Longer TTL acceptable since word properties rarely change mid-conversation.

Used by:
    - component_1_netzwerk_core.py: Word node caching
    - component_1_netzwerk_word_usage.py: N-gram lookups
"""

CACHE_TTL_EMBEDDINGS: int = 600
"""
TTL for embedding cache (10 minutes).

Caches vector embeddings for sentences/phrases to avoid recomputation.

Rationale:
    Embedding computation is expensive (sentence-transformers model inference).
    Observed 5600x speedup for cached embeddings. 10-minute TTL handles
    typical conversation durations without unbounded memory growth.

Used by:
    - component_11_embedding_service.py: Sentence embedding caching (1000 entries)
"""

CACHE_TTL_PROTOTYPES: int = 1800
"""
TTL for pattern prototype cache (30 minutes).

Caches loaded pattern prototypes and their metadata.

Rationale:
    Prototypes change infrequently (only when new patterns learned).
    Observed 52x speedup. Longer TTL acceptable since prototype updates
    are rare during normal operation. 30-minute window handles extended
    interactive sessions.

Used by:
    - component_8_prototype_matcher.py: Prototype matching cache
"""

CACHE_TTL_RESONANCE: int = 600
"""
TTL for resonance engine cache (10 minutes).

Caches spreading activation paths and resonance scores.

Rationale:
    Resonance patterns change as knowledge graph evolves, but computation
    is expensive (graph traversal with activation spreading). 10-minute TTL
    balances performance with graph evolution.

Used by:
    - component_44_resonance_engine.py: Activation spreading cache
"""

# Cache size limits (number of entries)

CACHE_MAXSIZE_SMALL: int = 50
"""
Small cache size for rarely-accessed data.

Use cases:
    - Infrequently used lookups
    - Low-volume operations
    - Development/testing caches
"""

CACHE_MAXSIZE_MEDIUM: int = 100
"""
Medium cache size for moderate-frequency operations.

Use cases:
    - Regular query patterns
    - Component-level caches
    - Session-scoped data

Used by:
    - component_44_resonance_engine.py: Resonance cache
    - component_1_netzwerk_production_rules.py: Rule repository cache
"""

CACHE_MAXSIZE_LARGE: int = 500
"""
Large cache size for high-frequency operations.

Use cases:
    - Hot-path queries
    - Frequently accessed facts
    - Common word lookups
"""

CACHE_MAXSIZE_XLARGE: int = 1000
"""
Extra-large cache size for critical performance paths.

Use cases:
    - Embedding cache (expensive computation, high reuse)
    - Core graph queries
    - Production-critical operations

Used by:
    - component_11_embedding_service.py: Sentence embeddings (observed 5600x speedup)
"""

# =============================================================================
# Confidence Thresholds (Autonomous Learning Decision Gates)
# =============================================================================

CONFIDENCE_AUTO_SAVE: float = 0.85
"""
Minimum confidence for autonomous knowledge storage (no user confirmation).

Statements with confidence >= 0.85 are automatically saved to the knowledge
graph without user interaction. This threshold represents "high confidence"
in definition/fact detection.

Rationale:
    Empirically validated through testing with well-formed German declarative
    statements (e.g., "Ein Apfel ist eine Frucht"). Pattern matching achieves
    0.92+ confidence for canonical IS_A patterns, 0.91+ for CAPABLE_OF, etc.
    Setting threshold at 0.85 allows slight pattern variations while avoiding
    false positives.

Used by:
    - component_4_goal_planner.py: decide_next_goal() decision gates
    - component_7_meaning_extractor.py: Definition detection confidence scoring

Pattern Examples (typical confidences):
    - "X ist ein/eine Y" -> 0.92 (IS_A, auto-save)
    - "X kann Y" -> 0.91 (CAPABLE_OF, auto-save)
    - "X ist Y" (adjective) -> 0.78 (HAS_PROPERTY, confirm)
"""

CONFIDENCE_CONFIRM_MIN: float = 0.70
"""
Lower bound of confirmation range (require user approval before saving).

Statements with confidence in [0.70, 0.84] are presented to the user for
confirmation before storage. This represents "medium confidence" - likely
correct but not certain enough for autonomous action.

Rationale:
    Bridges gap between autonomous learning (0.85+) and clarification requests
    (<0.70). Allows KAI to learn from less canonical patterns (e.g., implicit
    definitions, ambiguous phrasing) with user validation.

Used by:
    - component_4_goal_planner.py: Confirmation request logic
    - kai_sub_goal_executor.py: User interaction strategies
"""

CONFIDENCE_CONFIRM_MAX: float = 0.84
"""
Upper bound of confirmation range (require user approval before saving).

See CONFIDENCE_CONFIRM_MIN for rationale. Range is [0.70, 0.84].
"""

CONFIDENCE_CLARIFY_MAX: float = 0.40
"""
Maximum confidence for clarification requests (intent unclear).

Statements with confidence < 0.40 trigger clarification dialogues rather than
learning attempts. This represents "low confidence" - intent is too ambiguous
to proceed without user guidance.

Rationale:
    Very low confidence indicates KAI cannot reliably determine intent (question
    vs. statement, fact vs. opinion, etc.). Rather than guess or fail silently,
    request clarification to improve future understanding.

Used by:
    - component_4_goal_planner.py: Clarification decision logic
    - kai_response_formatter.py: Clarification message generation
"""

# =============================================================================
# Adaptive Pattern Recognition Thresholds
# =============================================================================

TYPO_DISTANCE_MIN: int = 3
"""
Minimum Levenshtein distance threshold for typo detection (cold phase).

During initial learning (cold phase), stricter threshold reduces false positives.
Only strings differing by 3+ characters are flagged as potential typos.

Rationale:
    Conservative initial threshold avoids learning incorrect corrections during
    bootstrapping. As system matures, threshold adapts upward (see TYPO_DISTANCE_MAX).

Used by:
    - component_19_pattern_recognition_char.py: Character-level pattern detection
    - component_25_adaptive_thresholds.py: Phase-based threshold adaptation

Adaptation:
    - Cold phase (< 10 samples): threshold = 3
    - Warming phase (10-100 samples): threshold = 5-7
    - Mature phase (100+ samples): threshold = 10
"""

TYPO_DISTANCE_MAX: int = 10
"""
Maximum Levenshtein distance threshold for typo detection (mature phase).

After sufficient training (100+ samples), threshold increases to capture more
distant typos and variations while maintaining acceptable precision.

Rationale:
    Mature system has learned typical typo patterns and can confidently identify
    larger-distance variations without excessive false positives.

Used by:
    - component_19_pattern_recognition_char.py: Character-level pattern detection
    - component_25_adaptive_thresholds.py: Phase-based threshold adaptation
"""

SEQUENCE_THRESHOLD_MIN: int = 2
"""
Minimum frequency threshold for sequence pattern detection (cold phase).

Sequences (bigrams/trigrams) must appear 2+ times before being considered
patterns during initial learning.

Rationale:
    Conservative threshold during bootstrapping avoids memorizing noise.
    Single occurrences likely random, not true patterns.

Used by:
    - component_20_pattern_recognition_sequence.py: N-gram pattern learning
    - component_25_adaptive_thresholds.py: Phase-based threshold adaptation

Adaptation:
    - Cold phase (< 50 sequences): threshold = 2
    - Warming phase (50-200 sequences): threshold = 3-4
    - Mature phase (200+ sequences): threshold = 5
"""

SEQUENCE_THRESHOLD_MAX: int = 5
"""
Maximum frequency threshold for sequence pattern detection (mature phase).

After sufficient training (200+ sequences), threshold increases to filter
out low-frequency noise and focus on robust patterns.

Rationale:
    Mature system should recognize only strong, frequently-occurring patterns
    to avoid overfitting to conversational artifacts.

Used by:
    - component_20_pattern_recognition_sequence.py: N-gram pattern learning
    - component_25_adaptive_thresholds.py: Phase-based threshold adaptation
"""

# =============================================================================
# Working Memory Configuration
# =============================================================================

MAX_WORKING_MEMORY_DEPTH: int = 10
"""
Maximum stack depth for working memory context tracking.

Working memory maintains a stack of conversation contexts (topics, entities,
goals). This limit prevents unbounded growth during extended conversations.

Rationale:
    Human working memory capacity is ~7±2 items (Miller's Law). Setting limit
    at 10 provides buffer while preventing memory leaks. Oldest contexts are
    evicted when limit reached (FIFO).

Used by:
    - component_13_working_memory.py: Context stack management

Implementation:
    Uses collections.deque(maxlen=10) for automatic FIFO eviction.
"""

# =============================================================================
# Production System Configuration
# =============================================================================

PRODUCTION_BATCH_UPDATE_THRESHOLD: int = 10
"""
Number of rule applications before batch-updating Neo4j statistics.

Production rules track usage statistics (application_count, success_count,
last_applied). To reduce database write load, statistics are batched and
written every N applications.

Rationale:
    Balances data freshness (for rule introspection queries) with performance
    (avoiding write transaction per rule firing). 10 applications is short
    enough for near-real-time stats while reducing write load by 10x.

Used by:
    - component_1_netzwerk_production_rules.py: update_production_rule_stats()
    - component_54_production_system.py: Stats accumulation during rule firing

Tuning:
    - Lower values (e.g., 5): Fresher stats, higher write load
    - Higher values (e.g., 20): Less write load, staleness acceptable
"""

# =============================================================================
# End of Constants
# =============================================================================
