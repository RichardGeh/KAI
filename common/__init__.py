"""
Common utilities and constants for the KAI project.

This package provides centralized configuration values, shared utilities,
and architectural constants used throughout the KAI codebase.
"""

from common.constants import *

__all__ = [
    # Embedding/Similarity Thresholds
    "EMBEDDING_DISTANCE_THRESHOLD",
    "SIMILARITY_THRESHOLD",
    # File Size Limits
    "MAX_FILE_LINES",
    # Cache Configuration
    "CACHE_TTL_FACTS",
    "CACHE_TTL_WORDS",
    "CACHE_TTL_EMBEDDINGS",
    "CACHE_TTL_PROTOTYPES",
    "CACHE_TTL_RESONANCE",
    "CACHE_MAXSIZE_SMALL",
    "CACHE_MAXSIZE_MEDIUM",
    "CACHE_MAXSIZE_LARGE",
    "CACHE_MAXSIZE_XLARGE",
    # Confidence Thresholds
    "CONFIDENCE_AUTO_SAVE",
    "CONFIDENCE_CONFIRM_MIN",
    "CONFIDENCE_CONFIRM_MAX",
    "CONFIDENCE_CLARIFY_MAX",
    # Adaptive Pattern Recognition
    "TYPO_DISTANCE_MIN",
    "TYPO_DISTANCE_MAX",
    "SEQUENCE_THRESHOLD_MIN",
    "SEQUENCE_THRESHOLD_MAX",
    # Working Memory
    "MAX_WORKING_MEMORY_DEPTH",
    # Production System
    "PRODUCTION_BATCH_UPDATE_THRESHOLD",
]
