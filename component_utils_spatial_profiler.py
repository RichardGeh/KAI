"""
component_utils_spatial_profiler.py

Performance Profiling Utilities for Spatial Reasoning

Provides profiling tools for:
- Grid operation timing
- Query performance analysis
- Cache hit rate monitoring
- Memory usage tracking
"""

import functools
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List

from component_15_logging_config import get_logger

logger = get_logger(__name__)


# ==================== Data Structures ====================


@dataclass
class ProfileResult:
    """Results from a single profiling run."""

    operation_name: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""


@dataclass
class ProfileSummary:
    """Aggregated profiling statistics."""

    operation_name: str
    total_calls: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Profiler Class ====================


class SpatialOperationProfiler:
    """
    Profiler for spatial reasoning operations.

    Tracks timing, success rates, and performance metrics.
    """

    def __init__(self):
        """Initialize the profiler."""
        self.results: List[ProfileResult] = []
        self.enabled = True

        logger.info("SpatialOperationProfiler initialized")

    def profile(self, operation_name: str = None):
        """
        Decorator for profiling functions.

        Args:
            operation_name: Name of the operation (defaults to function name)

        Example:
            @profiler.profile("grid_creation")
            def create_grid(rows, cols):
                return Grid2D(rows, cols)
        """

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                name = operation_name or func.__name__
                start_time = time.perf_counter()
                success = True
                error = ""
                result = None

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000

                    # Record result
                    profile_result = ProfileResult(
                        operation_name=name,
                        duration_ms=duration_ms,
                        success=success,
                        error=error,
                        metadata={
                            "args_count": len(args),
                            "kwargs_count": len(kwargs),
                        },
                    )
                    self.results.append(profile_result)

                    # Log if slow
                    if duration_ms > 100:
                        logger.warning(
                            f"Slow operation detected: {name}",
                            extra={
                                "duration_ms": duration_ms,
                                "success": success,
                            },
                        )

                return result

            return wrapper

        return decorator

    def profile_context(self, operation_name: str, metadata: Dict[str, Any] = None):
        """
        Context manager for profiling code blocks.

        Args:
            operation_name: Name of the operation
            metadata: Optional metadata to attach

        Example:
            with profiler.profile_context("complex_query"):
                result = perform_complex_query()
        """
        return _ProfileContext(self, operation_name, metadata or {})

    def get_summary(self, operation_name: str = None) -> List[ProfileSummary]:
        """
        Get profiling summary statistics.

        Args:
            operation_name: Optional filter for specific operation

        Returns:
            List of ProfileSummary objects
        """
        # Group results by operation name
        grouped: Dict[str, List[ProfileResult]] = {}
        for result in self.results:
            if operation_name and result.operation_name != operation_name:
                continue

            if result.operation_name not in grouped:
                grouped[result.operation_name] = []
            grouped[result.operation_name].append(result)

        # Calculate summaries
        summaries = []
        for name, results in grouped.items():
            durations = [r.duration_ms for r in results]
            successes = sum(1 for r in results if r.success)

            summary = ProfileSummary(
                operation_name=name,
                total_calls=len(results),
                total_duration_ms=sum(durations),
                avg_duration_ms=sum(durations) / len(durations),
                min_duration_ms=min(durations),
                max_duration_ms=max(durations),
                success_rate=successes / len(results),
            )
            summaries.append(summary)

        return summaries

    def log_summary(self, operation_name: str = None):
        """
        Log profiling summary to logger.

        Args:
            operation_name: Optional filter for specific operation
        """
        summaries = self.get_summary(operation_name)

        if not summaries:
            logger.info("No profiling data available")
            return

        logger.info("=== Spatial Operation Profiling Summary ===")
        for summary in summaries:
            logger.info(
                f"{summary.operation_name}",
                extra={
                    "total_calls": summary.total_calls,
                    "avg_duration_ms": f"{summary.avg_duration_ms:.2f}",
                    "min_duration_ms": f"{summary.min_duration_ms:.2f}",
                    "max_duration_ms": f"{summary.max_duration_ms:.2f}",
                    "success_rate": f"{summary.success_rate:.1%}",
                },
            )

    def clear(self):
        """Clear all profiling results."""
        self.results.clear()
        logger.debug("Profiling results cleared")

    def disable(self):
        """Disable profiling."""
        self.enabled = False
        logger.debug("Profiling disabled")

    def enable(self):
        """Enable profiling."""
        self.enabled = True
        logger.debug("Profiling enabled")


class _ProfileContext:
    """Context manager for profiling code blocks."""

    def __init__(
        self,
        profiler: SpatialOperationProfiler,
        operation_name: str,
        metadata: Dict[str, Any],
    ):
        self.profiler = profiler
        self.operation_name = operation_name
        self.metadata = metadata
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000

        success = exc_type is None
        error = str(exc_val) if exc_val else ""

        result = ProfileResult(
            operation_name=self.operation_name,
            duration_ms=duration_ms,
            success=success,
            error=error,
            metadata=self.metadata,
        )
        self.profiler.results.append(result)


# ==================== Global Profiler Instance ====================


# Global profiler instance for convenience
spatial_profiler = SpatialOperationProfiler()


# ==================== Convenience Decorators ====================


def profile_spatial_operation(operation_name: str = None):
    """
    Convenience decorator using global profiler.

    Example:
        @profile_spatial_operation("grid_query")
        def query_grid(grid, position):
            return grid.get_cell_data(position)
    """
    return spatial_profiler.profile(operation_name)


# ==================== Performance Benchmarking ====================


class SpatialPerformanceBenchmark:
    """
    Benchmark suite for spatial reasoning performance.

    Provides standardized benchmarks for:
    - Grid creation (various sizes)
    - Neighbor queries
    - Distance calculations
    - Path finding
    """

    def __init__(self, profiler: SpatialOperationProfiler = None):
        """
        Initialize benchmark suite.

        Args:
            profiler: Optional profiler instance (uses global if None)
        """
        self.profiler = profiler or spatial_profiler

    def run_all_benchmarks(self) -> Dict[str, ProfileSummary]:
        """
        Run all benchmarks and return summaries.

        Returns:
            Dictionary mapping benchmark name to ProfileSummary
        """
        logger.info("Starting spatial reasoning benchmarks...")

        benchmarks = [
            ("grid_creation_small", self.benchmark_grid_creation_small),
            ("grid_creation_medium", self.benchmark_grid_creation_medium),
            ("grid_creation_large", self.benchmark_grid_creation_large),
            ("neighbor_queries", self.benchmark_neighbor_queries),
            ("distance_calculations", self.benchmark_distance_calculations),
        ]

        results = {}
        for name, benchmark_func in benchmarks:
            try:
                benchmark_func()
                summary = self.profiler.get_summary(name)[0]
                results[name] = summary

                logger.info(
                    f"Benchmark '{name}' completed",
                    extra={
                        "avg_duration_ms": f"{summary.avg_duration_ms:.2f}",
                        "calls": summary.total_calls,
                    },
                )
            except Exception as e:
                logger.error(f"Benchmark '{name}' failed: {e}")

        return results

    def benchmark_grid_creation_small(self):
        """Benchmark small grid creation (8x8)."""
        from component_42_spatial_reasoning import Grid

        @self.profiler.profile("grid_creation_small")
        def create_small_grid():
            return Grid(height=8, width=8)

        for _ in range(100):
            create_small_grid()

    def benchmark_grid_creation_medium(self):
        """Benchmark medium grid creation (50x50)."""
        from component_42_spatial_reasoning import Grid

        @self.profiler.profile("grid_creation_medium")
        def create_medium_grid():
            return Grid(height=50, width=50)

        for _ in range(50):
            create_medium_grid()

    def benchmark_grid_creation_large(self):
        """Benchmark large grid creation (100x100)."""
        from component_42_spatial_reasoning import Grid

        @self.profiler.profile("grid_creation_large")
        def create_large_grid():
            return Grid(height=100, width=100)

        for _ in range(10):
            create_large_grid()

    def benchmark_neighbor_queries(self):
        """Benchmark neighbor queries."""
        from component_42_spatial_reasoning import Grid, NeighborhoodType, Position

        grid = Grid(height=100, width=100)
        center = Position(50, 50)

        @self.profiler.profile("neighbor_queries")
        def query_neighbors():
            return grid.get_neighbors(center, NeighborhoodType.MOORE)

        for _ in range(1000):
            query_neighbors()

    def benchmark_distance_calculations(self):
        """Benchmark distance calculations."""
        from component_42_spatial_reasoning import Position

        pos1 = Position(0, 0)
        pos2 = Position(100, 100)

        @self.profiler.profile("distance_calculations")
        def calculate_distance():
            return pos1.distance_to(pos2)

        for _ in range(10000):
            calculate_distance()


# ==================== Test Coverage Utilities ====================


def generate_coverage_report():
    """
    Generate a test coverage report for spatial reasoning.

    Requires pytest-cov to be installed.
    """
    import subprocess
    import sys

    logger.info("Generating test coverage report...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_spatial_reasoning_integration.py",
                "--cov=component_42_spatial_reasoning",
                "--cov=component_43_spatial_grid_widget",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_report_spatial",
                "-v",
            ],
            capture_output=True,
            text=True,
        )

        logger.info("Coverage report generated")
        logger.info(f"Exit code: {result.returncode}")

        if result.returncode == 0:
            logger.info("HTML report available at: coverage_report_spatial/index.html")

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Failed to generate coverage report: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Running spatial reasoning benchmarks...")

    benchmark = SpatialPerformanceBenchmark()
    results = benchmark.run_all_benchmarks()

    print("\n=== Benchmark Results ===")
    for name, summary in results.items():
        print(f"{name}:")
        print(f"  Avg: {summary.avg_duration_ms:.2f} ms")
        print(f"  Min: {summary.min_duration_ms:.2f} ms")
        print(f"  Max: {summary.max_duration_ms:.2f} ms")
        print(f"  Calls: {summary.total_calls}")

    print("\n Generating test coverage report...")
    generate_coverage_report()
