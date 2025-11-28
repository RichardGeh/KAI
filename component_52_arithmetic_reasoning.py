"""
Arithmetic Reasoning for KAI
FACADE MODULE - Maintains backward compatibility by re-exporting from split modules

This module serves as a compatibility layer after the refactoring that split
the original 1,838-line file into 3 focused modules:
- component_52_arithmetic_operations.py (800 lines): Individual operations
- component_52_comparison_engine.py (600 lines): Comparisons and property checking
- component_52_arithmetic_engine.py (200 lines): Main orchestration

All original functionality is preserved and re-exported here for backward compatibility.
"""

# Re-export core data structures from engine module
from component_52_arithmetic_engine import (
    ArithmeticConfig,
    ArithmeticEngine,
    ArithmeticResult,
)

# Re-export operation classes from operations module
from component_52_arithmetic_operations import (
    Addition,
    BaseOperation,
    DecimalArithmetic,
    Division,
    MathematicalConstants,
    ModuloArithmetic,
    Multiplication,
    OperationRegistry,
    PowerArithmetic,
    RationalArithmetic,
    Subtraction,
)

# Re-export comparison and property classes from comparison engine module
from component_52_comparison_engine import ComparisonEngine, PropertyChecker

# For backward compatibility, make all classes available at top level
__all__ = [
    # Core types
    "ArithmeticResult",
    "ArithmeticConfig",
    # Main engine
    "ArithmeticEngine",
    # Operations
    "BaseOperation",
    "Addition",
    "Subtraction",
    "Multiplication",
    "Division",
    "OperationRegistry",
    # Comparison and properties
    "ComparisonEngine",
    "PropertyChecker",
    # Specialized arithmetic
    "RationalArithmetic",
    "DecimalArithmetic",
    "PowerArithmetic",
    "ModuloArithmetic",
    "MathematicalConstants",
]
