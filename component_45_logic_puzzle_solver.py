"""
component_45_logic_puzzle_solver.py
====================================
FACADE for logic puzzle solver - maintains 100% backward compatibility.

This facade delegates to specialized modules:
- component_45_logic_puzzle_parser.py: Natural language parsing
- component_45_logic_puzzle_solver_core.py: SAT solving and answer formatting

ARCHITECTURE:
- Split on 2025-11-29 from monolithic 1,068-line file
- All modules <800 lines
- 100% backward compatible via facade pattern
- Thread-safe (parser/solver are stateless per-solve)

For new code: Import from specific modules directly.
For legacy code: This facade ensures no breaking changes.

Author: KAI Development Team
Date: 2025-11-29 (Facade after architectural split)
"""

# Re-export all public APIs for backward compatibility
from component_45_logic_puzzle_parser import (
    LogicCondition,
    LogicConditionParser,
    LogicVariable,
    _get_nlp_model,
)
from component_45_logic_puzzle_solver_core import LogicPuzzleSolver

# Expose everything that was previously importable
__all__ = [
    "LogicVariable",
    "LogicCondition",
    "LogicConditionParser",
    "LogicPuzzleSolver",
    "_get_nlp_model",
]
