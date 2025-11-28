# component_9_sat_utils.py
"""
Shared utilities for SAT solver integration in Logic Engine.

Provides conversion functions between Logic Engine structures (Fact, Rule)
and SAT solver structures (Literal, Clause, CNFFormula).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from component_9_logik_engine_core import Fact


def fact_to_literal_name(fact: "Fact", include_args: bool = True) -> str:
    """
    Converts Fact to normalized literal name for SAT solver.

    Args:
        fact: Fact object
        include_args: Whether to include arguments in name (default: True)

    Returns:
        Normalized literal name (alphanumeric + underscore only)

    Examples:
        >>> fact = Fact(pred="HAS_PROPERTY", args={"subject": "apfel", "object": "rot"})
        >>> fact_to_literal_name(fact)
        'HAS_PROPERTY_apfel_rot'

        >>> fact_to_literal_name(fact, include_args=False)
        'HAS_PROPERTY'
    """
    lit_name = fact.pred

    if include_args and fact.args:
        # Sort args for deterministic ordering
        args_str = "_".join(str(v) for k, v in sorted(fact.args.items()))
        lit_name = f"{lit_name}_{args_str}"

    # Normalize: Remove special characters
    lit_name = lit_name.replace(" ", "_").replace("-", "_").replace(":", "_")
    # Remove all non-alphanumeric characters except underscore
    lit_name = "".join(c for c in lit_name if c.isalnum() or c == "_")

    return lit_name
