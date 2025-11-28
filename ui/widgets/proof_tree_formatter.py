"""
ui.widgets.proof_tree_formatter

Mathematical text formatting and step type formatting for proof tree visualization.

Handles conversion of ASCII mathematical operators to Unicode symbols for display,
confidence level icons, and step type icons.

WARNING: This module uses Unicode symbols ONLY for UI rendering (Qt widgets).
DO NOT use these symbols in logging or console output (cp1252 incompatible).
For logging, use ASCII alternatives from CLAUDE.md encoding policy.
"""

from typing import Optional

try:
    from component_17_proof_explanation import StepType

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    StepType = None


class ProofTreeFormatter:
    """
    Formatter for mathematical text and proof step metadata.

    Provides methods for:
    - Converting ASCII math operators to Unicode symbols
    - Generating confidence level icons
    - Generating step type icons
    - Formatting confidence display text

    All methods are safe for Qt UI rendering but MUST NOT be used for logging.
    """

    @staticmethod
    def format_mathematical_text(text: str) -> str:
        """
        Format text with mathematical Unicode symbols.

        Replaces ASCII operators with Unicode equivalents:
        - * → ×
        - / → ÷
        - != → ≠
        - <= → ≤
        - >= → ≥
        - ** → ^
        - sqrt → √
        - pi/Pi → π

        WARNING: Output is UI-ONLY. DO NOT log to console (cp1252 incompatible).
        Use ONLY in QGraphicsItem rendering and Qt tooltips.

        Args:
            text: Text to format

        Returns:
            Formatted text with mathematical symbols
        """
        if not text:
            return text

        # Mathematical operators
        replacements = {
            " * ": " × ",
            " / ": " ÷ ",
            "!=": "≠",
            "<=": "≤",
            ">=": "≥",
            " mod ": " % ",
            "sqrt(": "√(",
            "**": "^",
            # Greek letters (if used)
            "pi": "π",
            "Pi": "π",
            "alpha": "α",
            "beta": "β",
            "gamma": "γ",
            "delta": "δ",
            "theta": "θ",
            "lambda": "λ",
            "mu": "μ",
            "sigma": "σ",
            "phi": "φ",
            "omega": "ω",
            # Mathematical symbols
            "infinity": "∞",
            "sum": "∑",
            "product": "∏",
            "integral": "∫",
            "partial": "∂",
            "nabla": "∇",
            "element_of": "∈",
            "not_element_of": "∉",
            "subset": "⊂",
            "superset": "⊃",
            "union": "∪",
            "intersection": "∩",
            "empty_set": "∅",
            "for_all": "∀",
            "exists": "∃",
            "therefore": "∴",
            "because": "∵",
            "approximately": "≈",
            "equivalent": "≡",
            "not_equal": "≠",
            "less_equal": "≤",
            "greater_equal": "≥",
        }

        formatted = text
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)

        return formatted

    @staticmethod
    def get_confidence_icon(confidence: float) -> str:
        """
        Get icon representing confidence level.

        Integrated with ConfidenceManager levels:
        - >= 0.8: [OK] (High)
        - 0.5-0.8: ~ (Medium)
        - 0.3-0.5: ! (Low)
        - < 0.3: ? (Unknown)

        Args:
            confidence: Confidence value (0.0-1.0)

        Returns:
            Icon string for display
        """
        if confidence >= 0.8:
            return "[OK]"  # Check mark for high confidence
        elif confidence >= 0.5:
            return "~"  # Tilde for medium confidence
        elif confidence >= 0.3:
            return "!"  # Exclamation for low confidence
        else:
            return "?"  # Question mark for unknown

    @staticmethod
    def get_step_icon(step_type: "StepType") -> str:
        """
        Get icon for step type.

        NOTE: Unicode symbols below are ONLY for UI rendering (QGraphicsItem.paint()).
        NEVER use these in logging or console output (violates CLAUDE.md encoding policy).
        For logging, use ASCII alternatives: [OK], *, ->, etc.

        Args:
            step_type: The step type to get icon for

        Returns:
            Icon string (may contain Unicode emoji)
        """
        if not PROOF_SYSTEM_AVAILABLE or step_type is None:
            return "*"

        icons = {
            StepType.FACT_MATCH: "[INFO]",
            StepType.RULE_APPLICATION: "⚙️",
            StepType.INFERENCE: "💡",
            StepType.HYPOTHESIS: "🔬",
            StepType.GRAPH_TRAVERSAL: "🗺️",
            StepType.PROBABILISTIC: "🎲",
            StepType.DECOMPOSITION: "🔀",
            StepType.UNIFICATION: "🔗",
            StepType.PREMISE: "📋",
            StepType.ASSUMPTION: "🤔",
            StepType.CONCLUSION: "✓",
            StepType.CONTRADICTION: "⚠️",
        }
        return icons.get(step_type, "*")

    @staticmethod
    def format_confidence_text(confidence: float) -> str:
        """
        Format confidence value as percentage text.

        Args:
            confidence: Confidence value (0.0-1.0)

        Returns:
            Formatted percentage string (e.g., "85%")
        """
        return f"{confidence:.0%}"

    @staticmethod
    def truncate_with_ellipsis(text: str, max_length: int) -> str:
        """
        Truncate text and add ellipsis if too long.

        Args:
            text: Text to truncate
            max_length: Maximum length before truncation

        Returns:
            Truncated text with "..." if necessary
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
