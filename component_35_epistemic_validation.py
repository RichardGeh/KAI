"""
component_35_epistemic_validation.py

Input Validation for Epistemic Reasoning Components

Provides security validation for all epistemic reasoning inputs to prevent:
- Cypher/SQL Injection attacks (OWASP A03:2021)
- Invalid parameter values
- DoS attacks via malicious inputs

Autor: KAI Development Team
Erstellt: 2025-11-23 (Security Hardening)
"""

import re
from typing import Any


class EpistemicValidator:
    """Input validation for epistemic reasoning components"""

    # Regex patterns (whitelist approach for security)
    AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
    PROP_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,200}$")

    @staticmethod
    def validate_agent_id(agent_id: Any) -> str:
        """
        Validate and sanitize agent ID

        Args:
            agent_id: Input to validate

        Returns:
            Validated agent_id (str)

        Raises:
            TypeError: If agent_id is not a string
            ValueError: If agent_id contains invalid characters

        Example:
            >>> EpistemicValidator.validate_agent_id("alice")
            'alice'
            >>> EpistemicValidator.validate_agent_id("alice'}) DELETE")
            ValueError: Invalid agent_id
        """
        if not isinstance(agent_id, str):
            raise TypeError(f"agent_id must be string, got {type(agent_id).__name__}")

        if not EpistemicValidator.AGENT_ID_PATTERN.match(agent_id):
            raise ValueError(
                f"Invalid agent_id: '{agent_id}'. "
                "Must be alphanumeric, underscore, hyphen (1-100 chars)"
            )

        return agent_id

    @staticmethod
    def validate_proposition_id(prop_id: Any) -> str:
        """
        Validate and sanitize proposition ID

        Args:
            prop_id: Input to validate

        Returns:
            Validated prop_id (str)

        Raises:
            TypeError: If prop_id is not a string
            ValueError: If prop_id contains invalid characters

        Example:
            >>> EpistemicValidator.validate_proposition_id("sky_is_blue")
            'sky_is_blue'
            >>> EpistemicValidator.validate_proposition_id("p'}) MATCH")
            ValueError: Invalid proposition_id
        """
        if not isinstance(prop_id, str):
            raise TypeError(
                f"proposition_id must be string, got {type(prop_id).__name__}"
            )

        if not EpistemicValidator.PROP_ID_PATTERN.match(prop_id):
            raise ValueError(
                f"Invalid proposition_id: '{prop_id}'. "
                "Must be alphanumeric, underscore, hyphen (1-200 chars)"
            )

        return prop_id

    @staticmethod
    def validate_certainty(certainty: Any) -> float:
        """
        Validate certainty value

        Args:
            certainty: Input to validate

        Returns:
            Validated certainty (float)

        Raises:
            TypeError: If certainty is not numeric
            ValueError: If certainty is not in [0.0, 1.0]

        Example:
            >>> EpistemicValidator.validate_certainty(0.8)
            0.8
            >>> EpistemicValidator.validate_certainty(1.5)
            ValueError: certainty must be in [0.0, 1.0]
        """
        if not isinstance(certainty, (int, float)):
            raise TypeError(
                f"certainty must be numeric, got {type(certainty).__name__}"
            )

        if not (0.0 <= certainty <= 1.0):
            raise ValueError(f"certainty must be in [0.0, 1.0], got {certainty}")

        return float(certainty)

    @staticmethod
    def validate_meta_level(meta_level: Any) -> int:
        """
        Validate meta-level depth

        Args:
            meta_level: Input to validate

        Returns:
            Validated meta_level (int)

        Raises:
            TypeError: If meta_level is not an integer
            ValueError: If meta_level is not in [1, 10]

        Example:
            >>> EpistemicValidator.validate_meta_level(2)
            2
            >>> EpistemicValidator.validate_meta_level(100)
            ValueError: meta_level must be in [1, 10]
        """
        if not isinstance(meta_level, int):
            raise TypeError(f"meta_level must be int, got {type(meta_level).__name__}")

        if not (1 <= meta_level <= 10):
            raise ValueError(f"meta_level must be in [1, 10], got {meta_level}")

        return meta_level

    @staticmethod
    def validate_max_depth(max_depth: Any, min_val: int = 1, max_val: int = 10) -> int:
        """
        Validate max_depth parameter

        Args:
            max_depth: Input to validate
            min_val: Minimum allowed value (default: 1)
            max_val: Maximum allowed value (default: 10)

        Returns:
            Validated max_depth (int)

        Raises:
            TypeError: If max_depth is not an integer
            ValueError: If max_depth is outside [min_val, max_val]

        Example:
            >>> EpistemicValidator.validate_max_depth(5)
            5
            >>> EpistemicValidator.validate_max_depth(999)
            ValueError: max_depth must be in [1, 10]
        """
        if not isinstance(max_depth, int):
            raise TypeError(f"max_depth must be int, got {type(max_depth).__name__}")

        if not (min_val <= max_depth <= max_val):
            raise ValueError(
                f"max_depth must be in [{min_val}, {max_val}], got {max_depth}"
            )

        return max_depth

    @staticmethod
    def validate_max_iterations(max_iterations: Any) -> int:
        """
        Validate max_iterations parameter

        Args:
            max_iterations: Input to validate

        Returns:
            Validated max_iterations (int)

        Raises:
            TypeError: If max_iterations is not an integer
            ValueError: If max_iterations is not in [1, 1000]
        """
        if not isinstance(max_iterations, int):
            raise TypeError(
                f"max_iterations must be int, got {type(max_iterations).__name__}"
            )

        if not (1 <= max_iterations <= 1000):
            raise ValueError(
                f"max_iterations must be in [1, 1000], got {max_iterations}"
            )

        return max_iterations


if __name__ == "__main__":
    print("\n=== EpistemicValidator Test ===\n")

    # Test valid inputs
    print("Testing valid inputs...")
    assert EpistemicValidator.validate_agent_id("alice") == "alice"
    assert EpistemicValidator.validate_agent_id("agent_123") == "agent_123"
    assert EpistemicValidator.validate_proposition_id("sky_is_blue") == "sky_is_blue"
    assert EpistemicValidator.validate_certainty(0.8) == 0.8
    assert EpistemicValidator.validate_meta_level(3) == 3
    assert EpistemicValidator.validate_max_depth(5) == 5
    print("[OK] Valid inputs test passed")

    # Test injection attacks blocked
    print("\nTesting injection prevention...")
    try:
        EpistemicValidator.validate_agent_id("alice'}) DELETE")
        print("[FEHLER] Injection attack not blocked!")
    except ValueError as e:
        print(f"[OK] Injection blocked: {e}")

    try:
        EpistemicValidator.validate_proposition_id("p'}) MATCH (x) DELETE x")
        print("[FEHLER] Injection attack not blocked!")
    except ValueError as e:
        print(f"[OK] Injection blocked: {e}")

    # Test invalid types
    print("\nTesting type validation...")
    try:
        EpistemicValidator.validate_agent_id(123)
        print("[FEHLER] Type check failed!")
    except TypeError as e:
        print(f"[OK] Type error caught: {e}")

    # Test boundary violations
    print("\nTesting boundary validation...")
    try:
        EpistemicValidator.validate_certainty(1.5)
        print("[FEHLER] Boundary check failed!")
    except ValueError as e:
        print(f"[OK] Boundary error caught: {e}")

    try:
        EpistemicValidator.validate_max_depth(999)
        print("[FEHLER] Boundary check failed!")
    except ValueError as e:
        print(f"[OK] Boundary error caught: {e}")

    print("\n[OK] All validation tests passed!")
