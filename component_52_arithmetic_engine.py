"""
Arithmetic Engine for KAI
Main orchestration engine for arithmetic reasoning with Neo4j integration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofTree

logger = get_logger(__name__)


@dataclass
class ArithmeticResult:
    """Result of an arithmetic operation"""

    value: Any  # int, float, Fraction, Decimal
    proof_tree: ProofTree
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArithmeticConfig:
    """Configuration for ArithmeticEngine"""

    # Transitive Inference
    max_transitive_rounds: int = 10  # Prevent infinite loops in relation chains
    max_total_relations: int = 1000  # Hard limit for memory protection

    # Prime Number Checking
    use_sqrt_optimization: bool = True  # Check divisors only up to square root n

    # Decimal Arithmetic
    default_decimal_precision: int = 10  # Significant digits for Decimal operations

    # Constant Precision
    constant_precision_digits: int = 50  # Digits for pi, e, phi, etc.

    # Neo4j Persistence
    persist_to_neo4j: bool = True  # Enable/disable graph persistence

    # Confidence update factors
    correct_factor: float = 1.1
    incorrect_factor: float = 0.85
    min_confidence: float = 0.0
    max_confidence: float = 1.0

    # Input validation
    max_number_value: int = 10_000_000  # Max value for find_divisors brute force

    def __post_init__(self):
        """Validate configuration"""
        if self.max_transitive_rounds < 1:
            raise ValueError("max_transitive_rounds must be >= 1")
        if self.default_decimal_precision < 1:
            raise ValueError("default_decimal_precision must be >= 1")
        if not 0.0 <= self.min_confidence <= self.max_confidence <= 1.0:
            raise ValueError(
                f"Invalid confidence bounds: [{self.min_confidence}, {self.max_confidence}]"
            )
        if self.max_total_relations < 100:
            raise ValueError("max_total_relations must be >= 100")


class ArithmeticEngine:
    """Main engine for arithmetic reasoning (thread-safe)"""

    def __init__(
        self, netzwerk: KonzeptNetzwerkCore, config: Optional[ArithmeticConfig] = None
    ):
        self.netzwerk = netzwerk
        self.config = config or ArithmeticConfig()

        # Import specialized components
        from component_52_arithmetic_operations import (
            DecimalArithmetic,
            MathematicalConstants,
            ModuloArithmetic,
            OperationRegistry,
            PowerArithmetic,
            RationalArithmetic,
        )
        from component_52_comparison_engine import ComparisonEngine, PropertyChecker

        self.registry = OperationRegistry()
        self.comparison_engine = ComparisonEngine(netzwerk, self.config)
        self.property_checker = PropertyChecker(netzwerk, self.config)
        self.rational_arithmetic = RationalArithmetic()
        self.decimal_arithmetic = DecimalArithmetic(
            self.config.default_decimal_precision
        )
        self.power_arithmetic = PowerArithmetic()
        self.modulo_arithmetic = ModuloArithmetic()
        self.math_constants = MathematicalConstants()
        self._register_operations()

        logger.info(
            "ArithmeticEngine initialized with config: max_transitive_rounds=%d, "
            "max_total_relations=%d, decimal_precision=%d",
            self.config.max_transitive_rounds,
            self.config.max_total_relations,
            self.config.default_decimal_precision,
        )

    def _register_operations(self):
        """Register all standard operations"""
        from component_52_arithmetic_operations import (
            Addition,
            Division,
            Multiplication,
            Subtraction,
        )

        self.registry.register(Addition())
        self.registry.register(Subtraction())
        self.registry.register(Multiplication())
        self.registry.register(Division())

    def calculate(self, operation: str, *operands) -> ArithmeticResult:
        """
        Execute calculation

        Args:
            operation: Operation (symbol or name)
            operands: Operands (already converted to numbers)

        Returns:
            ArithmeticResult with value, proof and confidence
        """
        logger.debug(
            "calculate() called: operation=%s, operands=%s", operation, operands
        )

        op = self.registry.get(operation)
        if not op:
            logger.error("Unknown operation requested: %s", operation)
            raise ValueError(f"Unbekannte Operation: {operation}")

        # Validation
        valid, error = op.validate(*operands)
        if not valid:
            logger.warning(
                "Validation failed for %s with operands %s: %s",
                operation,
                operands,
                error,
            )
            raise ValueError(f"Validierung fehlgeschlagen: {error}")

        # Execution
        result = op.execute(*operands)
        logger.debug(
            "Operation %s completed: result=%s, confidence=%f",
            operation,
            result.value,
            result.confidence,
        )

        # Optional: Store in Neo4j
        self._persist_calculation(operation, operands, result)

        return result

    def compare(self, a, b, operator: str) -> ArithmeticResult:
        """
        Compare two numbers (delegates to ComparisonEngine)

        Args:
            a, b: Numbers
            operator: "<", ">", "=", "<=", ">="

        Returns:
            ArithmeticResult with bool value
        """
        return self.comparison_engine.compare(a, b, operator)

    def check_property(self, n: int, property_name: str) -> ArithmeticResult:
        """
        Check property of a number (delegates to PropertyChecker)

        Args:
            n: Number
            property_name: "even", "odd", "prime"

        Returns:
            ArithmeticResult with bool value
        """
        property_methods = {
            "even": self.property_checker.is_even,
            "odd": self.property_checker.is_odd,
            "prime": self.property_checker.is_prime,
        }

        if property_name not in property_methods:
            raise ValueError(f"Unbekannte Eigenschaft: {property_name}")

        return property_methods[property_name](n)

    def find_divisors(self, n: int) -> ArithmeticResult:
        """Find all divisors (delegates to PropertyChecker)"""
        return self.property_checker.find_divisors(n)

    def transitive_inference(
        self, relations: List[Tuple[Any, str, Any]]
    ) -> ArithmeticResult:
        """Transitive inference (delegates to ComparisonEngine)"""
        return self.comparison_engine.build_transitive_proof(relations)

    def _persist_calculation(
        self, operation: str, operands: tuple, result: ArithmeticResult
    ):
        """Store calculation in Neo4j (optional)"""
        # TODO: Implement if needed
