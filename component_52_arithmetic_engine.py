"""
Arithmetic Engine for KAI
Main orchestration engine for arithmetic reasoning with Neo4j integration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofTree
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult

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


class ArithmeticEngine(BaseReasoningEngine):
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

    # ========================================================================
    # BaseReasoningEngine Interface Implementation
    # ========================================================================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute arithmetic reasoning on the query.

        Args:
            query: Natural language query
            context: Context with:
                - "operation": Arithmetic operation ("+", "-", "*", "/", "<", ">", "=", etc.)
                - "operands": List of operands
                - "property": Property to check ("even", "odd", "prime")
                - "number": Number for property checking
                - "relations": List of relations for transitive inference

        Returns:
            ReasoningResult with calculation result and proof tree
        """
        try:
            # Check for property checking
            if "property" in context and "number" in context:
                property_name = context["property"]
                number = context["number"]

                result = self.check_property(number, property_name)

                return ReasoningResult(
                    success=True,
                    answer=f"{number} is {'indeed' if result.value else 'not'} {property_name}",
                    confidence=result.confidence,
                    proof_tree=result.proof_tree,
                    strategy_used="arithmetic_property_checking",
                    metadata={
                        "property": property_name,
                        "number": number,
                        "result": result.value,
                    },
                )

            # Check for transitive inference
            elif "relations" in context:
                relations = context["relations"]
                result = self.transitive_inference(relations)

                return ReasoningResult(
                    success=True,
                    answer=f"Transitive inference result: {result.value}",
                    confidence=result.confidence,
                    proof_tree=result.proof_tree,
                    strategy_used="arithmetic_transitive_inference",
                    metadata={"num_relations": len(relations)},
                )

            # Check for comparison
            elif "operation" in context and context["operation"] in [
                "<",
                ">",
                "=",
                "<=",
                ">=",
            ]:
                operation = context["operation"]
                operands = context.get("operands", [])

                if len(operands) != 2:
                    return ReasoningResult(
                        success=False,
                        answer="Comparison requires exactly 2 operands",
                        confidence=0.0,
                        strategy_used="arithmetic_reasoning",
                    )

                result = self.compare(operands[0], operands[1], operation)

                return ReasoningResult(
                    success=True,
                    answer=f"{operands[0]} {operation} {operands[1]} is {result.value}",
                    confidence=result.confidence,
                    proof_tree=result.proof_tree,
                    strategy_used="arithmetic_comparison",
                    metadata={
                        "operation": operation,
                        "operands": operands,
                        "result": result.value,
                    },
                )

            # Standard arithmetic operation
            elif "operation" in context and "operands" in context:
                operation = context["operation"]
                operands = context["operands"]

                result = self.calculate(operation, *operands)

                return ReasoningResult(
                    success=True,
                    answer=f"{operation}({', '.join(map(str, operands))}) = {result.value}",
                    confidence=result.confidence,
                    proof_tree=result.proof_tree,
                    strategy_used="arithmetic_calculation",
                    metadata={
                        "operation": operation,
                        "operands": operands,
                        "result": result.value,
                    },
                )

            else:
                # No valid context provided
                logger.warning(
                    "ArithmeticEngine.reason() called without valid context",
                    extra={"query": query, "context_keys": list(context.keys())},
                )
                return ReasoningResult(
                    success=False,
                    answer="Insufficient context for arithmetic reasoning",
                    confidence=0.0,
                    strategy_used="arithmetic_reasoning",
                )

        except Exception as e:
            logger.error(
                "Error in arithmetic reasoning",
                extra={"query": query, "error": str(e)},
                exc_info=True,
            )
            return ReasoningResult(
                success=False,
                answer=f"Arithmetic reasoning error: {str(e)}",
                confidence=0.0,
                strategy_used="arithmetic_reasoning",
            )

    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        return [
            "arithmetic",
            "mathematical_operations",
            "comparison",
            "property_checking",
            "transitive_inference",
            "rational_arithmetic",
            "decimal_arithmetic",
            "modulo_arithmetic",
            "power_arithmetic",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for arithmetic reasoning.

        Returns:
            Cost estimate in [0.0, 1.0] range
            Base cost: 0.2 (cheap, direct computation)
        """
        # Arithmetic reasoning is generally cheap:
        # - Direct calculations are O(1) or O(n) for operands
        # - Comparison is O(1)
        # - Property checking varies (prime is O(sqrt(n)))
        # - Transitive inference can be more expensive O(n^2) worst case
        base_cost = 0.2

        # Query complexity has minimal impact
        query_complexity = min(len(query) / 500.0, 0.05)

        return min(base_cost + query_complexity, 1.0)
