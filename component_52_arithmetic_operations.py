"""
Arithmetic Operations Implementation for KAI
Contains individual operation classes (Addition, Subtraction, Multiplication, Division, etc.)
"""

import math
import threading
import uuid
from abc import ABC, abstractmethod
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofStep, ProofTree, StepType

logger = get_logger(__name__)


class BaseOperation(ABC):
    """Abstract Base Class for arithmetic operations"""

    def __init__(self, symbol: str, name: str, arity: int):
        self.symbol = symbol
        self.name = name
        self.arity = arity

    @abstractmethod
    def execute(self, *operands):
        """Execute operation and create proof"""

    @abstractmethod
    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validate operands (e.g. division by zero)"""


class Addition(BaseOperation):
    """Addition of numbers"""

    def __init__(self):
        super().__init__("+", "addition", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validate operands for addition"""
        if len(operands) != self.arity:
            return (
                False,
                f"Addition benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        return True, None

    def execute(self, *operands):
        """Execute addition"""
        a, b = operands
        result = a + b

        # Create detailed proof tree
        proof = ProofTree(query=f"{a} + {b} = ?")

        # Step 1: Identify operands (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Operanden a={a} und b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Step 2: Apply operation (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} + {b}",
            explanation_text=f"Wende Addition an: {a} + {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Addition", "operation": "+"},
        )
        step1.add_subgoal(step2)

        # Step 3: Calculate result (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} + {b}"],
            output=str(result),
            explanation_text=f"Ergebnis: {a} + {b} = {result}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result},
        )
        step2.add_subgoal(step3)

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": self.name, "operands": [a, b]},
        )


class Subtraction(BaseOperation):
    """Subtraction of numbers"""

    def __init__(self):
        super().__init__("-", "subtraction", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validate operands for subtraction"""
        if len(operands) != self.arity:
            return (
                False,
                f"Subtraktion benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        return True, None

    def execute(self, *operands):
        """Execute subtraction"""
        a, b = operands
        result = a - b

        # Create detailed proof tree
        proof = ProofTree(query=f"{a} - {b} = ?")

        # Step 1: Identify operands (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Minuend a={a} und Subtrahend b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Step 2: Apply operation (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} - {b}",
            explanation_text=f"Wende Subtraktion an: {a} - {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Subtraktion", "operation": "-"},
        )
        step1.add_subgoal(step2)

        # Step 3: Calculate result (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} - {b}"],
            output=str(result),
            explanation_text=f"Ergebnis: {a} - {b} = {result}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result},
        )
        step2.add_subgoal(step3)

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": self.name, "operands": [a, b]},
        )


class Multiplication(BaseOperation):
    """Multiplication of numbers"""

    def __init__(self):
        super().__init__("*", "multiplication", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validate operands for multiplication"""
        if len(operands) != self.arity:
            return (
                False,
                f"Multiplikation benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        return True, None

    def execute(self, *operands):
        """Execute multiplication"""
        a, b = operands
        result = a * b

        # Create detailed proof tree
        proof = ProofTree(query=f"{a} * {b} = ?")

        # Step 1: Identify operands (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Faktoren a={a} und b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Step 2: Apply operation (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} * {b}",
            explanation_text=f"Wende Multiplikation an: {a} * {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Multiplikation", "operation": "*"},
        )
        step1.add_subgoal(step2)

        # Step 3: Calculate result (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} * {b}"],
            output=str(result),
            explanation_text=f"Ergebnis: {a} * {b} = {result}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result},
        )
        step2.add_subgoal(step3)

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": self.name, "operands": [a, b]},
        )


class Division(BaseOperation):
    """Division of numbers"""

    def __init__(self):
        super().__init__("/", "division", 2)

    def validate(self, *operands) -> Tuple[bool, Optional[str]]:
        """Validate operands for division"""
        if len(operands) != self.arity:
            return (
                False,
                f"Division benötigt {self.arity} Operanden, {len(operands)} gegeben",
            )

        for i, op in enumerate(operands):
            if not isinstance(op, (int, float, Decimal, Fraction)):
                return False, f"Operand {i+1} ist keine Zahl: {type(op)}"

        # Check division by zero
        if operands[1] == 0:
            return False, "Division durch Null ist nicht erlaubt"

        return True, None

    def execute(self, *operands):
        """Execute division (with Fraction for exact fractions)"""
        a, b = operands

        # Use Fraction for exact fractions (when both are integers)
        if isinstance(a, int) and isinstance(b, int):
            result = Fraction(a, b)
            result_str = str(result)
        else:
            result = a / b
            result_str = str(result)

        # Create detailed proof tree
        proof = ProofTree(query=f"{a} / {b} = ?")

        # Step 1: Identify operands (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: Dividend a={a} und Divisor b={b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Step 2: Constraint check - division by zero (PREMISE for constraint)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[str(b)],
            output=f"{b} != 0: [OK]",
            explanation_text=f"Prüfe Constraint: Divisor {b} != 0",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"constraint": "division_by_zero", "check_passed": True},
        )
        step1.add_subgoal(step2)

        # Step 3: Apply operation (RULE_APPLICATION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} / {b}",
            explanation_text=f"Wende Division an: {a} / {b}",
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"rule": "Arithmetik: Division", "operation": "/"},
        )
        step2.add_subgoal(step3)

        # Step 4: Calculate result (CONCLUSION)
        if isinstance(result, Fraction):
            explanation = f"Ergebnis: {a} / {b} = {result_str} (exakter Bruch)"
        else:
            explanation = f"Ergebnis: {a} / {b} = {result_str}"

        step4 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} / {b}"],
            output=result_str,
            explanation_text=explanation,
            confidence=1.0,
            source_component="arithmetic_engine",
            metadata={"result": result, "result_type": type(result).__name__},
        )
        step3.add_subgoal(step4)

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=result,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": self.name,
                "operands": [a, b],
                "result_type": type(result).__name__,
            },
        )


class OperationRegistry:
    """Registry for all available operations (thread-safe)"""

    def __init__(self):
        self._operations: Dict[str, BaseOperation] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested calls

    def register(self, operation: BaseOperation):
        """Register an operation"""
        with self._lock:
            self._operations[operation.symbol] = operation
            self._operations[operation.name] = operation

    def get(self, key: str) -> Optional[BaseOperation]:
        """Get operation by symbol or name"""
        with self._lock:
            return self._operations.get(key)

    def list_operations(self) -> List[str]:
        """List all registered operations"""
        with self._lock:
            return list(set(self._operations.keys()))


class RationalArithmetic:
    """Rational arithmetic with exact results (Python fractions)"""

    def __init__(self):
        pass

    def add(self, a: Fraction, b: Fraction) -> Fraction:
        """Add two fractions"""
        return a + b  # Python Fraction supports + directly

    def subtract(self, a: Fraction, b: Fraction) -> Fraction:
        """Subtract two fractions"""
        return a - b

    def multiply(self, a: Fraction, b: Fraction) -> Fraction:
        """Multiply two fractions"""
        return a * b

    def divide(self, a: Fraction, b: Fraction) -> Fraction:
        """Divide two fractions"""
        if b == 0:
            raise ValueError("Division durch Null ist nicht erlaubt")
        return a / b

    def simplify(self, fraction: Fraction) -> Fraction:
        """Simplify fraction (automatic with Fraction)"""
        return fraction

    def to_mixed_number(self, fraction: Fraction) -> Tuple[int, Fraction]:
        """
        Convert to mixed number

        Example: 7/3 -> (2, 1/3)

        Args:
            fraction: Fraction to convert

        Returns:
            Tuple (whole part, fraction remainder)
        """
        whole = fraction.numerator // fraction.denominator
        remainder = fraction.numerator % fraction.denominator
        return whole, Fraction(remainder, fraction.denominator)

    def from_mixed_number(self, whole: int, fraction: Fraction) -> Fraction:
        """
        Convert mixed number to fraction

        Example: (2, 1/3) -> 7/3

        Args:
            whole: Whole part
            fraction: Fraction part

        Returns:
            Fraction
        """
        return Fraction(
            whole * fraction.denominator + fraction.numerator, fraction.denominator
        )

    def gcd(self, a: int, b: int) -> int:
        """Greatest common divisor"""
        return math.gcd(a, b)

    def lcm(self, a: int, b: int) -> int:
        """Least common multiple"""
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // math.gcd(a, b)

    def compare(self, a: Fraction, b: Fraction, operator: str) -> bool:
        """Compare two fractions"""
        ops = {
            "<": lambda x, y: x < y,
            ">": lambda x, y: x > y,
            "=": lambda x, y: x == y,
            "==": lambda x, y: x == y,
            "<=": lambda x, y: x <= y,
            ">=": lambda x, y: x >= y,
        }
        if operator not in ops:
            raise ValueError(f"Unbekannter Operator: {operator}")
        return ops[operator](a, b)


class DecimalArithmetic:
    """Decimal number arithmetic with configurable precision (thread-safe)"""

    def __init__(self, precision: int = 10):
        """
        Initialize DecimalArithmetic with precision

        Args:
            precision: Number of significant digits (default: 10)
        """
        self.precision = precision
        # DO NOT set global context here - use localcontext in methods

    def set_precision(self, precision: int):
        """Set new precision for this instance"""
        self.precision = precision
        # DO NOT mutate global context

    def calculate(self, operation: str, *operands) -> Decimal:
        """
        Execute operation with Decimal precision (with local context)

        Args:
            operation: Operation ("+", "-", "*", "/")
            operands: Operands (will be converted to Decimal)

        Returns:
            Decimal result
        """
        # Use localcontext to avoid global state mutation
        with localcontext() as ctx:
            ctx.prec = self.precision
            ctx.rounding = ROUND_HALF_UP

            # Convert to Decimal
            operands_decimal = [Decimal(str(op)) for op in operands]

            if operation == "+":
                return sum(operands_decimal)
            elif operation == "-":
                result = operands_decimal[0]
                for op in operands_decimal[1:]:
                    result -= op
                return result
            elif operation == "*":
                result = operands_decimal[0]
                for op in operands_decimal[1:]:
                    result *= op
                return result
            elif operation == "/":
                result = operands_decimal[0]
                for op in operands_decimal[1:]:
                    if op == 0:
                        raise ValueError("Division durch Null ist nicht erlaubt")
                    result /= op
                return result
            else:
                raise ValueError(f"Unbekannte Operation: {operation}")

    def round(self, value: float, decimals: int) -> float:
        """
        Round to n decimal places

        Args:
            value: Value to round
            decimals: Number of decimal places

        Returns:
            Rounded value
        """
        return round(value, decimals)

    def round_decimal(self, value: Decimal, decimals: int) -> Decimal:
        """
        Round Decimal to n decimal places (with local context)

        Args:
            value: Decimal value
            decimals: Number of decimal places

        Returns:
            Rounded Decimal
        """
        with localcontext() as ctx:
            ctx.prec = self.precision
            quantize_str = "1." + "0" * decimals if decimals > 0 else "1"
            return value.quantize(Decimal(quantize_str))

    def to_fraction(self, value: Decimal) -> Fraction:
        """Convert Decimal to Fraction (exact)"""
        return Fraction(str(value))

    def from_fraction(self, fraction: Fraction) -> Decimal:
        """Convert Fraction to Decimal (with current precision)"""
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)


class PowerArithmetic:
    """Powers, roots and related operations"""

    def __init__(self):
        pass

    def power(self, base: float, exponent: float) -> float:
        """
        Calculate base^exponent

        Args:
            base: Base
            exponent: Exponent

        Returns:
            Result of exponentiation

        Examples:
            power(2, 3) = 8
            power(4, 0.5) = 2.0 (square root)
            power(2, -1) = 0.5
        """
        if base == 0 and exponent < 0:
            raise ValueError("0 kann nicht zu einer negativen Potenz erhoben werden")

        if base < 0 and not self._is_integer(exponent):
            raise ValueError(
                "Negative Basis mit nicht-ganzzahligem Exponenten "
                "ist nicht definiert (komplexe Zahl)"
            )

        return base**exponent

    def square(self, n: float) -> float:
        """Square: n²"""
        return n**2

    def cube(self, n: float) -> float:
        """Cube: n³"""
        return n**3

    def sqrt(self, n: float) -> float:
        """
        Square root: √n

        Args:
            n: Number (must be >= 0)

        Returns:
            Square root of n
        """
        if n < 0:
            raise ValueError(
                "Quadratwurzel von negativen Zahlen ist nicht definiert (komplexe Zahl)"
            )

        return math.sqrt(n)

    def cbrt(self, n: float) -> float:
        """
        Cube root: ∛n

        Args:
            n: Number (can be negative)

        Returns:
            Cube root of n
        """
        if n >= 0:
            return n ** (1 / 3)
        else:
            # Negative cube root
            return -((-n) ** (1 / 3))

    def nth_root(self, n: float, root: int) -> float:
        """
        nth root

        Args:
            n: Number
            root: Root degree (must be > 0)

        Returns:
            root-th root of n

        Examples:
            nth_root(8, 3) = 2.0
            nth_root(16, 4) = 2.0
        """
        if root <= 0:
            raise ValueError("Wurzelgrad muss > 0 sein")

        if root == 1:
            return n

        # Even root of negative number
        if root % 2 == 0 and n < 0:
            raise ValueError(
                f"{root}-te Wurzel von negativen Zahlen ist nicht definiert (komplexe Zahl)"
            )

        # Odd root (also defined for negative numbers)
        if root % 2 == 1 and n < 0:
            return -((-n) ** (1 / root))

        return n ** (1 / root)

    def exp(self, x: float) -> float:
        """
        Exponential function: e^x

        Args:
            x: Exponent

        Returns:
            e^x
        """
        return math.exp(x)

    def log(self, x: float, base: Optional[float] = None) -> float:
        """
        Logarithm

        Args:
            x: Argument (must be > 0)
            base: Base (optional, default: e for natural logarithm)

        Returns:
            Logarithm of x to base base

        Examples:
            log(10) = ln(10) ≈ 2.302585
            log(100, 10) = log₁₀(100) = 2.0
        """
        if x <= 0:
            raise ValueError("Logarithmus ist nur für positive Zahlen definiert")

        if base is None:
            return math.log(x)  # Natural logarithm

        if base <= 0 or base == 1:
            raise ValueError("Logarithmus-Basis muss > 0 und != 1 sein")

        return math.log(x, base)

    def log10(self, x: float) -> float:
        """Logarithm base 10"""
        if x <= 0:
            raise ValueError("Logarithmus ist nur für positive Zahlen definiert")
        return math.log10(x)

    def log2(self, x: float) -> float:
        """Logarithm base 2"""
        if x <= 0:
            raise ValueError("Logarithmus ist nur für positive Zahlen definiert")
        return math.log2(x)

    def _is_integer(self, n: float) -> bool:
        """Check if number is integer"""
        return n == int(n)


class ModuloArithmetic:
    """Modulo and remainder arithmetic"""

    def __init__(self):
        pass

    def modulo(self, a: int, m: int) -> int:
        """
        Modulo operation: a mod m

        Args:
            a: Dividend
            m: Modulus (must be != 0)

        Returns:
            Remainder of division a / m (always non-negative)

        Examples:
            modulo(7, 3) = 1
            modulo(-7, 3) = 2 (Python convention)
        """
        if m == 0:
            raise ValueError("Modulo durch Null ist nicht erlaubt")

        return a % m

    def remainder(self, a: int, m: int) -> int:
        """
        Remainder operation (same as modulo in Python)

        Args:
            a: Dividend
            m: Divisor

        Returns:
            Remainder of division
        """
        return self.modulo(a, m)

    def divmod_op(self, a: int, m: int) -> Tuple[int, int]:
        """
        Quotient and remainder simultaneously

        Args:
            a: Dividend
            m: Divisor (must be != 0)

        Returns:
            Tuple (quotient, remainder)

        Examples:
            divmod_op(7, 3) = (2, 1)  # 7 = 2*3 + 1
        """
        if m == 0:
            raise ValueError("Division durch Null ist nicht erlaubt")

        return divmod(a, m)

    def mod_add(self, a: int, b: int, m: int) -> int:
        """
        Modular addition: (a + b) mod m

        Args:
            a, b: Summands
            m: Modulus

        Returns:
            (a + b) mod m
        """
        return (a + b) % m

    def mod_subtract(self, a: int, b: int, m: int) -> int:
        """
        Modular subtraction: (a - b) mod m

        Args:
            a, b: Minuend, subtrahend
            m: Modulus

        Returns:
            (a - b) mod m
        """
        return (a - b) % m

    def mod_multiply(self, a: int, b: int, m: int) -> int:
        """
        Modular multiplication: (a * b) mod m

        Args:
            a, b: Factors
            m: Modulus

        Returns:
            (a * b) mod m
        """
        return (a * b) % m

    def mod_power(self, base: int, exponent: int, m: int) -> int:
        """
        Modular exponentiation: base^exponent mod m
        Uses efficient square-and-multiply (Python built-in pow)

        Args:
            base: Base
            exponent: Exponent (must be >= 0)
            m: Modulus

        Returns:
            base^exponent mod m

        Examples:
            mod_power(2, 10, 1000) = 24
            mod_power(3, 100, 7) = 4
        """
        if exponent < 0:
            raise ValueError(
                "Negative Exponenten nicht unterstützt (benötigt modulares Inverse)"
            )

        if m == 0:
            raise ValueError("Modulo durch Null ist nicht erlaubt")

        return pow(base, exponent, m)

    def is_congruent(self, a: int, b: int, m: int) -> bool:
        """
        Check congruence: a ≡ b (mod m)

        Args:
            a, b: Numbers to compare
            m: Modulus

        Returns:
            True if a ≡ b (mod m)

        Examples:
            is_congruent(7, 1, 3) = True  # 7 ≡ 1 (mod 3)
            is_congruent(10, 2, 4) = True  # 10 ≡ 2 (mod 4)
        """
        return (a % m) == (b % m)

    def mod_inverse(self, a: int, m: int) -> Optional[int]:
        """
        Modular inverse: Find x with a*x ≡ 1 (mod m)
        Uses extended Euclidean algorithm

        Args:
            a: Number
            m: Modulus

        Returns:
            Modular inverse of a mod m, or None if doesn't exist

        Examples:
            mod_inverse(3, 7) = 5  # 3*5 = 15 ≡ 1 (mod 7)
        """

        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            """Extended Euclidean algorithm"""
            if a == 0:
                return b, 0, 1

            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1

            return gcd, x, y

        gcd, x, _ = extended_gcd(a % m, m)

        # Inverse exists only when gcd(a, m) = 1
        if gcd != 1:
            return None

        return (x % m + m) % m


class MathematicalConstants:
    """Mathematical constants and functions"""

    # Constants (with high precision)
    PI = Decimal("3.1415926535897932384626433832795028841971693993751")
    E = Decimal("2.7182818284590452353602874713526624977572470936999")
    GOLDEN_RATIO = Decimal("1.6180339887498948482045868343656381177203091798057")
    SQRT_2 = Decimal("1.4142135623730950488016887242096980785696718753769")
    SQRT_3 = Decimal("1.7320508075688772935274463415058723669428052538103")
    SQRT_5 = Decimal("2.2360679774997896964091736687312762354406183596115")

    # Float versions for faster calculations
    PI_FLOAT = math.pi
    E_FLOAT = math.e
    TAU_FLOAT = 2 * math.pi  # τ = 2π

    def __init__(self, use_decimal: bool = False):
        """
        Initialize MathematicalConstants

        Args:
            use_decimal: If True, use Decimal precision, else Float
        """
        self.use_decimal = use_decimal

    def pi(self) -> Any:
        """Return π (Decimal or Float)"""
        return self.PI if self.use_decimal else self.PI_FLOAT

    def e(self) -> Any:
        """Return e (Decimal or Float)"""
        return self.E if self.use_decimal else self.E_FLOAT

    def tau(self) -> Any:
        """Return τ = 2π"""
        return self.PI * 2 if self.use_decimal else self.TAU_FLOAT

    def golden_ratio(self) -> Any:
        """Return golden ratio φ"""
        return self.GOLDEN_RATIO if self.use_decimal else float(self.GOLDEN_RATIO)

    def sqrt_2(self) -> Any:
        """Return √2"""
        return self.SQRT_2 if self.use_decimal else float(self.SQRT_2)

    def sqrt_3(self) -> Any:
        """Return √3"""
        return self.SQRT_3 if self.use_decimal else float(self.SQRT_3)

    def sqrt_5(self) -> Any:
        """Return √5"""
        return self.SQRT_5 if self.use_decimal else float(self.SQRT_5)

    def circle_area(self, radius: float) -> float:
        """Calculate circle area: A = πr²"""
        return self.PI_FLOAT * radius**2

    def circle_circumference(self, radius: float) -> float:
        """Calculate circle circumference: C = 2πr"""
        return 2 * self.PI_FLOAT * radius

    def sphere_volume(self, radius: float) -> float:
        """Calculate sphere volume: V = 4/3 πr³"""
        return (4 / 3) * self.PI_FLOAT * radius**3

    def sphere_surface(self, radius: float) -> float:
        """Calculate sphere surface: A = 4πr²"""
        return 4 * self.PI_FLOAT * radius**2

    def cylinder_volume(self, radius: float, height: float) -> float:
        """Calculate cylinder volume: V = πr²h"""
        return self.PI_FLOAT * radius**2 * height

    def degrees_to_radians(self, degrees: float) -> float:
        """Convert degrees to radians"""
        return math.radians(degrees)

    def radians_to_degrees(self, radians: float) -> float:
        """Convert radians to degrees"""
        return math.degrees(radians)

    def sin(self, x: float, use_degrees: bool = False) -> float:
        """Sine (input in radians or degrees)"""
        if use_degrees:
            x = self.degrees_to_radians(x)
        return math.sin(x)

    def cos(self, x: float, use_degrees: bool = False) -> float:
        """Cosine (input in radians or degrees)"""
        if use_degrees:
            x = self.degrees_to_radians(x)
        return math.cos(x)

    def tan(self, x: float, use_degrees: bool = False) -> float:
        """Tangent (input in radians or degrees)"""
        if use_degrees:
            x = self.degrees_to_radians(x)
        return math.tan(x)
