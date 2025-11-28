"""
Comparison Engine for KAI
Handles comparison operations (<, >, =, <=, >=) with transitive inference
"""

import re
import threading
import uuid
from typing import Any, List, Optional, Tuple

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from component_1_netzwerk_core import KonzeptNetzwerkCore
from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofStep, ProofTree, StepType

logger = get_logger(__name__)

# Validation patterns
ENTITY_NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,63}$", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"^-?\d+$")


class ComparisonEngine:
    """Engine for comparison operations with transitive inference"""

    def __init__(self, netzwerk: KonzeptNetzwerkCore, config: Optional[Any] = None):
        self.netzwerk = netzwerk
        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticConfig

        self.config = config or ArithmeticConfig()
        self._comparison_ops = {
            "<": lambda x, y: x < y,
            ">": lambda x, y: x > y,
            "=": lambda x, y: x == y,
            "<=": lambda x, y: x <= y,
            ">=": lambda x, y: x >= y,
        }
        self._op_names = {
            "<": "kleiner als",
            ">": "größer als",
            "=": "gleich",
            "<=": "kleiner gleich",
            ">=": "größer gleich",
        }

    def compare(self, a, b, operator: str):
        """
        Compare two numbers

        Args:
            a, b: Numbers to compare
            operator: "<", ">", "=", "<=", ">="

        Returns:
            ArithmeticResult with bool value and proof
        """
        if operator not in self._comparison_ops:
            raise ValueError(f"Unbekannter Vergleichsoperator: {operator}")

        result_value = self._comparison_ops[operator](a, b)
        op_name = self._op_names[operator]

        # Create proof tree
        proof = ProofTree(query=f"{a} {operator} {b} = ?")

        # Step 1: Given operands (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"a={a}, b={b}",
            explanation_text=f"Gegeben: a={a} und b={b}",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"operands": [a, b]},
        )
        proof.add_root_step(step1)

        # Step 2: Perform comparison (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(a), str(b)],
            output=f"{a} {operator} {b}",
            explanation_text=f"Vergleiche: Ist {a} {op_name} {b}?",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={
                "rule": f"Arithmetik: Vergleich {operator}",
                "operator": operator,
            },
        )
        step1.add_subgoal(step2)

        # Step 3: Result (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{a} {operator} {b}"],
            output=str(result_value),
            explanation_text=f"Ergebnis: {a} {operator} {b} ist {result_value}",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"result": result_value},
        )
        step2.add_subgoal(step3)

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=result_value,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": "comparison",
                "operator": operator,
                "op_name": op_name,
            },
        )

    def transitive_inference(
        self, relations: List[Tuple[Any, str, Any]]
    ) -> List[Tuple[Any, str, Any]]:
        """
        Derive transitive relations (with multiple rounds)

        Example:
            Input: [(3, "<", 5), (5, "<", 7)]
            Output: [(3, "<", 7)]

        Supports transitive operators: <, >, <=, >=

        Args:
            relations: List of (a, operator, b) tuples

        Returns:
            List of derived relations
        """
        inferred = []
        transitive_ops = {"<", ">", "<=", ">="}

        # Combine existing and derived relations for multiple rounds
        all_relations = list(relations)
        max_rounds = self.config.max_transitive_rounds
        max_total = self.config.max_total_relations

        # Hard limit validation
        if len(all_relations) > max_total:
            logger.warning(
                "Input relations (%d) exceed max_total_relations (%d), truncating",
                len(all_relations),
                max_total,
            )
            all_relations = all_relations[:max_total]

        for round_num in range(max_rounds):
            new_in_round = []

            # Performance optimization: Track seen pairs to avoid recomputation
            seen_pairs = set()

            # Check all pairs of relations
            for i, (a1, op1, b1) in enumerate(all_relations):
                if op1 not in transitive_ops:
                    continue

                for j, (a2, op2, b2) in enumerate(all_relations):
                    if i >= j or op2 not in transitive_ops:
                        continue

                    # Avoid duplicate work
                    pair_key = (i, j)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # A < B AND B < C -> A < C (and variants)
                    if b1 == a2 and op1 == op2:
                        new_relation = (a1, op1, b2)
                        if (
                            new_relation not in relations
                            and new_relation not in inferred
                            and new_relation not in new_in_round
                        ):
                            new_in_round.append(new_relation)

                    # A > B AND B > C -> A > C (and variants)
                    elif a1 == b2 and op1 == op2:
                        new_relation = (a2, op1, b1)
                        if (
                            new_relation not in relations
                            and new_relation not in inferred
                            and new_relation not in new_in_round
                        ):
                            new_in_round.append(new_relation)

            # If no new relations, terminate early
            if not new_in_round:
                logger.debug(
                    "Transitive inference converged at round %d", round_num + 1
                )
                break

            # Hard limit check BEFORE adding
            if len(all_relations) + len(new_in_round) > max_total:
                logger.warning(
                    "Transitive inference hit max_total_relations limit at round %d",
                    round_num + 1,
                )
                # Add as many as fit
                remaining_space = max_total - len(all_relations)
                new_in_round = new_in_round[:remaining_space]
                inferred.extend(new_in_round)
                all_relations.extend(new_in_round)
                break

            # Add new relations
            inferred.extend(new_in_round)
            all_relations.extend(new_in_round)

        logger.info(
            "Transitive inference: %d input relations -> %d inferred relations (%d rounds)",
            len(relations),
            len(inferred),
            round_num + 1,
        )

        return inferred

    def build_transitive_proof(self, relations: List[Tuple[Any, str, Any]]):
        """
        Build proof tree for transitive inference

        Args:
            relations: List of (a, operator, b) tuples

        Returns:
            ArithmeticResult with derived relations and proof
        """
        inferred = self.transitive_inference(relations)

        # Create proof tree
        proof = ProofTree(query=f"Transitive Inferenz aus {len(relations)} Relationen")

        # Step 1: Given relations (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output="; ".join([f"{a} {op} {b}" for a, op, b in relations]),
            explanation_text=f"Gegeben: {len(relations)} Relationen",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"relations": relations},
        )
        proof.add_root_step(step1)

        # Step 2: Apply transitive rule (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(r) for r in relations],
            output="Transitivitätsregel",
            explanation_text="Wende Transitivitätsregel an: (A op B) AND (B op C) -> (A op C)",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"rule": "Transitivität"},
        )
        step1.add_subgoal(step2)

        # Step 3: Derived relations (CONCLUSION)
        if inferred:
            conclusion_text = "; ".join([f"{a} {op} {b}" for a, op, b in inferred])
        else:
            conclusion_text = "Keine neuen Relationen ableitbar"

        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=["Transitivitätsregel"],
            output=conclusion_text,
            explanation_text=f"Abgeleitet: {len(inferred)} neue Relationen",
            confidence=1.0,
            source_component="comparison_engine",
            metadata={"inferred": inferred},
        )
        step2.add_subgoal(step3)

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=inferred,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": "transitive_inference"},
        )


class PropertyChecker:
    """Check mathematical properties of numbers (thread-safe)"""

    def __init__(self, netzwerk: KonzeptNetzwerkCore, config: Optional[Any] = None):
        self.netzwerk = netzwerk
        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticConfig

        self.config = config or ArithmeticConfig()
        self._lock = threading.Lock()  # For Neo4j writes

    def is_even(self, n: int):
        """Check if number is even"""
        if not isinstance(n, int):
            raise ValueError(f"is_even benötigt Integer, nicht {type(n)}")

        result_value = n % 2 == 0

        # Create proof tree
        proof = ProofTree(query=f"Ist {n} gerade?")

        # Step 1: Number given (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"n={n}",
            explanation_text=f"Gegeben: Zahl n={n}",
            confidence=1.0,
            source_component="property_checker",
            metadata={"number": n},
        )
        proof.add_root_step(step1)

        # Step 2: Modulo calculation (RULE_APPLICATION)
        remainder = n % 2
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(n)],
            output=f"{n} % 2 = {remainder}",
            explanation_text=f"Berechne Modulo: {n} % 2 = {remainder}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "rule": "Definition: n ist gerade <=> n % 2 = 0",
                "remainder": remainder,
            },
        )
        step1.add_subgoal(step2)

        # Step 3: Result (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"{n} % 2 = {remainder}"],
            output=str(result_value),
            explanation_text=f"{n} ist {'gerade' if result_value else 'ungerade'}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "result": result_value,
                "property": "even" if result_value else "odd",
            },
        )
        step2.add_subgoal(step3)

        # Store in Neo4j
        self._persist_property(n, "gerade" if result_value else "ungerade")

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=result_value,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": "is_even",
                "property": "even" if result_value else "odd",
            },
        )

    def is_odd(self, n: int):
        """Check if number is odd"""
        result = self.is_even(n)
        # Negate result
        result.value = not result.value
        result.metadata["operation"] = "is_odd"
        result.metadata["property"] = "odd" if result.value else "even"

        # Update proof tree conclusion
        if result.proof_tree.root_steps:
            root = result.proof_tree.root_steps[0]
            if root.subgoals:
                for step in root.subgoals:
                    if step.subgoals:
                        conclusion = step.subgoals[0]
                        conclusion.explanation_text = (
                            f"{n} ist {'ungerade' if result.value else 'gerade'}"
                        )

        return result

    def is_prime(self, n: int):
        """Check if number is prime"""
        if not isinstance(n, int):
            raise ValueError(f"is_prime benötigt Integer, nicht {type(n)}")

        # Prime check
        if n < 2:
            return self._build_prime_result(n, False, reason=f"{n} < 2")

        if n == 2:
            return self._build_prime_result(
                n, True, reason="2 ist die kleinste Primzahl"
            )

        if n % 2 == 0:
            return self._build_prime_result(n, False, divisor=2)

        # Check odd divisors up to √n
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return self._build_prime_result(n, False, divisor=i)

        return self._build_prime_result(n, True)

    def _build_prime_result(
        self, n: int, is_prime: bool, divisor: int = None, reason: str = None
    ):
        """Create ArithmeticResult for prime check"""
        proof = ProofTree(query=f"Ist {n} eine Primzahl?")

        # Step 1: Number given (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"n={n}",
            explanation_text=f"Gegeben: Zahl n={n}",
            confidence=1.0,
            source_component="property_checker",
            metadata={"number": n},
        )
        proof.add_root_step(step1)

        # Step 2: Check prime criterion (RULE_APPLICATION)
        if reason:
            explanation = reason
        elif divisor:
            explanation = f"{n} ist durch {divisor} teilbar: {n} % {divisor} = 0"
        else:
            explanation = f"Keine Teiler zwischen 2 und {int(n**0.5)} gefunden"

        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(n)],
            output=explanation,
            explanation_text=f"Prüfe Primzahl-Kriterium: {explanation}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "rule": "Definition: Primzahl hat nur 1 und sich selbst als Teiler",
                "divisor": divisor,
            },
        )
        step1.add_subgoal(step2)

        # Step 3: Result (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[explanation],
            output=str(is_prime),
            explanation_text=f"{n} ist {'eine Primzahl' if is_prime else 'keine Primzahl'}",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "result": is_prime,
                "property": "prime" if is_prime else "composite",
            },
        )
        step2.add_subgoal(step3)

        # Store in Neo4j
        if is_prime:
            self._persist_property(n, "primzahl")

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=is_prime,
            proof_tree=proof,
            confidence=1.0,
            metadata={
                "operation": "is_prime",
                "property": "prime" if is_prime else "composite",
            },
        )

    def find_divisors(self, n: int):
        """Find all divisors of a number"""
        if not isinstance(n, int):
            raise ValueError(f"find_divisors benötigt Integer, nicht {type(n)}")

        if n == 0:
            raise ValueError("0 hat unendlich viele Teiler")

        n_abs = abs(n)
        divisors = [i for i in range(1, n_abs + 1) if n_abs % i == 0]

        # Create proof tree
        proof = ProofTree(query=f"Finde alle Teiler von {n}")

        # Step 1: Number given (PREMISE)
        step1 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.PREMISE,
            inputs=[],
            output=f"n={n}",
            explanation_text=f"Gegeben: Zahl n={n}",
            confidence=1.0,
            source_component="property_checker",
            metadata={"number": n},
        )
        proof.add_root_step(step1)

        # Step 2: Search divisors (RULE_APPLICATION)
        step2 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.RULE_APPLICATION,
            inputs=[str(n)],
            output=f"Prüfe alle Zahlen von 1 bis {n_abs}",
            explanation_text=f"Suche Teiler: Prüfe i in [1, {n_abs}] mit {n_abs} % i = 0",
            confidence=1.0,
            source_component="property_checker",
            metadata={
                "rule": "Definition: d teilt n <=> n % d = 0",
                "range": [1, n_abs],
            },
        )
        step1.add_subgoal(step2)

        # Step 3: Result (CONCLUSION)
        step3 = ProofStep(
            step_id=str(uuid.uuid4()),
            step_type=StepType.CONCLUSION,
            inputs=[f"Prüfe 1 bis {n_abs}"],
            output=str(divisors),
            explanation_text=f"Teiler von {n}: {divisors} ({len(divisors)} Teiler)",
            confidence=1.0,
            source_component="property_checker",
            metadata={"result": divisors, "count": len(divisors)},
        )
        step2.add_subgoal(step3)

        # Store in Neo4j
        self._persist_divisors(n, divisors)

        # Import here to avoid circular dependency
        from component_52_arithmetic_engine import ArithmeticResult

        return ArithmeticResult(
            value=divisors,
            proof_tree=proof,
            confidence=1.0,
            metadata={"operation": "find_divisors", "count": len(divisors)},
        )

    def _validate_entity_name(self, name: str) -> None:
        """Validate entity name for Neo4j safety"""
        if not ENTITY_NAME_PATTERN.match(name.lower()):
            raise ValueError(f"Invalid entity name for Neo4j: {name}")

    def _validate_number_string(self, num_str: str) -> None:
        """Validate number string"""
        if not NUMBER_PATTERN.match(num_str):
            raise ValueError(f"Invalid number string: {num_str}")

    def _persist_property(self, number: int, property_name: str):
        """Store property in Neo4j (thread-safe)"""
        if not self.config.persist_to_neo4j:
            return

        with self._lock:
            try:
                # Validate inputs
                self._validate_entity_name(property_name)
                self._validate_number_string(str(number))

                # Create word for number
                number_word = self.netzwerk.get_or_create_wort(str(number), pos="NUM")

                # Create property relation
                self.netzwerk.create_relation(
                    start_node_id=number_word,
                    end_node_id=self.netzwerk.get_or_create_wort(
                        property_name, pos="ADJ"
                    ),
                    relation_type="HAS_PROPERTY",
                    confidence=1.0,
                    provenance="arithmetic_reasoning",
                )
                logger.debug("Persisted property: %d -> %s", number, property_name)
            except ValueError as e:
                logger.error("Validation error in _persist_property: %s", e)
                return  # Don't crash, just skip persistence
            except ServiceUnavailable as e:
                logger.error(
                    "Neo4j service unavailable for property persistence: %s", e
                )
                raise  # Re-raise critical errors
            except Neo4jError as e:
                logger.warning(
                    "Neo4j error persisting property %s for %d: %s",
                    property_name,
                    number,
                    e,
                )
            except (TypeError, AttributeError) as e:
                logger.error(
                    "Invalid data for property persistence: %s", e, exc_info=True
                )

    def _persist_divisors(self, number: int, divisors: List[int]):
        """Store divisor relations in Neo4j (thread-safe)"""
        if not self.config.persist_to_neo4j:
            return

        with self._lock:
            try:
                # Validate inputs
                self._validate_number_string(str(number))

                number_word = self.netzwerk.get_or_create_wort(str(number), pos="NUM")

                for divisor in divisors:
                    self._validate_number_string(str(divisor))
                    divisor_word = self.netzwerk.get_or_create_wort(
                        str(divisor), pos="NUM"
                    )
                    self.netzwerk.create_relation(
                        start_node_id=divisor_word,
                        end_node_id=number_word,
                        relation_type="DIVIDES",
                        confidence=1.0,
                        provenance="arithmetic_reasoning",
                    )
                logger.debug("Persisted %d divisors for %d", len(divisors), number)
            except ValueError as e:
                logger.error("Validation error in _persist_divisors: %s", e)
                return
            except ServiceUnavailable as e:
                logger.error("Neo4j service unavailable for divisor persistence: %s", e)
                raise
            except Neo4jError as e:
                logger.warning("Neo4j error persisting divisors for %d: %s", number, e)
            except (TypeError, AttributeError) as e:
                logger.error(
                    "Invalid data for divisor persistence: %s", e, exc_info=True
                )
