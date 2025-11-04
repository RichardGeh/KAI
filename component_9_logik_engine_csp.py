# component_9_logik_engine_csp.py
"""
Constraint Satisfaction Problem (CSP) Integration für Logic Engine.

Erweitert die Engine-Klasse um CSP-basiertes Backward-Chaining,
das Constraint-Validierung für Variablen-Zuweisungen nutzt.
"""

from typing import Any, Dict, List, Optional, Set
from component_15_logging_config import get_logger

# Import Constraint Reasoning Engine
try:
    from component_29_constraint_reasoning import (
        Variable,
        ConstraintProblem,
        ConstraintSolver,
    )

    CONSTRAINT_REASONING_AVAILABLE = True
except ImportError:
    CONSTRAINT_REASONING_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning("Constraint Reasoning Engine nicht verfügbar")

logger = get_logger(__name__)


class CSPReasoningMixin:
    """
    Mixin für CSP-basiertes Reasoning.

    Fügt der Engine constraint-basierte Solving-Fähigkeiten hinzu.
    """

    def solve_with_constraints(
        self,
        goal,  # Goal from core module
        constraints: Optional[List] = None,
        max_depth: int = 5,
    ):
        """
        Backward-Chaining mit Constraint-Validierung.

        Nutzt CSP-Solver für Variablen-Zuweisungen,
        die alle Constraints erfüllen.

        Args:
            goal: Das zu beweisende Ziel (kann Variablen enthalten)
            constraints: Liste von Constraints (component_29) oder None
            max_depth: Maximale Rekursionstiefe für Backward-Chaining

        Returns:
            ProofStep mit Constraint-Lösung oder None

        Beispiel:
            goal = Goal(pred="HAS_PROPERTY", args={"subject": "?x", "object": "rot"})
            constraints = [not_equal_constraint("?x", "?y"), ...]
            proof = engine.solve_with_constraints(goal, constraints)
        """
        if not CONSTRAINT_REASONING_AVAILABLE:
            logger.warning(
                "Constraint Reasoning nicht verfügbar - nutze normales Backward-Chaining"
            )
            return self.prove_goal(goal, max_depth)

        if constraints is None:
            constraints = []

        logger.info(f"[BC+CSP] Beweise Goal mit Constraints: {goal.pred} {goal.args}")

        # 1. Extrahiere Variablen aus Goal
        variables_dict = {}
        variable_positions = {}  # Mapping: variable_name -> (arg_key, position)

        for arg_key, arg_value in goal.args.items():
            if isinstance(arg_value, str) and arg_value.startswith("?"):
                var_name = arg_value
                if var_name not in variables_dict:
                    # Baue Domain aus Faktenbasis (berücksichtige konkrete Werte)
                    domain = self._build_domain_from_facts_constrained(
                        goal.pred, arg_key, goal.args
                    )
                    if domain:
                        variables_dict[var_name] = Variable(
                            name=var_name, domain=domain
                        )
                        variable_positions[var_name] = arg_key
                        logger.debug(f"Variable {var_name} hat Domain: {domain}")
                    else:
                        logger.warning(f"Keine Domain für Variable {var_name} gefunden")
                        return None

        # Keine Variablen gefunden → normales Backward-Chaining
        if not variables_dict:
            logger.info("Keine Variablen im Goal - nutze normales Backward-Chaining")
            return self.prove_goal(goal, max_depth)

        logger.info(
            f"Extrahierte {len(variables_dict)} Variablen: {list(variables_dict.keys())}"
        )

        # 2. Konvertiere implizite Fakten → Constraints (optional)
        fact_constraints = self._build_constraints_from_facts(goal, variables_dict)
        all_constraints = constraints + fact_constraints

        logger.info(
            f"Gesamt {len(all_constraints)} Constraints ({len(constraints)} explizit, {len(fact_constraints)} implizit)"
        )

        # 3. Erstelle ConstraintProblem und nutze ConstraintSolver
        problem = ConstraintProblem(
            name=f"CSP_{goal.pred}_{goal.id}",
            variables=variables_dict,
            constraints=all_constraints,
        )

        solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)
        solution, csp_proof_tree = solver.solve(problem, track_proof=True)

        if not solution:
            logger.info("CSP-Solver fand keine Lösung")
            return None

        # 4. Baue ProofStep mit Lösung
        logger.info(f"CSP-Lösung gefunden: {solution}")

        # Erstelle Fakten aus Lösung
        supporting_facts = []
        bindings = {}

        for var_name, value in solution.items():
            bindings[var_name] = value
            # Finde zugehörigen Fakt in Faktenbasis
            arg_key = variable_positions.get(var_name)
            if arg_key:
                fact = self._find_fact_for_assignment(goal.pred, arg_key, value)
                if fact:
                    supporting_facts.append(fact)

        # Erstelle Ziel-Goal mit konkreten Werten (für Verifikation)
        resolved_args = {}
        for arg_key, arg_value in goal.args.items():
            if isinstance(arg_value, str) and arg_value.startswith("?"):
                resolved_args[arg_key] = bindings.get(arg_value, arg_value)
            else:
                resolved_args[arg_key] = arg_value

        # Import Goal and ProofStep from core
        from component_9_logik_engine_core import Goal as GoalClass, ProofStep

        resolved_goal = GoalClass(
            pred=goal.pred,
            args=resolved_args,
            depth=goal.depth,
            parent_id=goal.parent_id,
        )

        proof = ProofStep(
            goal=goal,
            method="constraint_satisfaction",
            bindings=bindings,
            supporting_facts=supporting_facts,
            confidence=1.0,  # CSP-Lösungen sind deterministisch
        )

        # Optional: Verifikation durch normales Backward-Chaining
        if self._verify_csp_solution(resolved_goal, max_depth):
            logger.info("CSP-Lösung erfolgreich verifiziert")
            proof.confidence = 1.0
        else:
            logger.warning(
                "CSP-Lösung konnte nicht verifiziert werden - reduziere Confidence"
            )
            proof.confidence = 0.8

        return proof

    def _build_domain_from_facts(self, pred: str, arg_key: str) -> Set[Any]:
        """
        Baut Domain für eine Variable aus der Faktenbasis.

        Durchsucht alle Fakten mit passendem Prädikat und sammelt
        mögliche Werte für das gegebene Argument.

        Args:
            pred: Prädikat des Goals (z.B. "HAS_PROPERTY")
            arg_key: Argument-Schlüssel (z.B. "subject", "object")

        Returns:
            Set möglicher Werte für die Variable
        """
        domain = set()
        all_facts = self.wm + self.kb

        for fact in all_facts:
            if fact.pred == pred and arg_key in fact.args:
                domain.add(fact.args[arg_key])

        # Falls keine Fakten gefunden: Versuche aus Neo4j-Graph zu laden
        if not domain and self.netzwerk:
            domain = self._build_domain_from_graph(pred, arg_key)

        return domain

    def _build_domain_from_facts_constrained(
        self, pred: str, arg_key: str, all_args: Dict[str, Any]
    ) -> Set[Any]:
        """
        Baut Domain für eine Variable aus der Faktenbasis,
        berücksichtigt konkrete Werte in anderen Argumenten.

        Args:
            pred: Prädikat des Goals
            arg_key: Argument-Schlüssel für die Variable
            all_args: Alle Argumente des Goals (mit Variablen und konkreten Werten)

        Returns:
            Set möglicher Werte, die mit konkreten Werten kompatibel sind

        Beispiel:
            Goal: HAS_PROPERTY(subject=?x, object="rot")
            → Domain für ?x: nur Subjects die "rot" als object haben
        """
        domain = set()
        all_facts = self.wm + self.kb

        # Extrahiere konkrete Constraints aus all_args
        concrete_constraints = {}
        for key, value in all_args.items():
            if not (isinstance(value, str) and value.startswith("?")):
                # Konkreter Wert
                concrete_constraints[key] = value

        # Suche Fakten, die alle konkreten Constraints erfüllen
        for fact in all_facts:
            if fact.pred != pred:
                continue

            # Prüfe ob Fakt alle konkreten Constraints erfüllt
            matches = True
            for constraint_key, constraint_value in concrete_constraints.items():
                if (
                    constraint_key not in fact.args
                    or fact.args[constraint_key] != constraint_value
                ):
                    matches = False
                    break

            if matches and arg_key in fact.args:
                # Dieser Fakt erfüllt alle Constraints
                domain.add(fact.args[arg_key])

        # Falls keine Fakten gefunden: Versuche aus Neo4j-Graph zu laden
        if not domain and self.netzwerk:
            # TODO: Erweitere _build_domain_from_graph um Constraints
            domain = self._build_domain_from_graph(pred, arg_key)

        return domain

    def _build_domain_from_graph(self, pred: str, arg_key: str) -> Set[Any]:
        """
        Baut Domain aus Neo4j-Graph für relationale Prädikate.

        Args:
            pred: Prädikat (z.B. "IS_A", "HAS_PROPERTY")
            arg_key: Argument-Schlüssel ("subject" oder "object")

        Returns:
            Set möglicher Werte aus dem Graphen
        """
        domain = set()

        if not self.netzwerk or not self.netzwerk.driver:
            return domain

        # Nur für bekannte relationale Prädikate
        if pred not in ["IS_A", "HAS_PROPERTY", "PART_OF", "LOCATED_IN", "CAPABLE_OF"]:
            return domain

        try:
            with self.netzwerk.driver.session(database="neo4j") as session:
                if arg_key == "subject":
                    # Hole alle Source-Konzepte mit dieser Relation
                    query = f"""
                        MATCH (start:Konzept)-[r:{pred}]->()
                        RETURN DISTINCT start.name AS value
                        LIMIT 100
                    """
                elif arg_key == "object":
                    # Hole alle Target-Konzepte mit dieser Relation
                    query = f"""
                        MATCH ()-[r:{pred}]->(end:Konzept)
                        RETURN DISTINCT end.name AS value
                        LIMIT 100
                    """
                else:
                    return domain

                result = session.run(query)
                for record in result:
                    if record["value"]:
                        domain.add(record["value"])

        except Exception as e:
            logger.warning(f"Fehler beim Laden der Domain aus Graph: {e}")

        return domain

    def _build_constraints_from_facts(self, goal, variables: Dict[str, Any]) -> List:
        """
        Baut zusätzliche Constraints aus der Faktenbasis.

        Analysiert Fakten, um implizite Constraints zu extrahieren.
        Beispiel: Wenn zwei Eigenschaften nie zusammen auftreten,
        erstelle NOT-EQUAL Constraint.

        Args:
            goal: Das Goal
            variables: Dict von Variable-Objekten

        Returns:
            Liste von Constraints (component_29.Constraint)
        """
        constraints = []

        # TODO: Erweiterte Constraint-Extraktion für Phase 2
        # Für Phase 1.2 erstmal nur grundlegende Constraints

        # Beispiel: Wenn Goal nach "HAS_PROPERTY" fragt,
        # stelle sicher, dass subject ein valides Konzept ist
        # (wird bereits durch Domain-Building abgedeckt)

        return constraints

    def _find_fact_for_assignment(
        self, pred: str, arg_key: str, value: Any
    ):  # Returns Optional[Fact]
        """
        Findet Fakt in Faktenbasis, der einer CSP-Zuweisung entspricht.

        Args:
            pred: Prädikat (z.B. "HAS_PROPERTY")
            arg_key: Argument-Key (z.B. "subject")
            value: Zugewiesener Wert

        Returns:
            Passender Fakt oder None
        """
        all_facts = self.wm + self.kb

        for fact in all_facts:
            if fact.pred == pred and arg_key in fact.args:
                if fact.args[arg_key] == value:
                    return fact

        return None

    def _verify_csp_solution(self, resolved_goal, max_depth: int) -> bool:
        """
        Verifiziert CSP-Lösung durch normales Backward-Chaining.

        Args:
            resolved_goal: Goal mit konkreten Werten (keine Variablen)
            max_depth: Maximale Rekursionstiefe

        Returns:
            True wenn Lösung verifizierbar, False sonst
        """
        try:
            proof = self.prove_goal(resolved_goal, max_depth=max_depth)
            return proof is not None
        except Exception as e:
            logger.warning(f"Fehler bei CSP-Verifikation: {e}")
            return False
