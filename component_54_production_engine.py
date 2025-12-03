"""
Component 54: Production System - Engine

ProductionSystemEngine class implementing the Recognize-Act Cycle.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from component_17_proof_explanation import ProofStep, ProofTree, StepType
from component_54_production_rule import ProductionRule
from component_54_production_state import ResponseGenerationState
from component_54_production_types import RuleCategory
from infrastructure.interfaces import BaseReasoningEngine, ReasoningResult


class ProductionSystemEngine(BaseReasoningEngine):
    """
    Recognize-Act Cycle Engine für das Produktionssystem.

    Implementiert den klassischen Produktionssystem-Zyklus:
    1. Match: Finde alle anwendbaren Regeln
    2. Conflict Resolution: Wähle beste Regel
    3. Act: Wende Regel an
    4. Repeat: Bis Ziel erreicht oder Max-Cycles

    Usage:
        engine = ProductionSystemEngine()
        engine.add_rule(my_rule)
        response = engine.generate(initial_state)
    """

    def __init__(
        self, signals: Optional[Any] = None, neo4j_repository: Optional[Any] = None
    ):
        """
        Initialisiert das Produktionssystem.

        Args:
            signals: Optional KaiSignals Objekt für UI-Integration (PHASE 5)
            neo4j_repository: Optional KonzeptNetzwerkProductionRules für Persistierung (PHASE 9)
        """
        self.rules: List[ProductionRule] = []
        self.logger = logging.getLogger("ProductionSystemEngine")
        self.signals = signals  # Optional für UI-Updates
        self.neo4j_repository = neo4j_repository  # Optional für Persistierung (PHASE 9)
        self._query_counter = 0  # Zähler für periodisches Sync

    def add_rule(self, rule: ProductionRule) -> None:
        """Fügt eine Produktionsregel hinzu."""
        self.rules.append(rule)
        self.logger.debug(f"Added rule: {rule.name} (category={rule.category.value})")

    def add_rules(self, rules: List[ProductionRule]) -> None:
        """Fügt mehrere Regeln hinzu."""
        for rule in rules:
            self.add_rule(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """
        Entfernt eine Regel.

        Returns:
            True wenn Regel gefunden und entfernt, False sonst
        """
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        removed = len(self.rules) < initial_count
        if removed:
            self.logger.debug(f"Removed rule: {rule_name}")
        return removed

    def match_rules(self, state: ResponseGenerationState) -> List[ProductionRule]:
        """
        Match Phase: Findet alle anwendbaren Regeln.

        Returns:
            Liste von Regeln, deren Conditions erfüllt sind
        """
        matching_rules = [rule for rule in self.rules if rule.matches(state)]
        self.logger.debug(
            f"Matched {len(matching_rules)} rules out of {len(self.rules)}"
        )
        return matching_rules

    def resolve_conflict(
        self, matching_rules: List[ProductionRule]
    ) -> Optional[ProductionRule]:
        """
        Conflict Resolution: Wählt beste Regel aus.

        Strategie: Utility * Specificity (höher ist besser)
        Bei Gleichstand: Erste Regel gewinnt (FIFO)

        Returns:
            Beste Regel oder None wenn keine vorhanden
        """
        if not matching_rules:
            return None

        # Sortiere nach Priorität (absteigend)
        sorted_rules = sorted(
            matching_rules, key=lambda r: r.get_priority(), reverse=True
        )
        best_rule = sorted_rules[0]

        self.logger.debug(
            f"Conflict resolution: Selected {best_rule.name} "
            f"(priority={best_rule.get_priority():.2f}, utility={best_rule.utility:.2f}, "
            f"specificity={best_rule.specificity:.2f})"
        )

        return best_rule

    def apply_rule(self, rule: ProductionRule, state: ResponseGenerationState) -> None:
        """
        Act Phase: Wendet Regel auf State an.

        PHASE 6: Erweitert um ProofStep-Generierung

        Args:
            rule: Anzuwendende Regel
            state: Working Memory State (wird modifiziert)
        """
        # PHASE 6: State VORHER erfassen (für ProofStep inputs)
        state_before = state.to_serializable_snapshot()

        # Regel anwenden
        rule.apply(state)
        state.cycle_count += 1

        # PHASE 6: State NACHHER erfassen (für ProofStep outputs)
        state_after = state.to_serializable_snapshot()

        # PHASE 6: ProofStep erstellen und zum ProofTree hinzufügen
        if state.proof_tree is not None:
            proof_step = self._create_proof_step_for_rule(
                rule=rule,
                state_before=state_before,
                state_after=state_after,
                cycle=state.cycle_count,
            )
            state.proof_tree.add_root_step(proof_step)

        # PHASE 5: Emit Signal für UI-Update (wenn verfügbar)
        if self.signals is not None and hasattr(
            self.signals, "production_system_trace"
        ):
            try:
                # Erstelle Beschreibung aus Metadata
                description = rule.metadata.get("description", "Regel angewendet")
                category = rule.category.value

                self.signals.production_system_trace.emit(
                    rule.name, f"[{category}] {description}"
                )
            except Exception as e:
                self.logger.debug(f"Could not emit production_system_trace signal: {e}")

    def _create_proof_step_for_rule(
        self,
        rule: ProductionRule,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        cycle: int,
    ) -> ProofStep:
        """
        Erstellt einen ProofStep für eine Regelanwendung.

        PHASE 6: ProofTree Integration

        Args:
            rule: Die angewendete Regel
            state_before: State-Snapshot vor Regelanwendung
            state_after: State-Snapshot nach Regelanwendung
            cycle: Aktueller Zyklus

        Returns:
            ProofStep für diese Regelanwendung
        """
        # Erstelle Beschreibung
        description = rule.metadata.get("description", "Regel angewendet")
        category = rule.category.value

        # Identifiziere Änderungen
        changes = []
        if state_after["num_sentences"] > state_before["num_sentences"]:
            changes.append(
                f"+{state_after['num_sentences'] - state_before['num_sentences']} Satz/Sätze"
            )
        if state_after["num_pending_facts"] != state_before["num_pending_facts"]:
            diff = state_after["num_pending_facts"] - state_before["num_pending_facts"]
            changes.append(f"{'+'if diff > 0 else ''}{diff} pending facts")
        if state_after["mentioned_entities"] != state_before["mentioned_entities"]:
            new_entities = set(state_after["mentioned_entities"]) - set(
                state_before["mentioned_entities"]
            )
            if new_entities:
                changes.append(f"Erwähnt: {', '.join(list(new_entities)[:3])}")

        changes_str = "; ".join(changes) if changes else "Keine sichtbaren Änderungen"

        # Erstelle explanation_text
        explanation_text = (
            f"Zyklus {cycle}: Regel '{rule.name}' [{category}] angewendet. "
            f"{description}. Änderungen: {changes_str}"
        )

        # Erstelle ProofStep
        proof_step = ProofStep(
            step_id=f"production_rule_{cycle}_{uuid.uuid4().hex[:8]}",
            step_type=StepType.RULE_APPLICATION,
            inputs=[f"State (Zyklus {cycle-1}): {str(state_before)}"],
            rule_name=rule.name,
            output=f"State (Zyklus {cycle}): {changes_str}",
            confidence=1.0,  # Regelanwendung deterministisch
            explanation_text=explanation_text,
            metadata={
                "rule_category": category,
                "rule_utility": rule.utility,
                "rule_specificity": rule.specificity,
                "rule_priority": rule.get_priority(),
                "cycle": cycle,
                "state_before": state_before,
                "state_after": state_after,
                "changes": changes,
            },
            source_component="component_54_production_system",
        )

        return proof_step

    def load_rules_from_neo4j(self) -> int:
        """
        Lädt alle Production Rules aus Neo4j Repository.

        PHASE 9: Neo4j Rule Repository Integration

        Returns:
            int: Anzahl geladener Regeln
        """
        if self.neo4j_repository is None:
            self.logger.debug("load_rules_from_neo4j: Kein Repository verfügbar")
            return 0

        try:
            rule_datas = self.neo4j_repository.get_all_production_rules()

            if not rule_datas:
                self.logger.info(
                    "load_rules_from_neo4j: Keine Regeln in Neo4j gefunden"
                )
                return 0

            # NOTE: Wir können keine serialisierten Callables aus Neo4j laden
            # (condition_code/action_code), daher laden wir nur Metadaten.
            # Die eigentlichen Regeln müssen via create_production_rules() Factory
            # neu erstellt werden.
            #
            # Stattdessen synchronisieren wir nur die Stats von existierenden Regeln.
            self._sync_stats_from_neo4j(rule_datas)

            self.logger.info(
                f"load_rules_from_neo4j: Stats für {len(rule_datas)} Regeln synchronisiert"
            )
            return len(rule_datas)

        except Exception as e:
            self.logger.error(
                "load_rules_from_neo4j: Fehler beim Laden",
                extra={"error": str(e)},
                exc_info=True,
            )
            return 0

    def _sync_stats_from_neo4j(self, rule_datas: List[Dict[str, Any]]) -> None:
        """
        Synchronisiert Stats von Neo4j zu in-memory Regeln.

        Args:
            rule_datas: Liste von Rule-Dicts aus Neo4j
        """
        rule_stats_map = {rd["name"]: rd for rd in rule_datas}

        for rule in self.rules:
            if rule.name in rule_stats_map:
                stats = rule_stats_map[rule.name]
                rule.application_count = stats["application_count"]
                rule.success_count = stats["success_count"]
                if stats["last_applied"]:
                    # Konvertiere milliseconds timestamp zu datetime
                    rule.last_applied = datetime.fromtimestamp(
                        stats["last_applied"] / 1000
                    )

                self.logger.debug(
                    f"_sync_stats_from_neo4j: Synced stats for '{rule.name}' "
                    f"(app={rule.application_count}, succ={rule.success_count})"
                )

    def sync_rule_stats_to_neo4j(self, force: bool = False) -> bool:
        """
        Synchronisiert Regel-Statistiken zu Neo4j.

        PHASE 9: Periodisches Sync (alle 10 Queries) oder force=True

        Args:
            force: Sofort synchronisieren (überspringt Batching)

        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        if self.neo4j_repository is None:
            return True  # Kein Fehler, nur kein Repository

        try:
            for rule in self.rules:
                self.neo4j_repository.update_production_rule_stats(
                    name=rule.name,
                    application_count=rule.application_count,
                    success_count=rule.success_count,
                    last_applied=rule.last_applied,
                    force_sync=force,
                )

            # Flush wenn force=True
            if force and hasattr(self.neo4j_repository, "_flush_pending_stats"):
                self.neo4j_repository._flush_pending_stats()

            self.logger.debug(
                f"sync_rule_stats_to_neo4j: {len(self.rules)} Regeln synchronisiert "
                f"(force={force})"
            )
            return True

        except Exception as e:
            self.logger.error(
                "sync_rule_stats_to_neo4j: Fehler beim Synchronisieren",
                extra={"error": str(e)},
                exc_info=True,
            )
            return False

    def save_rules_to_neo4j(self) -> int:
        """
        Speichert alle in-memory Regeln zu Neo4j (initial setup).

        PHASE 9: Wird beim ersten Start aufgerufen um Regeln zu persistieren.

        Returns:
            int: Anzahl gespeicherter Regeln
        """
        if self.neo4j_repository is None:
            self.logger.debug("save_rules_to_neo4j: Kein Repository verfügbar")
            return 0

        saved_count = 0

        try:
            for rule in self.rules:
                # Serialisiere Callables (HINWEIS: pickle ist nicht ideal für
                # Code-Persistierung, aber für Prototyp akzeptabel)
                import pickle

                condition_code = pickle.dumps(rule.condition).hex()
                action_code = pickle.dumps(rule.action).hex()

                success = self.neo4j_repository.create_production_rule(
                    name=rule.name,
                    category=rule.category.value,
                    condition_code=condition_code,
                    action_code=action_code,
                    utility=rule.utility,
                    specificity=rule.specificity,
                    metadata=rule.metadata,
                )

                if success:
                    saved_count += 1

            self.logger.info(
                f"save_rules_to_neo4j: {saved_count}/{len(self.rules)} Regeln gespeichert"
            )
            return saved_count

        except Exception as e:
            self.logger.error(
                "save_rules_to_neo4j: Fehler beim Speichern",
                extra={"error": str(e)},
                exc_info=True,
            )
            return saved_count

    def generate(self, state: ResponseGenerationState) -> ResponseGenerationState:
        """
        Hauptschleife: Recognize-Act Cycle bis Ziel erreicht.

        PHASE 6: Erweitert um ProofTree-Generierung

        Args:
            state: Initialer ResponseGenerationState

        Returns:
            Finaler ResponseGenerationState mit generiertem Text und ProofTree
        """
        self.logger.info(
            f"Starting response generation (goal={state.primary_goal.goal_type.value})"
        )

        # PHASE 6: Initialisiere ProofTree
        if state.proof_tree is None:
            query_text = state.current_query or "Response Generation"
            state.proof_tree = ProofTree(
                query=query_text,
                metadata={
                    "goal_type": state.primary_goal.goal_type.value,
                    "max_cycles": state.max_cycles,
                    "component": "component_54_production_system",
                },
            )
            self.logger.debug(f"Initialized ProofTree for query: {query_text}")

        while state.cycle_count < state.max_cycles:
            # Check if goal completed
            if state.is_goal_completed():
                self.logger.info(f"Goal completed after {state.cycle_count} cycles")
                break

            # 1. Match Phase
            matching_rules = self.match_rules(state)

            if not matching_rules:
                self.logger.warning(
                    f"No matching rules at cycle {state.cycle_count}, stopping"
                )
                break

            # 2. Conflict Resolution
            selected_rule = self.resolve_conflict(matching_rules)

            if selected_rule is None:
                self.logger.error(
                    "Conflict resolution returned None despite matching rules"
                )
                break

            # 3. Act Phase
            self.apply_rule(selected_rule, state)

            self.logger.debug(
                f"Cycle {state.cycle_count}: Applied {selected_rule.name}, "
                f"sentences={len(state.text.completed_sentences)}"
            )

        # Final check
        if state.cycle_count >= state.max_cycles:
            self.logger.warning(
                f"Reached max cycles ({state.max_cycles}) without completing goal"
            )

        self.logger.info(
            f"Generation finished: {len(state.text.completed_sentences)} sentences, "
            f"{state.cycle_count} cycles"
        )

        # PHASE 9: Periodisches Sync zu Neo4j (alle 10 Queries)
        self._query_counter += 1
        if self._query_counter >= 10:
            self.sync_rule_stats_to_neo4j(force=True)
            self._query_counter = 0

        return state

    def get_rules_by_category(self, category: RuleCategory) -> List[ProductionRule]:
        """Gibt alle Regeln einer Kategorie zurück."""
        return [r for r in self.rules if r.category == category]

    def get_rule_by_name(self, name: str) -> Optional[ProductionRule]:
        """Findet eine Regel nach Name."""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über das Regelsystem zurück.

        Returns:
            Dict mit Statistiken (Anzahl Regeln, Kategorien, Anwendungen, etc.)
        """
        total_applications = sum(r.application_count for r in self.rules)
        category_counts = {}
        for category in RuleCategory:
            category_counts[category.value] = len(self.get_rules_by_category(category))

        return {
            "total_rules": len(self.rules),
            "rules_by_category": category_counts,
            "total_applications": total_applications,
            "most_used_rule": (
                max(self.rules, key=lambda r: r.application_count).name
                if self.rules
                else None
            ),
        }

    # ========================================================================
    # BaseReasoningEngine Interface Implementation
    # ========================================================================

    def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Execute production system reasoning for response generation.

        Args:
            query: Natural language query
            context: Context with:
                - "state": ResponseGenerationState (required)
                - "max_cycles": Maximum number of production cycles (default: 100)
                - "goal": Goal to achieve (optional, default from state)

        Returns:
            ReasoningResult with generated response and proof tree
        """
        try:
            # Extract state from context (required)
            state = context.get("state")
            if not state or not isinstance(state, ResponseGenerationState):
                return ReasoningResult(
                    success=False,
                    answer="Production system requires ResponseGenerationState in context",
                    confidence=0.0,
                    strategy_used="production_system",
                )

            # Set max cycles if provided
            max_cycles = context.get("max_cycles", 100)
            state.max_cycles = max_cycles

            # Run production system
            final_state = self.generate(state)

            # Extract result
            success = final_state.goal_achieved
            response_text = final_state.current_response
            confidence = final_state.confidence

            # Build answer
            if success:
                answer = response_text
            else:
                answer = (
                    response_text if response_text else "Failed to generate response"
                )

            return ReasoningResult(
                success=success,
                answer=answer,
                confidence=confidence,
                proof_tree=final_state.proof_tree,
                strategy_used="production_system_recognize_act",
                metadata={
                    "cycles": final_state.cycle_count,
                    "rules_applied": len(final_state.applied_rules),
                    "goal_achieved": final_state.goal_achieved,
                    "response_quality": final_state.response_quality,
                },
            )

        except Exception as e:
            self.logger.error(
                "Error in production system reasoning",
                extra={"query": query, "error": str(e)},
                exc_info=True,
            )
            return ReasoningResult(
                success=False,
                answer=f"Production system error: {str(e)}",
                confidence=0.0,
                strategy_used="production_system",
            )

    def get_capabilities(self) -> List[str]:
        """Return list of reasoning capabilities."""
        return [
            "production_rules",
            "response_generation",
            "conflict_resolution",
            "recognize_act_cycle",
            "rule_based_reasoning",
            "discourse_management",
            "content_generation",
            "lexical_selection",
            "syntactic_structuring",
        ]

    def estimate_cost(self, query: str) -> float:
        """
        Estimate computational cost for production system reasoning.

        Returns:
            Cost estimate in [0.0, 1.0] range
            Base cost: 0.5 (medium, depends on number of rules and cycles)
        """
        # Production system cost depends on:
        # - Number of rules (more rules = higher matching cost)
        # - Number of cycles (iterative process)
        # - Conflict resolution overhead
        # - ProofTree generation
        base_cost = 0.5

        # Rule count factor (more rules = higher cost)
        if self.rules:
            rule_factor = min(len(self.rules) / 100.0, 0.2)
        else:
            rule_factor = 0.0

        # Query complexity
        query_complexity = min(len(query) / 400.0, 0.1)

        return min(base_cost + rule_factor + query_complexity, 1.0)


# ============================================================================
# Utility Functions
# ============================================================================
