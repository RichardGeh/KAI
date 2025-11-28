# component_16_probabilistic_engine.py
"""
Probabilistic Reasoning Engine für KAI

Implementiert Bayesian Inference und Unsicherheitspropagierung für robustes Schließen
unter unvollständiger Information.

Kern-Features:
- Bayesian Networks für kausale Modellierung
- Confidence-Propagation durch Reasoning-Ketten
- Noisy-OR für redundante Evidenz
- Prior/Posterior Aktualisierung
- Unsicherheits-basierte Antwortgenerierung
"""

import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import ProofStep as UnifiedProofStep
    from component_17_proof_explanation import (
        StepType,
        generate_explanation_text,
    )

    UNIFIED_PROOFS_AVAILABLE = True
except ImportError:
    UNIFIED_PROOFS_AVAILABLE = False


# ==================== DATENSTRUKTUREN ====================


@dataclass
class ProbabilisticFact:
    """
    Ein Fakt mit Wahrscheinlichkeitsverteilung.

    Attributes:
        pred: Prädikat (z.B. "IS_A", "HAS_PROPERTY")
        args: Argumente des Fakts
        probability: Wahrscheinlichkeit P(Fakt ist wahr) in [0, 1]
        source: Quelle der Information
        evidence: Liste von Evidenzen, die diese Wahrscheinlichkeit stützen
    """

    pred: str
    args: Dict[str, str]
    probability: float  # P(Fakt | Evidenz)
    source: str = "unknown"
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validierung aller Felder."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"Wahrscheinlichkeit muss in [0, 1] liegen: {self.probability}"
            )

        if not self.pred or not isinstance(self.pred, str):
            raise ValueError(f"Prädikat muss nicht-leerer String sein: {self.pred!r}")

        if not isinstance(self.args, dict):
            raise ValueError(f"Args muss Dictionary sein: {type(self.args)}")

        # Validiere, dass args nur String-Keys und String-Values hat
        for key, value in self.args.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"Args muss str->str Mapping sein, gefunden: {key!r}={value!r}"
                )


@dataclass
class ConditionalProbability:
    """
    Bedingte Wahrscheinlichkeit P(Consequent | Antecedents).

    Repräsentiert eine probabilistische Regel:
    P(Consequent | Antecedent1, Antecedent2, ...) = probability
    """

    consequent: str  # Ziel-Prädikat
    antecedents: List[str]  # Vorbedingungen
    probability: float  # P(consequent | antecedents)
    rule_id: str = ""


@dataclass
class BeliefState:
    """
    Repräsentiert den aktuellen Glaubenszustand über ein Proposition.

    Verwendet Beta-Verteilung für robuste Bayes-Updates:
    - Alpha: Anzahl der positiven Beobachtungen
    - Beta: Anzahl der negativen Beobachtungen
    """

    proposition: str
    alpha: float = 1.0  # Prior: Laplace-Smoothing
    beta: float = 1.0

    @property
    def probability(self) -> float:
        """Erwartungswert der Beta-Verteilung: E[p] = α/(α+β)"""
        denominator = self.alpha + self.beta
        # Epsilon guard (sollte nie 0 sein bei korrekt initialisierten Werten, aber Sicherheit)
        if denominator == 0:
            logger.warning(
                f"Probability-Berechnung: Invalid BeliefState (alpha={self.alpha}, beta={self.beta}). "
                f"Verwende uninformativen Prior 0.5."
            )
            return 0.5
        return self.alpha / denominator

    @property
    def variance(self) -> float:
        """Varianz der Beta-Verteilung (Maß für Unsicherheit)"""
        a, b = self.alpha, self.beta
        denominator = (a + b) ** 2 * (a + b + 1)

        # Schutz vor Division durch Zero
        if denominator == 0:
            logger.warning(
                f"Varianz-Berechnung: Invalid BeliefState (alpha={a}, beta={b}). "
                f"Verwende maximale Unsicherheit."
            )
            return 1.0  # Maximale Unsicherheit

        return (a * b) / denominator

    @property
    def confidence(self) -> float:
        """
        Confidence als Inverse der Standardabweichung.
        Höhere Werte = sicherer.
        """
        # Epsilon guard für numerische Stabilität
        variance = max(
            0.0, self.variance
        )  # Verhindere negative Varianz durch Rundungsfehler
        return 1.0 / (1.0 + math.sqrt(variance))

    def update(self, observation: bool, weight: float = 1.0):
        """
        Bayesian Update mit einer neuen Beobachtung.

        Args:
            observation: True (positive Evidenz) oder False (negative Evidenz)
            weight: Gewichtung der Beobachtung (für gewichtete Updates)
        """
        if observation:
            self.alpha += weight
        else:
            self.beta += weight


# ==================== PROBABILISTIC ENGINE ====================


class ProbabilisticEngine:
    """
    Haupt-Engine für probabilistisches Schließen.

    Features:
    - Bayesian Inference mit Forward-Chaining
    - Confidence-Propagation durch Reasoning-Ketten
    - Noisy-OR für redundante Evidenz
    - Uncertainty-Aware Response Generation
    """

    def __init__(self, max_facts: int = 10000, max_rules: int = 1000):
        """
        Initialisiert die Probabilistic Engine.

        Args:
            max_facts: Maximale Anzahl gespeicherter Fakten (bounded queue)
            max_rules: Maximale Anzahl gespeicherter Regeln (bounded queue)
        """
        self.beliefs: Dict[str, BeliefState] = {}
        self.conditional_probs: Deque[ConditionalProbability] = deque(maxlen=max_rules)
        self.facts: Deque[ProbabilisticFact] = deque(maxlen=max_facts)
        self.max_facts = max_facts
        self.max_rules = max_rules
        # Index rule by antecedents for efficient lookup
        self._rule_index: Dict[str, List[ConditionalProbability]] = {}

        logger.info(
            f"ProbabilisticEngine initialisiert (max_facts={max_facts}, max_rules={max_rules})."
        )

    # ==================== CORE INFERENCE ====================

    def add_fact(self, fact: ProbabilisticFact):
        """
        Fügt einen probabilistischen Fakt hinzu.

        Args:
            fact: Der hinzuzufügende Fakt
        """
        self.facts.append(fact)

        # Aktualisiere Belief-State
        prop_id = self._fact_to_proposition(fact)
        if prop_id not in self.beliefs:
            self.beliefs[prop_id] = BeliefState(proposition=prop_id)

        # KORREKTUR: Interpretiere Wahrscheinlichkeit p als virtuelle Beobachtungen
        # p=0.85 bedeutet "starke Evidenz für wahr" -> simuliere 8.5 positive, 1.5 negative
        # Effektive Sample Size: 10 (balanciert zwischen Prior-Einfluss und Evidenz)
        effective_sample_size = 10.0
        positive_evidence = fact.probability * effective_sample_size
        negative_evidence = (1.0 - fact.probability) * effective_sample_size

        # Update mit positiver und negativer Evidenz
        self.beliefs[prop_id].update(observation=True, weight=positive_evidence)
        self.beliefs[prop_id].update(observation=False, weight=negative_evidence)

        # WICHTIG: Für einfache Lookups füge auch nur den Prädikat-Namen hinzu (ohne Args)
        # Dies ermöglicht Matching mit Regeln, die nur Prädikat-Namen verwenden
        if fact.args:
            # Hat Args: Speichere unter vollständigem Prop-ID
            pass
        else:
            # Keine Args: Speichere auch unter einfachem Prädikat-Namen
            if fact.pred not in self.beliefs:
                self.beliefs[fact.pred] = BeliefState(proposition=fact.pred)
            # Gleiche Evidenz-Interpretation
            self.beliefs[fact.pred].update(observation=True, weight=positive_evidence)
            self.beliefs[fact.pred].update(observation=False, weight=negative_evidence)

        logger.debug(f"Fakt hinzugefügt: {fact.pred} mit P={fact.probability:.3f}")

    def add_conditional(self, cond_prob: ConditionalProbability):
        """
        Fügt eine bedingte Wahrscheinlichkeit (probabilistische Regel) hinzu.

        Args:
            cond_prob: Die bedingte Wahrscheinlichkeit
        """
        self.conditional_probs.append(cond_prob)

        # Index rule by antecedents for efficient lookup
        for antecedent in cond_prob.antecedents:
            if antecedent not in self._rule_index:
                self._rule_index[antecedent] = []
            self._rule_index[antecedent].append(cond_prob)

        logger.debug(
            f"Regel hinzugefügt: P({cond_prob.consequent} | "
            f"{', '.join(cond_prob.antecedents)}) = {cond_prob.probability:.3f}"
        )

    def infer(self, max_iterations: int = 10) -> List[ProbabilisticFact]:
        """
        Führt probabilistisches Forward-Chaining durch.

        Wendet alle bedingten Wahrscheinlichkeiten an, bis Konvergenz oder
        max_iterations erreicht ist.

        Args:
            max_iterations: Maximale Anzahl an Inferenz-Runden

        Returns:
            Liste neu abgeleiteter Fakten
        """
        logger.info("=== Probabilistic Inference gestartet ===")
        derived_facts = []
        # Track bereits abgeleitete Propositionen um Duplikate zu vermeiden
        derived_propositions = set()

        for iteration in range(max_iterations):
            new_facts = self._apply_rules_once()

            if not new_facts:
                logger.info(f"Konvergenz erreicht nach {iteration} Iterationen.")
                break

            # Filtere Duplikate
            unique_new_facts = []
            for fact in new_facts:
                prop_id = self._fact_to_proposition(fact)
                if prop_id not in derived_propositions:
                    unique_new_facts.append(fact)
                    derived_propositions.add(prop_id)

            if not unique_new_facts:
                logger.info(
                    f"Konvergenz erreicht nach {iteration} Iterationen (keine neuen Fakten)."
                )
                break

            derived_facts.extend(unique_new_facts)

            # Füge neue Fakten zur Faktenbasis hinzu
            for fact in unique_new_facts:
                self.add_fact(fact)

        logger.info(
            f"=== Inference beendet: {len(derived_facts)} neue Fakten abgeleitet ==="
        )
        return derived_facts

    def _apply_rules_once(self) -> List[ProbabilisticFact]:
        """
        Wendet alle Regeln einmal an (ein Forward-Chaining-Schritt).

        Returns:
            Liste neu abgeleiteter Fakten
        """
        new_facts = []

        # Nur Regeln prüfen, deren Antecedents wir haben (indexed lookup)
        candidate_rules_dict = {}
        for antecedent in self.beliefs.keys():
            for rule in self._rule_index.get(antecedent, []):
                # Use rule_id as key to avoid duplicates
                if rule.rule_id not in candidate_rules_dict:
                    candidate_rules_dict[rule.rule_id] = rule

        for rule in candidate_rules_dict.values():
            # KORREKTUR: Prüfe ob alle Antecedents erfüllt sind
            antecedent_probs = []
            all_antecedents_found = True

            # Wenn keine Antecedents: Regel kann nicht feuern (benötigt Bedingungen)
            if not rule.antecedents:
                # Regel ohne Vorbedingungen würde immer feuern -> skip
                # (könnte auch als "Axiom" behandelt werden mit P=rule.probability)
                # Für jetzt: überspringen
                continue

            for antecedent in rule.antecedents:
                if antecedent in self.beliefs:
                    antecedent_probs.append(self.beliefs[antecedent].probability)
                else:
                    # Antecedent unbekannt -> kann Regel nicht anwenden
                    all_antecedents_found = False
                    break

            if not all_antecedents_found:
                continue

            # Alle Antecedents gefunden -> berechne Consequent-Wahrscheinlichkeit
            consequent_prob = self._combine_probabilities(
                antecedent_probs, rule.probability
            )

            # KORREKTUR: Prüfe ob Consequent NEU ist oder höhere Wahrscheinlichkeit hat
            if (
                rule.consequent not in self.beliefs
                or self.beliefs[rule.consequent].probability < consequent_prob
            ):

                # Erstelle neuen Fakt
                new_fact = ProbabilisticFact(
                    pred=rule.consequent,
                    args={},  # Wird von Regel-Matching gefüllt
                    probability=consequent_prob,
                    source=f"rule:{rule.rule_id}",
                    evidence=rule.antecedents,
                )
                new_facts.append(new_fact)

                logger.debug(
                    f"Abgeleitet: {rule.consequent} mit P={consequent_prob:.3f} "
                    f"via {rule.rule_id}"
                )

        return new_facts

    # ==================== CONFIDENCE PROPAGATION ====================

    def propagate_confidence(
        self, source_facts: List[ProbabilisticFact], reasoning_chain: List[str]
    ) -> float:
        """
        Propagiert Confidence durch eine Reasoning-Kette.

        Strategie:
        - Für konjunktive Ketten (AND): Minimum
        - Für disjunktive Ketten (OR): Noisy-OR
        - Für gemischte Ketten: kontextabhängig

        Args:
            source_facts: Initiale Fakten mit Wahrscheinlichkeiten
            reasoning_chain: Sequenz von Inferenzschritten (Regel-IDs)

        Returns:
            Finale Confidence-Wert
        """
        if not source_facts:
            return 0.0

        # Start mit initialer Evidenz
        current_confidence = min(f.probability for f in source_facts)

        # Propagiere durch Kette
        for step in reasoning_chain:
            # Finde entsprechende Regel
            rule = next((r for r in self.conditional_probs if r.rule_id == step), None)
            if rule:
                current_confidence *= rule.probability

        return current_confidence

    def noisy_or(self, probabilities: List[float]) -> float:
        """
        Noisy-OR für redundante Evidenz.

        Modelliert mehrere unabhängige Ursachen für ein Ereignis:
        P(E | C1, C2, ..., Cn) = 1 - ∏(1 - P(E | Ci))

        Interpretation: Mindestens eine Ursache ist ausreichend.

        Args:
            probabilities: Liste von P(E | Ci) für jede Ursache

        Returns:
            Kombinierte Wahrscheinlichkeit
        """
        if not probabilities:
            return 0.0

        # Berechne Produkt der Komplementärwahrscheinlichkeiten
        product = 1.0
        for p in probabilities:
            product *= 1.0 - p

        return 1.0 - product

    def _combine_probabilities(
        self, antecedent_probs: List[float], rule_strength: float
    ) -> float:
        """
        Kombiniert Antecedent-Wahrscheinlichkeiten zu Consequent-Wahrscheinlichkeit.

        Verwendet Minimum für konjunktive Verknüpfung (konservativ).

        Args:
            antecedent_probs: Wahrscheinlichkeiten der Vorbedingungen
            rule_strength: P(Consequent | Antecedents)

        Returns:
            P(Consequent)
        """
        if not antecedent_probs:
            return 0.0

        # Min für AND-Verknüpfung (alle müssen wahr sein)
        min_prob = min(antecedent_probs)

        # Gewichtet mit Regel-Stärke
        return min_prob * rule_strength

    # ==================== QUERY & RETRIEVAL ====================

    def query(self, proposition: str) -> Tuple[float, float]:
        """
        Fragt den Glaubenszustand für eine Proposition ab.

        Args:
            proposition: Die zu prüfende Proposition

        Returns:
            Tuple (probability, confidence):
            - probability: P(Proposition ist wahr)
            - confidence: Unsicherheitsmaß (höher = sicherer)
        """
        if proposition in self.beliefs:
            belief = self.beliefs[proposition]
            return (belief.probability, belief.confidence)
        else:
            # Unbekannte Proposition: Verwende uninformativen Prior
            return (0.5, 0.0)

    def explain_belief(self, proposition: str) -> Dict:
        """
        Erklärt, warum eine Proposition eine bestimmte Wahrscheinlichkeit hat.

        Args:
            proposition: Die zu erklärende Proposition

        Returns:
            Dictionary mit Erklärung (Evidenzen, Regeln, etc.)
        """
        if proposition not in self.beliefs:
            return {"error": "Proposition unbekannt"}

        belief = self.beliefs[proposition]

        # Finde unterstützende Fakten
        supporting_facts = [
            f for f in self.facts if self._fact_to_proposition(f) == proposition
        ]

        # KORREKTUR: Finde anwendbare Regeln mit flexiblem Matching
        # Normalisiere beide Formate: "A" und "A()" sollten matchen
        # Entferne leere Klammern für Vergleich
        normalized_prop = proposition.replace("()", "")
        supporting_rules = [
            r
            for r in self.conditional_probs
            if r.consequent == proposition or r.consequent == normalized_prop
        ]

        explanation = {
            "proposition": proposition,
            "probability": belief.probability,
            "confidence": belief.confidence,
            "variance": belief.variance,
            "alpha": belief.alpha,
            "beta": belief.beta,
            "direct_facts": len(supporting_facts),
            "applicable_rules": [r.rule_id for r in supporting_rules],
        }

        return explanation

    # ==================== RESPONSE GENERATION ====================

    def generate_response(
        self, proposition: str, threshold_high: float = 0.8, threshold_low: float = 0.2
    ) -> str:
        """
        Generiert eine uncertainty-aware Antwort auf eine Frage.

        Strategie:
        - P > threshold_high: Sichere Bejahung
        - P < threshold_low: Sichere Verneinung
        - Dazwischen: Unsichere Antwort mit Wahrscheinlichkeit

        Args:
            proposition: Die Frage/Proposition
            threshold_high: Schwelle für sichere Bejahung
            threshold_low: Schwelle für sichere Verneinung

        Returns:
            Natürlichsprachliche Antwort
        """
        prob, conf = self.query(proposition)

        # Template-basierte Generierung
        if prob >= threshold_high:
            if conf >= 0.8:
                return f"Ja, sehr wahrscheinlich (P={prob:.2f}, Konfidenz={conf:.2f})."
            else:
                return f"Wahrscheinlich ja (P={prob:.2f}), aber mit Unsicherheit (Konfidenz={conf:.2f})."

        elif prob <= threshold_low:
            if conf >= 0.8:
                return (
                    f"Nein, sehr unwahrscheinlich (P={prob:.2f}, Konfidenz={conf:.2f})."
                )
            else:
                return f"Wahrscheinlich nein (P={prob:.2f}), aber mit Unsicherheit (Konfidenz={conf:.2f})."

        else:
            return (
                f"Unsicher. Die Wahrscheinlichkeit liegt bei {prob:.2f} "
                f"(Konfidenz: {conf:.2f}). Weitere Evidenz wird benötigt."
            )

    # ==================== UTILITY FUNCTIONS ====================

    def _fact_to_proposition(self, fact: ProbabilisticFact) -> str:
        """
        Konvertiert einen Fakt zu einer eindeutigen Propositions-ID.

        Args:
            fact: Der zu konvertierende Fakt

        Returns:
            Proposition-ID (String)
        """
        args_str = ",".join(f"{k}={v}" for k, v in sorted(fact.args.items()))
        return f"{fact.pred}({args_str})"

    def get_most_certain_facts(self, top_k: int = 10) -> List[Tuple[str, float, float]]:
        """
        Gibt die sichersten Fakten zurück (höchste Confidence).

        Args:
            top_k: Anzahl zurückzugebender Fakten

        Returns:
            Liste von (proposition, probability, confidence)
        """
        sorted_beliefs = sorted(
            self.beliefs.items(), key=lambda x: x[1].confidence, reverse=True
        )

        return [
            (prop, belief.probability, belief.confidence)
            for prop, belief in sorted_beliefs[:top_k]
        ]

    def get_most_uncertain_facts(
        self, top_k: int = 10
    ) -> List[Tuple[str, float, float]]:
        """
        Gibt die unsichersten Fakten zurück (niedrigste Confidence).

        Nützlich für aktives Lernen: Wo brauchen wir mehr Evidenz?

        Args:
            top_k: Anzahl zurückzugebender Fakten

        Returns:
            Liste von (proposition, probability, confidence)
        """
        sorted_beliefs = sorted(self.beliefs.items(), key=lambda x: x[1].confidence)

        return [
            (prop, belief.probability, belief.confidence)
            for prop, belief in sorted_beliefs[:top_k]
        ]

    def reset(self):
        """Setzt die Engine zurück (löscht alle Fakten und Beliefs)."""
        self.beliefs.clear()
        self.conditional_probs.clear()
        self.facts.clear()
        self._rule_index.clear()
        logger.info("ProbabilisticEngine zurückgesetzt.")

    # ==================== UNIFIED PROOF EXPLANATION INTEGRATION ====================

    def create_proof_step_from_inference(
        self,
        derived_fact: ProbabilisticFact,
        source_facts: List[ProbabilisticFact],
        rule: Optional[ConditionalProbability] = None,
    ) -> Optional[UnifiedProofStep]:
        """
        Erstellt einen UnifiedProofStep aus einem probabilistischen Inferenzschritt.

        Args:
            derived_fact: Der abgeleitete Fakt
            source_facts: Die verwendeten Quell-Fakten
            rule: Die angewandte Regel (optional)

        Returns:
            UnifiedProofStep oder None
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            return None

        # Generiere Step-ID
        step_id = f"prob_{uuid.uuid4().hex[:8]}"

        # Erstelle Inputs aus source_facts
        inputs = [self._fact_to_proposition(f) for f in source_facts]

        # Output ist der abgeleitete Fakt
        output = self._fact_to_proposition(derived_fact)

        # Generiere Erklärung
        explanation = generate_explanation_text(
            step_type=StepType.PROBABILISTIC,
            inputs=inputs,
            output=output,
            rule_name=rule.rule_id if rule else None,
            metadata={
                "probability": derived_fact.probability,
                "source": derived_fact.source,
                "num_evidence": len(source_facts),
            },
        )

        return UnifiedProofStep(
            step_id=step_id,
            step_type=StepType.PROBABILISTIC,
            inputs=inputs,
            rule_name=rule.rule_id if rule else None,
            output=output,
            confidence=derived_fact.probability,
            explanation_text=explanation,
            parent_steps=[],
            bindings={},
            metadata={
                "probability": derived_fact.probability,
                "source": derived_fact.source,
                "evidence": derived_fact.evidence,
                "rule_strength": rule.probability if rule else 1.0,
            },
            source_component="component_16_probabilistic_engine",
            timestamp=datetime.now(),
        )

    def create_bayesian_update_proof(
        self,
        proposition: str,
        prior_alpha: float,
        prior_beta: float,
        observation: bool,
        weight: float,
    ) -> Optional[UnifiedProofStep]:
        """
        Erstellt einen ProofStep der einen Bayesian Update erklärt.

        Args:
            proposition: Die Proposition
            prior_alpha: Alpha-Wert vor dem Update
            prior_beta: Beta-Wert vor dem Update
            observation: Die Beobachtung (True/False)
            weight: Gewicht der Beobachtung

        Returns:
            UnifiedProofStep oder None
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            return None

        # Berechne Prior und Posterior
        prior_prob = prior_alpha / (prior_alpha + prior_beta)
        new_alpha = prior_alpha + (weight if observation else 0)
        new_beta = prior_beta + (weight if not observation else 0)
        posterior_prob = new_alpha / (new_alpha + new_beta)

        step_id = f"bayesian_update_{uuid.uuid4().hex[:8]}"

        # Inputs: Prior state
        inputs = [
            f"Prior: P({proposition})={prior_prob:.3f} (α={prior_alpha:.1f}, β={prior_beta:.1f})"
        ]

        # Output: Posterior state
        output = f"Posterior: P({proposition})={posterior_prob:.3f} (α={new_alpha:.1f}, β={new_beta:.1f})"

        # Erklärung
        obs_str = "positive" if observation else "negative"
        explanation = (
            f"Bayesian Update für '{proposition}': "
            f"Beobachtung ({obs_str}, Gewicht={weight:.2f}) "
            f"aktualisierte P von {prior_prob:.3f} auf {posterior_prob:.3f}"
        )

        return UnifiedProofStep(
            step_id=step_id,
            step_type=StepType.PROBABILISTIC,
            inputs=inputs,
            rule_name="bayesian_update",
            output=output,
            confidence=posterior_prob,
            explanation_text=explanation,
            parent_steps=[],
            bindings={},
            metadata={
                "prior_alpha": prior_alpha,
                "prior_beta": prior_beta,
                "posterior_alpha": new_alpha,
                "posterior_beta": new_beta,
                "observation": observation,
                "weight": weight,
            },
            source_component="component_16_probabilistic_engine",
            timestamp=datetime.now(),
        )

    def create_inference_chain(
        self, target_proposition: str, max_depth: int = 3
    ) -> List[UnifiedProofStep]:
        """
        Erstellt eine Proof-Kette die zeigt wie eine Proposition abgeleitet wurde.

        Args:
            target_proposition: Die Ziel-Proposition
            max_depth: Maximale Tiefe der Rückwärtsverfolgung

        Returns:
            Liste von UnifiedProofStep-Objekten (Kette)
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            return []

        proof_chain = []

        # Finde Regeln die diese Proposition erzeugen
        relevant_rules = [
            r
            for r in self.conditional_probs
            if r.consequent == target_proposition
            or r.consequent.replace("()", "") == target_proposition
        ]

        for rule in relevant_rules:
            # Prüfe ob alle Antecedents erfüllt sind
            antecedent_facts = []
            for antecedent in rule.antecedents:
                if antecedent in self.beliefs:
                    # Erstelle einen virtuellen Fakt
                    fact = ProbabilisticFact(
                        pred=antecedent,
                        args={},
                        probability=self.beliefs[antecedent].probability,
                        source="belief_state",
                    )
                    antecedent_facts.append(fact)

            if len(antecedent_facts) == len(rule.antecedents):
                # Alle Antecedents gefunden - erstelle ProofStep
                derived_prob = self._combine_probabilities(
                    [f.probability for f in antecedent_facts], rule.probability
                )

                derived_fact = ProbabilisticFact(
                    pred=target_proposition,
                    args={},
                    probability=derived_prob,
                    source=f"rule:{rule.rule_id}",
                )

                step = self.create_proof_step_from_inference(
                    derived_fact, antecedent_facts, rule
                )

                if step:
                    proof_chain.append(step)

        return proof_chain

    def explain_with_proof_step(self, proposition: str) -> List[UnifiedProofStep]:
        """
        Erklärt eine Proposition mit UnifiedProofSteps.

        Dies ist die Hauptschnittstelle für Integration mit dem Reasoning System.

        Args:
            proposition: Die zu erklärende Proposition

        Returns:
            Liste von UnifiedProofStep-Objekten mit Erklärungen
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            return []

        # Erstelle Inferenz-Kette
        proof_steps = self.create_inference_chain(proposition)

        # Füge Belief-State-Information hinzu
        if proposition in self.beliefs:
            belief = self.beliefs[proposition]

            # Erstelle einen ProofStep der den aktuellen Glaubenszustand zeigt
            state_step = UnifiedProofStep(
                step_id=f"belief_state_{uuid.uuid4().hex[:6]}",
                step_type=StepType.PROBABILISTIC,
                inputs=[],
                rule_name=None,
                output=f"Glaubenszustand: P({proposition})={belief.probability:.3f}",
                confidence=belief.confidence,
                explanation_text=(
                    f"Aktueller Glaubenszustand für '{proposition}': "
                    f"P={belief.probability:.3f}, Konfidenz={belief.confidence:.3f} "
                    f"(basierend auf α={belief.alpha:.1f}, β={belief.beta:.1f})"
                ),
                parent_steps=[],
                bindings={},
                metadata={
                    "probability": belief.probability,
                    "confidence": belief.confidence,
                    "alpha": belief.alpha,
                    "beta": belief.beta,
                    "variance": belief.variance,
                },
                source_component="component_16_probabilistic_engine",
            )

            proof_steps.insert(0, state_step)

        return proof_steps

    def create_detailed_explanation(self, proposition: str) -> str:
        """
        Erstellt eine detaillierte Erklärung mit Unified Explanation System.

        Args:
            proposition: Die zu erklärende Proposition

        Returns:
            Detaillierte natürlichsprachliche Erklärung
        """
        if not UNIFIED_PROOFS_AVAILABLE:
            # Fallback
            explanation = self.explain_belief(proposition)
            return str(explanation)

        # Erstelle ProofSteps
        proof_steps = self.explain_with_proof_step(proposition)

        if not proof_steps:
            return f"Keine Erklärung verfügbar für: {proposition}"

        # Formatiere mit Unified System
        from component_17_proof_explanation import format_proof_step

        parts = []
        parts.append(f"=== Probabilistische Erklärung für: {proposition} ===")
        parts.append("")

        for i, step in enumerate(proof_steps, 1):
            parts.append(f"Schritt {i}:")
            parts.append(format_proof_step(step, indent=1, show_details=True))
            parts.append("")

        return "\n".join(parts)


# ==================== INTEGRATION MIT LOGIK-ENGINE ====================


def convert_fact_to_probabilistic(fact) -> ProbabilisticFact:
    """
    Konvertiert einen deterministischen Fakt (aus component_9) zu einem probabilistischen Fakt.

    Args:
        fact: Fakt aus der Logik-Engine (component_9.Fact)

    Returns:
        ProbabilisticFact
    """
    return ProbabilisticFact(
        pred=fact.pred,
        args=fact.args,
        probability=fact.confidence,  # Confidence wird zu Probability
        source=fact.source,
        evidence=[s for s in fact.support],
    )


def convert_rule_to_conditional(rule) -> List[ConditionalProbability]:
    """
    Konvertiert eine deterministische Regel (aus component_9) zu bedingten Wahrscheinlichkeiten.

    Args:
        rule: Regel aus der Logik-Engine (component_9.Rule)

    Returns:
        Liste von ConditionalProbability
    """
    conditional_probs = []

    # Extrahiere Antecedents
    antecedents = [
        f"{when_clause['pred']}({when_clause.get('args', {})})"
        for when_clause in rule.when
    ]

    # Erstelle ConditionalProbability für jede THEN-Aktion
    for then_action in rule.then:
        if "assert" in then_action:
            consequent_data = then_action["assert"]
            consequent = f"{consequent_data['pred']}({consequent_data.get('args', {})})"

            conditional_probs.append(
                ConditionalProbability(
                    consequent=consequent,
                    antecedents=antecedents,
                    probability=rule.weight,  # Regel-Weight wird zu Conditional-Probability
                    rule_id=rule.id,
                )
            )

    return conditional_probs


# ==================== BEISPIEL-USAGE ====================

if __name__ == "__main__":
    # Beispiel: Medizinische Diagnose
    logging.basicConfig(level=logging.INFO)

    engine = ProbabilisticEngine()

    # Fakten (Symptome)
    engine.add_fact(
        ProbabilisticFact(
            pred="HAS_SYMPTOM",
            args={"patient": "max", "symptom": "fieber"},
            probability=0.9,
            source="observation",
        )
    )

    engine.add_fact(
        ProbabilisticFact(
            pred="HAS_SYMPTOM",
            args={"patient": "max", "symptom": "husten"},
            probability=0.7,
            source="observation",
        )
    )

    # Regeln (Diagnose-Regeln)
    engine.add_conditional(
        ConditionalProbability(
            consequent="HAS_DISEASE(patient=max,disease=grippe)",
            antecedents=[
                "HAS_SYMPTOM(patient=max,symptom=fieber)",
                "HAS_SYMPTOM(patient=max,symptom=husten)",
            ],
            probability=0.8,
            rule_id="diagnose_regel_grippe",
        )
    )

    # Inferenz
    derived = engine.infer()

    # Abfrage
    prob, conf = engine.query("HAS_DISEASE(patient=max,disease=grippe)")
    print(f"\nDiagnose: Grippe mit P={prob:.3f}, Konfidenz={conf:.3f}")

    # Erklärung
    explanation = engine.explain_belief("HAS_DISEASE(patient=max,disease=grippe)")
    print(f"\nErklärung: {explanation}")

    # Response-Generierung
    response = engine.generate_response("HAS_DISEASE(patient=max,disease=grippe)")
    print(f"\nAntwort: {response}")
