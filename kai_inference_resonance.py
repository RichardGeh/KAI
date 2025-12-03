# kai_inference_resonance.py
"""
Resonance-Based Inference Handler für KAI

Verantwortlichkeiten:
- Resonance-Based Question Answering via Activation Spreading
- Konzept-Extraktion aus Queries
- Widerspruchs-Erkennung
- Antwort-Generierung aus Activation Maps
- ProofTree-Generierung aus Resonance
"""
import logging
import string
from typing import Any, Dict, List, Optional

from component_1_netzwerk import KonzeptNetzwerk

logger = logging.getLogger(__name__)

# Import Unified Proof Explanation System
try:
    from component_17_proof_explanation import ProofStep, ProofTree, StepType

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    logger.warning("Unified Proof Explanation System nicht verfügbar")


class ResonanceInferenceHandler:
    """
    Handler für Resonance-Based Question Answering.

    Verwaltet:
    - Activation Spreading über semantische Netzwerke
    - Überschneidungs-Erkennung zwischen Activation Maps
    - Widerspruchs-Erkennung
    - Antwort-Generierung
    """

    def __init__(
        self,
        netzwerk: KonzeptNetzwerk,
        resonance_engine,  # ResonanceEngine
        working_memory,  # WorkingMemory
        signals,  # KaiSignals
        linguistik_engine=None,  # Optional: LinguistikEngine für bessere Konzept-Extraktion
    ):
        """
        Initialisiert den Resonance-Inference Handler.

        Args:
            netzwerk: KonzeptNetzwerk für Datenspeicherung
            resonance_engine: ResonanceEngine für Activation Spreading
            working_memory: WorkingMemory für Reasoning-Trace
            signals: KaiSignals für UI-Kommunikation
            linguistik_engine: Optional LinguistikEngine für Konzept-Extraktion
        """
        self.netzwerk = netzwerk
        self._resonance_engine = resonance_engine
        self.working_memory = working_memory
        self.signals = signals
        self.linguistik_engine = linguistik_engine

        logger.info("ResonanceInferenceHandler initialisiert")

    def handle_resonance_inference(
        self, query: str, context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Beantwortet Frage via Activation Spreading (Resonance-Based QA).

        Example: "Kann ein Pinguin fliegen?"
        1. Extrahiere Konzepte: ["Pinguin", "fliegen"]
        2. Aktiviere beide Konzepte parallel
        3. Finde Überschneidungen in Activation Maps
        4. Prüfe auf Widersprüche (CAPABLE_OF vs. NOT_CAPABLE_OF)
        5. Generiere Antwort basierend auf Resonance

        Args:
            query: Die Frage als Text
            context: Optional context dict

        Returns:
            Dictionary mit answer, proof_tree, activation_maps, confidence
            oder None wenn Resonance-Inference nicht möglich
        """
        if not self._resonance_engine:
            logger.warning("[Resonance QA] ResonanceEngine nicht verfügbar")
            return None

        if context is None:
            context = {}

        logger.info(f"[Resonance QA] Starte Resonance-Based QA für: '{query}'")

        try:
            # 1. Extrahiere Key-Konzepte aus der Query
            concepts = self._extract_key_concepts(query)

            if not concepts:
                logger.warning("[Resonance QA] Keine Konzepte extrahiert")
                return None

            logger.debug(f"[Resonance QA] Extrahierte Konzepte: {concepts}")

            # 2. Aktiviere alle Konzepte parallel
            activation_maps = {}
            for concept in concepts:
                try:
                    activation_map = self._resonance_engine.activate_concept(
                        start_word=concept, query_context=context
                    )
                    activation_maps[concept] = activation_map
                    logger.debug(
                        f"[Resonance QA] {concept}: {activation_map.concepts_activated} Konzepte aktiviert, "
                        f"{len(activation_map.resonance_points)} Resonanz-Punkte"
                    )
                except Exception as e:
                    logger.warning(
                        f"[Resonance QA] Aktivierung fehlgeschlagen für '{concept}': {e}"
                    )
                    continue

            if not activation_maps:
                logger.warning("[Resonance QA] Keine Activation Maps erstellt")
                return None

            # 3. Finde semantische Überschneidungen
            overlap = self._find_activation_overlap(activation_maps)

            logger.debug(
                f"[Resonance QA] Overlap gefunden: {len(overlap)} gemeinsame Konzepte"
            )

            # 4. Prüfe auf Widersprüche
            contradictions = self._detect_contradictions(
                overlap, concepts, activation_maps
            )

            # 5. Reasoning basierend auf Overlap
            if overlap or contradictions:
                # Konzepte sind semantisch verbunden
                answer = self._generate_answer_from_overlap(
                    overlap, contradictions, query, concepts
                )
                confidence = self._calculate_resonance_confidence(
                    overlap, contradictions, activation_maps
                )
            else:
                # Keine Verbindung gefunden
                answer = self._generate_negative_answer(concepts)
                confidence = 0.3  # Low confidence für negative Antwort

            # 6. Build Proof Tree aus Activation Paths
            proof_tree = None
            if PROOF_SYSTEM_AVAILABLE:
                try:
                    proof_tree = self._build_proof_from_activation(
                        activation_maps, overlap, contradictions, query, answer
                    )
                    # Emittiere an UI
                    self.signals.proof_tree_update.emit(proof_tree)
                except Exception as e:
                    logger.warning(
                        f"[Resonance QA] Proof Tree Generierung fehlgeschlagen: {e}"
                    )

            # 7. Tracke in Working Memory
            self.working_memory.add_reasoning_state(
                step_type="resonance_qa",
                description=f"Resonance-Based QA für '{query}'",
                data={
                    "query": query,
                    "concepts": concepts,
                    "overlap_size": len(overlap),
                    "contradictions": len(contradictions),
                    "answer": answer,
                },
                confidence=confidence,
            )

            logger.info(
                f"[Resonance QA] [OK] Antwort generiert (Konfidenz: {confidence:.2f})"
            )

            return {
                "answer": answer,
                "proof_tree": proof_tree,
                "activation_maps": activation_maps,
                "confidence": confidence,
                "overlap": overlap,
                "contradictions": contradictions,
            }

        except Exception as e:
            logger.error(f"[Resonance QA] Fehler: {e}", exc_info=True)
            return None

    def construct_query_from_topic(self, topic: str, relation_type: str) -> str:
        """
        Konstruiert natürlichsprachliche Query aus Topic und Relation.

        Args:
            topic: Das Thema (z.B. "pinguin")
            relation_type: Der Relationstyp (z.B. "CAPABLE_OF", "IS_A")

        Returns:
            Deutsche Frage als Text
        """
        topic_lower = topic.lower()

        # Mapping von Relationen zu Fragewörtern
        if relation_type == "IS_A":
            return f"Was ist ein {topic_lower}?"
        elif relation_type == "CAPABLE_OF":
            return f"Was kann ein {topic_lower}?"
        elif relation_type == "HAS_PROPERTY":
            return f"Welche Eigenschaften hat ein {topic_lower}?"
        elif relation_type == "PART_OF":
            return f"Teil von was ist {topic_lower}?"
        elif relation_type == "LOCATED_IN":
            return f"Wo befindet sich {topic_lower}?"
        else:
            # Fallback: Generic question
            return f"Was weißt du über {topic_lower}?"

    # ==================== PRIVATE HELPER METHODS ====================

    def _extract_key_concepts(self, query: str) -> List[str]:
        """
        Extrahiert Schlüssel-Konzepte aus einer Query.

        Uses spaCy linguistic engine to extract nouns and verbs.

        Args:
            query: Die Frage als Text

        Returns:
            Liste von Konzepten (lemmatisiert, lowercase)
        """
        try:
            # Nutze linguistik_engine falls verfügbar
            if self.linguistik_engine:
                # Parse query
                doc = self.linguistik_engine.nlp(query)

                # Extrahiere Nomen und Verben
                concepts = []
                for token in doc:
                    # Nur Nomen und Verben
                    if token.pos_ in ["NOUN", "VERB", "PROPN"]:
                        # Lemmatisiert und lowercase
                        lemma = token.lemma_.lower()
                        if lemma and len(lemma) > 2:  # Min 3 Zeichen
                            concepts.append(lemma)

                return concepts

            else:
                # Fallback: Einfaches Word-Splitting
                logger.warning(
                    "[Resonance QA] Linguistik-Engine nicht verfügbar, verwende Fallback"
                )
                # Entferne Satzzeichen
                query_clean = query.translate(str.maketrans("", "", string.punctuation))
                words = query_clean.lower().split()
                # Filtere Stopwords (einfach)
                stopwords = {
                    "ein",
                    "eine",
                    "der",
                    "die",
                    "das",
                    "ist",
                    "kann",
                    "hat",
                    "was",
                    "wie",
                    "wer",
                    "wo",
                    "wann",
                    "warum",
                }
                concepts = [w for w in words if w not in stopwords and len(w) > 2]
                return concepts

        except Exception as e:
            logger.warning(f"[Resonance QA] Konzept-Extraktion fehlgeschlagen: {e}")
            return []

    def _find_activation_overlap(
        self, activation_maps: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Findet semantische Überschneidungen zwischen Activation Maps.

        Args:
            activation_maps: Dict mit concept -> ActivationMap

        Returns:
            Dict mit overlapping concepts und deren Aktivierungen
            Format: {concept: {source1: activation1, source2: activation2, ...}}
        """
        if len(activation_maps) < 2:
            # Kein Overlap möglich mit nur einer Map
            return {}

        # Sammle alle aktivierten Konzepte pro Source
        concept_sources = {}  # concept -> {source: activation}

        for source_concept, activation_map in activation_maps.items():
            for concept, activation in activation_map.activations.items():
                if concept not in concept_sources:
                    concept_sources[concept] = {}
                concept_sources[concept][source_concept] = activation

        # Finde Konzepte die in mehreren Maps vorkommen
        overlap = {}
        for concept, sources in concept_sources.items():
            if len(sources) >= 2:  # Mindestens 2 Sources
                overlap[concept] = sources

        return overlap

    def _detect_contradictions(
        self,
        overlap: Dict[str, Dict[str, float]],
        query_concepts: List[str],
        activation_maps: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Erkennt Widersprüche in den Activation Maps.

        Prüft auf:
        - CAPABLE_OF vs. NOT_CAPABLE_OF
        - HAS_PROPERTY vs. NOT_HAS_PROPERTY
        - IS_A vs. NOT_IS_A

        Args:
            overlap: Overlapping concepts
            query_concepts: Konzepte aus der Query
            activation_maps: Alle Activation Maps

        Returns:
            Liste von Widersprüchen
        """
        contradictions = []

        # Prüfe für jedes Query-Konzept ob es widersprüchliche Paths gibt
        for concept in query_concepts:
            if concept not in activation_maps:
                continue

            activation_map = activation_maps[concept]

            # Sammle alle Relationstypen aus Paths
            positive_relations = {}  # target -> [paths]
            negative_relations = {}  # target -> [paths]

            for path in activation_map.reasoning_paths:
                for relation in path.relations:
                    # Prüfe ob negativ
                    is_negative = relation.startswith("NOT_")
                    _ = (
                        relation[4:] if is_negative else relation
                    )  # base_relation unused

                    target = path.target

                    if is_negative:
                        if target not in negative_relations:
                            negative_relations[target] = []
                        negative_relations[target].append(
                            {"relation": relation, "path": path, "source": concept}
                        )
                    else:
                        if target not in positive_relations:
                            positive_relations[target] = []
                        positive_relations[target].append(
                            {"relation": relation, "path": path, "source": concept}
                        )

            # Finde Widersprüche
            for target in positive_relations:
                if target in negative_relations:
                    # Widerspruch gefunden!
                    contradictions.append(
                        {
                            "concept": concept,
                            "target": target,
                            "positive_paths": positive_relations[target],
                            "negative_paths": negative_relations[target],
                            "type": "relation_negation",
                        }
                    )

        return contradictions

    def _generate_answer_from_overlap(
        self,
        overlap: Dict[str, Dict[str, float]],
        contradictions: List[Dict[str, Any]],
        query: str,
        concepts: List[str],
    ) -> str:
        """
        Generiert natürlichsprachliche Antwort aus Overlap und Widersprüchen.

        Args:
            overlap: Overlapping concepts
            contradictions: Erkannte Widersprüche
            query: Ursprüngliche Frage
            concepts: Extrahierte Konzepte

        Returns:
            Deutsche Antwort-Text
        """
        # Prüfe ob Ja/Nein-Frage
        is_yes_no = any(
            word in query.lower() for word in ["kann", "ist", "hat", "sind"]
        )

        # Fall 1: Widersprüche gefunden
        if contradictions:
            # Nehme ersten Widerspruch
            contr = contradictions[0]

            # Extrahiere Details
            concept = contr["concept"]
            target = contr["target"]
            _ = contr["positive_paths"]  # unused
            neg_paths = contr["negative_paths"]

            # Bevorzuge negative Paths (spezifischer)
            if neg_paths:
                neg_relation = neg_paths[0]["relation"]
                _ = (
                    neg_relation[4:]
                    if neg_relation.startswith("NOT_")
                    else neg_relation
                )  # base_relation unused

                answer = f"Nein, {concept} kann nicht {target}. "
                answer += f"Obwohl {concept} generell zur Kategorie gehören könnte, "
                answer += "gibt es eine explizite Ausnahme."
            else:
                answer = f"Es gibt widersprüchliche Informationen über {concept} und {target}."

            return answer

        # Fall 2: Starker Overlap (Konzepte sind verbunden)
        if overlap:
            # Finde Konzept mit höchster kombinierter Aktivierung
            best_overlap = max(overlap.items(), key=lambda x: sum(x[1].values()))
            overlap_concept, sources = best_overlap

            if is_yes_no:
                # Finde Pfade die die Relation bestätigen
                # (vereinfacht: schaue ob overlap_concept relevant ist)
                if any(concept in sources for concept in concepts):
                    answer = f"Ja, basierend auf semantischen Verbindungen über '{overlap_concept}'. "
                    answer += f"Aktivierung: {sum(sources.values()):.2f}"
                else:
                    answer = f"Die Konzepte sind semantisch verbunden über '{overlap_concept}', "
                    answer += "aber es gibt keine direkte Bestätigung."
            else:
                answer = (
                    f"Die Konzepte sind semantisch verbunden über '{overlap_concept}'. "
                )
                answer += f"{len(overlap)} gemeinsame Konzepte gefunden."

            return answer

        # Fall 3: Kein Overlap
        return self._generate_negative_answer(concepts)

    def _generate_negative_answer(self, concepts: List[str]) -> str:
        """
        Generiert negative Antwort bei fehlender semantischer Verbindung.

        Args:
            concepts: Die Query-Konzepte

        Returns:
            Deutsche Antwort
        """
        if len(concepts) >= 2:
            return f"Ich habe keine semantische Verbindung zwischen {' und '.join(concepts)} gefunden."
        elif concepts:
            return f"Ich habe nicht genügend Informationen über '{concepts[0]}'."
        else:
            return "Ich konnte die Frage nicht verstehen."

    def _build_proof_from_activation(
        self,
        activation_maps: Dict[str, Any],
        overlap: Dict[str, Dict[str, float]],
        contradictions: List[Dict[str, Any]],
        query: str,
        answer: str,
    ) -> Optional[Any]:
        """
        Baut Proof Tree aus Activation Maps.

        Args:
            activation_maps: Alle Activation Maps
            overlap: Overlapping concepts
            contradictions: Widersprüche
            query: Original-Query
            answer: Generierte Antwort

        Returns:
            ProofTree oder None
        """
        if not PROOF_SYSTEM_AVAILABLE:
            return None

        try:
            proof_tree = ProofTree(query=query)

            # Root Step: Query
            root_step = ProofStep(
                step_id="resonance_root",
                step_type=StepType.QUERY,
                inputs=list(activation_maps.keys()),
                output=answer,
                confidence=self._calculate_resonance_confidence(
                    overlap, contradictions, activation_maps
                ),
                explanation_text=f"Resonance-Based Question Answering für: {query}",
                source_component="resonance_qa",
                metadata={
                    "concepts": list(activation_maps.keys()),
                    "overlap_size": len(overlap),
                    "contradictions": len(contradictions),
                },
            )
            proof_tree.add_root_step(root_step)

            # Add Activation Summaries
            for concept, activation_map in activation_maps.items():
                summary = self._resonance_engine.get_activation_summary(activation_map)

                activation_step = ProofStep(
                    step_id=f"activation_{concept}",
                    step_type=StepType.INFERENCE,
                    inputs=[concept],
                    output=f"{activation_map.concepts_activated} Konzepte aktiviert",
                    confidence=min(activation_map.max_activation, 1.0),
                    explanation_text=summary,
                    source_component="resonance_engine",
                    metadata={
                        "waves": activation_map.waves_executed,
                        "resonance_points": len(activation_map.resonance_points),
                    },
                )
                proof_tree.add_child_step("resonance_root", activation_step)

                # Add Resonance Points
                for rp in activation_map.resonance_points[:3]:  # Top 3
                    explanation = self._resonance_engine.explain_activation(
                        rp.concept, activation_map, max_paths=2
                    )
                    rp_step = ProofStep(
                        step_id=f"resonance_{concept}_{rp.concept}",
                        step_type=StepType.FACT_MATCH,
                        inputs=[concept],
                        output=f"{rp.concept} (Resonanz)",
                        confidence=min(rp.resonance_boost, 1.0),
                        explanation_text=explanation,
                        source_component="resonance_engine",
                        metadata={"num_paths": rp.num_paths},
                    )
                    proof_tree.add_child_step(f"activation_{concept}", rp_step)

            # Add Contradictions if any
            for i, contr in enumerate(contradictions[:2]):  # Max 2
                contr_step = ProofStep(
                    step_id=f"contradiction_{i}",
                    step_type=StepType.CONSTRAINT,
                    inputs=[contr["concept"], contr["target"]],
                    output=f"Widerspruch erkannt: {contr['type']}",
                    confidence=0.9,
                    explanation_text=f"Positive und negative Relationen zu {contr['target']} gefunden",
                    source_component="resonance_qa",
                    metadata=contr,
                )
                proof_tree.add_child_step("resonance_root", contr_step)

            return proof_tree

        except Exception as e:
            logger.warning(f"[Resonance QA] Proof Tree Build fehlgeschlagen: {e}")
            return None

    def _calculate_resonance_confidence(
        self,
        overlap: Dict[str, Dict[str, float]],
        contradictions: List[Dict[str, Any]],
        activation_maps: Dict[str, Any],
    ) -> float:
        """
        Berechnet Confidence-Score aus Resonance-Daten.

        Args:
            overlap: Overlapping concepts
            contradictions: Widersprüche
            activation_maps: Alle Activation Maps

        Returns:
            Confidence zwischen 0.0 und 1.0
        """
        # Base Confidence aus Overlap-Größe
        if not overlap:
            base_conf = 0.3
        else:
            # Normalisiere auf Anzahl Query-Konzepte
            overlap_ratio = len(overlap) / max(len(activation_maps), 1)
            base_conf = 0.5 + (overlap_ratio * 0.3)  # 0.5 bis 0.8

        # Boost für Widersprüche (höhere Confidence da spezifisch)
        if contradictions:
            base_conf = min(base_conf + 0.15, 0.95)

        # Boost für Resonanz-Punkte
        total_resonance_points = sum(
            len(am.resonance_points) for am in activation_maps.values()
        )
        if total_resonance_points > 0:
            resonance_boost = min(total_resonance_points * 0.05, 0.15)
            base_conf = min(base_conf + resonance_boost, 0.95)

        # Berücksichtige max Aktivierung
        max_activation = max(
            (am.max_activation for am in activation_maps.values()), default=0.0
        )
        if max_activation > 1.0:
            # Hohe Aktivierung deutet auf starke Verbindung
            activation_boost = min((max_activation - 1.0) * 0.1, 0.1)
            base_conf = min(base_conf + activation_boost, 0.95)

        return base_conf
