# kai_response_pipeline.py
"""
Pipeline-Based Response Generation für KAI

Verantwortlichkeiten:
- KaiResponse Datenstruktur
- Format-Methoden für verschiedene Fragetypen (wer, wann, wie, warum, was)
- Episodische und räumliche Antwortformatierung
- Pipeline-basierte Textgenerierung (traditionelle Ansatz)

Extracted from kai_response_formatter.py für bessere Modularität.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KaiResponse:
    """Datenstruktur für KAI-Antworten"""

    text: str
    trace: List[str] = field(default_factory=list)
    answer_id: Optional[str] = None  # Für Feedback-Tracking
    confidence: Optional[float] = None
    strategy: Optional[str] = None
    evaluation: Optional[Any] = None  # EvaluationResult
    proof_tree: Optional[Any] = None  # PHASE 6: ProofTree für Transparenz


class PipelineResponseGenerator:
    """
    Pipeline-basierter Response Generator.

    Verwendet template-basierte Formatierung für verschiedene Fragetypen.
    Dies ist der traditionelle Ansatz vor dem Production System.
    """

    def __init__(self, confidence_manager: Any, self_evaluator: Any):
        """
        Args:
            confidence_manager: ConfidenceManager Instanz
            self_evaluator: SelfEvaluator Instanz
        """
        self.confidence_manager = confidence_manager
        self.self_evaluator = self_evaluator

    def format_confidence_prefix(
        self, confidence: float, reasoning_type: str = "standard"
    ) -> str:
        """
        Generiert einen Confidence-aware Präfix für Antworten.

        Args:
            confidence: Confidence-Wert (0.0-1.0)
            reasoning_type: Art des Reasoning ("standard", "backward_chaining",
                           "hypothesis", "probabilistic", "graph_traversal")

        Returns:
            Formatierter Präfix-String
        """
        from component_confidence_manager import ConfidenceLevel

        level = self.confidence_manager.classify_confidence(confidence)

        # Reasoning-spezifische Präfixe
        reasoning_prefixes = {
            "backward_chaining": "durch komplexe schlussfolgerung",
            "hypothesis": "basierend auf abduktiver schlussfolgerung",
            "probabilistic": "durch probabilistische inferenz",
            "graph_traversal": "über mehrere beziehungen hinweg",
            "standard": "",
        }

        reasoning_prefix = reasoning_prefixes.get(reasoning_type, "")

        # Confidence-Level-basierte Formulierungen
        if level == ConfidenceLevel.HIGH:
            if confidence >= 0.95:
                qualifier = "mit sehr hoher sicherheit"
            else:
                qualifier = "mit hoher sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} habe ich {qualifier} (konfidenz: {confidence:.0%}) herausgefunden:"

            return f"{qualifier} (konfidenz: {confidence:.0%}) weiß ich:"

        elif level == ConfidenceLevel.MEDIUM:
            qualifier = "mit mittlerer sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} vermute ich {qualifier} (konfidenz: {confidence:.0%}):"

            return f"{qualifier} (konfidenz: {confidence:.0%}) vermute ich:"

        elif level == ConfidenceLevel.LOW:
            qualifier = "mit geringer sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} vermute ich vorsichtig {qualifier} (konfidenz: {confidence:.0%}):"

            return f"{qualifier} (konfidenz: {confidence:.0%}) vermute ich vorsichtig:"

        else:  # UNKNOWN
            qualifier = "sehr unsicher"

            if reasoning_prefix:
                return f"{reasoning_prefix} bin ich {qualifier} (konfidenz: {confidence:.0%}), aber möglich:"

            return f"ich bin {qualifier} (konfidenz: {confidence:.0%}), aber möglicherweise:"

    def format_low_confidence_warning(self, confidence: float) -> str:
        """
        Generiert eine Warnung für niedrige Confidence-Werte.

        Args:
            confidence: Confidence-Wert (0.0-1.0)

        Returns:
            Warnungs-String oder leerer String (bei hoher Confidence)
        """
        from component_confidence_manager import ConfidenceLevel

        level = self.confidence_manager.classify_confidence(confidence)

        if level == ConfidenceLevel.UNKNOWN:
            return "\n[WARNING] WARNUNG: Diese Antwort basiert auf sehr unsicheren Informationen. Weitere Evidenz wird dringend benötigt."

        elif level == ConfidenceLevel.LOW:
            return "\n[WARNING] HINWEIS: Diese Antwort ist unsicher. Bitte mit Vorsicht interpretieren."

        elif level == ConfidenceLevel.MEDIUM:
            return "\n[INFO] HINWEIS: Diese Antwort hat mittlere Sicherheit. Bestätigung empfohlen."

        # HIGH: Keine Warnung
        return ""

    @staticmethod
    def format_person_answer(
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
    ) -> str:
        """
        Formatiert eine Antwort auf eine Wer-Frage (nach Personen/Akteuren).

        Priorisiert: IS_A (Personen-Typen), Synonyme, Bedeutungen

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen
            synonyms: Liste von Synonymen

        Returns:
            Formatierte Antwort als String
        """
        # Build response parts
        response_parts: List[str] = []

        # Wenn es Bedeutungen gibt, zeige sie
        if bedeutungen:
            response_parts.append(f"'{topic}': {bedeutungen[0]}")
        else:
            response_parts.append(f"über '{topic}' weiß ich:")

        # Synonyme/Alternative Namen
        if synonyms:
            response_parts.append(f"auch bekannt als: {', '.join(synonyms)}")

        # IS_A für Personen-Kategorien
        if "IS_A" in facts:
            is_a_str = ", ".join(facts["IS_A"])
            response_parts.append(f"ist ein/eine {is_a_str}")

        # CAPABLE_OF für Fähigkeiten/Rollen
        if "CAPABLE_OF" in facts:
            capable_str = ", ".join(facts["CAPABLE_OF"])
            response_parts.append(f"kann {capable_str}")

        # ASSOCIATED_WITH für Assoziationen (wichtig für Reverse Lookup bei WER-Fragen!)
        # Beispiel: "Wer trinkt Brandy?" -> facts = {"ASSOCIATED_WITH": ["nick"]}
        if "ASSOCIATED_WITH" in facts:
            # Für WER-Fragen ist das Topic das Objekt (z.B. "brandy")
            # und die facts sind die Personen (z.B. ["nick"])
            persons = facts["ASSOCIATED_WITH"]
            if len(persons) == 1:
                return f"{persons[0].capitalize()}"
            elif len(persons) > 1:
                persons_str = ", ".join([p.capitalize() for p in persons[:-1]])
                return f"{persons_str} und {persons[-1].capitalize()}"

        # Wenn keine relevanten Fakten gefunden wurden
        if len(response_parts) == 1 and not bedeutungen:
            return f"Ich habe keine spezifischen Informationen darüber, wer oder was '{topic}' ist."

        return ". ".join(response_parts) + "."

    @staticmethod
    def format_time_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """
        Formatiert eine Antwort auf eine Wann-Frage (nach Zeit/Zeitpunkten).

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen

        Returns:
            Formatierte Antwort als String
        """
        # Suche nach zeitbezogenen Relationen (falls vorhanden)
        time_relations = ["OCCURRED_AT", "HAPPENS_AT", "TIME", "DATE"]

        for relation in time_relations:
            if relation in facts:
                time_str = ", ".join(facts[relation])
                return f"'{topic}' findet statt: {time_str}."

        # Fallback: Zeige Bedeutung oder generelle Info
        if bedeutungen:
            return f"Ich weiß über '{topic}': {bedeutungen[0]}. Aber ich habe keine spezifischen zeitlichen Informationen."

        return f"Ich habe leider keine zeitlichen Informationen über '{topic}'. Ich kenne '{topic}' noch nicht im zeitlichen Kontext."

    @staticmethod
    def format_process_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """
        Formatiert eine Antwort auf eine Wie-Frage (nach Prozessen/Methoden).

        Priorisiert: Bedeutungen (enthalten oft Beschreibungen), CAPABLE_OF, HAS_PROPERTY

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen

        Returns:
            Formatierte Antwort als String
        """
        parts: List[str] = []

        # Bedeutungen enthalten oft Prozess-Beschreibungen
        if bedeutungen:
            parts.append(f"'{topic}': {bedeutungen[0]}")

        # CAPABLE_OF für Funktionen/Fähigkeiten
        if "CAPABLE_OF" in facts:
            capable_str = ", ".join(facts["CAPABLE_OF"])
            parts.append(f"es kann {capable_str}")

        # HAS_PROPERTY für Eigenschaften (wie etwas funktioniert)
        if "HAS_PROPERTY" in facts:
            properties = ", ".join(facts["HAS_PROPERTY"])
            parts.append(f"eigenschaften: {properties}")

        # Fallback
        if not parts:
            return f"Ich habe keine spezifischen Informationen darüber, wie '{topic}' funktioniert oder abläuft."

        return ". ".join(parts) + "."

    @staticmethod
    def format_reason_answer(
        topic: str, facts: Dict[str, List[str]], bedeutungen: List[str]
    ) -> str:
        """
        Formatiert eine Antwort auf eine Warum-Frage (nach Gründen/Ursachen).

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen

        Returns:
            Formatierte Antwort als String
        """
        # Suche nach kausal-relevanten Relationen
        causal_relations = ["CAUSED_BY", "REASON", "PURPOSE", "BECAUSE_OF"]

        for relation in causal_relations:
            if relation in facts:
                reason_str = ", ".join(facts[relation])
                return f"'{topic}' weil: {reason_str}."

        # Fallback: Zeige Bedeutung
        if bedeutungen:
            return f"Ich weiß über '{topic}': {bedeutungen[0]}. Aber ich habe keine spezifischen Informationen über Gründe oder Ursachen."

        return f"Ich habe leider keine Informationen über die Gründe oder Ursachen von '{topic}'. Ich kenne '{topic}' nicht im kausalen Zusammenhang."

    def format_standard_answer(
        self,
        topic: str,
        facts: Dict[str, List[str]],
        bedeutungen: List[str],
        synonyms: List[str],
        query_type: str = "normal",
        backward_chaining_used: bool = False,
        is_hypothesis: bool = False,
        confidence: Optional[float] = None,
    ) -> str:
        """
        Formatiert eine Standard-Antwort (für Was-Fragen und generische Fragen).

        Args:
            topic: Das Thema der Frage
            facts: Dictionary mit Relationstypen und Objekten
            bedeutungen: Liste von Bedeutungen/Definitionen
            synonyms: Liste von Synonymen
            query_type: Typ der Query ("normal" oder "show_all_knowledge")
            backward_chaining_used: Ob Backward-Chaining verwendet wurde
            is_hypothesis: Ob es sich um eine Hypothese handelt
            confidence: Optionale Konfidenz (0.0-1.0)

        Returns:
            Formatierte Antwort als String
        """
        # PHASE: Confidence-Based Learning - Wähle Einleitung basierend auf Methode und Confidence
        if confidence is not None:
            # Bestimme Reasoning-Typ für Context-aware Präfix
            if is_hypothesis:
                reasoning_type = "hypothesis"
            elif backward_chaining_used:
                reasoning_type = "backward_chaining"
            else:
                reasoning_type = "standard"

            # Generiere Confidence-aware Präfix
            prefix = self.format_confidence_prefix(confidence, reasoning_type)
            parts = [prefix]
        else:
            # Fallback auf alte Logik (für Backwards-Kompatibilität)
            if is_hypothesis:
                parts = ["basierend auf abduktiver schlussfolgerung vermute ich:"]
            elif backward_chaining_used:
                parts = ["durch komplexe schlussfolgerung habe ich herausgefunden:"]
            else:
                parts = [f"das weiß ich über {topic}:"]

        # PRIORITÄT 1: Zeige Bedeutungen/Definitionen zuerst (falls vorhanden)
        if bedeutungen:
            # Wenn es nur eine Bedeutung gibt, zeige sie direkt
            if len(bedeutungen) == 1:
                parts.append(bedeutungen[0])
            else:
                # Wenn es mehrere Bedeutungen gibt, liste sie auf
                for i, bed in enumerate(bedeutungen, 1):
                    parts.append(f"({i}) {bed}")

        # PRIORITÄT 2: Behandle Synonyme separat und zuerst
        if synonyms:
            syn_str = ", ".join(synonyms)
            parts.append(f"es ist auch bekannt als {syn_str}.")

        # TEIL_VON kann auch andere Beziehungen enthalten (nicht nur Synonyme)
        # Diese wurden bereits durch synonyms abgedeckt, also entfernen wir sie
        if "TEIL_VON" in facts:
            # Entferne Synonyme, die bereits erwähnt wurden
            other_parts = [
                p for p in facts["TEIL_VON"] if p not in synonyms and p != topic
            ]
            if other_parts:
                # Das sind echte "Teil von"-Beziehungen
                parts_str = ", ".join(other_parts)
                parts.append(f"es ist teil von {parts_str}.")
            # Entferne TEIL_VON aus facts, da wir es behandelt haben
            facts = {k: v for k, v in facts.items() if k != "TEIL_VON"}

        # Behandle IS_A
        if "IS_A" in facts:
            is_a_str = ", ".join(facts["IS_A"])
            parts.append(f"es ist eine art von {is_a_str}.")
            facts = {k: v for k, v in facts.items() if k != "IS_A"}

        # SPEZIAL: Bei "show_all_knowledge" Query, zeige ALLE Relationen detailliert
        if query_type == "show_all_knowledge":
            # Zeige alle verbleibenden Relationen ausführlich
            for relation, objects in facts.items():
                obj_str = ", ".join(objects)
                relation_str = relation.replace("_", " ").lower()
                parts.append(f"{relation_str}: {obj_str}.")
        else:
            # Normal: Nur zusammengefasst
            for relation, objects in facts.items():
                obj_str = ", ".join(objects)
                relation_str = relation.replace("_", " ").lower()
                parts.append(f"zudem: {relation_str} {obj_str}.")

        response = " ".join(parts)

        # PHASE: Confidence-Based Learning - Füge Warnung hinzu bei niedriger Confidence
        if confidence is not None:
            warning = self.format_low_confidence_warning(confidence)
            response += warning

        return response

    @staticmethod
    def format_episodic_answer(
        episodes: List[Dict], query_type: str, topic: Optional[str] = None
    ) -> str:
        """
        Formatiert eine Antwort für episodische Gedächtnis-Abfragen.

        Args:
            episodes: Liste von Episode-Dictionaries
            query_type: Typ der episodischen Query (when_learned, show_episodes, etc.)
            topic: Optionales Thema

        Returns:
            Formatierte Antwort als String
        """
        if not episodes:
            if topic:
                return f"Ich habe noch nichts über '{topic}' gelernt oder gefolgert."
            else:
                return "Ich habe noch keine Episoden gespeichert."

        # Header basierend auf Query-Typ
        if query_type == "when_learned":
            if topic:
                header = f"Ich habe {len(episodes)} Mal über '{topic}' gelernt:"
            else:
                header = f"Ich habe {len(episodes)} Lern-Episoden gespeichert:"
        elif query_type == "what_learned":
            header = f"Hier ist was ich über '{topic}' gelernt habe ({len(episodes)} Episoden):"
        elif query_type == "learning_history":
            if topic:
                header = f"Lernverlauf für '{topic}' ({len(episodes)} Episoden):"
            else:
                header = f"Gesamter Lernverlauf ({len(episodes)} Episoden):"
        else:  # show_episodes
            if topic:
                header = f"Episoden über '{topic}' ({len(episodes)} gesamt):"
            else:
                header = f"Alle gespeicherten Episoden ({len(episodes)} gesamt):"

        response_parts = [header, ""]

        # Zeige bis zu 5 Episoden in der Text-Antwort
        for i, episode in enumerate(episodes[:5], 1):
            # Zeitstempel formatieren
            timestamp = episode.get("timestamp")
            if timestamp:
                from datetime import datetime

                try:
                    dt = datetime.fromtimestamp(timestamp / 1000.0)
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    time_str = "?"
            else:
                time_str = "?"

            # Episode-Typ
            ep_type = episode.get("type", "?")

            # Inhalt/Query
            if "content" in episode:
                content = episode["content"][:80]
            elif "query" in episode:
                content = episode["query"][:80]
            else:
                content = "?"

            # Gelernte Fakten
            learned_facts = episode.get("learned_facts", [])
            learned_facts = [f for f in learned_facts if f is not None]
            facts_summary = ""
            if learned_facts:
                if len(learned_facts) == 1:
                    fact = learned_facts[0]
                    if isinstance(fact, dict):
                        facts_summary = f" -> [{fact.get('subject')} {fact.get('relation')} {fact.get('object')}]"
                else:
                    facts_summary = f" -> {len(learned_facts)} Fakten"

            response_parts.append(
                f"{i}. [{time_str}] {ep_type}: {content}{facts_summary}"
            )

        # Hinweis auf weitere Episoden
        if len(episodes) > 5:
            response_parts.append("")
            response_parts.append(f"... und {len(episodes) - 5} weitere Episoden.")

        # Hinweis auf UI-Tab (using emoji-free version)
        response_parts.append("")
        response_parts.append(
            "[INFO] Tipp: Nutze den Tab 'Episodisches Gedächtnis' für Details und Filter!"
        )

        return "\n".join(response_parts)

    @staticmethod
    def format_spatial_answer(
        model_type: str,
        spatial_query_type: str,
        entities: List[Dict] = None,
        positions: Dict = None,
        relations: List[Dict] = None,
        plan: List = None,
        plan_length: int = 0,
        reachable: bool = True,
    ) -> str:
        """
        Formatiert eine Antwort für räumliche Reasoning-Abfragen.

        Args:
            model_type: Typ des räumlichen Modells (grid, positions, relations, path_finding)
            spatial_query_type: Typ der räumlichen Query
            entities: Liste von extrahierten Entitäten
            positions: Dictionary mit Positionen
            relations: Liste von räumlichen Relationen
            plan: Optionaler Plan (für Path-Finding)
            plan_length: Länge des Plans
            reachable: Ob das Ziel erreichbar ist

        Returns:
            Formatierte Antwort als String
        """
        entities = entities or []
        positions = positions or {}
        relations = relations or []

        response_parts = []

        # FALL 1: Grid-basierte Queries
        if spatial_query_type == "grid_query":
            if entities:
                grid_entity = entities[0]
                grid_config = grid_entity.get("config", {})
                rows = grid_config.get("rows", 8)
                cols = grid_config.get("cols", 8)

                response_parts.append(
                    f"Ich habe ein {rows}×{cols} Grid-Modell erstellt."
                )
                response_parts.append(f"Das Grid hat {rows * cols} Felder insgesamt.")

        # FALL 2: Positions-Queries
        elif spatial_query_type == "position_query":
            if positions:
                obj_count = len(positions)
                response_parts.append(
                    f"Ich habe {obj_count} {'Objekt' if obj_count == 1 else 'Objekte'} im räumlichen Modell:"
                )
                response_parts.append("")

                for obj_name, pos_data in positions.items():
                    x, y = pos_data.get("x", "?"), pos_data.get("y", "?")
                    response_parts.append(f"  - {obj_name}: Position ({x}, {y})")
            else:
                response_parts.append("Ich konnte keine Positionsinformationen finden.")

        # FALL 3: Relations-Queries
        elif spatial_query_type == "relation_query":
            if relations:
                rel = relations[0]
                subject = rel.get("subject", "unbekannt")
                relation_type = rel.get("relation", "ADJACENT_TO")
                target = rel.get("target", "unbekannt")

                # Übersetze Relation ins Deutsche
                relation_translations = {
                    "NORTH_OF": "nördlich von",
                    "SOUTH_OF": "südlich von",
                    "EAST_OF": "östlich von",
                    "WEST_OF": "westlich von",
                    "ADJACENT_TO": "neben",
                    "BETWEEN": "zwischen",
                    "ABOVE": "über",
                    "BELOW": "unter",
                }
                relation_german = relation_translations.get(
                    relation_type, relation_type.lower()
                )

                # Hier würde man normalerweise das räumliche Modell abfragen
                # Vereinfachte Antwort:
                response_parts.append(
                    f"Ich prüfe ob '{subject}' {relation_german} '{target}' liegt."
                )
                response_parts.append(
                    "Diese Query erfordert ein vollständig initialisiertes räumliches Modell."
                )
            else:
                response_parts.append("Keine räumlichen Relationen zum Prüfen.")

        # FALL 4: Path-Finding-Queries
        elif spatial_query_type == "path_finding":
            if not reachable:
                response_parts.append(
                    "Es gibt keinen Pfad zum Ziel (nicht erreichbar)."
                )
            elif plan and plan_length > 0:
                response_parts.append(
                    f"Ich habe einen Pfad gefunden! Länge: {plan_length} Schritte."
                )
                response_parts.append("")
                response_parts.append("Pfad:")
                for i, action in enumerate(plan[:10], 1):  # Zeige max. 10 Schritte
                    action_name = (
                        action.name if hasattr(action, "name") else str(action)
                    )
                    response_parts.append(f"  {i}. {action_name}")

                if plan_length > 10:
                    response_parts.append(
                        f"  ... und {plan_length - 10} weitere Schritte"
                    )
            else:
                response_parts.append(
                    "Path-Finding wurde durchgeführt, aber kein Plan generiert."
                )

        # FALLBACK: Generische Antwort
        else:
            response_parts.append(
                f"Räumliche Abfrage vom Typ '{spatial_query_type}' wurde verarbeitet."
            )
            response_parts.append(
                f"Modell-Typ: {model_type}, {len(entities)} Entitäten"
            )

        return (
            "\n".join(response_parts)
            if response_parts
            else "Keine räumliche Antwort verfügbar."
        )
