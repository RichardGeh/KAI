# kai_response_formatter.py
"""
Response Formatting Module für KAI

Verantwortlichkeiten:
- Antwort-Formatierung basierend auf Fragetyp
- Confidence-aware Response Generation (integriert mit ConfidenceManager)
- Pure functions ohne State
- Delegiert Text-Normalisierung an zentrale component_utils_text_normalization
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from component_utils_text_normalization import clean_entity as normalize_entity
from component_confidence_manager import get_confidence_manager, ConfidenceLevel

logger = logging.getLogger(__name__)


@dataclass
class KaiResponse:
    """Datenstruktur für KAI-Antworten"""

    text: str
    trace: List[str] = field(default_factory=list)


class KaiResponseFormatter:
    """
    Formatter für KAI-Antworten basierend auf Fragetyp und Wissensstand.

    Diese Klasse ist zustandslos und enthält nur Formatierungs-Logik.
    Text-Normalisierung wurde in component_utils_text_normalization zentralisiert.

    PHASE: Confidence-Based Learning Integration
    - Verwendet ConfidenceManager für einheitliches Confidence-Feedback
    - Generiert confidence-aware Antworten für alle Reasoning-Typen
    """

    def __init__(self):
        """Initialisiert den Formatter mit globalem ConfidenceManager."""
        self.confidence_manager = get_confidence_manager()
        logger.info("KaiResponseFormatter initialisiert mit ConfidenceManager")

    @staticmethod
    def clean_entity(entity_text: str) -> str:
        """
        Entfernt führende Artikel, bereinigt den Text und normalisiert Plurale zu Singularen.

        REFACTORED: Delegiert an zentrale component_utils_text_normalization.

        Args:
            entity_text: Der zu bereinigende Text

        Returns:
            Bereinigter und normalisierter Text
        """
        return normalize_entity(entity_text)

    def format_confidence_prefix(
        self, confidence: float, reasoning_type: str = "standard"
    ) -> str:
        """
        Generiert einen Confidence-aware Präfix für Antworten.

        NEUE METHODE: Confidence-Based Learning Integration

        Args:
            confidence: Confidence-Wert (0.0-1.0)
            reasoning_type: Art des Reasoning ("standard", "backward_chaining",
                           "hypothesis", "probabilistic", "graph_traversal")

        Returns:
            Formatierter Präfix-String
        """
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
            else:
                return f"{qualifier} (konfidenz: {confidence:.0%}) weiß ich:"

        elif level == ConfidenceLevel.MEDIUM:
            qualifier = "mit mittlerer sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} vermute ich {qualifier} (konfidenz: {confidence:.0%}):"
            else:
                return f"{qualifier} (konfidenz: {confidence:.0%}) vermute ich:"

        elif level == ConfidenceLevel.LOW:
            qualifier = "mit geringer sicherheit"

            if reasoning_prefix:
                return f"{reasoning_prefix} vermute ich vorsichtig {qualifier} (konfidenz: {confidence:.0%}):"
            else:
                return (
                    f"{qualifier} (konfidenz: {confidence:.0%}) vermute ich vorsichtig:"
                )

        else:  # UNKNOWN
            qualifier = "sehr unsicher"

            if reasoning_prefix:
                return f"{reasoning_prefix} bin ich {qualifier} (konfidenz: {confidence:.0%}), aber möglich:"
            else:
                return f"ich bin {qualifier} (konfidenz: {confidence:.0%}), aber möglicherweise:"

    def format_low_confidence_warning(self, confidence: float) -> str:
        """
        Generiert eine Warnung für niedrige Confidence-Werte.

        NEUE METHODE: Confidence-Based Learning Integration

        Args:
            confidence: Confidence-Wert (0.0-1.0)

        Returns:
            Warnungs-String oder leerer String (bei hoher Confidence)
        """
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
        parts: List[str] = []

        # Wenn es Bedeutungen gibt, zeige sie
        if bedeutungen:
            parts.append(f"'{topic}': {bedeutungen[0]}")
        else:
            parts.append(f"über '{topic}' weiß ich:")

        # Synonyme/Alternative Namen
        if synonyms:
            parts.append(f"auch bekannt als: {', '.join(synonyms)}")

        # IS_A für Personen-Kategorien
        if "IS_A" in facts:
            is_a_str = ", ".join(facts["IS_A"])
            parts.append(f"ist ein/eine {is_a_str}")

        # CAPABLE_OF für Fähigkeiten/Rollen
        if "CAPABLE_OF" in facts:
            capable_str = ", ".join(facts["CAPABLE_OF"])
            parts.append(f"kann {capable_str}")

        # Wenn keine relevanten Fakten gefunden wurden
        if len(parts) == 1 and not bedeutungen:
            return f"Ich habe keine spezifischen Informationen darüber, wer oder was '{topic}' ist."

        return ". ".join(parts) + "."

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

        parts: List[str] = []
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

        parts: List[str] = []
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

        UPDATED: Integriert mit ConfidenceManager für einheitliches Feedback

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
                parts = [f"basierend auf abduktiver schlussfolgerung vermute ich:"]
            elif backward_chaining_used:
                parts = [f"durch komplexe schlussfolgerung habe ich herausgefunden:"]
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

    def format_episodic_answer(
        self, episodes: List[Dict], query_type: str, topic: Optional[str] = None
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
                except:
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

        # Hinweis auf UI-Tab
        response_parts.append("")
        response_parts.append(
            "💡 Tipp: Nutze den Tab 'Episodisches Gedächtnis' für Details und Filter!"
        )

        return "\n".join(response_parts)
