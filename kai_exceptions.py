"""
kai_exceptions.py

Zentrale Exception-Hierarchie für KAI-System.
Definiert spezialisierte Exception-Klassen für verschiedene Fehlerszenarien.

Exception-Hierarchie:
    KAIException (Basis)
    ├── DatabaseException
    │   ├── Neo4jConnectionError
    │   ├── Neo4jQueryError
    │   └── Neo4jWriteError
    ├── LinguisticException
    │   ├── ParsingError
    │   ├── SpaCyModelError
    │   ├── EmbeddingError
    │   └── DocumentParseError
    ├── KnowledgeException
    │   ├── ExtractionRuleError
    │   ├── PatternMatchingError
    │   ├── ConceptNotFoundError
    │   └── RelationValidationError
    ├── ReasoningException
    │   ├── LogicEngineError
    │   ├── UnificationError
    │   ├── InferenceError
    │   ├── GraphTraversalError
    │   ├── AbductiveReasoningError
    │   ├── ProbabilisticReasoningError
    │   ├── WorkingMemoryError
    │   └── ConstraintReasoningError
    ├── PlanningException
    │   ├── GoalPlanningError
    │   ├── InvalidMeaningPointError
    │   └── PlanExecutionError
    └── ConfigurationException
        ├── InvalidConfigError
        └── MissingDependencyError

Verwendung:
    from kai_exceptions import Neo4jConnectionError, KnowledgeException

    try:
        netzwerk.connect()
    except Neo4jConnectionError as e:
        logger.error(f"Datenbankverbindung fehlgeschlagen: {e}")
        logger.error(f"Kontext: {e.context}")
"""

from typing import Optional, Dict, Any


class KAIException(Exception):
    """
    Basis-Exception für alle KAI-spezifischen Fehler.

    Alle KAI-Exceptions unterstützen:
    - Detaillierte Fehlermeldungen
    - Kontextuelle Informationen (dict)
    - Original-Exception-Verkettung (via 'from')
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_exception = original_exception

    def __str__(self) -> str:
        base_msg = self.message

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" | Context: {context_str}"

        if self.original_exception:
            base_msg += f" | Caused by: {type(self.original_exception).__name__}: {str(self.original_exception)}"

        return base_msg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r})"


# ============================================================================
# DATABASE EXCEPTIONS
# ============================================================================


class DatabaseException(KAIException):
    """Basis-Exception für alle datenbankbezogenen Fehler."""


class Neo4jConnectionError(DatabaseException):
    """
    Verbindung zur Neo4j-Datenbank fehlgeschlagen.

    Ursachen:
    - Datenbank läuft nicht
    - Falsche Verbindungsparameter (URI, User, Passwort)
    - Netzwerkprobleme
    """


class Neo4jQueryError(DatabaseException):
    """
    Fehler beim Ausführen einer Neo4j-Cypher-Query.

    Ursachen:
    - Syntaxfehler in Cypher-Query
    - Ungültige Parameter
    - Constraint-Verletzungen
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context["query"] = query
        context["parameters"] = parameters
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class Neo4jWriteError(DatabaseException):
    """
    Fehler beim Schreiben in die Neo4j-Datenbank.

    Ursachen:
    - Autocommit-Probleme
    - Constraint-Verletzungen
    - Inkonsistente Daten
    """


# ============================================================================
# LINGUISTIC EXCEPTIONS
# ============================================================================


class LinguisticException(KAIException):
    """Basis-Exception für linguistische Verarbeitungsfehler."""


class ParsingError(LinguisticException):
    """
    Fehler beim Parsen von Texteingabe.

    Ursachen:
    - spaCy-Verarbeitungsfehler
    - Ungültige Zeichenkodierung
    - Leere/Ungültige Eingabe
    """


class SpaCyModelError(LinguisticException):
    """
    spaCy-Modell konnte nicht geladen werden.

    Ursachen:
    - Modell nicht installiert (de_core_news_sm)
    - Falsche spaCy-Version
    - Beschädigte Modell-Dateien
    """


class EmbeddingError(LinguisticException):
    """
    Fehler beim Generieren von Embeddings.

    Ursachen:
    - Embedding-Modell nicht verfügbar
    - Ungültige Eingabe für Embedding
    - Dimension-Mismatch
    """


class DocumentParseError(LinguisticException):
    """
    Fehler beim Parsen von Dokumenten (DOCX, PDF, etc.).

    Ursachen:
    - Datei nicht gefunden oder nicht lesbar
    - Korrupte Dokumentdatei
    - Nicht unterstütztes Dateiformat
    - Fehlende Parser-Bibliothek (python-docx, pdfplumber)
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_format: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context["file_path"] = file_path
        context["file_format"] = file_format
        kwargs["context"] = context
        super().__init__(message, **kwargs)


# ============================================================================
# KNOWLEDGE EXCEPTIONS
# ============================================================================


class KnowledgeException(KAIException):
    """Basis-Exception für Wissensverarbeitungsfehler."""


class ExtractionRuleError(KnowledgeException):
    """
    Fehler bei der Verarbeitung von ExtractionRules.

    Ursachen:
    - Ungültiges Regex-Pattern
    - Falsche Anzahl Capture-Groups
    - Regel-Konflikt
    """

    def __init__(
        self,
        message: str,
        rule_id: Optional[str] = None,
        pattern: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context["rule_id"] = rule_id
        context["pattern"] = pattern
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class PatternMatchingError(KnowledgeException):
    """
    Fehler beim Pattern-Matching (Prototype-System).

    Ursachen:
    - Embedding-Fehler
    - Ungültige Distanz-Berechnung
    - Fehlende TRIGGERS-Beziehung
    """


class ConceptNotFoundError(KnowledgeException):
    """
    Gesuchtes Konzept/Wort existiert nicht im Knowledge Graph.

    Ursachen:
    - Konzept wurde nie gelernt
    - Tippfehler in Konzept-Name
    - Normalisierungsfehler
    """

    def __init__(self, message: str, concept_name: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        context["concept_name"] = concept_name
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class RelationValidationError(KnowledgeException):
    """
    Ungültige Relation oder Relationstyp.

    Ursachen:
    - Unbekannter Relationstyp (nicht in [IS_A, HAS_PROPERTY, ...])
    - Fehlende Subject/Object-Angaben
    - Inkonsistente Relationsdaten
    """

    def __init__(self, message: str, relation_type: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        context["relation_type"] = relation_type
        kwargs["context"] = context
        super().__init__(message, **kwargs)


# ============================================================================
# REASONING EXCEPTIONS
# ============================================================================


class ReasoningException(KAIException):
    """Basis-Exception für logische Reasoning-Fehler."""


class LogicEngineError(ReasoningException):
    """
    Fehler in der Logic Engine (component_9).

    Ursachen:
    - Inkonsistente Regeln
    - Zirkeldefinitionen
    - Stack-Overflow bei Inferenz
    """


class UnificationError(ReasoningException):
    """
    Variablen-Unifikation fehlgeschlagen.

    Ursachen:
    - Inkompatible Bindings
    - Unifikation nicht möglich
    - Typ-Mismatch
    """


class InferenceError(ReasoningException):
    """
    Fehler beim Ableiten neuer Fakten.

    Ursachen:
    - Fehlende Prämissen
    - Regelanwendung fehlgeschlagen
    - Inkonsistentes Weltmodell
    """


class GraphTraversalError(ReasoningException):
    """
    Fehler bei der Graph-Traversierung (component_12).

    Ursachen:
    - Kein Pfad zwischen Konzepten gefunden
    - Zyklische Abhängigkeiten
    - Zu tiefe Traversierung (max_depth erreicht)
    """

    def __init__(
        self,
        message: str,
        source_concept: Optional[str] = None,
        target_concept: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context["source_concept"] = source_concept
        context["target_concept"] = target_concept
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class AbductiveReasoningError(ReasoningException):
    """
    Fehler beim Abduktiven Reasoning (component_14).

    Ursachen:
    - Keine Hypothesen generierbar
    - Template-Matching fehlgeschlagen
    - Analogy-Basis leer
    """


class ProbabilisticReasoningError(ReasoningException):
    """
    Fehler beim Probabilistischen Reasoning (component_16).

    Ursachen:
    - Ungültige Wahrscheinlichkeiten (nicht in [0,1])
    - Bayesian Update fehlgeschlagen
    - Inkonsistente Belief States
    """

    def __init__(
        self, message: str, probability_value: Optional[float] = None, **kwargs
    ):
        context = kwargs.get("context", {})
        context["probability_value"] = probability_value
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class WorkingMemoryError(ReasoningException):
    """
    Fehler im Working Memory System (component_13).

    Ursachen:
    - Stack underflow (pop auf leerem Stack)
    - Context restoration fehlgeschlagen
    - Ungültige ReasoningState
    """


class ConstraintReasoningError(ReasoningException):
    """
    Fehler beim Constraint-basierten Reasoning (component_29).

    Ursachen:
    - Inkonsistente Constraints
    - Keine gültige Variable für Auswahl
    - Domain-Inkonsistenz
    - Solver-Fehler
    """

    def __init__(self, message: str, problem_name: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        context["problem_name"] = problem_name
        kwargs["context"] = context
        super().__init__(message, **kwargs)


# ============================================================================
# PLANNING EXCEPTIONS
# ============================================================================


class PlanningException(KAIException):
    """Basis-Exception für Goal-Planning-Fehler."""


class GoalPlanningError(PlanningException):
    """
    Fehler beim Erstellen eines Execution-Plans.

    Ursachen:
    - Unbekannter Goal-Typ
    - Fehlende Informationen für Planerstellung
    - Plan-Konflikt
    """


class InvalidMeaningPointError(PlanningException):
    """
    Ungültiger oder nicht erkannter MeaningPoint.

    Ursachen:
    - MeaningPoint-Typ unbekannt
    - Fehlende Felder (subject, object, etc.)
    - Nicht unterstützte Intention
    """

    def __init__(
        self, message: str, meaning_point_type: Optional[str] = None, **kwargs
    ):
        context = kwargs.get("context", {})
        context["meaning_point_type"] = meaning_point_type
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class PlanExecutionError(PlanningException):
    """
    Fehler während der Ausführung eines Plans.

    Ursachen:
    - SubGoal fehlgeschlagen
    - Timeout
    - Unerwarteter Zustand
    """

    def __init__(
        self,
        message: str,
        goal_type: Optional[str] = None,
        subgoal_index: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context["goal_type"] = goal_type
        context["subgoal_index"] = subgoal_index
        kwargs["context"] = context
        super().__init__(message, **kwargs)


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================


class ConfigurationException(KAIException):
    """Basis-Exception für Konfigurationsfehler."""


class InvalidConfigError(ConfigurationException):
    """
    Ungültige Konfiguration.

    Ursachen:
    - Fehlende Konfigurationsparameter
    - Ungültige Werte
    - Inkonsistente Einstellungen
    """


class MissingDependencyError(ConfigurationException):
    """
    Erforderliche Abhängigkeit fehlt.

    Ursachen:
    - spaCy-Modell nicht installiert
    - Neo4j nicht erreichbar
    - Python-Package fehlt
    """

    def __init__(self, message: str, dependency_name: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        context["dependency_name"] = dependency_name
        kwargs["context"] = context
        super().__init__(message, **kwargs)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def wrap_exception(
    exc: Exception, kai_exception_class: type[KAIException], message: str, **context
) -> KAIException:
    """
    Wandelt eine generische Exception in eine KAI-spezifische Exception um.

    Args:
        exc: Original-Exception
        kai_exception_class: Ziel-Exception-Klasse (z.B. Neo4jConnectionError)
        message: Benutzerdefinierte Fehlermeldung
        **context: Zusätzliche Kontextinformationen

    Returns:
        KAI-spezifische Exception mit Original-Exception verkettet

    Beispiel:
        try:
            driver.verify_connectivity()
        except Exception as e:
            raise wrap_exception(e, Neo4jConnectionError, "Verbindung fehlgeschlagen", uri="bolt://...")
    """
    return kai_exception_class(message=message, context=context, original_exception=exc)


def get_user_friendly_message(exc: Exception, include_details: bool = False) -> str:
    """
    Generiert eine benutzerfreundliche Fehlermeldung aus einer Exception.

    Args:
        exc: Exception-Objekt
        include_details: Ob technische Details angezeigt werden sollen (für Debug-Modus)

    Returns:
        Benutzerfreundliche Fehlermeldung in deutscher Sprache

    Beispiel:
        try:
            process_query()
        except Exception as e:
            user_message = get_user_friendly_message(e)
            show_error_dialog(user_message)
    """
    # Mapping von Exception-Typen zu benutzerfreundlichen Nachrichten
    friendly_messages = {
        Neo4jConnectionError: "[ERROR] Verbindung zur Wissensdatenbank fehlgeschlagen. Bitte stelle sicher, dass Neo4j läuft.",
        Neo4jQueryError: "[ERROR] Fehler beim Abfragen der Wissensdatenbank. Die Daten könnten inkonsistent sein.",
        Neo4jWriteError: "[ERROR] Konnte Wissen nicht speichern. Bitte versuche es erneut.",
        ParsingError: "[ERROR] Ich konnte deine Eingabe nicht verstehen. Bitte formuliere anders.",
        SpaCyModelError: "[ERROR] Das Sprachmodell ist nicht verfügbar. Bitte installiere 'de_core_news_sm'.",
        EmbeddingError: "[ERROR] Fehler bei der Bedeutungsanalyse. Das Embedding-Modell ist nicht verfügbar.",
        DocumentParseError: "[ERROR] Fehler beim Lesen der Dokumentdatei. Die Datei ist möglicherweise beschädigt oder im falschen Format.",
        ExtractionRuleError: "[ERROR] Fehler bei der Regelverarbeitung. Die Extraktionsregel ist ungültig.",
        PatternMatchingError: "[ERROR] Fehler beim Musterabgleich. Konnte kein passendes Muster finden.",
        ConceptNotFoundError: "🤔 Ich kenne dieses Konzept noch nicht. Möchtest du es mir beibringen?",
        RelationValidationError: "[ERROR] Ungültige Beziehung. Bitte überprüfe die Eingabe.",
        LogicEngineError: "[ERROR] Fehler bei der logischen Schlussfolgerung. Die Regeln könnten inkonsistent sein.",
        UnificationError: "[ERROR] Konnte Variablen nicht vereinheitlichen. Die Regel passt nicht auf die Fakten.",
        InferenceError: "🤔 Ich kann daraus keine Schlussfolgerung ziehen. Mir fehlen Informationen.",
        GraphTraversalError: "🤔 Ich finde keine Verbindung zwischen diesen Konzepten.",
        AbductiveReasoningError: "🤔 Ich kann keine plausible Erklärung finden. Mir fehlt Hintergrundwissen.",
        ProbabilisticReasoningError: "[ERROR] Fehler bei der Wahrscheinlichkeitsberechnung. Die Werte sind inkonsistent.",
        WorkingMemoryError: "[ERROR] Fehler im Arbeitsgedächtnis. Der Kontext ist verloren gegangen.",
        ConstraintReasoningError: "🤔 Ich kann keine gültige Lösung für das Constraint-Problem finden.",
        GoalPlanningError: "[ERROR] Ich konnte keinen Plan für deine Anfrage erstellen.",
        InvalidMeaningPointError: "[ERROR] Ich konnte deine Absicht nicht erkennen. Bitte formuliere klarer.",
        PlanExecutionError: "[ERROR] Fehler bei der Ausführung. Ein Teilschritt ist fehlgeschlagen.",
        InvalidConfigError: "[ERROR] Ungültige Konfiguration. Bitte überprüfe die Einstellungen.",
        MissingDependencyError: "[ERROR] Eine erforderliche Komponente fehlt. Bitte überprüfe die Installation.",
    }

    # Standard-Nachricht für unbekannte Exceptions
    default_message = "[ERROR] Ein unerwarteter Fehler ist aufgetreten."

    # Hole benutzerfreundliche Nachricht
    exc_type = type(exc)
    user_message = friendly_messages.get(exc_type, default_message)

    # Füge spezifische Details hinzu, falls verfügbar
    if isinstance(exc, ConceptNotFoundError) and exc.context.get("concept_name"):
        user_message = f"🤔 Ich kenne das Konzept '{exc.context['concept_name']}' noch nicht. Möchtest du es mir beibringen?"

    elif isinstance(exc, GraphTraversalError):
        source = exc.context.get("source_concept", "?")
        target = exc.context.get("target_concept", "?")
        user_message = (
            f"🤔 Ich finde keine Verbindung zwischen '{source}' und '{target}'."
        )

    elif (
        isinstance(exc, ProbabilisticReasoningError)
        and exc.context.get("probability_value") is not None
    ):
        prob = exc.context["probability_value"]
        user_message = f"[ERROR] Ungültige Wahrscheinlichkeit ({prob}). Werte müssen zwischen 0 und 1 liegen."

    # Füge technische Details hinzu, wenn erwünscht
    if include_details and isinstance(exc, KAIException):
        user_message += f"\n\n💡 Technische Details: {exc.message}"
        if exc.context:
            user_message += f"\n   Kontext: {exc.context}"

    return user_message


if __name__ == "__main__":
    # Test-Code für Exception-Hierarchie
    print("=== Testing KAI Exception Hierarchy ===\n")

    # Test 1: Basis-Exception
    try:
        raise KAIException("Generischer Fehler", context={"test": "value"})
    except KAIException as e:
        print(f"1. {e}\n")

    # Test 2: Database Exception mit Query-Kontext
    try:
        raise Neo4jQueryError(
            "Query fehlgeschlagen", query="MATCH (n) RETURN n", parameters={"limit": 10}
        )
    except Neo4jQueryError as e:
        print(f"2. {e}\n")

    # Test 3: Exception-Wrapping
    try:
        try:
            int("invalid")
        except ValueError as original:
            raise wrap_exception(
                original,
                ParsingError,
                "Konnte Eingabe nicht parsen",
                input_text="invalid",
            )
    except ParsingError as e:
        print(f"3. {e}\n")

    # Test 4: ConceptNotFoundError
    try:
        raise ConceptNotFoundError(
            "Konzept nicht gefunden", concept_name="unbekanntes_wort"
        )
    except ConceptNotFoundError as e:
        print(f"4. {e}\n")

    # Test 5: Exception-Verkettung (via 'from')
    try:
        try:
            raise ValueError("Original-Fehler")
        except ValueError as original:
            raise LogicEngineError("Inferenz fehlgeschlagen") from original
    except LogicEngineError as e:
        print(f"5. {e}")
        print(f"   Caused by: {e.__cause__}\n")

    print("=== Alle Tests erfolgreich ===")
