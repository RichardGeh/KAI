# component_10_knowledge_extractor.py

from component_1_netzwerk import KonzeptNetzwerk
from spacy.tokens import Doc
from component_15_logging_config import get_logger

logger = get_logger(__name__)


class KnowledgeExtractor:
    """
    Analysiert Rohtext und stellt sicher, dass die grundlegenden
    Wort- und Konzept-Knoten im Wissensgraphen existieren und
    mit grammatikalischen Informationen angereichert sind.
    """

    def __init__(self, netzwerk: KonzeptNetzwerk):
        self.netzwerk = netzwerk
        self._network_available = self._check_network_availability()

    def _check_network_availability(self) -> bool:
        """Prüft, ob die Netzwerkverbindung verfügbar ist."""
        if not self.netzwerk:
            logger.error("KnowledgeExtractor: Kein Netzwerk-Objekt übergeben.")
            return False

        if not self.netzwerk.driver:
            logger.error("KnowledgeExtractor: Netzwerk-Driver ist nicht verfügbar.")
            return False

        return True

    def enrich_network_from_doc(self, doc: Doc) -> int:
        """
        Iteriert durch relevante Tokens eines Satzes, stellt deren Existenz
        im Graphen sicher und speichert grammatikalische Attribute.

        Returns:
            Anzahl der verarbeiteten Entitäten (0 bei Fehler)
        """
        if not doc:
            logger.warning("KnowledgeExtractor: Leeres Doc-Objekt übergeben.")
            return 0

        if not self._network_available:
            logger.warning(
                "KnowledgeExtractor: Netzwerk nicht verfügbar. "
                f"Überspringe Anreicherung für: '{doc.text[:50]}...'"
            )
            return 0

        logger.debug(f"Reichere Netzwerk mit Entitäten aus: '{doc.text}'")

        # Betrachte Nomen, Eigennamen und Verben, die keine Stoppwörter sind
        relevante_tokens = [
            t for t in doc if t.pos_ in ("NOUN", "PROPN", "VERB") and not t.is_stop
        ]

        processed_count = 0
        for token in relevante_tokens:
            lemma = token.lemma_.lower()
            if not lemma or len(lemma) < 2:  # Filtere sehr kurze Tokens
                continue

            try:
                # 1. Sicherstellen, dass die Wort/Konzept-Einheit existiert
                self.netzwerk.ensure_wort_und_konzept(lemma)

                # 2. Grammatikalische Informationen als Attribute am :Wort-Knoten speichern
                if token.pos_:
                    self.netzwerk.set_wort_attribut(lemma, "pos", token.pos_)

                processed_count += 1
            except Exception as e:
                logger.error(
                    f"KnowledgeExtractor: Fehler beim Verarbeiten von '{lemma}': {e}",
                    exc_info=True,
                )
                # Weitermachen trotz Fehler bei einzelnem Token
                continue

        if processed_count > 0:
            logger.debug(
                f"KnowledgeExtractor: {processed_count} Entitäten verarbeitet."
            )

        return processed_count

    def is_available(self) -> bool:
        """Public-Methode zur Verfügbarkeitsprüfung."""
        return self._network_available
