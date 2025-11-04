"""
component_28_document_parser.py

Dokumentparser-Modul für KAI-System.
Extrahiert Text aus verschiedenen Dokumentformaten (DOCX, PDF).

Hauptkomponenten:
    - DocumentParser: Abstrakte Basisklasse für Dokumentparser
    - DocxParser: Parser für Microsoft Word-Dokumente (.docx)
    - PdfParser: Parser für PDF-Dokumente (.pdf)
    - DocumentParserFactory: Factory für automatische Parser-Auswahl

Verwendung:
    from component_28_document_parser import DocumentParserFactory

    # Automatische Parser-Auswahl
    parser = DocumentParserFactory.create_parser("document.pdf")
    text = parser.extract_text("document.pdf")

    # Oder direkt einen spezifischen Parser nutzen
    from component_28_document_parser import PdfParser
    pdf_parser = PdfParser()
    text = pdf_parser.extract_text("document.pdf")

Architektur:
    - Factory-Pattern für Parser-Auswahl
    - ABC (Abstract Base Class) für gemeinsames Interface
    - Robuste Fehlerbehandlung mit DocumentParseError
    - Logging für alle wichtigen Operationen
    - Layout-Awareness für bessere Textextraktion
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from kai_exceptions import DocumentParseError, MissingDependencyError

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================


class DocumentParser(ABC):
    """
    Abstrakte Basisklasse für alle Dokumentparser.

    Definiert das gemeinsame Interface für Textextraktion aus Dokumenten.
    Alle konkreten Parser müssen extract_text() implementieren.
    """

    @abstractmethod
    def extract_text(self, filepath: str) -> str:
        """
        Extrahiert Text aus einem Dokument.

        Args:
            filepath: Absoluter oder relativer Pfad zur Dokumentdatei

        Returns:
            Extrahierter Text als String

        Raises:
            DocumentParseError: Wenn Extraktion fehlschlägt
            FileNotFoundError: Wenn Datei nicht existiert
        """

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Gibt unterstützte Dateierweiterungen zurück.

        Returns:
            Liste von Dateierweiterungen (z.B. ['.pdf', '.PDF'])
        """

    def validate_file(self, filepath: str) -> None:
        """
        Validiert Existenz und Lesbarkeit einer Datei.

        Args:
            filepath: Pfad zur Datei

        Raises:
            FileNotFoundError: Wenn Datei nicht existiert
            DocumentParseError: Wenn Datei nicht lesbar ist
        """
        path = Path(filepath)

        if not path.exists():
            logger.error(f"Datei nicht gefunden: {filepath}")
            raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")

        if not path.is_file():
            logger.error(f"Pfad ist keine Datei: {filepath}")
            raise DocumentParseError("Pfad ist keine Datei", file_path=filepath)

        if not os.access(filepath, os.R_OK):
            logger.error(f"Datei nicht lesbar: {filepath}")
            raise DocumentParseError(
                "Datei nicht lesbar (fehlende Leserechte)", file_path=filepath
            )


# ============================================================================
# DOCX PARSER
# ============================================================================


class DocxParser(DocumentParser):
    """
    Parser für Microsoft Word-Dokumente (.docx).

    Nutzt die python-docx Bibliothek zur Textextraktion.
    Extrahiert Absätze sequenziell und erhält Zeilenumbrüche.

    Beispiel:
        parser = DocxParser()
        text = parser.extract_text("dokument.docx")
    """

    def __init__(self):
        """Initialisiert DocxParser und prüft Verfügbarkeit von python-docx."""
        try:
            import docx

            self._docx_module = docx
            logger.debug("DocxParser initialisiert (python-docx verfügbar)")
        except ImportError as e:
            logger.error("python-docx Library nicht installiert")
            raise MissingDependencyError(
                "python-docx ist nicht installiert. Installiere mit: pip install python-docx",
                dependency_name="python-docx",
                original_exception=e,
            )

    def get_supported_extensions(self) -> List[str]:
        """Unterstützte Erweiterungen: .docx"""
        return [".docx", ".DOCX"]

    def extract_text(self, filepath: str) -> str:
        """
        Extrahiert Text aus DOCX-Datei.

        Args:
            filepath: Pfad zur .docx Datei

        Returns:
            Extrahierter Text (Absätze getrennt durch doppelte Zeilenumbrüche)

        Raises:
            DocumentParseError: Bei Parsing-Fehler
        """
        self.validate_file(filepath)

        try:
            logger.info(f"Extrahiere Text aus DOCX: {filepath}")
            doc = self._docx_module.Document(filepath)

            # Extrahiere alle Absätze
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # Nur nicht-leere Absätze
                    paragraphs.append(text)

            extracted_text = "\n\n".join(paragraphs)

            logger.info(
                f"DOCX erfolgreich geparst: {len(paragraphs)} Absätze, "
                f"{len(extracted_text)} Zeichen"
            )

            return extracted_text

        except Exception as e:
            logger.error(f"Fehler beim Parsen von DOCX-Datei: {e}")
            raise DocumentParseError(
                f"Konnte DOCX-Datei nicht parsen: {str(e)}",
                file_path=filepath,
                file_format="docx",
                original_exception=e,
            )


# ============================================================================
# TXT PARSER
# ============================================================================


class TxtParser(DocumentParser):
    """
    Parser für Plain Text-Dateien (.txt).

    Liest Textdateien mit automatischer Encoding-Erkennung (UTF-8, Latin-1).

    Beispiel:
        parser = TxtParser()
        text = parser.extract_text("dokument.txt")
    """

    def __init__(self):
        """Initialisiert TxtParser."""
        logger.debug("TxtParser initialisiert")

    def get_supported_extensions(self) -> List[str]:
        """Unterstützte Erweiterungen: .txt"""
        return [".txt", ".TXT"]

    def extract_text(self, filepath: str) -> str:
        """
        Extrahiert Text aus TXT-Datei mit automatischer Encoding-Erkennung.

        Args:
            filepath: Pfad zur .txt Datei

        Returns:
            Extrahierter Text

        Raises:
            DocumentParseError: Bei Lese-Fehler
        """
        self.validate_file(filepath)

        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                logger.info(
                    f"Extrahiere Text aus TXT: {filepath} (Encoding: {encoding})"
                )
                with open(filepath, "r", encoding=encoding) as f:
                    text = f.read()

                logger.info(
                    f"TXT erfolgreich geparst: {len(text)} Zeichen (Encoding: {encoding})"
                )
                return text

            except UnicodeDecodeError:
                logger.debug(
                    f"Encoding {encoding} fehlgeschlagen, versuche nächstes..."
                )
                continue
            except Exception as e:
                logger.error(f"Fehler beim Parsen von TXT-Datei: {e}")
                raise DocumentParseError(
                    f"Konnte TXT-Datei nicht parsen: {str(e)}",
                    file_path=filepath,
                    file_format="txt",
                    original_exception=e,
                )

        # Wenn alle Encodings fehlschlagen
        logger.error(f"Konnte keine passende Encoding für {filepath} finden")
        raise DocumentParseError(
            f"Konnte TXT-Datei nicht dekodieren. Versuchte Encodings: {', '.join(encodings)}",
            file_path=filepath,
            file_format="txt",
        )


# ============================================================================
# MARKDOWN PARSER
# ============================================================================


class MarkdownParser(DocumentParser):
    """
    Parser für Markdown-Dateien (.md).

    Extrahiert reinen Text aus Markdown (entfernt Formatierung).

    Beispiel:
        parser = MarkdownParser()
        text = parser.extract_text("dokument.md")
    """

    def __init__(self):
        """Initialisiert MarkdownParser."""
        logger.debug("MarkdownParser initialisiert")

    def get_supported_extensions(self) -> List[str]:
        """Unterstützte Erweiterungen: .md"""
        return [".md", ".MD", ".markdown", ".MARKDOWN"]

    def extract_text(self, filepath: str) -> str:
        """
        Extrahiert Text aus Markdown-Datei.

        Entfernt Markdown-Formatierung (##, **, [], etc.) und behält nur Text.

        Args:
            filepath: Pfad zur .md Datei

        Returns:
            Extrahierter Text ohne Markdown-Formatierung

        Raises:
            DocumentParseError: Bei Lese-Fehler
        """
        self.validate_file(filepath)

        try:
            logger.info(f"Extrahiere Text aus Markdown: {filepath}")

            # Lese Datei (UTF-8 Standard für Markdown)
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Entferne Markdown-Formatierung
            import re

            text = raw_text
            # Entferne Code-Blöcke (```...```)
            text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
            # Entferne Inline-Code (`...`)
            text = re.sub(r"`[^`]+`", "", text)
            # Entferne Headers (# ## ###)
            text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
            # Entferne Links ([text](url)) -> behält nur text
            text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
            # Entferne Bilder (![alt](url))
            text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)
            # Entferne Bold/Italic (**, *, __, _)
            text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
            text = re.sub(r"\*([^\*]+)\*", r"\1", text)
            text = re.sub(r"__([^_]+)__", r"\1", text)
            text = re.sub(r"_([^_]+)_", r"\1", text)
            # Entferne Listen-Marker (-, *, +, 1.)
            text = re.sub(r"^\s*[-\*\+]\s+", "", text, flags=re.MULTILINE)
            text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
            # Entferne Blockquotes (>)
            text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
            # Entferne horizontale Linien (---, ***)
            text = re.sub(r"^[-\*]{3,}$", "", text, flags=re.MULTILINE)

            logger.info(f"Markdown erfolgreich geparst: {len(text)} Zeichen")

            return text.strip()

        except UnicodeDecodeError as e:
            logger.error(f"Encoding-Fehler beim Parsen von Markdown: {e}")
            raise DocumentParseError(
                f"Konnte Markdown-Datei nicht dekodieren (erwartet UTF-8): {str(e)}",
                file_path=filepath,
                file_format="md",
                original_exception=e,
            )
        except Exception as e:
            logger.error(f"Fehler beim Parsen von Markdown-Datei: {e}")
            raise DocumentParseError(
                f"Konnte Markdown-Datei nicht parsen: {str(e)}",
                file_path=filepath,
                file_format="md",
                original_exception=e,
            )


# ============================================================================
# HTML PARSER
# ============================================================================


class HtmlParser(DocumentParser):
    """
    Parser für HTML-Dateien (.html).

    Extrahiert reinen Text aus HTML (entfernt alle Tags).

    Beispiel:
        parser = HtmlParser()
        text = parser.extract_text("dokument.html")
    """

    def __init__(self):
        """Initialisiert HtmlParser und prüft Verfügbarkeit von html.parser."""
        logger.debug("HtmlParser initialisiert")

    def get_supported_extensions(self) -> List[str]:
        """Unterstützte Erweiterungen: .html, .htm"""
        return [".html", ".HTML", ".htm", ".HTM"]

    def extract_text(self, filepath: str) -> str:
        """
        Extrahiert Text aus HTML-Datei.

        Entfernt alle HTML-Tags und behält nur Text-Inhalt.

        Args:
            filepath: Pfad zur .html Datei

        Returns:
            Extrahierter Text ohne HTML-Tags

        Raises:
            DocumentParseError: Bei Parsing-Fehler
        """
        self.validate_file(filepath)

        try:
            logger.info(f"Extrahiere Text aus HTML: {filepath}")

            # Lese Datei
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()

            # HTML-Parser
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                """Extrahiert reinen Text aus HTML."""

                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip_tags = {"script", "style", "head"}
                    self.current_tag = None

                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag

                def handle_endtag(self, tag):
                    if tag in (
                        "p",
                        "div",
                        "br",
                        "li",
                        "tr",
                        "h1",
                        "h2",
                        "h3",
                        "h4",
                        "h5",
                        "h6",
                    ):
                        self.text.append("\n")
                    self.current_tag = None

                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        text = data.strip()
                        if text:
                            self.text.append(text)

                def get_text(self):
                    return " ".join(self.text)

            extractor = TextExtractor()
            extractor.feed(html_content)
            text = extractor.get_text()

            logger.info(f"HTML erfolgreich geparst: {len(text)} Zeichen")

            return text.strip()

        except UnicodeDecodeError as e:
            logger.error(f"Encoding-Fehler beim Parsen von HTML: {e}")
            raise DocumentParseError(
                f"Konnte HTML-Datei nicht dekodieren (erwartet UTF-8): {str(e)}",
                file_path=filepath,
                file_format="html",
                original_exception=e,
            )
        except Exception as e:
            logger.error(f"Fehler beim Parsen von HTML-Datei: {e}")
            raise DocumentParseError(
                f"Konnte HTML-Datei nicht parsen: {str(e)}",
                file_path=filepath,
                file_format="html",
                original_exception=e,
            )


# ============================================================================
# PDF PARSER
# ============================================================================


class PdfParser(DocumentParser):
    """
    Parser für PDF-Dokumente (.pdf).

    Nutzt die pdfplumber Bibliothek zur Textextraktion.
    Extrahiert Text Seite für Seite mit Layout-Awareness für bessere Qualität.

    Beispiel:
        parser = PdfParser()
        text = parser.extract_text("dokument.pdf")
    """

    def __init__(self):
        """Initialisiert PdfParser und prüft Verfügbarkeit von pdfplumber."""
        try:
            import pdfplumber

            self._pdfplumber_module = pdfplumber
            logger.debug("PdfParser initialisiert (pdfplumber verfügbar)")
        except ImportError as e:
            logger.error("pdfplumber Library nicht installiert")
            raise MissingDependencyError(
                "pdfplumber ist nicht installiert. Installiere mit: pip install pdfplumber",
                dependency_name="pdfplumber",
                original_exception=e,
            )

    def get_supported_extensions(self) -> List[str]:
        """Unterstützte Erweiterungen: .pdf"""
        return [".pdf", ".PDF"]

    def extract_text(self, filepath: str) -> str:
        """
        Extrahiert Text aus PDF-Datei.

        Args:
            filepath: Pfad zur .pdf Datei

        Returns:
            Extrahierter Text (Seiten getrennt durch doppelte Zeilenumbrüche)

        Raises:
            DocumentParseError: Bei Parsing-Fehler
        """
        self.validate_file(filepath)

        try:
            logger.info(f"Extrahiere Text aus PDF: {filepath}")

            pages_text = []
            with self._pdfplumber_module.open(filepath) as pdf:
                total_pages = len(pdf.pages)
                logger.debug(f"PDF hat {total_pages} Seiten")

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()

                    if page_text:
                        pages_text.append(page_text.strip())
                        logger.debug(
                            f"Seite {page_num}/{total_pages}: "
                            f"{len(page_text)} Zeichen extrahiert"
                        )
                    else:
                        logger.warning(
                            f"Seite {page_num}/{total_pages}: Kein Text gefunden "
                            f"(möglicherweise Bild-basiert)"
                        )

            extracted_text = "\n\n".join(pages_text)

            logger.info(
                f"PDF erfolgreich geparst: {total_pages} Seiten, "
                f"{len(extracted_text)} Zeichen"
            )

            return extracted_text

        except Exception as e:
            logger.error(f"Fehler beim Parsen von PDF-Datei: {e}")
            raise DocumentParseError(
                f"Konnte PDF-Datei nicht parsen: {str(e)}",
                file_path=filepath,
                file_format="pdf",
                original_exception=e,
            )


# ============================================================================
# DOCUMENT PARSER FACTORY
# ============================================================================


class DocumentParserFactory:
    """
    Factory-Klasse für automatische Parser-Auswahl basierend auf Dateierweiterung.

    Unterstützt:
        - .docx → DocxParser (Microsoft Word)
        - .pdf  → PdfParser (PDF-Dokumente)
        - .txt  → TxtParser (Plain Text)
        - .md   → MarkdownParser (Markdown)
        - .html → HtmlParser (HTML-Dokumente)

    Beispiel:
        parser = DocumentParserFactory.create_parser("dokument.pdf")
        text = parser.extract_text("dokument.pdf")
    """

    # Mapping von Dateierweiterungen zu Parser-Klassen
    _PARSER_REGISTRY = {
        ".docx": DocxParser,
        ".DOCX": DocxParser,
        ".pdf": PdfParser,
        ".PDF": PdfParser,
        ".txt": TxtParser,
        ".TXT": TxtParser,
        ".md": MarkdownParser,
        ".MD": MarkdownParser,
        ".markdown": MarkdownParser,
        ".MARKDOWN": MarkdownParser,
        ".html": HtmlParser,
        ".HTML": HtmlParser,
        ".htm": HtmlParser,
        ".HTM": HtmlParser,
    }

    @classmethod
    def create_parser(cls, filepath: str) -> DocumentParser:
        """
        Erstellt passenden Parser für gegebene Datei.

        Args:
            filepath: Pfad zur Datei (Erweiterung wird ausgewertet)

        Returns:
            Instanz des passenden DocumentParser

        Raises:
            DocumentParseError: Wenn Dateierweiterung nicht unterstützt wird
        """
        extension = Path(filepath).suffix

        if not extension:
            logger.error(f"Datei hat keine Erweiterung: {filepath}")
            raise DocumentParseError("Datei hat keine Erweiterung", file_path=filepath)

        parser_class = cls._PARSER_REGISTRY.get(extension)

        if parser_class is None:
            supported = ", ".join(cls._PARSER_REGISTRY.keys())
            logger.error(
                f"Nicht unterstützte Dateierweiterung: {extension}. "
                f"Unterstützt: {supported}"
            )
            raise DocumentParseError(
                f"Dateierweiterung '{extension}' wird nicht unterstützt. "
                f"Unterstützte Formate: {supported}",
                file_path=filepath,
                file_format=extension,
            )

        logger.debug(f"Erstelle {parser_class.__name__} für {filepath}")
        return parser_class()

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Gibt Liste aller unterstützten Dateierweiterungen zurück.

        Returns:
            Liste von Erweiterungen (z.B. ['.docx', '.pdf'])
        """
        return list(cls._PARSER_REGISTRY.keys())

    @classmethod
    def is_supported(cls, filepath: str) -> bool:
        """
        Prüft, ob Dateiformat unterstützt wird.

        Args:
            filepath: Pfad zur Datei

        Returns:
            True wenn Format unterstützt wird, sonst False
        """
        extension = Path(filepath).suffix
        return extension in cls._PARSER_REGISTRY


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def extract_text_from_document(filepath: str) -> str:
    """
    Convenience-Funktion für schnelle Textextraktion.

    Wählt automatisch passenden Parser und extrahiert Text.

    Args:
        filepath: Pfad zum Dokument

    Returns:
        Extrahierter Text

    Raises:
        DocumentParseError: Bei Parsing-Fehler oder nicht unterstütztem Format

    Beispiel:
        text = extract_text_from_document("mein_dokument.pdf")
    """
    parser = DocumentParserFactory.create_parser(filepath)
    return parser.extract_text(filepath)


def get_document_info(filepath: str) -> dict:
    """
    Gibt Metainformationen über ein Dokument zurück.

    Args:
        filepath: Pfad zum Dokument

    Returns:
        Dictionary mit Informationen (Größe, Format, etc.)

    Beispiel:
        info = get_document_info("dokument.pdf")
        print(f"Format: {info['format']}, Größe: {info['size_bytes']} Bytes")
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")

    return {
        "filepath": str(path.absolute()),
        "filename": path.name,
        "format": path.suffix,
        "size_bytes": path.stat().st_size,
        "is_supported": DocumentParserFactory.is_supported(filepath),
    }


# ============================================================================
# MAIN (für Testing)
# ============================================================================

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=== DocumentParser Module Test ===\n")

    # Test 1: Unterstützte Formate
    print("1. Unterstützte Formate:")
    extensions = DocumentParserFactory.get_supported_extensions()
    print(f"   {extensions}\n")

    # Test 2: Format-Prüfung
    print("2. Format-Prüfung:")
    test_files = ["test.pdf", "test.docx", "test.txt"]
    for filename in test_files:
        is_supported = DocumentParserFactory.is_supported(filename)
        print(
            f"   {filename}: {'✓ unterstützt' if is_supported else '✗ nicht unterstützt'}"
        )

    print("\n=== Test abgeschlossen ===")
