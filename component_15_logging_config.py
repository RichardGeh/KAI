"""
component_12_logging_config.py

Zentrales Logging-System für KAI
Bietet strukturiertes Logging mit verschiedenen Log-Levels und Formatierungen.

Features:
- Konsolen- und Datei-basiertes Logging
- Unterschiedliche Log-Levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Strukturierte Formatierung mit Timestamps und Komponenten-Namen
- Performance-Tracking für kritische Operationen
- Kontextuelle Logging-Informationen
- UTF-8 Encoding-Fix für Windows (automatisch beim Import)

Verwendung:
    from component_15_logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Operation erfolgreich", extra={"user_query": "Was ist ein Hund?"})
    logger.error("Fehler beim Speichern", extra={"concept": "hund", "relation": "IS_A"})
"""

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Literal, MutableMapping, Optional, Tuple, Type

# WICHTIG: Encoding-Fix MUSS vor anderen Imports stehen
# Behebt Windows cp1252 -> UTF-8 Probleme für Unicode-Zeichen in Logs
import kai_encoding_fix  # noqa: F401 (automatische Aktivierung beim Import)

# Globale Logging-Konfiguration
LOG_DIR: Path = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_LOG_FILE: Path = LOG_DIR / "kai.log"
ERROR_LOG_FILE: Path = LOG_DIR / "kai_errors.log"
PERFORMANCE_LOG_FILE: Path = LOG_DIR / "kai_performance.log"

DEFAULT_LOG_LEVEL: int = logging.INFO
CONSOLE_LOG_LEVEL: int = logging.INFO
FILE_LOG_LEVEL: int = logging.DEBUG


class KAILogFormatter(logging.Formatter):
    """
    Benutzerdefinierter Formatter für strukturierte Log-Ausgaben.
    Fügt Farben für Konsolen-Output hinzu (optional).
    """

    # ANSI Color Codes für Konsolen-Output
    COLORS: Dict[str, str] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = False, include_extra: bool = True) -> None:
        self.use_colors: bool = use_colors
        self.include_extra: bool = include_extra

        # Format: [TIMESTAMP] [LEVEL] [COMPONENT] MESSAGE
        fmt = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        # Basis-Formatierung
        log_message = super().format(record)

        # Füge Extra-Informationen hinzu, falls vorhanden
        if self.include_extra and hasattr(record, "extra_info"):
            extra_str = " | ".join(f"{k}={v}" for k, v in record.extra_info.items())
            log_message += f" | {extra_str}"

        # Färbe Konsolen-Output
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            log_message = f"{color}{log_message}{reset}"

        return log_message


class WindowsSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Windows-sicherer RotatingFileHandler, der Rotation-Fehler graceful behandelt.

    Problem: Auf Windows kann os.rename() mit PermissionError fehlschlagen, wenn die
    Datei noch von einem anderen Prozess oder Thread geöffnet ist.

    Lösung: Bei Rotation-Fehlern wird die Rotation übersprungen und das Logging
    fortgesetzt. Nach mehreren Versuchen wird erneut versucht zu rotieren.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rotation_errors: int = 0
        self.max_rotation_errors: int = 10  # Nach 10 Fehlern erneut versuchen

    def doRollover(self) -> None:
        """
        Überschreibt doRollover() mit robustem Error Handling.
        Bei PermissionError wird die Rotation übersprungen.
        """
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore

        # Versuche Rotation durchzuführen
        try:
            # Standard-Rotation-Logik
            if self.backupCount > 0:
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                    dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                    if os.path.exists(sfn):
                        if os.path.exists(dfn):
                            os.remove(dfn)
                        os.rename(sfn, dfn)

                dfn = self.rotation_filename(f"{self.baseFilename}.1")
                if os.path.exists(dfn):
                    os.remove(dfn)

                # Kritischer Schritt: Umbenennen der aktuellen Datei
                self.rotate(self.baseFilename, dfn)

            # Rotation erfolgreich - Counter zurücksetzen
            self.rotation_errors = 0

        except (OSError, PermissionError) as e:
            # Rotation fehlgeschlagen - tracke Fehler
            self.rotation_errors += 1

            # Schreibe Fehler direkt in stderr (um Logging-Rekursion zu vermeiden)
            if self.rotation_errors == 1:
                # Nur beim ersten Fehler ausgeben, um Spam zu vermeiden
                print(
                    f"WARNING: Log rotation failed for {self.baseFilename}: {e}. "
                    f"Continuing without rotation.",
                    file=sys.stderr,
                )

            # Nach max_rotation_errors Versuchen wieder von vorne beginnen
            if self.rotation_errors >= self.max_rotation_errors:
                self.rotation_errors = 0

        # Öffne Stream wieder (immer, auch bei Fehler)
        if not self.stream:
            self.stream = self._open()

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """
        Bestimmt, ob Rollover durchgeführt werden soll.
        Überschrieben, um bei vielen Fehlern die Rotation zu deaktivieren.
        """
        # Standard-Logik
        if self.stream is None:
            self.stream = self._open()  # type: ignore[unreachable]

        if self.maxBytes > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= self.maxBytes:
                # Nur rotieren, wenn wir nicht zu viele Fehler hatten
                # (gibt dem System Zeit, File-Handles zu schließen)
                if self.rotation_errors < self.max_rotation_errors // 2:
                    return True

        return False


class PerformanceLogger:
    """
    Kontext-Manager für Performance-Tracking kritischer Operationen.

    Verwendung:
        with PerformanceLogger(logger, "Neo4j Query"):
            netzwerk.query_graph_for_facts("hund")
    """

    def __init__(
        self, logger: logging.Logger, operation_name: str, **context: Any
    ) -> None:
        self.logger: logging.Logger = logger
        self.operation_name: str = operation_name
        self.context: Dict[str, Any] = context
        self.start_time: Optional[datetime] = None

    def __enter__(self) -> "PerformanceLogger":
        self.start_time = datetime.now()
        self.logger.debug(
            f"START: {self.operation_name}", extra={"extra_info": self.context}
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        assert (
            self.start_time is not None
        ), "PerformanceLogger wurde nicht korrekt initialisiert"
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000

        if exc_type is None:
            self.logger.debug(
                f"END: {self.operation_name} (duration: {duration_ms:.2f}ms)",
                extra={"extra_info": {**self.context, "duration_ms": duration_ms}},
            )

            # Schreibe Performance-Log
            perf_logger = logging.getLogger("kai.performance")
            perf_logger.info(
                f"{self.operation_name}: {duration_ms:.2f}ms",
                extra={"extra_info": {**self.context, "duration_ms": duration_ms}},
            )
        else:
            self.logger.error(
                f"FAILED: {self.operation_name} (duration: {duration_ms:.2f}ms)",
                extra={
                    "extra_info": {
                        **self.context,
                        "duration_ms": duration_ms,
                        "error": str(exc_val),
                    }
                },
            )

        # Propagiere Exception weiter
        return False


class StructuredLogger(logging.LoggerAdapter[logging.Logger]):
    """
    Erweiterter Logger, der strukturierte Extra-Informationen unterstützt.
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, MutableMapping[str, Any]]:
        # Extrahiere 'extra' Dict und speichere als 'extra_info' im LogRecord
        extra = kwargs.get("extra", {})
        if extra:
            kwargs["extra"] = {"extra_info": extra}
        return msg, kwargs

    def log_exception(self, exc: Exception, message: str = "", **context: Any) -> None:
        """
        Loggt eine Exception mit vollständigem Traceback und Kontext.
        """
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        self.error(
            f"{message}: {type(exc).__name__}: {str(exc)}\n{tb_str}", extra=context
        )


def setup_logging(
    console_level: int = CONSOLE_LOG_LEVEL,
    file_level: int = FILE_LOG_LEVEL,
    log_file: Optional[Path] = None,
    enable_performance_logging: bool = True,
) -> None:
    """
    Konfiguriert das globale Logging-System für KAI.

    Args:
        console_level: Log-Level für Konsolen-Output
        file_level: Log-Level für Datei-Output
        log_file: Pfad zur Haupt-Log-Datei (Standard: logs/kai.log)
        enable_performance_logging: Aktiviert separates Performance-Logging
    """

    # Root-Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture alles, Filter auf Handler-Ebene

    # Entferne existierende Handler (verhindert Duplikate bei mehrfachem Setup)
    root_logger.handlers.clear()

    # === Konsolen-Handler ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(KAILogFormatter(use_colors=True, include_extra=True))
    root_logger.addHandler(console_handler)

    # === Haupt-Log-Datei ===
    file_path = log_file or DEFAULT_LOG_FILE
    file_handler = WindowsSafeRotatingFileHandler(
        file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10 MB
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(KAILogFormatter(use_colors=False, include_extra=True))
    root_logger.addHandler(file_handler)

    # === Error-Only Log-Datei ===
    error_handler = WindowsSafeRotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(KAILogFormatter(use_colors=False, include_extra=True))
    root_logger.addHandler(error_handler)

    # === Performance-Logger ===
    if enable_performance_logging:
        perf_logger = logging.getLogger("kai.performance")
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False  # Verhindere Duplikate im Root-Logger

        perf_handler = WindowsSafeRotatingFileHandler(
            PERFORMANCE_LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        perf_handler.setFormatter(KAILogFormatter(use_colors=False, include_extra=True))
        perf_logger.addHandler(perf_handler)

    # Initialisierungs-Log
    logger = logging.getLogger("kai.logging_config")
    logger.info(
        "Logging-System initialisiert",
        extra={
            "console_level": logging.getLevelName(console_level),
            "file_level": logging.getLevelName(file_level),
            "log_file": str(file_path),
            "performance_logging": enable_performance_logging,
        },
    )


def get_logger(name: str) -> StructuredLogger:
    """
    Erstellt einen strukturierten Logger für eine Komponente.

    Args:
        name: Name der Komponente (üblicherweise __name__)

    Returns:
        StructuredLogger-Instanz mit strukturierten Logging-Fähigkeiten

    Beispiel:
        logger = get_logger(__name__)
        logger.info("Query ausgeführt", extra={"query": "was ist ein hund"})
    """
    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger, {})


# === Convenience-Funktionen ===


def log_component_start(
    logger: StructuredLogger, component_name: str, **context: Any
) -> None:
    """Loggt den Start einer Komponenten-Operation."""
    logger.info(f"START: {component_name}", extra=context)


def log_component_end(
    logger: StructuredLogger, component_name: str, **context: Any
) -> None:
    """Loggt das erfolgreiche Ende einer Komponenten-Operation."""
    logger.info(f"END: {component_name}", extra=context)


def log_component_error(
    logger: StructuredLogger, component_name: str, error: Exception, **context: Any
) -> None:
    """Loggt einen Fehler in einer Komponente mit vollem Traceback."""
    logger.log_exception(error, message=f"ERROR in {component_name}", **context)


# Automatische Initialisierung beim Import
# Kann durch expliziten setup_logging()-Aufruf überschrieben werden
if not logging.getLogger().handlers:
    setup_logging()


if __name__ == "__main__":
    # Test-Code für Logging-System
    setup_logging(console_level=logging.DEBUG)

    logger = get_logger("test_component")

    logger.debug("Debug-Nachricht", extra={"test_param": "value1"})
    logger.info("Info-Nachricht", extra={"user_query": "Was ist ein Hund?"})
    logger.warning("Warnung", extra={"threshold": 15.0, "distance": 18.5})
    logger.error("Fehler", extra={"component": "Neo4j", "operation": "write"})

    # Performance-Tracking Test
    with PerformanceLogger(logger.logger, "Test-Operation", param1="value1", param2=42):
        import time

        time.sleep(0.1)

    # Exception-Logging Test
    try:
        raise ValueError("Test-Exception mit Kontext")
    except Exception as e:
        logger.log_exception(e, message="Test-Fehler", context_param="test_value")

    print("\nLog-Dateien erstellt in:", LOG_DIR.absolute())
