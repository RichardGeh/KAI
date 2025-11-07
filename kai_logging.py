"""
kai_logging.py

DEPRECATED: Wrapper für Abwärtskompatibilität.
Verwenden Sie stattdessen direkt component_12_logging_config.

Dieses Modul leitet einfach zum neuen Logging-System weiter.
"""

import logging

from component_15_logging_config import setup_logging as new_setup_logging


def setup_logging():
    """
    Konfiguriert das zentrale Logging-System für KAI.

    DEPRECATED: Diese Funktion ist ein Wrapper für das neue Logging-System.
    Verwenden Sie stattdessen: from component_12_logging_config import setup_logging
    """
    # Rufe neues Setup auf mit Standard-Einstellungen
    new_setup_logging(
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        enable_performance_logging=True,
    )

    # Stille zu gesprächige Logger von Drittanbieter-Bibliotheken
    logging.getLogger("spacy").setLevel(logging.WARNING)

    logger = logging.getLogger("KAI_SETUP")
    logger.info(
        "Logging initialisiert via kai_logging.py (DEPRECATED - verwende component_12_logging_config)"
    )
