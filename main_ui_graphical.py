# main_ui_graphical.py
"""
Main entry point for KAI GUI application.

This module has been refactored into focused UI modules:
- ui/plan_monitor.py - PlanMonitor and TaskDisplay widgets
- ui/chat_interface.py - ChatInterface widget
- ui/main_window.py - MainWindow, AnalysisWindow, InnerPictureDisplay, and theme functions

This file now serves as a thin launcher that imports and starts the application.
"""

import sys

from PySide6.QtWidgets import QApplication

# WICHTIG: Encoding-Fix MUSS früh importiert werden
# Behebt Windows cp1252 -> UTF-8 Probleme für Unicode-Zeichen ([OK], [X], ->, etc.)
import kai_encoding_fix  # noqa: F401 (automatische Aktivierung beim Import)
from component_15_logging_config import setup_logging
from kai_config import get_config
from ui.main_window import MainWindow
from ui.themes import set_stylesheet


def main():
    """Main entry point for KAI GUI application."""
    setup_logging()
    app = QApplication(sys.argv)

    # Load theme from config
    cfg = get_config()
    theme = cfg.get("theme", "dark")
    set_stylesheet(app, theme)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
