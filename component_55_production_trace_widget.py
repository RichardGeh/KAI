"""
component_55_production_trace_widget.py

Widget für Production System Trace Visualisierung (PHASE 8.2)

Zeigt Regelanwendungen in chronologischer Reihenfolge mit Details.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ProductionTraceWidget(QWidget):
    """
    Widget für Production System Trace Visualisierung (PHASE 8.2)

    Zeigt alle Regelanwendungen in chronologischer Reihenfolge an.
    Bei Klick auf eine Regel werden Details angezeigt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Trace-Daten (chronologisch)
        self.trace_entries: List[Dict[str, Any]] = []
        self.current_cycle = 0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Header: Statistiken ===
        header_group = QGroupBox("Trace Statistiken")
        header_layout = QHBoxLayout()

        self.stats_label = QLabel("Regelanwendungen: 0 | Zyklen: 0")
        self.stats_label.setStyleSheet(
            "color: #3498db; font-size: 12px; font-weight: bold;"
        )

        self.clear_button = QPushButton("Trace löschen")
        self.clear_button.clicked.connect(self.clear_trace)
        self.clear_button.setStyleSheet(
            "background-color: #e74c3c; color: white; font-weight: bold; padding: 5px;"
        )

        header_layout.addWidget(self.stats_label)
        header_layout.addStretch()
        header_layout.addWidget(self.clear_button)
        header_group.setLayout(header_layout)

        # === Splitter: Liste (links) + Details (rechts) ===
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Liste der Regelanwendungen
        list_group = QGroupBox("Regelanwendungen (chronologisch)")
        list_layout = QVBoxLayout()

        self.trace_list = QListWidget()
        self.trace_list.setAlternatingRowColors(True)
        self.trace_list.itemClicked.connect(self.show_trace_details)

        list_layout.addWidget(self.trace_list)
        list_group.setLayout(list_layout)

        # Details-Anzeige
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout()

        self.details_browser = QTextBrowser()
        self.details_browser.setOpenExternalLinks(False)
        self.details_browser.setHtml(
            "<i>Klicke auf eine Regelanwendung, um Details zu sehen.</i>"
        )

        details_layout.addWidget(self.details_browser)
        details_group.setLayout(details_layout)

        # Splitter konfigurieren
        splitter.addWidget(list_group)
        splitter.addWidget(details_group)
        splitter.setStretchFactor(0, 1)  # Liste: 50%
        splitter.setStretchFactor(1, 1)  # Details: 50%

        # === Layout ===
        layout.addWidget(header_group)
        layout.addWidget(splitter)

    @Slot(str, str)
    def add_trace_entry(self, rule_name: str, description: str):
        """
        Fügt eine neue Regelanwendung zum Trace hinzu.

        Args:
            rule_name: Name der angewendeten Regel
            description: Beschreibung (Format: "[category] description")
        """
        # Extrahiere Kategorie aus description
        category = "UNKNOWN"
        desc_text = description
        if description.startswith("["):
            end_bracket = description.find("]")
            if end_bracket > 0:
                category = description[1:end_bracket]
                desc_text = description[end_bracket + 1 :].strip()

        # Erstelle Trace-Eintrag
        entry = {
            "cycle": self.current_cycle,
            "rule_name": rule_name,
            "category": category,
            "description": desc_text,
            "timestamp": datetime.now(),
        }

        self.trace_entries.append(entry)

        # Update UI
        self._add_trace_item(entry)
        self._update_stats()

        logger.debug(
            f"[TRACE] Cycle {self.current_cycle}: {rule_name} [{category}] - {desc_text}"
        )

    def increment_cycle(self):
        """Erhöht den Zyklus-Zähler (für nächste Generation)."""
        self.current_cycle += 1

    def _add_trace_item(self, entry: Dict[str, Any]):
        """Fügt einen Eintrag zur Liste hinzu."""
        # Format: "[Zyklus X] [CATEGORY] RuleName: description"
        item_text = (
            f"[Zyklus {entry['cycle']}] "
            f"[{entry['category']}] {entry['rule_name']}: "
            f"{entry['description'][:50]}..."
        )

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, entry)  # Store entry data

        # Farbe je nach Kategorie
        if entry["category"] == "content_selection":
            item.setForeground(Qt.GlobalColor.cyan)
        elif entry["category"] == "lexicalization":
            item.setForeground(Qt.GlobalColor.green)
        elif entry["category"] == "discourse":
            item.setForeground(Qt.GlobalColor.yellow)
        elif entry["category"] == "syntax":
            item.setForeground(Qt.GlobalColor.magenta)

        self.trace_list.addItem(item)

        # Scroll to bottom (newest entry)
        self.trace_list.scrollToBottom()

    def _update_stats(self):
        """Aktualisiert die Statistik-Anzeige."""
        num_entries = len(self.trace_entries)
        num_cycles = self.current_cycle + 1

        self.stats_label.setText(
            f"Regelanwendungen: {num_entries} | Zyklen: {num_cycles}"
        )

    def show_trace_details(self, item):
        """Zeigt Details für ausgewählte Regelanwendung."""
        entry = item.data(Qt.ItemDataRole.UserRole)

        if entry is None:
            return

        # HTML-formatierte Details
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
                h2 {{ color: #3498db; margin-top: 5px; }}
                h3 {{ color: #2ecc71; margin-top: 15px; }}
                .field {{ margin: 5px 0; }}
                .label {{ font-weight: bold; color: #7f8c8d; }}
                .value {{ color: #ecf0f1; }}
                .category {{ color: #e74c3c; font-weight: bold; }}
                .timestamp {{ color: #95a5a6; font-size: 11px; }}
            </style>
        </head>
        <body>
            <h2>Regelanwendung: {entry['rule_name']}</h2>

            <div class="field">
                <span class="label">Zyklus:</span>
                <span class="value">{entry['cycle']}</span>
            </div>

            <div class="field">
                <span class="label">Kategorie:</span>
                <span class="category">{entry['category'].upper()}</span>
            </div>

            <div class="field">
                <span class="label">Beschreibung:</span>
                <span class="value">{entry['description']}</span>
            </div>

            <div class="field">
                <span class="label">Zeitstempel:</span>
                <span class="timestamp">{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}</span>
            </div>

            <h3>Hinweis</h3>
            <div class="field">
                <span class="value">
                Für detaillierte State-Informationen (Working Memory vorher/nachher)
                siehe den ProofTree-Tab, der alle Regelanwendungen mit State-Snapshots zeigt.
                </span>
            </div>
        </body>
        </html>
        """

        self.details_browser.setHtml(html)

    def clear_trace(self):
        """Löscht alle Trace-Einträge."""
        self.trace_entries.clear()
        self.trace_list.clear()
        self.current_cycle = 0
        self._update_stats()
        self.details_browser.setHtml(
            "<i>Trace gelöscht. Neue Regelanwendungen werden hier angezeigt.</i>"
        )
        logger.info("Production trace cleared")


if __name__ == "__main__":
    """Test-Code"""
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Dark Theme
    app.setStyleSheet(
        """
        QWidget {
            background-color: #34495e;
            color: #ecf0f1;
            font-size: 14px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #7f8c8d;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
    """
    )

    widget = ProductionTraceWidget()
    widget.setWindowTitle("Production Trace Viewer - Test")
    widget.resize(900, 600)
    widget.show()

    # Test-Daten
    widget.add_trace_entry("SELECT_IS_A_FACT", "[content_selection] Select IS_A facts")
    widget.add_trace_entry(
        "VERBALIZE_IS_A_SIMPLE", "[lexicalization] Erzeuge einfachen IS_A Satz"
    )
    widget.increment_cycle()
    widget.add_trace_entry(
        "SELECT_PROPERTY_FACT", "[content_selection] Select HAS_PROPERTY facts"
    )
    widget.add_trace_entry(
        "VERBALIZE_PROPERTY", "[lexicalization] Erzeuge Property-Beschreibung"
    )

    sys.exit(app.exec())
