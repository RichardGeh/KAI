"""
component_56_ab_testing_dashboard.py

A/B Testing Dashboard für Response Generation (PHASE 8.3)

Zeigt Performance-Vergleich zwischen Pipeline und Production System.
Ermöglicht Umschaltung zwischen Systemen für Testing.
"""

import logging
from typing import Any, Dict

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ABTestingDashboard(QWidget):
    """
    A/B Testing Dashboard für Response Generation (PHASE 8.3)

    Zeigt Side-by-Side Vergleich zwischen Pipeline und Production System.
    """

    def __init__(self, meta_learning_engine=None, parent=None):
        super().__init__(parent)

        self.meta_learning = meta_learning_engine

        # Production Weight (0.0 = Pipeline, 1.0 = Production System)
        self.production_weight = 0.5  # Default: 50/50 Split

        self.init_ui()

        # Auto-Refresh Timer (alle 5 Sekunden)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_stats)
        self.refresh_timer.start(5000)  # 5 seconds

    def init_ui(self):
        layout = QVBoxLayout(self)

        # === Header ===
        header_group = QGroupBox("A/B Testing: Pipeline vs. Production System")
        header_layout = QVBoxLayout()

        header_text = QLabel(
            "Vergleiche die Performance zwischen dem klassischen Pipeline-System\n"
            "und dem neuen Production-System für Response Generation."
        )
        header_text.setWordWrap(True)
        header_text.setStyleSheet("color: #95a5a6; font-size: 12px;")

        header_layout.addWidget(header_text)
        header_group.setLayout(header_layout)

        # === Production Weight Slider ===
        weight_group = QGroupBox("System-Auswahl (Production Weight)")
        weight_layout = QVBoxLayout()

        weight_info = QLabel(
            "Steuert, welches System für neue Queries verwendet wird:\n"
            "0% = Nur Pipeline | 50% = Random Split | 100% = Nur Production System"
        )
        weight_info.setWordWrap(True)
        weight_info.setStyleSheet("color: #95a5a6; font-size: 11px;")

        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setMinimum(0)
        self.weight_slider.setMaximum(100)
        self.weight_slider.setValue(int(self.production_weight * 100))
        self.weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.weight_slider.setTickInterval(10)
        self.weight_slider.valueChanged.connect(self.on_weight_changed)

        self.weight_label = QLabel(
            f"Production Weight: {self.production_weight:.2f} (50% Pipeline / 50% Production)"
        )
        self.weight_label.setStyleSheet("font-weight: bold; color: #3498db;")

        # Quick-Select Buttons
        button_row = QHBoxLayout()
        self.pipeline_only_btn = QPushButton("100% Pipeline")
        self.pipeline_only_btn.clicked.connect(lambda: self.set_weight(0.0))
        self.pipeline_only_btn.setStyleSheet("background-color: #2ecc71;")

        self.split_btn = QPushButton("50/50 Split")
        self.split_btn.clicked.connect(lambda: self.set_weight(0.5))
        self.split_btn.setStyleSheet("background-color: #3498db;")

        self.production_only_btn = QPushButton("100% Production")
        self.production_only_btn.clicked.connect(lambda: self.set_weight(1.0))
        self.production_only_btn.setStyleSheet("background-color: #e74c3c;")

        button_row.addWidget(self.pipeline_only_btn)
        button_row.addWidget(self.split_btn)
        button_row.addWidget(self.production_only_btn)

        weight_layout.addWidget(weight_info)
        weight_layout.addWidget(self.weight_slider)
        weight_layout.addWidget(self.weight_label)
        weight_layout.addLayout(button_row)
        weight_group.setLayout(weight_layout)

        # === Performance Comparison ===
        stats_group = QGroupBox("Performance-Vergleich")
        stats_layout = QVBoxLayout()

        self.stats_browser = QTextBrowser()
        self.stats_browser.setOpenExternalLinks(False)
        self.stats_browser.setHtml(self._generate_loading_html())

        refresh_btn = QPushButton("Statistiken aktualisieren")
        refresh_btn.clicked.connect(self.refresh_stats)
        refresh_btn.setStyleSheet(
            "background-color: #3498db; color: white; font-weight: bold; padding: 5px;"
        )

        stats_layout.addWidget(self.stats_browser)
        stats_layout.addWidget(refresh_btn)
        stats_group.setLayout(stats_layout)

        # === Layout ===
        layout.addWidget(header_group)
        layout.addWidget(weight_group)
        layout.addWidget(stats_group)

        # Initial Refresh
        QTimer.singleShot(500, self.refresh_stats)

    def on_weight_changed(self, value: int):
        """Callback für Slider-Änderung."""
        self.production_weight = value / 100.0
        self._update_weight_label()

    def set_weight(self, weight: float):
        """Setzt Production Weight programmatisch."""
        self.production_weight = weight
        self.weight_slider.setValue(int(weight * 100))
        self._update_weight_label()

    def _update_weight_label(self):
        """Aktualisiert das Weight-Label."""
        pipeline_pct = (1.0 - self.production_weight) * 100
        production_pct = self.production_weight * 100

        self.weight_label.setText(
            f"Production Weight: {self.production_weight:.2f} "
            f"({pipeline_pct:.0f}% Pipeline / {production_pct:.0f}% Production)"
        )

    @Slot()
    def refresh_stats(self):
        """Aktualisiert die Statistiken."""
        if self.meta_learning is None:
            self.stats_browser.setHtml(self._generate_no_data_html())
            return

        try:
            # Lade Vergleichsdaten
            comparison = self.meta_learning.get_generation_system_comparison()
            html = self._generate_comparison_html(comparison)
            self.stats_browser.setHtml(html)
            logger.debug("AB Testing Dashboard refreshed")
        except Exception as e:
            logger.error(f"Error refreshing AB Testing Dashboard: {e}")
            self.stats_browser.setHtml(self._generate_error_html(str(e)))

    def _generate_loading_html(self) -> str:
        """Generiert HTML für Lade-Zustand."""
        return """
        <html>
        <body style="font-family: 'Segoe UI', sans-serif; color: #ecf0f1;">
            <p style="color: #3498db; font-size: 14px;">
                Lade Statistiken...
            </p>
        </body>
        </html>
        """

    def _generate_no_data_html(self) -> str:
        """Generiert HTML wenn keine Daten verfügbar."""
        return """
        <html>
        <body style="font-family: 'Segoe UI', sans-serif; color: #ecf0f1;">
            <p style="color: #e74c3c; font-size: 14px; font-weight: bold;">
                ⚠ Meta-Learning Engine nicht verfügbar
            </p>
            <p style="color: #95a5a6; font-size: 12px;">
                Statistiken können nicht geladen werden.
            </p>
        </body>
        </html>
        """

    def _generate_error_html(self, error: str) -> str:
        """Generiert HTML für Fehler-Zustand."""
        return f"""
        <html>
        <body style="font-family: 'Segoe UI', sans-serif; color: #ecf0f1;">
            <p style="color: #e74c3c; font-size: 14px; font-weight: bold;">
                ❌ Fehler beim Laden der Statistiken
            </p>
            <p style="color: #95a5a6; font-size: 12px;">
                {error}
            </p>
        </body>
        </html>
        """

    def _generate_comparison_html(self, comparison: Dict[str, Any]) -> str:
        """Generiert HTML für Performance-Vergleich."""
        pipeline = comparison.get("pipeline", {})
        production = comparison.get("production", {})
        comp = comparison.get("comparison", {})

        # Winner bestimmen
        winner = comp.get("winner", "Unentschieden")
        winner_emoji = "🏆" if winner != "Unentschieden" else "🤝"

        # Metriken extrahieren
        p_queries = pipeline.get("queries_count", 0)
        pr_queries = production.get("queries_count", 0)

        p_conf = pipeline.get("avg_confidence", 0.0)
        pr_conf = production.get("avg_confidence", 0.0)

        p_time = pipeline.get("avg_response_time", 0.0)
        pr_time = production.get("avg_response_time", 0.0)

        p_success = pipeline.get("success_rate", 0.0)
        pr_success = production.get("success_rate", 0.0)

        # HTML generieren
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #ecf0f1; }}
                h2 {{ color: #3498db; margin-top: 5px; }}
                h3 {{ color: #2ecc71; margin-top: 15px; margin-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #7f8c8d; }}
                th {{ background-color: #2c3e50; color: #3498db; font-weight: bold; }}
                .winner {{ color: #2ecc71; font-weight: bold; font-size: 16px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .pipeline {{ color: #3498db; }}
                .production {{ color: #e74c3c; }}
                .better {{ color: #2ecc71; font-weight: bold; }}
                .worse {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h2>Performance-Vergleich</h2>

            <div class="winner">
                {winner_emoji} Winner: {winner}
            </div>

            <h3>Metriken</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metrik</th>
                        <th class="pipeline">Pipeline</th>
                        <th class="production">Production System</th>
                        <th>Differenz</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="metric-label">Queries</td>
                        <td>{p_queries}</td>
                        <td>{pr_queries}</td>
                        <td>{abs(p_queries - pr_queries)}</td>
                    </tr>
                    <tr>
                        <td class="metric-label">Durchschn. Confidence</td>
                        <td class="{'better' if p_conf > pr_conf else ''}">{p_conf:.3f}</td>
                        <td class="{'better' if pr_conf > p_conf else ''}">{pr_conf:.3f}</td>
                        <td>{abs(p_conf - pr_conf):.3f}</td>
                    </tr>
                    <tr>
                        <td class="metric-label">Durchschn. Response Time (s)</td>
                        <td class="{'better' if p_time < pr_time else 'worse'}">{p_time:.3f}</td>
                        <td class="{'better' if pr_time < p_time else 'worse'}">{pr_time:.3f}</td>
                        <td>{abs(p_time - pr_time):.3f}</td>
                    </tr>
                    <tr>
                        <td class="metric-label">Success Rate</td>
                        <td class="{'better' if p_success > pr_success else ''}">{p_success:.1%}</td>
                        <td class="{'better' if pr_success > p_success else ''}">{pr_success:.1%}</td>
                        <td>{abs(p_success - pr_success):.1%}</td>
                    </tr>
                </tbody>
            </table>

            <h3>Hinweise</h3>
            <ul>
                <li>Confidence: Höher ist besser</li>
                <li>Response Time: Niedriger ist besser</li>
                <li>Success Rate: Höher ist besser (basierend auf User-Feedback)</li>
                <li>Winner: Gesamtwertung basierend auf mehreren Metriken</li>
            </ul>
        </body>
        </html>
        """

        return html

    def get_production_weight(self) -> float:
        """Gibt aktuelles Production Weight zurück."""
        return self.production_weight


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

    widget = ABTestingDashboard()
    widget.setWindowTitle("A/B Testing Dashboard - Test")
    widget.resize(900, 700)
    widget.show()

    sys.exit(app.exec())
