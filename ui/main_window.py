# ui/main_window.py
"""
Main Window UI component for KAI application.

Contains MainWindow, AnalysisWindow, InnerPictureDisplay, and theme functions.
"""

import logging
import sys

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from component_1_netzwerk import KonzeptNetzwerk
from component_11_embedding_service import EmbeddingService
from kai_exceptions import KAIException, get_user_friendly_message
from kai_worker import KaiWorker
from logging_ui import LogViewerWindow
from settings_ui import SettingsDialog
from test_runner_ui import TestRunnerWindow
from ui.chat_interface import ChatInterface
from ui.plan_monitor import PlanMonitor
from ui.themes import set_stylesheet

logger = logging.getLogger(__name__)

# PHASE 3.4 (User Feedback Loop)
try:
    from component_51_feedback_handler import FeedbackHandler, FeedbackType

    FEEDBACK_HANDLER_AVAILABLE = True
except ImportError:
    FEEDBACK_HANDLER_AVAILABLE = False
    print("[UI] FeedbackHandler nicht verfuegbar - Feedback-Loop deaktiviert")

# Import ProofTreeWidget (with fallback)
try:
    from component_18_proof_tree_widget import ProofTreeWidget

    PROOF_TREE_AVAILABLE = True
except ImportError:
    PROOF_TREE_AVAILABLE = False
    print("[UI] ProofTreeWidget nicht verfuegbar - Beweisbaum-Tab wird nicht angezeigt")

# Import EpisodicMemoryWidget (with fallback)
try:
    from component_19_episodic_memory_widget import EpisodicMemoryWidget

    EPISODIC_MEMORY_WIDGET_AVAILABLE = True
except ImportError:
    EPISODIC_MEMORY_WIDGET_AVAILABLE = False
    print(
        "[UI] EpisodicMemoryWidget nicht verfuegbar - Episodisches-Gedaechtnis-Tab wird nicht angezeigt"
    )

# Import ContextVisualizationWidget (with fallback)
try:
    from context_visualization_widget import ContextVisualizationWidget

    CONTEXT_VISUALIZATION_AVAILABLE = True
except ImportError:
    CONTEXT_VISUALIZATION_AVAILABLE = False
    print(
        "[UI] ContextVisualizationWidget nicht verfuegbar - Kontext-Tracker-Tab wird nicht angezeigt"
    )

# Import SpatialGridWidget (with fallback)
try:
    from component_43_spatial_grid_widget import SpatialGridWidget

    SPATIAL_GRID_WIDGET_AVAILABLE = True
except ImportError:
    SPATIAL_GRID_WIDGET_AVAILABLE = False
    print(
        "[UI] SpatialGridWidget nicht verfuegbar - Raeumliches-Grid-Tab wird nicht angezeigt"
    )

# Import ResonanceViewWidget (with fallback)
try:
    from component_45_resonance_view_widget import ResonanceViewWidget

    RESONANCE_VIEW_AVAILABLE = True
except ImportError:
    RESONANCE_VIEW_AVAILABLE = False
    print(
        "[UI] ResonanceViewWidget nicht verfuegbar - Resonance-View-Tab wird nicht angezeigt"
    )

# Import ProductionTraceWidget (with fallback) - PHASE 8.2
try:
    from component_55_production_trace_widget import ProductionTraceWidget

    PRODUCTION_TRACE_AVAILABLE = True
except ImportError:
    PRODUCTION_TRACE_AVAILABLE = False
    print(
        "[UI] ProductionTraceWidget nicht verfuegbar - Production-Trace-Tab wird nicht angezeigt"
    )

# Import ABTestingDashboard (with fallback) - PHASE 8.3
try:
    from component_56_ab_testing_dashboard import ABTestingDashboard

    AB_TESTING_DASHBOARD_AVAILABLE = True
except ImportError:
    AB_TESTING_DASHBOARD_AVAILABLE = False
    print(
        "[UI] ABTestingDashboard nicht verfuegbar - A/B-Testing-Tab wird nicht angezeigt"
    )


class InnerPictureDisplay(QWidget):
    """Displays internal reasoning trace (thought process)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.label = QLabel("Inneres Bild (Gedankengang)")
        self.label.setStyleSheet(
            "font-weight: bold; font-size: 16px; margin-bottom: 5px;"
        )

        self.trace_view = QTextEdit()
        self.trace_view.setReadOnly(True)
        self.trace_view.setStyleSheet(
            "font-family: Consolas, monospace; color: #2ecc71;"
        )

        layout.addWidget(self.label)
        layout.addWidget(self.trace_view)

    def update_trace(self, trace_steps: list):
        if not trace_steps:
            self.trace_view.setText("Kein spezifischer Gedankengang fuer diese Antwort.")
        else:
            formatted_trace = "\n".join(f"-> {step}" for step in trace_steps)
            self.trace_view.setText(formatted_trace)

    @Slot(str)
    def update_trace_from_string(self, trace_str: str):
        """PHASE 2: Update trace from formatted string (Working Memory)"""
        self.trace_view.setText(trace_str)

    @Slot(str, str)
    def update_production_trace(self, rule_name: str, description: str):
        """
        PHASE 5: Update trace with Production System rule applications.

        Formatiert Regelanwendungen des Production Systems und fuegt sie
        zum bestehenden Trace hinzu.

        Args:
            rule_name: Name der angewendeten Regel
            description: Beschreibung der Regelanwendung
        """
        current_text = self.trace_view.toPlainText()

        # Formatiere Production System Eintrag
        production_entry = f"[PRODUCTION] {rule_name}: {description}"

        # Fuege zum bestehenden Trace hinzu (wenn vorhanden)
        if current_text and current_text.strip():
            updated_text = current_text + "\n" + production_entry
        else:
            updated_text = production_entry

        self.trace_view.setText(updated_text)

        # Auto-scroll zum Ende
        cursor = self.trace_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.trace_view.setTextCursor(cursor)
        self.trace_view.ensureCursorVisible()


class AnalysisWindow(QMainWindow):
    """Separates Fenster fuer Inneres Bild, Beweisbaum, Episodisches Gedaechtnis und Kontext-Tracker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KAI - Analyse & Visualisierungen")
        self.setGeometry(150, 150, 1000, 700)

        # Central Widget mit Tab-Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Tab 1: Inneres Bild
        self.inner_picture = InnerPictureDisplay()
        self.tab_widget.addTab(self.inner_picture, "Inneres Bild")

        # Tab 2: Beweisbaum (only if available)
        if PROOF_TREE_AVAILABLE:
            self.proof_tree_widget = ProofTreeWidget()
            self.tab_widget.addTab(self.proof_tree_widget, "Beweisbaum")
        else:
            self.proof_tree_widget = None  # type: ignore[assignment]

        # Tab 3: Episodisches Gedaechtnis (only if available)
        if EPISODIC_MEMORY_WIDGET_AVAILABLE:
            self.episodic_memory_widget = EpisodicMemoryWidget()
            self.tab_widget.addTab(
                self.episodic_memory_widget, "Episodisches Gedaechtnis"
            )
        else:
            self.episodic_memory_widget = None  # type: ignore[assignment]

        # Tab 4: Kontext-Tracker (only if available)
        if CONTEXT_VISUALIZATION_AVAILABLE:
            self.context_visualization_widget = ContextVisualizationWidget()
            self.tab_widget.addTab(self.context_visualization_widget, "Kontext-Tracker")
        else:
            self.context_visualization_widget = None  # type: ignore[assignment]

        # Tab 5: Raeumliches Grid (only if available)
        if SPATIAL_GRID_WIDGET_AVAILABLE:
            self.spatial_grid_widget = SpatialGridWidget()
            self.tab_widget.addTab(self.spatial_grid_widget, "Raeumliches Grid")
        else:
            self.spatial_grid_widget = None  # type: ignore[assignment]

        # Tab 6: Resonance View (only if available)
        if RESONANCE_VIEW_AVAILABLE:
            self.resonance_view_widget = ResonanceViewWidget()
            self.tab_widget.addTab(self.resonance_view_widget, "Resonance View")
        else:
            self.resonance_view_widget = None  # type: ignore[assignment]

        # Tab 7: Production Trace (only if available) - PHASE 8.2
        if PRODUCTION_TRACE_AVAILABLE:
            self.production_trace_widget = ProductionTraceWidget()
            self.tab_widget.addTab(self.production_trace_widget, "Regelanwendungen")
        else:
            self.production_trace_widget = None  # type: ignore[assignment]

        # Tab 8: A/B Testing Dashboard (only if available) - PHASE 8.3
        if AB_TESTING_DASHBOARD_AVAILABLE:
            self.ab_testing_dashboard = ABTestingDashboard(meta_learning_engine=None)
            self.tab_widget.addTab(self.ab_testing_dashboard, "A/B Testing")
        else:
            self.ab_testing_dashboard = None  # type: ignore[assignment]


class MainWindow(QMainWindow):
    """Main application window for KAI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KAI - Konzeptueller AI Prototyp (v3.0 Fokus-UI)")
        self.setGeometry(100, 100, 1200, 800)

        self.setup_kai_backend()
        self.create_menu_bar()
        self.create_status_bar()

        # Separates Analysis-Fenster (initial geschlossen)
        self.analysis_window = None

        # Central Widget mit vertikalem Splitter
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Vertikaler Splitter: PlanMonitor (oben) + Chat (unten)
        self.splitter = QSplitter(Qt.Orientation.Vertical)

        # PlanMonitor (kollabierbar)
        self.plan_monitor_container = QFrame()
        self.plan_monitor_container.setFrameShape(QFrame.Shape.StyledPanel)
        plan_monitor_layout = QVBoxLayout(self.plan_monitor_container)
        plan_monitor_layout.setContentsMargins(0, 0, 0, 0)

        # Header mit Toggle-Button
        plan_monitor_header = QWidget()
        header_layout = QHBoxLayout(plan_monitor_header)
        header_layout.setContentsMargins(5, 5, 5, 5)

        self.plan_monitor_toggle_btn = QPushButton("v Plan Monitor")
        self.plan_monitor_toggle_btn.setCheckable(True)
        self.plan_monitor_toggle_btn.setChecked(True)
        self.plan_monitor_toggle_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:checked {
                background-color: #2c3e50;
            }
        """
        )
        self.plan_monitor_toggle_btn.clicked.connect(self.toggle_plan_monitor)
        header_layout.addWidget(self.plan_monitor_toggle_btn)
        header_layout.addStretch()

        self.plan_monitor = PlanMonitor()
        plan_monitor_layout.addWidget(plan_monitor_header)
        plan_monitor_layout.addWidget(self.plan_monitor)

        # Chat Interface
        self.chat_interface = ChatInterface()

        # Splitter konfigurieren
        self.splitter.addWidget(self.plan_monitor_container)
        self.splitter.addWidget(self.chat_interface)
        self.splitter.setStretchFactor(0, 1)  # PlanMonitor: weniger Platz
        self.splitter.setStretchFactor(1, 3)  # Chat: mehr Platz
        self.splitter.setSizes([150, 650])  # Initial: 150px PlanMonitor, 650px Chat

        main_layout.addWidget(self.splitter)

        self.connect_signals()
        self.chat_interface.add_message(
            "KAI",
            "Hallo. Ich bin bereit. Nutze das Analyse-Fenster (Menue) fuer detaillierte Visualisierungen.",
        )

    def setup_kai_backend(self):
        try:
            self.netzwerk = KonzeptNetzwerk()
            embedding_service = EmbeddingService()
            self.kai_thread = QThread()
            self.kai_worker = KaiWorker(self.netzwerk, embedding_service)
            self.kai_worker.moveToThread(self.kai_thread)
            self.kai_thread.start()

            # PHASE 3.4 (User Feedback Loop): Initialize FeedbackHandler
            if FEEDBACK_HANDLER_AVAILABLE:
                try:
                    meta_learning = (
                        self.kai_worker.meta_learning
                        if hasattr(self.kai_worker, "meta_learning")
                        else None
                    )
                    self.feedback_handler = FeedbackHandler(
                        netzwerk=self.netzwerk, meta_learning=meta_learning
                    )
                    # Connect FeedbackHandler to ResponseFormatter
                    if hasattr(self.kai_worker, "response_formatter"):
                        self.kai_worker.response_formatter.feedback_handler = (
                            self.feedback_handler
                        )
                        print("[UI] FeedbackHandler mit ResponseFormatter verbunden")
                    print("[UI] FeedbackHandler initialisiert")
                except Exception as e:
                    print(
                        f"[UI] [WARNING] FeedbackHandler konnte nicht initialisiert werden: {e}"
                    )
                    self.feedback_handler = None  # type: ignore[assignment]
            else:
                self.feedback_handler = None  # type: ignore[assignment]

            # Pruefe ob Worker erfolgreich initialisiert wurde
            if not self.kai_worker.is_initialized_successfully:
                error_msg = (
                    self.kai_worker.initialization_error_message
                    or "[ERROR] Unbekannter Initialisierungsfehler"
                )
                print(f"[UI] {error_msg}")
            else:
                print("[UI] KAI Backend erfolgreich initialisiert.")
        except Exception as e:
            # Nutzerfreundliche Fehlermeldung generieren
            if isinstance(e, KAIException):
                user_friendly_msg = get_user_friendly_message(e, include_details=False)
                print(f"[UI] {user_friendly_msg}")
            else:
                print(
                    f"[UI] [ERROR] KRITISCHER FEHLER beim Starten des KAI Backends: {e}"
                )

    def create_menu_bar(self):
        """Erstellt die Menu-Leiste mit Einstellungen und Logging-Optionen"""
        menubar = self.menuBar()

        # === Ansicht-Menu (NEU: fuer UI-Optionen) ===
        view_menu = menubar.addMenu("&Ansicht")

        # Analyse-Fenster oeffnen
        analysis_window_action = QAction("&Analyse-Fenster oeffnen", self)
        analysis_window_action.setShortcut("Ctrl+A")
        analysis_window_action.setStatusTip(
            "Oeffnet das Analyse-Fenster (Inneres Bild, Beweisbaum, etc.)"
        )
        analysis_window_action.triggered.connect(self.open_analysis_window)
        view_menu.addAction(analysis_window_action)

        view_menu.addSeparator()

        # Plan Monitor ein-/ausblenden
        toggle_plan_action = QAction("Plan Monitor &umschalten", self)
        toggle_plan_action.setShortcut("Ctrl+P")
        toggle_plan_action.setStatusTip("Plan Monitor ein-/ausblenden")
        toggle_plan_action.triggered.connect(self.toggle_plan_monitor)
        view_menu.addAction(toggle_plan_action)

        # === Einstellungen-Menu ===
        settings_menu = menubar.addMenu("&Einstellungen")

        # Einstellungen-Dialog (Logging + Tests)
        settings_dialog_action = QAction("&Einstellungen oeffnen...", self)
        settings_dialog_action.setShortcut("Ctrl+Shift+S")
        settings_dialog_action.setStatusTip(
            "Oeffnet den Einstellungen-Dialog (Logging, Tests)"
        )
        settings_dialog_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(settings_dialog_action)

        # Test-Runner oeffnen (direkter Zugriff)
        test_runner_action = QAction("&Test-Runner oeffnen...", self)
        test_runner_action.setShortcut("Ctrl+T")
        test_runner_action.setStatusTip("Oeffnet den Test-Runner direkt")
        test_runner_action.triggered.connect(self.open_test_runner)
        settings_menu.addAction(test_runner_action)

        settings_menu.addSeparator()

        # Log-Viewer oeffnen
        viewer_action = QAction("Log-&Viewer oeffnen", self)
        viewer_action.setShortcut("Ctrl+L")
        viewer_action.setStatusTip("Oeffnet das Log-Viewer-Fenster")
        viewer_action.triggered.connect(self.open_log_viewer)
        settings_menu.addAction(viewer_action)

        # Quick-Access zu Log-Dateien
        open_logs_folder_action = QAction("Log-Ordner oeffnen", self)
        open_logs_folder_action.setStatusTip("Oeffnet den Ordner mit allen Log-Dateien")
        open_logs_folder_action.triggered.connect(self.open_logs_folder)
        settings_menu.addAction(open_logs_folder_action)

    def open_settings_dialog(self):
        """Oeffnet den neuen Einstellungen-Dialog mit Tabs (Logging + Tests)"""
        settings_dialog = SettingsDialog(self)

        # Callback wenn Einstellungen geaendert wurden
        def on_settings_changed(new_settings):
            print(f"[UI] Einstellungen aktualisiert: {new_settings}")
            self.chat_interface.add_message(
                "System", f"Einstellungen geaendert: {new_settings}"
            )

        # Callback fuer Theme-Aenderungen
        def on_theme_changed(theme: str):
            print(f"[UI] Theme geaendert zu: {theme}")
            # Apply new theme to application
            app = QApplication.instance()
            if app:
                set_stylesheet(app, theme)
            self.chat_interface.add_message(
                "System",
                f"Theme geaendert zu: {theme}. Einige Elemente aktualisieren sich sofort, andere nach Neustart.",
            )

        settings_dialog.settings_changed.connect(on_settings_changed)
        # Connect theme_changed signal from AppearanceTab
        if hasattr(settings_dialog, "appearance_tab"):
            settings_dialog.appearance_tab.theme_changed.connect(on_theme_changed)
        settings_dialog.exec()

    def open_test_runner(self):
        """Oeffnet den Test-Runner in einem separaten Fenster"""
        test_runner_window = TestRunnerWindow(self)
        test_runner_window.exec()

    def open_log_viewer(self):
        """Oeffnet das Log-Viewer-Fenster"""
        viewer = LogViewerWindow(self)
        viewer.exec()

    def open_logs_folder(self):
        """Oeffnet den Log-Ordner im Datei-Explorer"""
        import os
        import subprocess

        from component_15_logging_config import LOG_DIR

        try:
            if os.name == "nt":  # Windows
                os.startfile(LOG_DIR)
            elif os.name == "posix":  # macOS, Linux
                subprocess.call(
                    ["open" if sys.platform == "darwin" else "xdg-open", str(LOG_DIR)]
                )

            self.chat_interface.add_message("System", f"Log-Ordner geoeffnet: {LOG_DIR}")
        except Exception as e:
            self.chat_interface.add_message(
                "System", f"Fehler beim Oeffnen des Log-Ordners: {e}"
            )

    def toggle_plan_monitor(self):
        """Blendet den Plan Monitor ein oder aus"""
        is_visible = self.plan_monitor.isVisible()
        self.plan_monitor.setVisible(not is_visible)

        # Update Button-Text
        if is_visible:
            self.plan_monitor_toggle_btn.setText("v Plan Monitor")
            self.plan_monitor_toggle_btn.setChecked(False)
        else:
            self.plan_monitor_toggle_btn.setText("^ Plan Monitor")
            self.plan_monitor_toggle_btn.setChecked(True)

    def open_analysis_window(self):
        """Oeffnet das separate Analyse-Fenster"""
        if self.analysis_window is None:
            self.analysis_window = AnalysisWindow(self)
            # Verbinde Signale mit Analysis-Fenster
            self.connect_analysis_window_signals()

        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()

    def connect_analysis_window_signals(self):
        """Verbindet Worker-Signale mit dem Analysis-Fenster"""
        if self.analysis_window is None:
            return

        signals = self.kai_worker.signals

        # Inner Picture
        signals.inner_picture_update.connect(
            self.analysis_window.inner_picture.update_trace_from_string
        )

        # Production System Trace (PHASE 5: Production System Integration)
        signals.production_system_trace.connect(
            self.analysis_window.inner_picture.update_production_trace
        )

        # Production System Trace Widget (PHASE 8.2)
        if PRODUCTION_TRACE_AVAILABLE and self.analysis_window.production_trace_widget:
            signals.production_system_trace.connect(
                self.analysis_window.production_trace_widget.add_trace_entry
            )

        # Proof Tree
        if PROOF_TREE_AVAILABLE and self.analysis_window.proof_tree_widget:
            signals.proof_tree_update.connect(
                lambda proof_tree: self.update_analysis_proof_tree(proof_tree)
            )

        # Episodic Memory
        if (
            EPISODIC_MEMORY_WIDGET_AVAILABLE
            and self.analysis_window.episodic_memory_widget
        ):
            signals.episodic_data_update.connect(
                self.analysis_window.episodic_memory_widget.update_episodes
            )

    def update_analysis_proof_tree(self, proof_tree):
        """Updates proof tree in analysis window and switches to that tab"""
        if (
            self.analysis_window
            and PROOF_TREE_AVAILABLE
            and self.analysis_window.proof_tree_widget
        ):
            self.analysis_window.proof_tree_widget.set_proof_tree(proof_tree)
            # Switch to Beweisbaum tab
            self.analysis_window.tab_widget.setCurrentWidget(
                self.analysis_window.proof_tree_widget
            )

    def create_status_bar(self):
        """Erstellt die Statusleiste mit Neo4j-Verbindungsindikator"""
        status_bar = self.statusBar()

        # Neo4j-Status-Label
        self.neo4j_status_label = QLabel("Neo4j: Pruefe Verbindung...")
        self.neo4j_status_label.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        status_bar.addPermanentWidget(self.neo4j_status_label)

        # Timer fuer regelmaessige Status-Updates (alle 30 Sekunden)
        self.status_check_timer = QTimer(self)
        self.status_check_timer.timeout.connect(self.check_neo4j_status)
        self.status_check_timer.start(30000)  # 30 Sekunden

        # Initiale Ueberpruefung
        QTimer.singleShot(1000, self.check_neo4j_status)  # Nach 1 Sekunde

    def check_neo4j_status(self):
        """Prueft den Neo4j-Verbindungsstatus"""
        try:
            # Pruefe ob netzwerk verfuegbar ist
            if (
                hasattr(self, "netzwerk")
                and self.netzwerk
                and hasattr(self.netzwerk, "driver")
            ):
                # Versuche eine einfache Cypher-Query
                try:
                    with self.netzwerk.driver.session(database="neo4j") as session:
                        result = session.run("RETURN 1 AS test")
                        result.single()

                    # Verbindung erfolgreich
                    self.neo4j_status_label.setText("[OK] Neo4j: Verbunden")
                    self.neo4j_status_label.setStyleSheet(
                        "background-color: #2ecc71; color: white; padding: 2px 8px; "
                        "border-radius: 3px; font-size: 11px;"
                    )
                except Exception as e:
                    # Verbindung fehlgeschlagen
                    self.neo4j_status_label.setText("[X] Neo4j: Getrennt")
                    self.neo4j_status_label.setStyleSheet(
                        "background-color: #e74c3c; color: white; padding: 2px 8px; "
                        "border-radius: 3px; font-size: 11px;"
                    )
                    print(f"[UI] Neo4j-Verbindungsfehler: {e}")
            else:
                # Netzwerk nicht initialisiert
                self.neo4j_status_label.setText("[WARNING] Neo4j: Nicht initialisiert")
                self.neo4j_status_label.setStyleSheet(
                    "background-color: #f39c12; color: white; padding: 2px 8px; "
                    "border-radius: 3px; font-size: 11px;"
                )
        except Exception as e:
            # Fehler beim Status-Check
            self.neo4j_status_label.setText("[ERROR] Neo4j: Fehler")
            self.neo4j_status_label.setStyleSheet(
                "background-color: #95a5a6; color: white; padding: 2px 8px; "
                "border-radius: 3px; font-size: 11px;"
            )
            print(f"[UI] Fehler beim Neo4j-Status-Check: {e}")

    def connect_signals(self):
        # Strg+Enter zum Senden (statt Enter)
        send_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"), self.chat_interface.input_text
        )
        send_shortcut.activated.connect(self.send_query_to_kai)

        signals = self.kai_worker.signals
        signals.clear_goals.connect(self.plan_monitor.clear_goals)
        signals.set_main_goal.connect(self.plan_monitor.set_main_goal)
        signals.add_sub_goal.connect(self.plan_monitor.add_sub_goal)
        signals.update_sub_goal_status.connect(self.plan_monitor.update_sub_goal_status)
        signals.finished.connect(self.on_kai_finished)
        # Context-Update fuer Chat-Interface
        signals.context_update.connect(self.chat_interface.update_context)
        # File-Progress fuer Chat-Interface
        signals.file_progress_update.connect(self.chat_interface.update_file_progress)
        # Preview-Confirmation fuer File-Upload
        signals.preview_confirmation_needed.connect(
            self.show_preview_confirmation_dialog
        )

    def closeEvent(self, event):
        print("UI wird geschlossen. Speichere finales Netzwerk.")
        if self.netzwerk:
            self.netzwerk.close()
        if self.kai_thread:
            self.kai_thread.quit()
            self.kai_thread.wait()
        event.accept()

    def send_query_to_kai(self):
        query = self.chat_interface.get_input()
        if query:
            self.chat_interface.add_message("Du", query)
            self.chat_interface.clear_input()
            self.chat_interface.input_text.setEnabled(False)
            # Emit signal to process query in worker thread (non-blocking)
            self.kai_worker.signals.query_submitted.emit(query)

    @Slot(object)
    def on_kai_finished(self, response_obj):
        # Check for reprocessing request (from context manager)
        if response_obj.text == "__REPROCESS_QUERY__":
            # Extract corrected query from trace
            if response_obj.trace and len(response_obj.trace) > 0:
                corrected_query = response_obj.trace[0]
                # Reprocess the corrected query automatically via signal (non-blocking)
                self.kai_worker.signals.query_submitted.emit(corrected_query)
                return  # Don't display "__REPROCESS_QUERY__" or enable input yet
            else:
                # Fallback: Enable input if no corrected query found
                self.chat_interface.add_message(
                    "KAI", "Es gab einen Fehler bei der Verarbeitung."
                )
                self.chat_interface.input_text.setEnabled(True)
                self.chat_interface.input_text.setFocus()
                return

        # Normal response handling
        self.chat_interface.add_message("KAI", response_obj.text)

        # PHASE 3.4 (User Feedback Loop): Show feedback buttons if answer_id available
        if hasattr(response_obj, "answer_id") and response_obj.answer_id:
            self.chat_interface.show_feedback_buttons(response_obj.answer_id)

        self.chat_interface.input_text.setEnabled(True)
        self.chat_interface.input_text.setFocus()

    def process_feedback(self, answer_id: str, feedback_type: str):
        """
        PHASE 3.4 (User Feedback Loop): Verarbeitet User-Feedback fuer eine Antwort.

        Args:
            answer_id: Eindeutige ID der Antwort
            feedback_type: Art des Feedbacks ('correct', 'incorrect', 'unsure', 'partially_correct')
        """
        if not FEEDBACK_HANDLER_AVAILABLE or not self.feedback_handler:
            print("[UI] [WARNING] FeedbackHandler nicht verfuegbar")
            return

        # Map string to FeedbackType enum
        feedback_type_map = {
            "correct": FeedbackType.CORRECT,
            "incorrect": FeedbackType.INCORRECT,
            "unsure": FeedbackType.UNSURE,
            "partially_correct": FeedbackType.PARTIALLY_CORRECT,
        }

        fb_type = feedback_type_map.get(feedback_type)
        if not fb_type:
            print(f"[UI] [WARNING] Unbekannter Feedback-Type: {feedback_type}")
            return

        try:
            result = self.feedback_handler.process_user_feedback(
                answer_id=answer_id, feedback_type=fb_type
            )

            if result["success"]:
                print(f"[UI] Feedback verarbeitet: {feedback_type} fuer {answer_id[:8]}")
                # Optional: Stats anzeigen
                stats = self.feedback_handler.get_feedback_stats()
                print(
                    f"[UI] Feedback-Stats: Accuracy={stats['accuracy']:.2%}, "
                    f"Total={stats['total_feedbacks']}"
                )
            else:
                print(
                    f"[UI] [WARNING] Feedback-Verarbeitung fehlgeschlagen: "
                    f"{result.get('message', 'Unknown')}"
                )

        except Exception as e:
            print(f"[UI] [ERROR] Fehler beim Feedback-Processing: {e}")

    @Slot(str, str, int)
    def show_preview_confirmation_dialog(
        self, preview: str, file_name: str, char_count: int
    ):
        """
        PHASE 8 (Extended Features): Zeigt Preview-Dialog und fordert User-Bestaetigung an.

        Args:
            preview: Text-Preview (erste 500 Zeichen)
            file_name: Name der Datei
            char_count: Gesamtanzahl Zeichen in der Datei
        """
        # Erstelle Message Box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Datei-Ingestion bestaetigen")
        msg_box.setIcon(QMessageBox.Icon.Question)

        # Formatiere Text
        msg_box.setText(
            f"Moechtest du die Datei '{file_name}' ({char_count:,} Zeichen) einlesen und lernen?"
        )

        # Zeige Preview als detaillierten Text
        msg_box.setDetailedText(f"Vorschau:\n\n{preview}")

        # Buttons
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

        # Zeige Dialog (modal, blockiert UI-Thread bis Antwort)
        result = msg_box.exec()

        # Emittiere Antwort an Worker
        confirmed = result == QMessageBox.StandardButton.Yes
        self.kai_worker.signals.preview_confirmation_response.emit(confirmed)

        # Zeige Feedback in Chat
        if confirmed:
            self.chat_interface.add_message(
                "System", f"Starte Ingestion von '{file_name}'..."
            )
        else:
            self.chat_interface.add_message(
                "System", f"Ingestion von '{file_name}' abgebrochen."
            )
