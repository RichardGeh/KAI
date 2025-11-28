# ui/chat_interface.py
"""
Chat Interface UI component for KAI application.

Contains ChatInterface widget for user interaction with chat history,
input field, feedback buttons, and context display.
"""

import logging

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# PHASE 3.4 (User Feedback Loop)
try:
    from component_51_feedback_handler import FeedbackHandler, FeedbackType

    FEEDBACK_HANDLER_AVAILABLE = True
except ImportError:
    FEEDBACK_HANDLER_AVAILABLE = False


class ChatInterface(QWidget):
    """Der Chat-Bereich am unteren Rand des Fensters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.history = QTextEdit()
        self.history.setReadOnly(True)

        # PHASE 8 (Extended Features): Progress Bar für Datei-Ingestion
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setVisible(False)
        self.file_progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #34495e;
                border-radius: 5px;
                text-align: center;
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """
        )
        self.file_progress_label = QLabel("")
        self.file_progress_label.setVisible(False)
        self.file_progress_label.setStyleSheet(
            "color: #3498db; font-size: 11px; padding: 2px;"
        )

        # PHASE 2 (Multi-Turn): Context-Anzeige oberhalb der Eingabezeile
        self.context_label = QLabel("")
        self.context_label.setStyleSheet(
            """
            background-color: #2c3e50;
            color: #f39c12;
            padding: 5px;
            border-radius: 3px;
            font-size: 11px;
            font-style: italic;
        """
        )
        self.context_label.setVisible(False)  # Initial versteckt
        self.context_label.setWordWrap(True)

        # Input-Bereich: Mehrzeiliges Textfeld statt einzeilig
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "Stelle eine Frage an KAI... (Strg+Enter zum Senden)"
        )
        self.input_text.setMaximumHeight(100)  # Begrenzt auf ~4 Zeilen
        self.input_text.setMinimumHeight(60)  # Mindestens ~2 Zeilen
        self.input_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #2c3e50;
                border: 2px solid #7f8c8d;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QTextEdit:focus {
                border: 2px solid #3498db;
            }
        """
        )

        # Bottom controls layout (Datei-Button + Checkbox)
        controls_layout = QHBoxLayout()

        # File-Picker-Button (kompakt, nur Icon)
        self.file_picker_button = QPushButton("[+]")
        self.file_picker_button.setToolTip(
            "Datei einlesen (DOCX, PDF, TXT, MD, HTML)\nUnterstützt auch mehrere Dateien"
        )
        self.file_picker_button.setFixedSize(40, 40)
        self.file_picker_button.setStyleSheet(
            """
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: 2px solid #7f8c8d;
                border-radius: 6px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #3498db;
                border: 2px solid #2980b9;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """
        )
        self.file_picker_button.clicked.connect(self._on_file_picker_clicked)

        self.curiosity_checkbox = QCheckBox("Aktives Nachfragen")
        self.curiosity_checkbox.setChecked(True)
        self.curiosity_checkbox.setToolTip(
            "Wenn aktiviert, fragt KAI von sich aus nach, wenn es Wörter nicht kennt."
        )

        controls_layout.addWidget(self.file_picker_button)
        controls_layout.addWidget(self.curiosity_checkbox)
        controls_layout.addStretch()  # Push alles nach links

        # PHASE 3.4 (User Feedback Loop): Feedback-Buttons
        self.feedback_widget = self._create_feedback_widget()
        self.current_answer_id = None  # Track aktuellen Answer für Feedback

        layout.addWidget(self.history)
        layout.addWidget(self.file_progress_label)  # Progress Label
        layout.addWidget(self.file_progress_bar)  # Progress Bar
        layout.addWidget(self.feedback_widget)  # Feedback-Buttons (dynamisch ein/aus)
        layout.addWidget(self.context_label)  # Context-Label zwischen History und Input
        layout.addWidget(self.input_text)  # Mehrzeiliges Eingabefeld
        layout.addLayout(controls_layout)  # Datei-Button + Checkbox

    def _create_feedback_widget(self):
        """
        PHASE 3.4: Erstellt Feedback-Widget mit Buttons

        Returns:
            QWidget mit Feedback-Buttons
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Label
        label = QLabel("War diese Antwort hilfreich?")
        label.setStyleSheet("color: #95a5a6; font-size: 12px; font-weight: bold;")

        # Buttons
        button_style = """
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: 2px solid #7f8c8d;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #3498db;
                border: 2px solid #2980b9;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """

        self.feedback_btn_correct = QPushButton("[OK] Richtig")
        self.feedback_btn_correct.setStyleSheet(button_style)
        self.feedback_btn_correct.setToolTip("Diese Antwort war korrekt")
        self.feedback_btn_correct.clicked.connect(
            lambda: self._on_feedback_clicked("correct")
        )

        self.feedback_btn_incorrect = QPushButton("[X] Falsch")
        self.feedback_btn_incorrect.setStyleSheet(button_style)
        self.feedback_btn_incorrect.setToolTip("Diese Antwort war falsch")
        self.feedback_btn_incorrect.clicked.connect(
            lambda: self._on_feedback_clicked("incorrect")
        )

        self.feedback_btn_unsure = QPushButton("[?] Unsicher")
        self.feedback_btn_unsure.setStyleSheet(button_style)
        self.feedback_btn_unsure.setToolTip("Ich bin mir bei dieser Antwort unsicher")
        self.feedback_btn_unsure.clicked.connect(
            lambda: self._on_feedback_clicked("unsure")
        )

        self.feedback_btn_partial = QPushButton("[/] Teilweise")
        self.feedback_btn_partial.setStyleSheet(button_style)
        self.feedback_btn_partial.setToolTip("Diese Antwort war teilweise korrekt")
        self.feedback_btn_partial.clicked.connect(
            lambda: self._on_feedback_clicked("partially_correct")
        )

        layout.addWidget(label)
        layout.addWidget(self.feedback_btn_correct)
        layout.addWidget(self.feedback_btn_incorrect)
        layout.addWidget(self.feedback_btn_unsure)
        layout.addWidget(self.feedback_btn_partial)
        layout.addStretch()

        # Widget-Stil
        widget.setStyleSheet(
            """
            QWidget {
                background-color: #2c3e50;
                border: 2px solid #7f8c8d;
                border-radius: 8px;
                padding: 5px;
            }
        """
        )

        # Initial versteckt
        widget.setVisible(False)

        return widget

    def _on_feedback_clicked(self, feedback_type: str):
        """
        PHASE 3.4: Handler für Feedback-Button Klicks

        Args:
            feedback_type: 'correct', 'incorrect', 'unsure', 'partially_correct'
        """
        if not self.current_answer_id:
            logger.warning("Kein current_answer_id für Feedback vorhanden")
            return

        # Emit [?]ignal an MainWindow
        # Das MainWindow sollte ein [?]ignal haben: feedback_given(answer_id, feedback_type)
        # Wir nutzen parent() um zum MainWindow zu kommen
        main_window = self.parent()
        while main_window and not isinstance(main_window, QMainWindow):
            main_window = main_window.parent()

        if main_window and hasattr(main_window, "process_feedback"):
            main_window.process_feedback(self.current_answer_id, feedback_type)

        # Zeige Danke-Nachricht und blende Buttons aus
        self._show_feedback_thanks(feedback_type)

    def _show_feedback_thanks(self, feedback_type: str):
        """Zeigt Danke-Nachricht und blendet Feedback-Widget aus"""
        # Map feedback_type zu ASCII icons
        icon_map = {
            "correct": "[OK]",
            "incorrect": "[X]",
            "unsure": "[?]",
            "partially_correct": "[/]",
        }
        icon = icon_map.get(feedback_type, "")

        # Zeige Danke-Nachricht im Chat
        self.history.append(
            f'<i style="color:#95a5a6; font-size:12px;">{icon} '
            f"Danke für dein Feedback! Ich lerne daraus.</i>"
        )

        # Blende Feedback-Widget aus
        self.feedback_widget.setVisible(False)
        self.current_answer_id = None

    def show_feedback_buttons(self, answer_id: str):
        """
        PHASE 3.4: Zeigt Feedback-Buttons für eine trackbare Antwort

        Args:
            answer_id: ID der Antwort, für die Feedback gegeben werden kann
        """
        if answer_id:
            self.current_answer_id = answer_id
            self.feedback_widget.setVisible(True)
            logger.debug(f"Feedback-Buttons aktiviert für answer_id={answer_id[:8]}")
        else:
            self.feedback_widget.setVisible(False)

    def add_message(self, sender, message):
        color = "#3498db" if sender == "Du" else "#e74c3c"
        self.history.append(f'<b style="color:{color};">{sender}:</b> {message}')

    def get_input(self):
        return self.input_text.toPlainText().strip()

    def clear_input(self):
        self.input_text.clear()

    @Slot(str)
    def update_context(self, context_summary: str):
        """
        PHASE 2 (Multi-Turn): Aktualisiert die Context-Anzeige.

        Args:
            context_summary: Formatierte Zusammenfassung des aktuellen Kontexts
        """
        if context_summary:
            self.context_label.setText(f"> Kontext: {context_summary}")
            self.context_label.setVisible(True)
        else:
            self.context_label.setText("")
            self.context_label.setVisible(False)

    @Slot(int, int, str)
    def update_file_progress(self, current: int, total: int, message: str):
        """
        PHASE 8 (Extended Features): Aktualisiert die Datei-Ingestion Progress Bar.

        Args:
            current: Aktueller Fortschritt
            total: Gesamtanzahl
            message: [?]tatus-Nachricht
        """
        if total > 0:
            percent = int((current / total) * 100)
            self.file_progress_bar.setMaximum(total)
            self.file_progress_bar.setValue(current)
            self.file_progress_bar.setFormat(f"{percent}%")
            self.file_progress_label.setText(message)

            # WICHTIG: Stelle sicher dass beide Widgets sichtbar sind
            if not self.file_progress_bar.isVisible():
                self.file_progress_bar.setVisible(True)
                self.file_progress_label.setVisible(True)

            # Process events to keep UI responsive during file processing
            QApplication.processEvents()

        # Verstecke Progress-Bar wenn fertig
        if current >= total and total > 0:
            # Kurz anzeigen, dann ausblenden (2 [?]ekunden Delay)
            QTimer.singleShot(2000, lambda: self.file_progress_bar.setVisible(False))
            QTimer.singleShot(2000, lambda: self.file_progress_label.setVisible(False))

    @Slot()
    def _on_file_picker_clicked(self):
        """
        Öffnet File-Picker für Dokument-Auswahl (einzeln oder mehrere).

        Workflow:
            1. Öffnet QFileDialog mit Mehrfachauswahl
            2. Fügt Command-[?]tring in Eingabefeld ein
            3. User kann Command prüfen/anpassen vor Enter
        """
        # Öffne File-Dialog mit Mehrfachauswahl
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Wähle eine oder mehrere Dateien zum Einlesen",
            "",  # [?]tartverzeichnis (leer = letztes verwendetes)
            "Alle unterstützten Formate (*.docx *.pdf *.txt *.md *.markdown *.html *.htm);;"
            "Word-Dokumente (*.docx);;"
            "PDF-Dokumente (*.pdf);;"
            "Text-Dateien (*.txt);;"
            "Markdown-Dateien (*.md *.markdown);;"
            "HTML-Dateien (*.html *.htm);;"
            "Alle Dateien (*.*)",
        )

        # Wenn User Datei(en) gewählt hat
        if file_paths:
            # Erstelle Command (einzelne Datei oder Batch)
            if len(file_paths) == 1:
                command = f"Lese Datei: {file_paths[0]}"
            else:
                files_str = ";".join(file_paths)
                command = f"Lese Dateien: {files_str}"

            self.input_text.setPlainText(command)
            self.input_text.setFocus()

            # Cursor ans Ende bewegen
            cursor = self.input_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.input_text.setTextCursor(cursor)
