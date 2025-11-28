# ui/plan_monitor.py
"""
Plan Monitor UI components for KAI application.

Contains PlanMonitor and TaskDisplay widgets for displaying goals and sub-goals.
"""

from typing import Dict

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class PlanMonitor(QWidget):
    """Zeigt das Hauptziel und eine dynamische Liste von Unterzielen an."""

    STATUS_ICONS = {
        "PENDING": "[ ]",
        "IN_PROGRESS": "[*]",
        "SUCCESS": "[OK]",
        "FAILED": "[X]",
    }
    STATUS_COLORS = {
        "PENDING": "#95a5a6",  # Grau
        "IN_PROGRESS": "#3498db",  # Blau
        "SUCCESS": "#2ecc71",  # Grün
        "FAILED": "#e74c3c",  # Rot
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sub_goal_labels: Dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_goal_label = QLabel("Warte auf Aufgabe...")
        self.main_goal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_goal_label.setStyleSheet(
            """
            background-color: #2c3e50; color: #ecf0f1; padding: 10px;
            font-size: 16px; font-weight: bold; border-radius: 5px;
        """
        )

        sub_goal_widget = QWidget()
        self.sub_goal_layout = QVBoxLayout(sub_goal_widget)
        self.sub_goal_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(sub_goal_widget)
        self.scroll_area.setVisible(True)  # Immer sichtbar

        layout.addWidget(self.main_goal_label)
        layout.addWidget(self.scroll_area)

    @Slot()
    def clear_goals(self):
        """Entfernt alle alten Ziele von der Anzeige."""
        self.main_goal_label.setText("Warte auf Aufgabe...")
        for label in self.sub_goal_labels.values():
            label.deleteLater()
        self.sub_goal_labels.clear()

    @Slot(str)
    def set_main_goal(self, text: str):
        """Setzt den Text des Hauptziels."""
        self.main_goal_label.setText(text)

    @Slot(str, str)
    def add_sub_goal(self, sg_id: str, description: str):
        """Fügt ein neues Unterziel in der Anzeige hinzu."""
        if sg_id in self.sub_goal_labels:
            return

        label = QLabel(f"{self.STATUS_ICONS['PENDING']} {description}")
        label.setStyleSheet(
            f"color: {self.STATUS_COLORS['PENDING']}; "
            "padding: 5px; font-size: 12px; border-bottom: 1px solid #34495e;"
        )
        self.sub_goal_layout.addWidget(label)
        self.sub_goal_labels[sg_id] = label

    @Slot(str, str)
    def update_sub_goal_status(self, sg_id: str, status_name: str):
        """Aktualisiert den Status (Farbe, Icon) eines bestehenden Unterziels."""
        if sg_id not in self.sub_goal_labels:
            return

        label = self.sub_goal_labels[sg_id]
        # Bewahre die alte Beschreibung, ohne das alte Icon
        current_text = label.text().split(" ", 1)[-1]

        icon = self.STATUS_ICONS.get(status_name, "[?]")
        color = self.STATUS_COLORS.get(status_name, "#ecf0f1")

        label.setText(f"{icon} {current_text}")
        label.setStyleSheet(
            f"color: {color}; padding: 5px; font-size: 12px; "
            "border-bottom: 1px solid #34495e;"
        )


class TaskDisplay(QWidget):
    """Zeigt das Hauptziel und eine ausklappbare Liste von Unterzielen an."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_goal_label = QLabel("Warte auf Aufgabe...")
        self.main_goal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_goal_label.setStyleSheet(
            """
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
        """
        )

        self.sub_goal_container = QFrame()
        self.sub_goal_container.setObjectName("subGoalContainer")
        self.sub_goal_container.setStyleSheet(
            "#subGoalContainer { border: 1px solid #34495e; border-top: none; }"
        )
        self.sub_goal_layout = QVBoxLayout(self.sub_goal_container)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.sub_goal_container)
        self.scroll_area.setMaximumHeight(120)
        self.scroll_area.setVisible(False)

        layout.addWidget(self.main_goal_label)
        layout.addWidget(self.scroll_area)

        self.main_goal_label.mousePressEvent = self.toggle_sub_goals

    def toggle_sub_goals(self, event):
        self.scroll_area.setVisible(not self.scroll_area.isVisible())

    def update_main_goal(self, text):
        self.main_goal_label.setText(text)

    def add_sub_goal(self, text):
        MAX_SUB_GOALS = 20
        if self.sub_goal_layout.count() >= MAX_SUB_GOALS:
            # Entferne ältestes Goal
            child = self.sub_goal_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        """ANGEPASST: Formatiert die Anzeige der Ziele basierend auf ihrer Priorität."""
        label = QLabel(text)
        style = "padding: 5px; font-size: 12px; border-bottom: 1px solid #34495e;"

        if "[MUST]" in text:
            style += "font-weight: bold; color: #f1c40f;"  # Gelb für MUST
        elif "[SHOULD]" in text:
            style += "color: #ecf0f1;"  # Standardfarbe für SHOULD
        elif "[COULD]" in text:
            style += "color: #95a5a6;"  # Grau für COULD

        label.setStyleSheet(style)
        self.sub_goal_layout.addWidget(label)

    def clear_goals(self):
        self.update_main_goal("Warte auf Aufgabe...")
        while self.sub_goal_layout.count():
            child = self.sub_goal_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.scroll_area.setVisible(False)
