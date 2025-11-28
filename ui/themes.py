# ui/themes.py
"""
Theme management for KAI application.

Contains theme stylesheets and theme switching logic.
"""


def get_dark_theme():
    """Dark Theme Stylesheet"""
    return """
        QWidget {
            background-color: #34495e;
            color: #ecf0f1;
            font-size: 14px;
        }
        QMainWindow {
            background-color: #2c3e50;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #7f8c8d;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        QTextEdit, QLineEdit {
            background-color: #2c3e50;
            border: 1px solid #7f8c8d;
            border-radius: 4px;
            padding: 5px;
        }
        QLineEdit:focus {
            border: 1px solid #3498db;
        }
    """


def get_light_theme():
    """Light Theme Stylesheet"""
    return """
        QWidget {
            background-color: #ecf0f1;
            color: #2c3e50;
            font-size: 14px;
        }
        QMainWindow {
            background-color: #bdc3c7;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #7f8c8d;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        QTextEdit, QLineEdit {
            background-color: #ffffff;
            border: 1px solid #95a5a6;
            border-radius: 4px;
            padding: 5px;
            color: #2c3e50;
        }
        QLineEdit:focus {
            border: 1px solid #3498db;
        }
    """


def set_stylesheet(app, theme="dark"):
    """
    Setzt das Stylesheet der Anwendung basierend auf dem gewählten Theme.

    Args:
        app: QApplication Instanz
        theme: "dark" oder "light"
    """
    if theme == "light":
        app.setStyleSheet(get_light_theme())
    else:  # default to dark
        app.setStyleSheet(get_dark_theme())
