"""
component_43_spatial_grid_widget.py

Interactive Spatial Grid Visualization Widget for KAI

Provides a PySide6-based graphical grid visualization for spatial reasoning
with support for position highlighting, path visualization, and interactive exploration.

Features:
- QGraphicsView-based grid rendering
- Position highlighting (marked cells)
- Path visualization (for path-finding results)
- Color-coded cells (obstacles, start, goal, path)
- Interactive zoom and pan
- Export to image
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import (
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from component_15_logging_config import get_logger
from kai_exceptions import KAIException

logger = get_logger(__name__)


class FileSystemError(KAIException):
    """Exception für Dateisystem-Operationen im SpatialGridWidget."""


# ==================== Data Structures ====================


@dataclass
class GridCell:
    """Represents a single cell in the grid."""

    row: int
    col: int
    type: str = "empty"  # empty, obstacle, start, goal, path, highlighted
    label: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SpatialGridData:
    """Data structure for grid visualization."""

    rows: int
    cols: int
    cells: List[GridCell]
    title: str = "Spatial Grid"
    show_coordinates: bool = True


# ==================== Custom Graphics Items ====================


class GridCellItem(QGraphicsRectItem):
    """
    Custom graphics item for grid cells.

    Color coding:
    - White: Empty cell
    - Black: Obstacle
    - Green: Start position
    - Red: Goal position
    - Blue: Path cell
    - Yellow: Highlighted/selected
    """

    def __init__(self, cell: GridCell, cell_size: float = 50.0, parent=None):
        super().__init__(parent)
        self.cell = cell
        self.cell_size = cell_size
        self.is_animated = False

        # Set cell dimensions
        self.setRect(0, 0, cell_size, cell_size)

        # Set cell appearance based on type
        self._update_appearance()

        # Make item interactive
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)

        # Set tooltip
        self._update_tooltip()

    def _update_appearance(self):
        """Update cell colors based on type."""
        colors = {
            "empty": QColor(255, 255, 255),  # White
            "obstacle": QColor(50, 50, 50),  # Dark gray
            "start": QColor(0, 255, 0),  # Green
            "goal": QColor(255, 0, 0),  # Red
            "path": QColor(100, 150, 255),  # Light blue
            "highlighted": QColor(255, 255, 0),  # Yellow
        }

        color = colors.get(self.cell.type, QColor(255, 255, 255))
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(0, 0, 0), 1))

    def _update_tooltip(self):
        """Generate tooltip with cell information."""
        tooltip_lines = [
            f"Position: ({self.cell.row}, {self.cell.col})",
            f"Typ: {self.cell.type}",
        ]

        if self.cell.label:
            tooltip_lines.append(f"Label: {self.cell.label}")

        if self.cell.metadata:
            tooltip_lines.append("")
            tooltip_lines.append("Metadata:")
            for key, value in self.cell.metadata.items():
                tooltip_lines.append(f"  {key}: {value}")

        self.setToolTip("\n".join(tooltip_lines))

    def hoverEnterEvent(self, event):
        """Highlight on hover."""
        if self.cell.type == "empty":
            self.setBrush(QBrush(QColor(240, 240, 240)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Remove highlight on leave."""
        self._update_appearance()
        super().hoverLeaveEvent(event)


class GridLabelItem(QGraphicsItem):
    """Text label for grid coordinates."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.text = text
        self.font = QFont("Arial", 10)

    def boundingRect(self):
        return QRectF(-20, -10, 40, 20)

    def paint(self, painter, option, widget):
        painter.setFont(self.font)
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, self.text)


# ==================== Main Widget ====================


class SpatialGridWidget(QWidget):
    """
    Main widget for spatial grid visualization.

    Displays grids with position highlighting, obstacles, and paths.
    """

    # Signals
    cell_clicked = Signal(int, int)  # row, col
    grid_updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_data: Optional[SpatialGridData] = None
        self.cell_size = 50.0
        self.cell_items: Dict[Tuple[int, int], GridCellItem] = {}

        self._init_ui()

        logger.info("SpatialGridWidget initialized")

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title label
        self.title_label = QLabel("Räumliches Grid")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Graphics view for grid
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self.view)

        # Control panel
        control_layout = QHBoxLayout()

        # Zoom controls
        zoom_label = QLabel("Zoom:")
        control_layout.addWidget(zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        control_layout.addWidget(self.zoom_slider)

        # Cell size controls
        cell_size_label = QLabel("Zellengröße:")
        control_layout.addWidget(cell_size_label)

        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setMinimum(20)
        self.cell_size_spin.setMaximum(100)
        self.cell_size_spin.setValue(50)
        self.cell_size_spin.setSuffix(" px")
        self.cell_size_spin.valueChanged.connect(self._on_cell_size_changed)
        control_layout.addWidget(self.cell_size_spin)

        # Show coordinates checkbox
        self.show_coords_checkbox = QCheckBox("Koordinaten anzeigen")
        self.show_coords_checkbox.setChecked(True)
        self.show_coords_checkbox.stateChanged.connect(self._on_show_coords_changed)
        control_layout.addWidget(self.show_coords_checkbox)

        control_layout.addStretch()

        # Export button
        export_btn = QPushButton("Exportieren als Bild")
        export_btn.clicked.connect(self._export_to_image)
        control_layout.addWidget(export_btn)

        # Clear button
        clear_btn = QPushButton("Leeren")
        clear_btn.clicked.connect(self.clear_grid)
        control_layout.addWidget(clear_btn)

        layout.addLayout(control_layout)

        # Info label
        self.info_label = QLabel("Kein Grid geladen")
        self.info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.info_label)

    def set_grid_data(self, grid_data: SpatialGridData):
        """
        Load and display grid data.

        Args:
            grid_data: SpatialGridData instance with grid configuration
        """
        self.grid_data = grid_data
        self._render_grid()
        self.grid_updated.emit()

        logger.info(
            "Grid data loaded",
            extra={
                "rows": grid_data.rows,
                "cols": grid_data.cols,
                "cells": len(grid_data.cells),
            },
        )

    def _render_grid(self):
        """Render the grid in the graphics scene."""
        if not self.grid_data:
            return

        # Clear scene
        self.scene.clear()
        self.cell_items.clear()

        # Update title
        self.title_label.setText(self.grid_data.title)

        # Create cell items
        for cell in self.grid_data.cells:
            cell_item = GridCellItem(cell, self.cell_size)

            # Position cell
            x = cell.col * self.cell_size
            y = cell.row * self.cell_size
            cell_item.setPos(x, y)

            self.scene.addItem(cell_item)
            self.cell_items[(cell.row, cell.col)] = cell_item

            # Add coordinate labels
            if self.grid_data.show_coordinates:
                if cell.row == 0:
                    # Column labels at top
                    col_label = GridLabelItem(str(cell.col))
                    col_label.setPos(x + self.cell_size / 2, -15)
                    self.scene.addItem(col_label)

                if cell.col == 0:
                    # Row labels at left
                    row_label = GridLabelItem(str(cell.row))
                    row_label.setPos(-15, y + self.cell_size / 2)
                    self.scene.addItem(row_label)

        # Update info label
        obstacle_count = sum(1 for c in self.grid_data.cells if c.type == "obstacle")
        path_count = sum(1 for c in self.grid_data.cells if c.type == "path")

        info_text = f"Grid: {self.grid_data.rows}×{self.grid_data.cols}"
        if obstacle_count > 0:
            info_text += f" | Hindernisse: {obstacle_count}"
        if path_count > 0:
            info_text += f" | Pfad-Zellen: {path_count}"

        self.info_label.setText(info_text)

        # Fit view to scene
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def highlight_cells(
        self, positions: List[Tuple[int, int]], color: str = "highlighted"
    ):
        """
        Highlight specific cells.

        Args:
            positions: List of (row, col) tuples
            color: Cell type/color to use
        """
        for row, col in positions:
            if (row, col) in self.cell_items:
                cell_item = self.cell_items[(row, col)]
                cell_item.cell.type = color
                cell_item._update_appearance()

        logger.debug(f"Highlighted {len(positions)} cells with color '{color}'")

    def show_path(self, path: List[Tuple[int, int]], animate: bool = False):
        """
        Visualize a path through the grid.

        Args:
            path: List of (row, col) positions
            animate: Whether to animate the path
        """
        if not path:
            return

        # Mark start and goal
        if len(path) > 0:
            start_pos = path[0]
            if start_pos in self.cell_items:
                self.cell_items[start_pos].cell.type = "start"
                self.cell_items[start_pos]._update_appearance()

        if len(path) > 1:
            goal_pos = path[-1]
            if goal_pos in self.cell_items:
                self.cell_items[goal_pos].cell.type = "goal"
                self.cell_items[goal_pos]._update_appearance()

        # Mark path cells
        for i, pos in enumerate(path[1:-1], 1):
            if pos in self.cell_items:
                self.cell_items[pos].cell.type = "path"
                self.cell_items[pos].cell.label = str(i)
                self.cell_items[pos]._update_appearance()
                self.cell_items[pos]._update_tooltip()

        logger.info(f"Path displayed with {len(path)} steps (animate={animate})")

    def clear_grid(self):
        """
        Clear the grid and free all resources.

        Properly removes all QGraphicsItems to prevent memory leaks.
        """
        # Step 1: Explicitly remove all items from scene
        for item in list(self.scene.items()):
            # Break parent-child relationships
            item.setParentItem(None)
            # Remove from scene
            self.scene.removeItem(item)

        # Step 2: Clear scene (belt and suspenders)
        self.scene.clear()

        # Step 3: Clear cell items dictionary
        self.cell_items.clear()

        # Step 4: Clear grid data
        self.grid_data = None

        # Step 5: Reset UI
        self.title_label.setText("Räumliches Grid")
        self.info_label.setText("Kein Grid geladen")

        logger.debug("Grid cleared (all items removed, memory freed)")

    def _on_zoom_changed(self, value: int):
        """Handle zoom slider changes."""
        scale = value / 100.0
        self.view.resetTransform()
        self.view.scale(scale, scale)

    def _on_cell_size_changed(self, value: int):
        """Handle cell size changes."""
        self.cell_size = float(value)
        if self.grid_data:
            self._render_grid()

    def _on_show_coords_changed(self, state: int):
        """Handle show coordinates checkbox."""
        if self.grid_data:
            self.grid_data.show_coordinates = bool(state)
            self._render_grid()

    def _export_to_image(self):
        """
        Export the grid to an image file.

        Raises:
            FileSystemError: Wenn Export fehlschlägt
        """
        if not self.grid_data:
            logger.warning("No grid to export")
            QMessageBox.warning(
                self,
                "Export fehlgeschlagen",
                "Kein Grid vorhanden zum Exportieren.",
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Grid exportieren",
            "spatial_grid.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )

        if filename:
            try:
                # Create pixmap from scene
                pixmap = QPixmap(self.scene.sceneRect().size().toSize())
                pixmap.fill(Qt.GlobalColor.white)

                painter = QPainter(pixmap)
                self.scene.render(painter)
                painter.end()

                # Save to file
                success = pixmap.save(filename)
                if not success:
                    raise FileSystemError(
                        "Pixmap konnte nicht gespeichert werden",
                        context={"filename": filename},
                    )

                logger.info(f"Grid exported to {filename}")
                QMessageBox.information(
                    self,
                    "Export erfolgreich",
                    f"Grid wurde erfolgreich exportiert:\n{filename}",
                )

            except FileSystemError as e:
                logger.error(f"Export fehlgeschlagen: {e}")
                QMessageBox.critical(
                    self,
                    "Export fehlgeschlagen",
                    f"Fehler beim Speichern der Datei:\n{e.message}\n\nPfad: {filename}",
                )
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Export: {e}", exc_info=True)
                QMessageBox.critical(
                    self,
                    "Export fehlgeschlagen",
                    f"Unerwarteter Fehler:\n{type(e).__name__}: {str(e)}\n\nPfad: {filename}",
                )


# ==================== Utility Functions ====================


def create_grid_from_dimensions(
    rows: int, cols: int, title: str = "Grid"
) -> SpatialGridData:
    """
    Create an empty grid with specified dimensions.

    Args:
        rows: Number of rows
        cols: Number of columns
        title: Grid title

    Returns:
        SpatialGridData instance
    """
    cells = []
    for r in range(rows):
        for c in range(cols):
            cells.append(GridCell(row=r, col=c, type="empty"))

    return SpatialGridData(rows=rows, cols=cols, cells=cells, title=title)


def create_grid_from_spatial_model(
    spatial_model, title: str = "Spatial Model"
) -> SpatialGridData:
    """
    Create grid data from a spatial reasoning model.

    Args:
        spatial_model: Grid2D or similar spatial model
        title: Grid title

    Returns:
        SpatialGridData instance
    """
    try:
        rows = spatial_model.height
        cols = spatial_model.width

        cells = []
        for r in range(rows):
            for c in range(cols):
                cells.append(GridCell(row=r, col=c, type="empty"))

        return SpatialGridData(rows=rows, cols=cols, cells=cells, title=title)
    except AttributeError:
        logger.error("Invalid spatial model provided")
        return create_grid_from_dimensions(8, 8, title)
