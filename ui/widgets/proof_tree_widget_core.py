"""
ui.widgets.proof_tree_widget_core

Core proof tree widget class and UI structure.

Provides interactive proof tree visualization with:
- QGraphicsView-based rendering
- Toolbar with zoom, filter, and export controls
- Progressive rendering for large trees
- Interactive node selection and path highlighting
"""

from typing import List, Optional

from PySide6.QtCore import QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

try:
    from component_17_proof_explanation import (
        ProofTree,
        ProofTreeNode,
        StepType,
        export_proof_to_json,
    )

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    ProofTree = None
    ProofTreeNode = None
    StepType = None

from ui.widgets.proof_tree_formatter import ProofTreeFormatter
from ui.widgets.proof_tree_renderer import ProofTreeRenderer


class ProofTreeWidget(QWidget):
    """
    Interactive proof tree visualization widget.

    Displays proof trees with hierarchical layout, interactive features,
    and export capabilities.

    Signals:
        node_selected: Emitted when a node is clicked (ProofStep)
    """

    node_selected = Signal(object)  # ProofStep

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_tree: Optional[ProofTree] = None
        self.tree_nodes: List[ProofTreeNode] = []
        self.selected_node: Optional[ProofTreeNode] = None

        # Renderer
        self.renderer = ProofTreeRenderer()

        # Filter state
        self.min_confidence: float = 0.0  # Filter threshold
        self.filter_enabled: bool = False
        self.enabled_step_types: set = set(StepType) if StepType else set()

        # Performance settings
        self.max_nodes_threshold = 100  # Auto-collapse if tree exceeds this
        self.rendered_node_count = 0

        # Progressive rendering settings
        self.progressive_rendering_enabled: bool = True
        self.render_batch_size: int = 50
        self.current_render_index: int = 0
        self.pending_nodes: List[ProofTreeNode] = []

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Graphics view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setBackgroundBrush(QBrush(QColor("#1e1e1e")))

        layout.addWidget(self.view)

        # Status bar
        self.status_label = QLabel("Kein Beweisbaum geladen")
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        layout.addWidget(self.status_label)

    def _create_toolbar(self) -> QWidget:
        """Create toolbar with controls"""
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        # Expand/Collapse All
        btn_expand_all = QPushButton("Alles Aufklappen")
        btn_expand_all.clicked.connect(self.expand_all)
        toolbar_layout.addWidget(btn_expand_all)

        btn_collapse_all = QPushButton("Alles Zuklappen")
        btn_collapse_all.clicked.connect(self.collapse_all)
        toolbar_layout.addWidget(btn_collapse_all)

        toolbar_layout.addStretch()

        # Zoom controls
        toolbar_layout.addWidget(QLabel("Zoom:"))

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.clicked.connect(lambda: self.view.scale(1.2, 1.2))
        btn_zoom_in.setFixedWidth(30)
        toolbar_layout.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("-")
        btn_zoom_out.clicked.connect(lambda: self.view.scale(0.8, 0.8))
        btn_zoom_out.setFixedWidth(30)
        toolbar_layout.addWidget(btn_zoom_out)

        btn_zoom_fit = QPushButton("Fit")
        btn_zoom_fit.clicked.connect(self.fit_to_view)
        toolbar_layout.addWidget(btn_zoom_fit)

        toolbar_layout.addStretch()

        # Filter controls
        toolbar_layout.addWidget(QLabel("Filter:"))

        self.filter_checkbox = QCheckBox("Konfidenz ≥")
        self.filter_checkbox.setChecked(False)
        self.filter_checkbox.toggled.connect(self._on_filter_toggled)
        toolbar_layout.addWidget(self.filter_checkbox)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(0)
        self.confidence_slider.setFixedWidth(100)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        toolbar_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0.00")
        self.confidence_label.setFixedWidth(35)
        toolbar_layout.addWidget(self.confidence_label)

        # StepType filter button
        btn_steptype_filter = QPushButton("StepType-Filter")
        btn_steptype_filter.clicked.connect(self._show_steptype_filter_menu)
        toolbar_layout.addWidget(btn_steptype_filter)

        toolbar_layout.addStretch()

        # Filter presets
        toolbar_layout.addWidget(QLabel("Presets:"))

        btn_preset_high_conf = QPushButton("High Conf")
        btn_preset_high_conf.setToolTip("Nur Schritte mit Konfidenz >= 0.8")
        btn_preset_high_conf.clicked.connect(self._apply_preset_high_confidence)
        toolbar_layout.addWidget(btn_preset_high_conf)

        btn_preset_rules = QPushButton("Rules")
        btn_preset_rules.setToolTip("Nur Rule Applications & Inferences")
        btn_preset_rules.clicked.connect(self._apply_preset_rules_only)
        toolbar_layout.addWidget(btn_preset_rules)

        btn_reset_filters = QPushButton("Reset")
        btn_reset_filters.setToolTip("Alle Filter zuruecksetzen")
        btn_reset_filters.clicked.connect(self._reset_all_filters)
        toolbar_layout.addWidget(btn_reset_filters)

        toolbar_layout.addStretch()

        # Progressive Rendering Toggle
        self.progressive_checkbox = QCheckBox("Progressive Rendering")
        self.progressive_checkbox.setChecked(self.progressive_rendering_enabled)
        self.progressive_checkbox.setToolTip(
            "Rendert grosse Bäume schrittweise fuer bessere Performance"
        )
        self.progressive_checkbox.toggled.connect(self._on_progressive_toggled)
        toolbar_layout.addWidget(self.progressive_checkbox)

        toolbar_layout.addStretch()

        # Export buttons
        btn_export_json = QPushButton("Export JSON")
        btn_export_json.clicked.connect(self.export_to_json)
        toolbar_layout.addWidget(btn_export_json)

        btn_export_image = QPushButton("Export Bild")
        btn_export_image.clicked.connect(self.export_to_image)
        toolbar_layout.addWidget(btn_export_image)

        # Style toolbar
        toolbar.setStyleSheet(
            """
            QWidget {
                background-color: #2c3e50;
            }
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QLabel {
                color: #ecf0f1;
            }
        """
        )

        return toolbar

    def set_proof_tree(self, tree: ProofTree):
        """
        Set and display a proof tree.

        Args:
            tree: ProofTree to visualize
        """
        self.current_tree = tree
        self.tree_nodes = tree.to_tree_nodes()
        self.selected_node = None

        # Render tree
        self._render_tree()

        # Update status (only if not already set by performance warning)
        if "Auto-Collapse" not in self.status_label.text():
            self._update_status_with_counter()

    def _render_tree(self):
        """Render the current tree to the scene"""
        # Clear scene
        self.scene.clear()
        self.renderer.clear()

        if not self.tree_nodes:
            return

        # Performance optimization: Check total node count
        total_nodes = sum(
            ProofTreeRenderer.count_nodes(root) for root in self.tree_nodes
        )

        # Auto-collapse for large trees (>100 nodes)
        if total_nodes > self.max_nodes_threshold:
            self._auto_collapse_large_tree()
            self.status_label.setText(
                f"[WARNING] Grosser Baum ({total_nodes} Knoten) - Auto-Collapse aktiviert"
            )

        # Decide between progressive and full rendering
        if self.progressive_rendering_enabled and total_nodes > 50:
            self._render_tree_progressive()
        else:
            self._render_tree_full()

    def _render_tree_full(self):
        """Full tree rendering"""
        # Layout parameters
        horizontal_spacing = 200
        vertical_spacing = 120

        # Calculate layout positions
        self.renderer.layout_tree(self.tree_nodes, horizontal_spacing, vertical_spacing)

        # Create graphics items
        self.renderer.create_node_items_recursive(
            self.tree_nodes[0] if self.tree_nodes else None,
            self.scene,
            self._should_display_node,
        )

        # Create all node items for all roots
        for root in self.tree_nodes:
            self.renderer.create_node_items_recursive(
                root, self.scene, self._should_display_node
            )

        # Create edge items
        for root in self.tree_nodes:
            self.renderer.create_edge_items_recursive(
                root, self.scene, self._should_display_node
            )

        # Track rendered node count
        self.rendered_node_count = len(self.renderer.node_items)

        # Update status with counter (unless performance warning is active)
        if "Auto-Collapse" not in self.status_label.text():
            self._update_status_with_counter()

        # Fit to view
        QTimer.singleShot(100, self.fit_to_view)

    def _render_tree_progressive(self):
        """Progressive tree rendering in batches"""
        # Layout parameters
        horizontal_spacing = 200
        vertical_spacing = 120

        # Calculate layout positions
        self.renderer.layout_tree(self.tree_nodes, horizontal_spacing, vertical_spacing)

        # Collect all nodes to render (flatten)
        all_nodes = []
        for root in self.tree_nodes:
            all_nodes.extend(ProofTreeRenderer.flatten_tree(root))

        # Filter nodes
        filtered_nodes = [node for node in all_nodes if self._should_display_node(node)]

        # Store pending nodes
        self.pending_nodes = filtered_nodes
        self.current_render_index = 0

        # Render first batch
        self._render_next_batch()

        # Update status
        self.status_label.setText(
            f"Progressive Rendering: {self.rendered_node_count} von {len(filtered_nodes)} Knoten geladen"
        )

        # Fit to view after initial rendering
        QTimer.singleShot(100, self.fit_to_view)

    def _render_next_batch(self):
        """Render the next batch of nodes"""
        if self.current_render_index >= len(self.pending_nodes):
            # All nodes rendered - now render edges
            for root in self.tree_nodes:
                self.renderer.create_edge_items_recursive(
                    root, self.scene, self._should_display_node
                )
            self._update_status_with_counter()
            return

        # Determine batch
        batch_end = min(
            self.current_render_index + self.render_batch_size, len(self.pending_nodes)
        )

        batch_nodes = self.pending_nodes[self.current_render_index : batch_end]

        # Render batch
        for node in batch_nodes:
            from ui.widgets.proof_tree_renderer import ProofNodeItem

            item = ProofNodeItem(node)
            item.setPos(node.position[0], node.position[1])
            self.scene.addItem(item)
            self.renderer.node_items[node.step.step_id] = item

        # Update index
        self.current_render_index = batch_end
        self.rendered_node_count = len(self.renderer.node_items)

        # Update status
        if self.current_render_index < len(self.pending_nodes):
            self.status_label.setText(
                f"Progressive Rendering: {self.rendered_node_count} von {len(self.pending_nodes)} Knoten"
            )
            # Schedule next batch
            QTimer.singleShot(50, self._render_next_batch)
        else:
            # Render edges and update status
            for root in self.tree_nodes:
                self.renderer.create_edge_items_recursive(
                    root, self.scene, self._should_display_node
                )
            self._update_status_with_counter()

    def mousePressEvent(self, event):
        """Handle mouse press for node selection"""
        # Find clicked item
        scene_pos = self.view.mapToScene(
            self.view.mapFromGlobal(event.globalPosition().toPoint())
        )
        item = self.scene.itemAt(scene_pos, self.view.transform())

        from ui.widgets.proof_tree_renderer import ProofNodeItem

        if isinstance(item, ProofNodeItem):
            self._select_node(item.tree_node)

        super().mousePressEvent(event)

    def _select_node(self, node: ProofTreeNode):
        """
        Select a node and highlight path to root.

        Args:
            node: The ProofTreeNode to select
        """
        # Clear previous selection
        if self.selected_node:
            for step_id, item in self.renderer.node_items.items():
                item.set_selected_state(False)
                item.set_highlighted(False)
            for edge in self.renderer.edge_items:
                edge.set_highlighted(False)

        # Set new selection
        self.selected_node = node

        # Highlight selected node
        selected_item = self.renderer.node_items.get(node.step.step_id)
        if selected_item:
            selected_item.set_selected_state(True)

        # Highlight path to root
        path = node.get_path_to_root()
        for path_node in path:
            item = self.renderer.node_items.get(path_node.step.step_id)
            if item:
                item.set_highlighted(True)

        # Highlight edges in path
        for i in range(len(path) - 1):
            parent_item = self.renderer.node_items.get(path[i].step.step_id)
            child_item = self.renderer.node_items.get(path[i + 1].step.step_id)

            for edge in self.renderer.edge_items:
                if edge.parent_item == parent_item and edge.child_item == child_item:
                    edge.set_highlighted(True)

        # Emit signal
        self.node_selected.emit(node.step)

    def expand_all(self):
        """Expand all nodes"""
        for root in self.tree_nodes:
            self._expand_recursive(root)
        self._render_tree()

    def collapse_all(self):
        """Collapse all nodes"""
        for root in self.tree_nodes:
            self._collapse_recursive(root)
        self._render_tree()

    def _expand_recursive(self, node: ProofTreeNode):
        """Recursively expand node and children"""
        node.expand()
        for child in node.children:
            self._expand_recursive(child)

    def _collapse_recursive(self, node: ProofTreeNode):
        """Recursively collapse node and children"""
        node.collapse()
        for child in node.children:
            self._collapse_recursive(child)

    def fit_to_view(self):
        """Fit the entire tree to the view"""
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def export_to_json(self):
        """Export proof tree to JSON file"""
        if not self.current_tree:
            self.status_label.setText("[WARNING] Kein Beweisbaum zum Exportieren")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Beweisbaum", "", "JSON Files (*.json)"
        )

        if not filename:
            return

        try:
            export_proof_to_json(self.current_tree, filename)
            self.status_label.setText(f"[SUCCESS] Exportiert nach: {filename}")

        except (IOError, OSError) as e:
            self.status_label.setText(f"[ERROR] Dateizugriff fehlgeschlagen: {e}")

        except (TypeError, ValueError) as e:
            self.status_label.setText(f"[ERROR] Daten nicht serialisierbar: {e}")

        except Exception as e:
            self.status_label.setText(
                f"[ERROR] Export fehlgeschlagen: {type(e).__name__}"
            )
            raise

    def export_to_image(self):
        """Export proof tree to image file"""
        if not self.current_tree:
            self.status_label.setText("[WARNING] Kein Beweisbaum zum Exportieren")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Beweisbaum als Bild",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf)",
        )

        if not filename:
            return

        try:
            # Get scene dimensions
            rect = self.scene.sceneRect()
            width = int(rect.width())
            height = int(rect.height())

            # Validate dimensions (prevent memory exhaustion)
            MAX_DIMENSION = 16384  # QPixmap max size on most platforms
            MAX_PIXELS = 100_000_000  # ~100 megapixels

            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                self.status_label.setText(
                    f"[ERROR] Bild zu gross ({width}x{height}). Max: {MAX_DIMENSION}px. "
                    "Verwende Filter oder Zoom."
                )
                return

            if width * height > MAX_PIXELS:
                self.status_label.setText(
                    f"[ERROR] Bild zu gross ({width * height / 1_000_000:.1f} Megapixel). "
                    f"Max: {MAX_PIXELS / 1_000_000:.0f} MP. Verwende Filter."
                )
                return

            # Warn for large images
            if width * height > 25_000_000:  # >25 megapixels
                self.status_label.setText(
                    f"[WARNING] Grosses Bild ({width}x{height}) wird erstellt..."
                )
                self.status_label.repaint()

            # Create pixmap
            pixmap = QPixmap(width, height)
            if pixmap.isNull():
                self.status_label.setText(
                    "[ERROR] Pixmap-Erstellung fehlgeschlagen (Speicher voll?)"
                )
                return

            pixmap.fill(QColor("#1e1e1e"))

            # Render scene
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.scene.render(painter)
            painter.end()

            # Save with error checking
            success = pixmap.save(filename)
            if success:
                self.status_label.setText(f"[SUCCESS] Bild exportiert: {filename}")
            else:
                self.status_label.setText(
                    f"[ERROR] Speichern fehlgeschlagen: {filename}"
                )

        except Exception as e:
            self.status_label.setText(
                f"[ERROR] Export fehlgeschlagen: {type(e).__name__}"
            )
            raise

    def clear(self):
        """Clear the current proof tree"""
        self.current_tree = None
        self.tree_nodes = []
        self.selected_node = None
        self.scene.clear()
        self.renderer.clear()

        # Clear progressive rendering state
        self.pending_nodes.clear()
        self.current_render_index = 0
        self.rendered_node_count = 0

        self.status_label.setText("Kein Beweisbaum geladen")

    # ==================== Filter Methods ====================

    def _on_filter_toggled(self, checked: bool):
        """Handle filter checkbox toggle"""
        self.filter_enabled = checked
        if self.current_tree:
            self._render_tree()

    def _on_confidence_changed(self, value: int):
        """Handle confidence slider change"""
        self.min_confidence = value / 100.0
        self.confidence_label.setText(f"{self.min_confidence:.2f}")

        if self.filter_enabled and self.current_tree:
            self._render_tree()

    def _on_progressive_toggled(self, checked: bool):
        """Handle progressive rendering toggle"""
        self.progressive_rendering_enabled = checked
        if self.current_tree:
            self._render_tree()

    def _should_display_node(self, node: ProofTreeNode) -> bool:
        """
        Determine if a node should be displayed based on active filters.

        Args:
            node: ProofTreeNode to check

        Returns:
            True if node passes all active filters
        """
        # Confidence filter
        if self.filter_enabled:
            if node.step.confidence < self.min_confidence:
                return False

        # StepType filter
        if node.step.step_type not in self.enabled_step_types:
            return False

        return True

    def _show_steptype_filter_menu(self):
        """Show popup menu for StepType filter selection"""
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
            }
            QMenu::item:selected {
                background-color: #3498db;
            }
        """
        )

        # Add "All" option
        all_action = menu.addAction(
            "[OK] Alle aktivieren"
            if len(self.enabled_step_types) == len(StepType)
            else "Alle aktivieren"
        )
        all_action.triggered.connect(self._enable_all_steptypes)

        none_action = menu.addAction("Alle deaktivieren")
        none_action.triggered.connect(self._disable_all_steptypes)

        menu.addSeparator()

        # Add checkbox for each StepType
        for step_type in StepType:
            icon = ProofTreeFormatter.get_step_icon(step_type)
            is_enabled = step_type in self.enabled_step_types
            action = menu.addAction(
                f"{'[OK]' if is_enabled else '  '} {icon} {step_type.value}"
            )
            action.setCheckable(True)
            action.setChecked(is_enabled)
            action.triggered.connect(
                lambda checked, st=step_type: self._toggle_steptype(st, checked)
            )

        # Show menu below button
        btn = self.sender()
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _toggle_steptype(self, step_type: StepType, enabled: bool):
        """Toggle a specific StepType filter"""
        if enabled:
            self.enabled_step_types.add(step_type)
        else:
            self.enabled_step_types.discard(step_type)

        if self.current_tree:
            self._render_tree()

    def _enable_all_steptypes(self):
        """Enable all StepTypes"""
        self.enabled_step_types = set(StepType)
        if self.current_tree:
            self._render_tree()

    def _disable_all_steptypes(self):
        """Disable all StepTypes"""
        self.enabled_step_types.clear()
        if self.current_tree:
            self._render_tree()

    # ==================== Filter Presets ====================

    def _apply_preset_high_confidence(self):
        """Apply preset: Only high confidence steps (>= 0.8)"""
        self.filter_enabled = True
        self.filter_checkbox.setChecked(True)
        self.min_confidence = 0.8
        self.confidence_slider.setValue(80)
        self.confidence_label.setText("0.80")

        if self.current_tree:
            self._render_tree()

    def _apply_preset_rules_only(self):
        """Apply preset: Only rule applications and inferences"""
        self.enabled_step_types = {StepType.RULE_APPLICATION, StepType.INFERENCE}

        if self.current_tree:
            self._render_tree()

    def _reset_all_filters(self):
        """Reset all filters to default (show everything)"""
        # Reset confidence filter
        self.filter_enabled = False
        self.filter_checkbox.setChecked(False)
        self.min_confidence = 0.0
        self.confidence_slider.setValue(0)
        self.confidence_label.setText("0.00")

        # Reset StepType filter
        self.enabled_step_types = set(StepType)

        if self.current_tree:
            self._render_tree()

    def _update_status_with_counter(self):
        """Update status bar with node counter"""
        if not self.current_tree:
            self.status_label.setText("Kein Beweisbaum geladen")
            return

        total_steps = len(self.current_tree.get_all_steps())
        visible_steps = self.rendered_node_count

        if visible_steps < total_steps:
            # Filters active
            self.status_label.setText(
                f"Beweisbaum: {visible_steps} von {total_steps} Knoten sichtbar "
                f"({len(self.tree_nodes)} Wurzeln) - Filter aktiv"
            )
        else:
            # No filters
            self.status_label.setText(
                f"Beweisbaum: {total_steps} Schritte, {len(self.tree_nodes)} Wurzeln"
            )

    # ==================== Performance Methods ====================

    def _auto_collapse_large_tree(self):
        """
        Auto-collapse large trees for performance.

        Collapses all nodes beyond depth 2 to reduce rendering load.
        """
        for root in self.tree_nodes:
            self._collapse_beyond_depth(root, current_depth=0, max_depth=2)

    def _collapse_beyond_depth(
        self, node: ProofTreeNode, current_depth: int, max_depth: int
    ):
        """
        Recursively collapse nodes beyond a certain depth.

        Args:
            node: Current node
            current_depth: Current depth in tree (0 = root)
            max_depth: Maximum depth to keep expanded
        """
        if current_depth >= max_depth:
            node.collapse()
        else:
            node.expand()

        # Recurse for children
        for child in node.children:
            self._collapse_beyond_depth(child, current_depth + 1, max_depth)


class ComparisonProofTreeWidget(QWidget):
    """
    Side-by-side comparison widget for two proof trees.

    Displays two ProofTreeWidget instances in a horizontal splitter
    for comparing different reasoning strategies or proof variants.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize comparison UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)

        control_layout.addWidget(QLabel("Vergleichsmodus"))
        control_layout.addStretch()

        # Synchronized controls checkbox
        self.sync_checkbox = QCheckBox("Zoom synchronisieren")
        self.sync_checkbox.setChecked(False)
        self.sync_checkbox.toggled.connect(self._on_sync_toggled)
        control_layout.addWidget(self.sync_checkbox)

        # Export both button
        btn_export_both = QPushButton("Beide exportieren")
        btn_export_both.clicked.connect(self._export_both_trees)
        control_layout.addWidget(btn_export_both)

        control_bar.setStyleSheet(
            """
            QWidget {
                background-color: #34495e;
                color: #ecf0f1;
                padding: 5px;
            }
            QPushButton {
                background-color: #2c3e50;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """
        )

        layout.addWidget(control_bar)

        # Splitter with two tree widgets
        from PySide6.QtWidgets import QSplitter

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left tree (Tree A)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_label = QLabel("Baum A")
        left_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; padding: 5px; font-weight: bold;"
        )
        left_layout.addWidget(left_label)

        self.tree_widget_a = ProofTreeWidget()
        left_layout.addWidget(self.tree_widget_a)

        # Right tree (Tree B)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_label = QLabel("Baum B")
        right_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; padding: 5px; font-weight: bold;"
        )
        right_layout.addWidget(right_label)

        self.tree_widget_b = ProofTreeWidget()
        right_layout.addWidget(self.tree_widget_b)

        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right_container)
        self.splitter.setSizes([500, 500])

        layout.addWidget(self.splitter)

    def set_trees(self, tree_a: ProofTree, tree_b: ProofTree):
        """Set both proof trees for comparison"""
        self.tree_widget_a.set_proof_tree(tree_a)
        self.tree_widget_b.set_proof_tree(tree_b)

    def set_tree_a(self, tree: ProofTree):
        """Set Tree A"""
        self.tree_widget_a.set_proof_tree(tree)

    def set_tree_b(self, tree: ProofTree):
        """Set Tree B"""
        self.tree_widget_b.set_proof_tree(tree)

    def _on_sync_toggled(self, checked: bool):
        """Handle zoom synchronization toggle"""
        if checked:
            self.tree_widget_a.fit_to_view()
            self.tree_widget_b.fit_to_view()

    def _export_both_trees(self):
        """Export both trees side-by-side as single image"""
        # Get scenes
        scene_a = self.tree_widget_a.scene
        scene_b = self.tree_widget_b.scene

        rect_a = scene_a.sceneRect()
        rect_b = scene_b.sceneRect()

        # Calculate combined dimensions
        combined_width = int(rect_a.width() + rect_b.width() + 20)
        combined_height = int(max(rect_a.height(), rect_b.height()))

        # Create combined pixmap
        pixmap = QPixmap(combined_width, combined_height)
        pixmap.fill(QColor("#1e1e1e"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Render Tree A (left)
        scene_a.render(painter, target=QRectF(0, 0, rect_a.width(), rect_a.height()))

        # Render Tree B (right)
        scene_b.render(
            painter,
            target=QRectF(rect_a.width() + 20, 0, rect_b.width(), rect_b.height()),
        )

        painter.end()

        # Save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Vergleich", "", "PNG Files (*.png)"
        )

        if filename:
            pixmap.save(filename)
