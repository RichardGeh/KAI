"""
ui.widgets.proof_tree_renderer

Proof tree rendering and graphics item creation for visualization.

Handles conversion of ProofTree data structures to QGraphicsItems,
node/edge creation, shape rendering, and color mapping.
"""

from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsLineItem

try:
    from component_17_proof_explanation import ProofTreeNode, StepType

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False
    ProofTreeNode = None
    StepType = None

from ui.widgets.proof_tree_formatter import ProofTreeFormatter


class ProofNodeItem(QGraphicsItem):
    """
    Custom graphics item for proof tree nodes.

    Supports different shapes based on step type:
    - Rectangle: Facts, inferences
    - Diamond: Rules, rule applications
    - Circle: Hypotheses, conclusions
    """

    def __init__(self, tree_node: "ProofTreeNode", parent=None):
        super().__init__(parent)
        self.tree_node = tree_node
        self.node_width = 150
        self.node_height = 60
        self.is_highlighted = False
        self.is_selected_item = False

        # Make item interactive
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

        # Set tooltip
        self._update_tooltip()

    def _update_tooltip(self):
        """Generate tooltip with full explanation"""
        step = self.tree_node.step

        # Format output with mathematical symbols
        formatted_output = ProofTreeFormatter.format_mathematical_text(step.output)

        tooltip_lines = [
            f"<b>Schritt:</b> {step.step_type.value}",
            (
                f"<b>Ausgabe:</b> {formatted_output[:100]}..."
                if len(formatted_output) > 100
                else f"<b>Ausgabe:</b> {formatted_output}"
            ),
            f"<b>Konfidenz:</b> {step.confidence:.2f}",
            "",
        ]

        if step.explanation_text:
            tooltip_lines.append(f"<b>Erklärung:</b><br>{step.explanation_text}")

        if step.rule_name:
            tooltip_lines.append(f"<b>Regel:</b> {step.rule_name}")

        if step.inputs:
            tooltip_lines.append(f"<b>Eingaben:</b> {len(step.inputs)}")

        # Enhanced: Source component
        tooltip_lines.append("")
        tooltip_lines.append(f"<b>Quelle:</b> {step.source_component}")

        # Enhanced: Timestamp (formatted)
        if step.timestamp:
            timestamp_str = step.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            tooltip_lines.append(f"<b>Zeitstempel:</b> {timestamp_str}")

        # Enhanced: Metadata (formatted if present)
        if step.metadata:
            tooltip_lines.append("")
            tooltip_lines.append("<b>Metadata:</b>")
            for key, value in list(step.metadata.items())[:5]:  # Limit to 5 entries
                value_str = str(value)[:50]  # Truncate long values
                if len(str(value)) > 50:
                    value_str += "..."
                tooltip_lines.append(f"  * {key}: {value_str}")
            if len(step.metadata) > 5:
                tooltip_lines.append(f"  ... (+{len(step.metadata) - 5} weitere)")

        self.setToolTip("<br>".join(tooltip_lines))

    def boundingRect(self) -> QRectF:
        """Define bounding rectangle for the item"""
        return QRectF(
            -self.node_width / 2,
            -self.node_height / 2,
            self.node_width,
            self.node_height,
        )

    def shape(self) -> QPainterPath:
        """Define the shape for collision detection"""
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def paint(self, painter: QPainter, option, widget=None):
        """Paint the node with appropriate shape and color"""
        step = self.tree_node.step

        # Determine shape based on step type
        shape_type = self._get_shape_type(step.step_type)

        # Determine color based on confidence
        fill_color = self._get_confidence_color(step.confidence)

        # Adjust color if highlighted or selected
        if self.is_selected_item:
            pen = QPen(QColor("#3498db"), 3)  # Blue border for selection
        elif self.is_highlighted:
            pen = QPen(QColor("#f39c12"), 3)  # Orange border for highlight
        else:
            pen = QPen(QColor("#2c3e50"), 2)  # Dark border

        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        # Draw shape
        if shape_type == "rectangle":
            painter.drawRect(self.boundingRect())
        elif shape_type == "diamond":
            self._draw_diamond(painter)
        elif shape_type == "circle":
            self._draw_circle(painter)

        # Draw text (abbreviated)
        painter.setPen(QPen(QColor("#ecf0f1")))
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))

        # Draw step type icon
        icon = ProofTreeFormatter.get_step_icon(step.step_type)
        painter.drawText(
            QRectF(-self.node_width / 2 + 5, -self.node_height / 2 + 5, 30, 20),
            Qt.AlignmentFlag.AlignLeft,
            icon,
        )

        # Draw truncated output (with mathematical notation)
        output_text = ProofTreeFormatter.format_mathematical_text(step.output)
        output_text = ProofTreeFormatter.truncate_with_ellipsis(output_text, 30)
        painter.setFont(QFont("Arial", 8))
        painter.drawText(
            QRectF(
                -self.node_width / 2 + 5,
                -self.node_height / 2 + 25,
                self.node_width - 10,
                self.node_height - 30,
            ),
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
            | Qt.AlignmentFlag.AlignHCenter,
            output_text,
        )

        # Draw confidence indicator (small bar at bottom)
        conf_bar_width = (self.node_width - 20) * step.confidence
        conf_bar_color = self._get_confidence_bar_color(step.confidence)
        painter.setBrush(QBrush(conf_bar_color))
        painter.drawRect(
            QRectF(
                -self.node_width / 2 + 10, self.node_height / 2 - 10, conf_bar_width, 5
            )
        )

        # Draw confidence percentage text
        painter.setPen(QPen(QColor("#ecf0f1")))
        painter.setFont(QFont("Arial", 7, QFont.Weight.Bold))
        conf_text = ProofTreeFormatter.format_confidence_text(step.confidence)
        painter.drawText(
            QRectF(self.node_width / 2 - 35, self.node_height / 2 - 25, 30, 15),
            Qt.AlignmentFlag.AlignRight,
            conf_text,
        )

        # Draw confidence level icon
        conf_icon = ProofTreeFormatter.get_confidence_icon(step.confidence)
        painter.setFont(QFont("Arial", 10))
        painter.drawText(
            QRectF(self.node_width / 2 - 20, -self.node_height / 2 + 5, 15, 15),
            Qt.AlignmentFlag.AlignCenter,
            conf_icon,
        )

    def _get_shape_type(self, step_type: "StepType") -> str:
        """Determine shape based on step type"""
        if step_type in [StepType.FACT_MATCH, StepType.INFERENCE]:
            return "rectangle"
        elif step_type in [StepType.RULE_APPLICATION, StepType.DECOMPOSITION]:
            return "diamond"
        else:  # HYPOTHESIS, PROBABILISTIC, GRAPH_TRAVERSAL, UNIFICATION
            return "circle"

    def _get_confidence_color(self, confidence: float) -> QColor:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return QColor("#27ae60")  # Green (high confidence)
        elif confidence >= 0.5:
            return QColor("#f39c12")  # Yellow/Orange (medium)
        else:
            return QColor("#e74c3c")  # Red (low confidence)

    def _get_confidence_bar_color(self, confidence: float) -> QColor:
        """
        Get color for confidence bar (brighter variants for better visibility).

        Integrated with ConfidenceManager thresholds:
        - >= 0.8: GREEN (High confidence)
        - 0.5-0.8: ORANGE (Medium confidence)
        - 0.3-0.5: RED (Low confidence)
        - < 0.3: DARK RED (Unknown/Very low)
        """
        if confidence >= 0.8:
            return QColor("#2ecc71")  # Bright green
        elif confidence >= 0.5:
            return QColor("#f39c12")  # Orange
        elif confidence >= 0.3:
            return QColor("#e74c3c")  # Red
        else:
            return QColor("#c0392b")  # Dark red

    def _draw_diamond(self, painter: QPainter):
        """Draw diamond shape"""
        w, h = self.node_width / 2, self.node_height / 2
        points = [
            QPointF(0, -h),  # Top
            QPointF(w, 0),  # Right
            QPointF(0, h),  # Bottom
            QPointF(-w, 0),  # Left
        ]
        painter.drawPolygon(QPolygonF(points))

    def _draw_circle(self, painter: QPainter):
        """Draw circular shape"""
        radius = min(self.node_width, self.node_height) / 2
        painter.drawEllipse(QPointF(0, 0), radius, radius)

    def set_highlighted(self, highlighted: bool):
        """Set highlight state"""
        self.is_highlighted = highlighted
        self.update()

    def set_selected_state(self, selected: bool):
        """Set selection state"""
        self.is_selected_item = selected
        self.update()

    def hoverEnterEvent(self, event):
        """Handle mouse hover"""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse leave"""
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)


class ProofEdgeItem(QGraphicsLineItem):
    """
    Custom graphics item for edges between proof nodes.
    """

    def __init__(self, parent_item: ProofNodeItem, child_item: ProofNodeItem):
        super().__init__()
        self.parent_item = parent_item
        self.child_item = child_item
        self.is_highlighted = False

        self._update_line()

        # Set pen
        self.setPen(QPen(QColor("#7f8c8d"), 2))

    def _update_line(self):
        """Update line position based on parent/child positions"""
        parent_pos = self.parent_item.scenePos()
        child_pos = self.child_item.scenePos()

        # Start from bottom of parent
        start = QPointF(
            parent_pos.x(), parent_pos.y() + self.parent_item.node_height / 2
        )
        # End at top of child
        end = QPointF(child_pos.x(), child_pos.y() - self.child_item.node_height / 2)

        self.setLine(start.x(), start.y(), end.x(), end.y())

    def set_highlighted(self, highlighted: bool):
        """Set highlight state"""
        self.is_highlighted = highlighted
        if highlighted:
            self.setPen(QPen(QColor("#f39c12"), 3))
        else:
            self.setPen(QPen(QColor("#7f8c8d"), 2))
        self.update()


class ProofTreeRenderer:
    """
    Renderer for converting ProofTree structures to QGraphicsItems.

    Handles tree layout, node positioning, and graphics item creation.
    """

    def __init__(self):
        self.node_items: Dict[str, ProofNodeItem] = {}
        self.edge_items: List[ProofEdgeItem] = []

    def create_node_items_recursive(
        self, node: "ProofTreeNode", scene, filter_func=None
    ):
        """
        Recursively create graphics items for nodes.

        Args:
            node: Current ProofTreeNode
            scene: QGraphicsScene to add items to
            filter_func: Optional filter function (node) -> bool
        """
        # Apply filter if provided
        if filter_func and not filter_func(node):
            return  # Skip this node and its children

        # Create item for this node
        item = ProofNodeItem(node)
        item.setPos(node.position[0], node.position[1])
        scene.addItem(item)
        self.node_items[node.step.step_id] = item

        # Connect click handler
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

        # Recurse for children (if expanded)
        if node.expanded:
            for child in node.children:
                self.create_node_items_recursive(child, scene, filter_func)

    def create_edge_items_recursive(self, node: "ProofTreeNode", scene, filter_func=None):
        """
        Recursively create edge items.

        Args:
            node: Current ProofTreeNode
            scene: QGraphicsScene to add items to
            filter_func: Optional filter function (node) -> bool
        """
        # Skip if node is filtered out
        if filter_func and not filter_func(node):
            return

        if node.expanded:
            parent_item = self.node_items.get(node.step.step_id)
            for child in node.children:
                # Only create edge if child is also displayed
                if filter_func and not filter_func(child):
                    continue

                child_item = self.node_items.get(child.step.step_id)
                if parent_item and child_item:
                    edge = ProofEdgeItem(parent_item, child_item)
                    scene.addItem(edge)
                    edge.setZValue(-1)  # Behind nodes
                    self.edge_items.append(edge)

                # Recurse
                self.create_edge_items_recursive(child, scene, filter_func)

    def layout_tree(
        self, roots: List["ProofTreeNode"], h_spacing: float, v_spacing: float
    ):
        """
        Calculate positions for all nodes using hierarchical layout.

        Uses a top-down tree layout algorithm that positions nodes
        to minimize edge crossings and maintain visual clarity.

        Args:
            roots: List of root ProofTreeNodes
            h_spacing: Horizontal spacing between nodes
            v_spacing: Vertical spacing between levels
        """
        # Process each root separately
        x_offset = 0
        for root in roots:
            x_offset = self._layout_subtree(root, x_offset, 0, h_spacing, v_spacing)
            x_offset += h_spacing  # Space between separate trees

    def _layout_subtree(
        self,
        node: "ProofTreeNode",
        x_start: float,
        depth: int,
        h_spacing: float,
        v_spacing: float,
    ) -> float:
        """
        Layout a subtree recursively.

        Args:
            node: Current ProofTreeNode
            x_start: Starting x position
            depth: Current depth in tree
            h_spacing: Horizontal spacing
            v_spacing: Vertical spacing

        Returns:
            The next available x position after this subtree
        """
        y = depth * v_spacing

        if not node.expanded or not node.children:
            # Leaf node or collapsed
            node.position = (x_start, y)
            return x_start + h_spacing

        # Layout children first
        child_x = x_start
        child_positions = []
        for child in node.children:
            child_x_start = child_x
            child_x = self._layout_subtree(
                child, child_x, depth + 1, h_spacing, v_spacing
            )
            child_positions.append(child_x_start)

        # Position parent at midpoint of children
        if child_positions:
            first_child_x = child_positions[0]
            last_child_x = child_positions[-1]
            parent_x = (first_child_x + last_child_x) / 2
        else:
            parent_x = x_start

        node.position = (parent_x, y)
        return child_x

    def clear(self):
        """Clear all cached items"""
        self.node_items.clear()
        self.edge_items.clear()

    @staticmethod
    def flatten_tree(node: "ProofTreeNode") -> List["ProofTreeNode"]:
        """
        Flatten tree to list of all nodes.

        Args:
            node: Root ProofTreeNode

        Returns:
            List of all nodes in subtree
        """
        nodes = [node]
        if node.expanded:
            for child in node.children:
                nodes.extend(ProofTreeRenderer.flatten_tree(child))
        return nodes

    @staticmethod
    def count_nodes(node: "ProofTreeNode") -> int:
        """
        Count total nodes in subtree (including collapsed).

        Args:
            node: Root of subtree

        Returns:
            Total number of nodes
        """
        count = 1  # Count this node
        for child in node.children:
            count += ProofTreeRenderer.count_nodes(child)
        return count
