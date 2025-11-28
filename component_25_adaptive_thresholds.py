# component_25_adaptive_thresholds.py
"""
Adaptive Threshold Management für Pattern Recognition.

Berechnet dynamische Bootstrap-Schwellenwerte basierend auf:
- Vocabulary-Größe
- Connection-Dichte
- Learning-Phase (cold_start, warming, mature)

Statistische Grundlage:
- Typo Threshold: min(10, max(3, vocab_size^0.4))
- Sequence Threshold: min(5, max(2, connection_count^0.35))
- Phase-abhängige Confidence Gates
"""

import math
from enum import Enum
from typing import Any, Dict, Optional

from component_15_logging_config import get_logger

logger = get_logger(__name__)


class BootstrapPhase(Enum):
    """Learning-Phasen des Systems"""

    COLD_START = "cold_start"  # <100 words
    WARMING = "warming"  # 100-1000 words
    MATURE = "mature"  # >1000 words


class AdaptiveThresholdManager:
    """
    Berechnet adaptive Schwellenwerte für Pattern Recognition.

    Features:
    - Dynamische Bootstrap-Thresholds basierend auf Datenmenge
    - Phase-Detection (cold_start, warming, mature)
    - Phase-abhängige Confidence-Gates
    - Graceful Degradation bei wenig Daten
    """

    # Phase-Grenzen
    COLD_START_LIMIT = 100
    WARMING_LIMIT = 1000

    # Typo-Detection Thresholds
    TYPO_MIN_THRESHOLD = 3  # Minimum (bei sehr kleinem Vocab)
    TYPO_MAX_THRESHOLD = 10  # Maximum (klassisch)
    TYPO_SCALE_EXPONENT = 0.4  # Skalierungs-Exponent

    # Sequence-Prediction Thresholds
    SEQ_MIN_THRESHOLD = 2  # Minimum
    SEQ_MAX_THRESHOLD = 5  # Maximum (klassisch)
    SEQ_SCALE_EXPONENT = 0.35  # Skalierungs-Exponent

    def __init__(self, netzwerk):
        """
        Args:
            netzwerk: KonzeptNetzwerk Instanz für Daten-Zugriff
        """
        self.netzwerk = netzwerk

    def get_bootstrap_phase(self, vocab_size: Optional[int] = None) -> BootstrapPhase:
        """
        Ermittelt aktuelle Bootstrap-Phase basierend auf Vocabulary-Größe.

        Args:
            vocab_size: Optional override, sonst wird Vocab aus Netzwerk geladen

        Returns:
            BootstrapPhase Enum
        """
        if vocab_size is None:
            vocab_size = len(self.netzwerk.get_all_known_words())

        if vocab_size < self.COLD_START_LIMIT:
            return BootstrapPhase.COLD_START
        elif vocab_size < self.WARMING_LIMIT:
            return BootstrapPhase.WARMING
        else:
            return BootstrapPhase.MATURE

    def get_typo_threshold(self, vocab_size: Optional[int] = None) -> int:
        """
        Berechnet adaptive Typo-Detection Threshold.

        Formel: min(MAX, max(MIN, vocab_size^0.4))

        Beispiele:
        - 10 words -> 3 occurrences
        - 100 words -> 4 occurrences
        - 1000 words -> 6 occurrences
        - 10000 words -> 10 occurrences (max)

        Args:
            vocab_size: Optional override

        Returns:
            Threshold als Integer
        """
        if vocab_size is None:
            vocab_size = len(self.netzwerk.get_all_known_words())

        if vocab_size == 0:
            return self.TYPO_MIN_THRESHOLD

        # Power-Law Scaling
        adaptive_threshold = math.pow(vocab_size, self.TYPO_SCALE_EXPONENT)

        # Clamp zwischen Min/Max
        threshold = max(
            self.TYPO_MIN_THRESHOLD,
            min(self.TYPO_MAX_THRESHOLD, int(adaptive_threshold)),
        )

        logger.debug(
            "Typo threshold berechnet",
            extra={
                "vocab_size": vocab_size,
                "threshold": threshold,
                "raw_value": adaptive_threshold,
            },
        )

        return threshold

    def get_sequence_threshold(self, connection_count: Optional[int] = None) -> int:
        """
        Berechnet adaptive Sequence-Prediction Threshold.

        Formel: min(MAX, max(MIN, connection_count^0.35))

        Args:
            connection_count: Anzahl aktiver Connections, optional

        Returns:
            Threshold als Integer
        """
        if connection_count is None:
            # Hole Connection-Count aus Netzwerk
            connection_count = self._get_total_connection_count()

        if connection_count == 0:
            return self.SEQ_MIN_THRESHOLD

        # Power-Law Scaling
        adaptive_threshold = math.pow(connection_count, self.SEQ_SCALE_EXPONENT)

        # Clamp zwischen Min/Max
        threshold = max(
            self.SEQ_MIN_THRESHOLD, min(self.SEQ_MAX_THRESHOLD, int(adaptive_threshold))
        )

        logger.debug(
            "Sequence threshold berechnet",
            extra={
                "connection_count": connection_count,
                "threshold": threshold,
                "raw_value": adaptive_threshold,
            },
        )

        return threshold

    def get_confidence_gates(
        self, phase: Optional[BootstrapPhase] = None
    ) -> Dict[str, float]:
        """
        Gibt phase-abhängige Confidence-Schwellenwerte zurück.

        Strategie:
        - cold_start: Sehr konservativ (nur bei hoher Sicherheit)
        - warming: Standard-Gates
        - mature: Aggressiver (System hat genug Daten)

        Args:
            phase: Optional phase override

        Returns:
            Dict mit "auto_correct", "ask_user", "min_confidence"
        """
        if phase is None:
            phase = self.get_bootstrap_phase()

        if phase == BootstrapPhase.COLD_START:
            return {
                "auto_correct": 999.0,  # Permanently disabled - ask_user strategy preferred for safety (prevents incorrect automatic corrections)
                "ask_user": 0.80,  # Frage bei hoher Confidence
                "min_confidence": 0.70,
                "description": "cold_start: Keine Auto-Korrektur (nur ask_user)",
            }
        elif phase == BootstrapPhase.WARMING:
            return {
                "auto_correct": 999.0,  # Permanently disabled - ask_user strategy preferred for safety (prevents incorrect automatic corrections)
                "ask_user": 0.60,  # Frage bei mittlerer Confidence
                "min_confidence": 0.50,
                "description": "warming: Keine Auto-Korrektur (nur ask_user)",
            }
        else:  # MATURE
            return {
                "auto_correct": 999.0,  # Permanently disabled - ask_user strategy preferred for safety (prevents incorrect automatic corrections)
                "ask_user": 0.50,  # Frage bei mittlerer Confidence
                "min_confidence": 0.40,
                "description": "mature: Keine Auto-Korrektur (nur ask_user)",
            }

    def get_bootstrap_confidence_multiplier(
        self, actual_count: int, threshold: int
    ) -> float:
        """
        Berechnet Confidence-Multiplikator für bootstrapped Patterns.

        Logik:
        - actual_count < threshold -> Downgrade (0.5 - 1.0)
        - actual_count >= threshold -> Normal (1.0)
        - actual_count >> threshold -> Boost (1.0 - 1.3)

        Args:
            actual_count: Tatsächliche Vorkommen
            threshold: Aktueller Bootstrap-Threshold

        Returns:
            Multiplikator (0.5 - 1.3)
        """
        if threshold == 0:
            return 1.0

        ratio = actual_count / threshold

        if ratio < 0.5:
            # Sehr wenig Daten -> starker Downgrade
            return 0.5
        elif ratio < 1.0:
            # Unter Threshold -> linearer Downgrade
            return 0.5 + 0.5 * ratio
        elif ratio <= 2.0:
            # Normal bis doppelt -> sanfter Boost
            return 1.0 + 0.15 * (ratio - 1.0)
        else:
            # Sehr viele Daten -> Max Boost
            return 1.3

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Gibt umfassende System-Statistiken zurück.

        Nützlich für Debugging und UI-Anzeige.

        Returns:
            Dict mit allen relevanten Metriken
        """
        vocab_size = len(self.netzwerk.get_all_known_words())
        connection_count = self._get_total_connection_count()
        phase = self.get_bootstrap_phase(vocab_size)

        typo_threshold = self.get_typo_threshold(vocab_size)
        seq_threshold = self.get_sequence_threshold(connection_count)
        gates = self.get_confidence_gates(phase)

        return {
            "vocab_size": vocab_size,
            "connection_count": connection_count,
            "phase": phase.value,
            "typo_threshold": typo_threshold,
            "sequence_threshold": seq_threshold,
            "confidence_gates": gates,
            "system_maturity": self._calculate_maturity_score(vocab_size),
        }

    def _get_total_connection_count(self) -> int:
        """
        Holt Anzahl aller CONNECTION Edges aus Neo4j.

        Returns:
            Anzahl als Integer
        """
        if not self.netzwerk.driver:
            return 0

        try:
            with self.netzwerk.driver.session() as session:
                query = """
                MATCH ()-[c:CONNECTION]->()
                RETURN count(c) AS total
                """
                result = session.run(query)
                record = result.single()

                if record:
                    return int(record["total"])
                else:
                    return 0

        except Exception as e:
            logger.warning(
                "Fehler beim Abrufen von Connection Count", extra={"error": str(e)}
            )
            return 0

    def _calculate_maturity_score(self, vocab_size: int) -> float:
        """
        Berechnet System-Maturity Score (0.0 - 1.0).

        Nutzt Sigmoid-Funktion für sanften Übergang.

        Args:
            vocab_size: Vocabulary-Größe

        Returns:
            Score zwischen 0.0 und 1.0
        """
        # Sigmoid mit Midpoint bei 500 words
        x = vocab_size / 500.0
        sigmoid = 1.0 / (1.0 + math.exp(-x + 1.0))
        return min(1.0, sigmoid)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_adaptive_thresholds(netzwerk) -> Dict[str, int]:
    """
    Standalone-Funktion für schnellen Threshold-Zugriff.

    Args:
        netzwerk: KonzeptNetzwerk Instanz

    Returns:
        Dict mit typo_threshold, sequence_threshold
    """
    manager = AdaptiveThresholdManager(netzwerk)
    return {
        "typo_threshold": manager.get_typo_threshold(),
        "sequence_threshold": manager.get_sequence_threshold(),
        "phase": manager.get_bootstrap_phase().value,
    }


if __name__ == "__main__":
    print("=== Adaptive Threshold Manager ===\n")

    # Test-Berechnungen (ohne Neo4j)
    print("Typo-Thresholds für verschiedene Vocab-Größen:")
    manager = type(
        "obj", (object,), {"driver": None, "get_all_known_words": lambda: []}
    )()

    atm = AdaptiveThresholdManager(manager)

    test_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    for size in test_sizes:
        threshold = atm.get_typo_threshold(size)
        phase = atm.get_bootstrap_phase(size)
        maturity = atm._calculate_maturity_score(size)
        print(
            f"  {size:>5} words -> threshold={threshold:>2}, phase={phase.value:>11}, maturity={maturity:.2f}"
        )

    print("\nSequence-Thresholds für verschiedene Connection-Counts:")
    test_connections = [10, 50, 100, 500, 1000, 5000]
    for count in test_connections:
        threshold = atm.get_sequence_threshold(count)
        print(f"  {count:>4} connections -> threshold={threshold}")

    print("\nConfidence-Gates pro Phase:")
    for phase in BootstrapPhase:
        gates = atm.get_confidence_gates(phase)
        print(
            f"  {phase.value:>11}: auto={gates['auto_correct']:.2f}, ask={gates['ask_user']:.2f}, min={gates['min_confidence']:.2f}"
        )

    print("\nBootstrap Confidence-Multiplier:")
    test_cases = [(3, 10), (5, 10), (10, 10), (15, 10), (30, 10)]
    for actual, threshold in test_cases:
        multiplier = atm.get_bootstrap_confidence_multiplier(actual, threshold)
        print(f"  {actual:>2} / {threshold:>2} -> {multiplier:.2f}x")
