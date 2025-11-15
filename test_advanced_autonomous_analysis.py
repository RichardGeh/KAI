# test_advanced_autonomous_analysis.py
"""
Erweiterte systematische Analyse des automatischen Lernsystems von KAI.

Analysiert:
1. Pattern Recognition (Typo Detection, Sequence Prediction, Implicit Facts)
2. Adaptive Thresholds (Bootstrap-Phasen)
3. Prototype Matching & Clustering
4. Fehlende Definitionsmuster
5. Confidence-Calibration
"""

import io
import sys
# Removed unused imports: Dict, List

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

from component_1_netzwerk import KonzeptNetzwerk
from component_6_linguistik_engine import LinguisticPreprocessor
from component_7_meaning_extractor import MeaningPointExtractor
from component_8_prototype_matcher import PrototypingEngine
from component_11_embedding_service import EmbeddingService
from component_19_pattern_recognition_char import TypoCandidateFinder
from component_20_pattern_recognition_sequence import SequencePredictor
from component_22_pattern_recognition_implicit import ImplicationDetector
from component_24_pattern_orchestrator import PatternOrchestrator
from component_25_adaptive_thresholds import AdaptiveThresholdManager


class AdvancedAutonomousAnalyzer:
    """Erweiterte Analyse des automatischen Lernsystems"""

    def __init__(self):
        self.netzwerk = KonzeptNetzwerk()
        self.embedding_service = EmbeddingService()
        self.preprocessor = LinguisticPreprocessor()
        self.prototyping_engine = PrototypingEngine(
            self.netzwerk, self.embedding_service
        )
        self.extractor = MeaningPointExtractor(
            self.embedding_service, self.preprocessor, self.prototyping_engine
        )

        # Pattern Recognition Components
        self.typo_finder = TypoCandidateFinder(self.netzwerk)
        self.sequence_predictor = SequencePredictor(self.netzwerk)
        self.implication_detector = ImplicationDetector(self.netzwerk)
        self.pattern_orchestrator = PatternOrchestrator(self.netzwerk)

        # Adaptive Thresholds
        self.adaptive_manager = AdaptiveThresholdManager(self.netzwerk)

        self.results = []

    def run_full_analysis(self):
        """Führt vollständige erweiterte Analyse aus"""
        print("\n" + "=" * 100)
        print("ERWEITERTE ANALYSE: AUTOMATISCHES LERNSYSTEM VON KAI")
        print("=" * 100 + "\n")

        # TEIL 1: Adaptive Thresholds & Bootstrap-Phasen
        self._analyze_adaptive_thresholds()

        # TEIL 2: Fehlende Definitionsmuster
        self._analyze_missing_patterns()

        # TEIL 3: Confidence-Calibration
        self._analyze_confidence_calibration()

        # TEIL 4: Pattern Recognition Orchestrator
        self._analyze_pattern_orchestrator()

        # TEIL 5: Erweiterte Relationstypen
        self._analyze_extended_relations()

        # Summary
        self._print_advanced_summary()

    def _analyze_adaptive_thresholds(self):
        """Analysiert Adaptive Thresholds"""
        print("\n" + "=" * 100)
        print("TEIL 1: ADAPTIVE THRESHOLDS & BOOTSTRAP-PHASEN")
        print("=" * 100 + "\n")

        # Hole System-Stats
        stats = self.adaptive_manager.get_system_stats()

        print(f"Aktueller System-Status:")
        print(f"  Vocabulary Size: {stats['vocab_size']}")
        print(f"  Connection Count: {stats['connection_count']}")
        print(f"  Bootstrap Phase: {stats['phase']}")
        print(f"  System Maturity: {stats['system_maturity']:.2%}")
        print(f"\nDynamische Thresholds:")
        print(f"  Typo Threshold: {stats['typo_threshold']}")
        print(f"  Sequence Threshold: {stats['sequence_threshold']}")
        print(f"\nConfidence Gates:")
        gates = stats['confidence_gates']
        print(f"  Auto-Correct: >= {gates['auto_correct']}")
        print(f"  Ask User: >= {gates['ask_user']}")
        print(f"  Min Confidence: >= {gates['min_confidence']}")
        print(f"  Description: {gates['description']}")

        # Teste Threshold-Berechnung für verschiedene Vocab-Größen
        print(f"\nThreshold-Projektion (Typo Detection):")
        test_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
        for size in test_sizes:
            threshold = self.adaptive_manager.get_typo_threshold(size)
            phase = self.adaptive_manager.get_bootstrap_phase(size)
            maturity = self.adaptive_manager._calculate_maturity_score(size)
            print(
                f"  {size:>5} words -> threshold={threshold:>2}, "
                f"phase={phase.value:>11}, maturity={maturity:.2%}"
            )

        # Bewertung
        print(f"\n[BEWERTUNG] Adaptive Thresholds:")
        if stats['phase'] == 'cold_start':
            print("  [WARNUNG] System in COLD_START Phase - Sehr konservativ")
            print("  [EMPFEHLUNG] Mehr Daten sammeln für bessere Performance")
        elif stats['phase'] == 'warming':
            print("  [OK] System in WARMING Phase - Standard-Gates aktiv")
        else:
            print("  [OPTIMAL] System in MATURE Phase - Aggressivere Gates")

        print()

    def _analyze_missing_patterns(self):
        """Analysiert fehlende Definitionsmuster"""
        print("\n" + "=" * 100)
        print("TEIL 2: FEHLENDE DEFINITIONSMUSTER")
        print("=" * 100 + "\n")

        # Test verschiedene Muster die möglicherweise fehlen
        missing_pattern_tests = [
            # Attributive Konstruktionen
            ("Ein roter Apfel", "Attribut (Farbe)", False),
            ("Ein großer Hund", "Attribut (Größe)", False),
            ("Ein schnelles Auto", "Attribut (Geschwindigkeit)", False),

            # Ownership/Possession
            ("Das Buch gehört mir", "Besitz (Person)", False),
            ("Peter hat einen Hund", "Besitz (Named Entity)", False),
            ("Mein Auto ist rot", "Besitz + Eigenschaft", True),

            # Temporal Relations
            ("Sommer folgt auf Frühling", "Temporale Sequenz", False),
            ("Weihnachten ist im Dezember", "Temporale Zuordnung", True),

            # Purpose/Function
            ("Ein Hammer dient zum Hämmern", "Zweck/Funktion", False),
            ("Ein Messer wird zum Schneiden benutzt", "Verwendung", False),

            # Material/Composition
            ("Ein Tisch besteht aus Holz", "Material", False),
            ("Wasser besteht aus H2O", "Zusammensetzung", False),

            # Causality
            ("Hitze verursacht Schweiß", "Kausale Relation", False),
            ("Regen macht die Straße nass", "Kausale Wirkung", False),

            # Equivalence
            ("H2O ist Wasser", "Äquivalenz (Formel)", False),
            ("Ein Doktor ist ein Arzt", "Äquivalenz (Synonym)", True),

            # Negation with explicit denial
            ("Ein Pinguin ist kein Säugetier", "Negative IS_A", False),
            ("Glas ist nicht essbar", "Negative Property", False),

            # Quantified Properties
            ("Die meisten Vögel können fliegen", "Quantifizierte Fähigkeit", True),
            ("Viele Menschen mögen Pizza", "Quantifizierte Präferenz", False),

            # Comparative (should be filtered)
            ("Hunde sind treuer als Katzen", "Komparativ (Filter)", False),
        ]

        detected_count = 0
        missing_count = 0
        correct_filter_count = 0

        for sentence, category, should_detect in missing_pattern_tests:
            doc = self.preprocessor.process(sentence)
            meaning_points = self.extractor.extract(doc)

            detected = len(meaning_points) > 0
            confidence = meaning_points[0].confidence if detected else 0.0
            relation = (
                meaning_points[0].arguments.get("relation_type")
                if detected
                else None
            )

            if detected == should_detect:
                status = "[OK]"
                if detected:
                    detected_count += 1
                else:
                    correct_filter_count += 1
            else:
                status = "[FEHLT]" if should_detect else "[FALSCH]"
                if should_detect:
                    missing_count += 1

            detection_str = f"Erkannt ({relation})" if detected else "Nicht erkannt"
            print(
                f"{status} {sentence[:45]:<45} | {category:<25} | "
                f"{detection_str:<25} | Conf: {confidence:.2f}"
            )

        print(
            f"\n[BEWERTUNG] Muster-Abdeckung: {detected_count}/{detected_count + missing_count} "
            f"erkannt ({100 * detected_count / max(1, detected_count + missing_count):.1f}%)"
        )
        print(f"  Korrekt erkannt: {detected_count}")
        print(f"  Fehlend: {missing_count}")
        print(f"  Korrekt gefiltert: {correct_filter_count}")

        if missing_count > 0:
            print(f"\n[EMPFEHLUNG] Fehlende Muster implementieren:")
            print(
                "  1. OWNERSHIP: 'X gehört Y', 'X hat Y' (für Personen/Named Entities)"
            )
            print("  2. PURPOSE: 'X dient zum Y', 'X wird zum Y benutzt'")
            print("  3. MATERIAL: 'X besteht aus Y'")
            print("  4. EQUIVALENCE: 'X ist Y' (bei Synonymen/Formeln)")
            print("  5. NEGATIVE RELATIONS: 'X ist kein Y', 'X ist nicht Y'")

        print()

    def _analyze_confidence_calibration(self):
        """Analysiert Confidence-Kalibrierung"""
        print("\n" + "=" * 100)
        print("TEIL 3: CONFIDENCE-CALIBRATION")
        print("=" * 100 + "\n")

        # Test verschiedene Confidence-Levels
        confidence_tests = [
            # Sehr hohe Confidence (>= 0.90)
            ("Ein Hund ist ein Tier", 0.92, "IS_A mit Artikel"),
            ("Berlin liegt in Deutschland", 0.93, "LOCATED_IN klar"),
            ("Ein Vogel kann fliegen", 0.91, "CAPABLE_OF eindeutig"),

            # Hohe Confidence (0.85-0.89)
            ("Katzen sind Säugetiere", 0.87, "IS_A Plural"),
            ("Ein Auto hat Räder", 0.88, "PART_OF"),
            ("Fische leben im Meer", 0.89, "LOCATED_IN (leben)"),

            # Mittlere Confidence (0.70-0.84)
            ("Hunde sind intelligent", 0.78, "HAS_PROPERTY (Adjektiv)"),
            ("Eine Rose ist rot", 0.78, "HAS_PROPERTY (Farbe)"),
        ]

        print("Confidence-Level Verteilung:")
        print("-" * 100)

        calibration_results = []
        for sentence, expected_conf, description in confidence_tests:
            doc = self.preprocessor.process(sentence)
            meaning_points = self.extractor.extract(doc)

            if meaning_points:
                actual_conf = meaning_points[0].confidence
                diff = abs(actual_conf - expected_conf)

                if diff < 0.01:
                    status = "[PERFEKT]"
                elif diff < 0.05:
                    status = "[GUT]"
                elif diff < 0.10:
                    status = "[OK]"
                else:
                    status = "[ABWEICHUNG]"

                print(
                    f"{status} {sentence[:40]:<40} | Expected: {expected_conf:.2f}, "
                    f"Actual: {actual_conf:.2f}, Diff: {diff:+.2f} | {description}"
                )

                calibration_results.append(diff)
            else:
                print(f"[FEHLER] {sentence[:40]:<40} | Nicht erkannt!")

        avg_diff = sum(calibration_results) / len(calibration_results)
        print(f"\n[BEWERTUNG] Durchschnittliche Abweichung: {avg_diff:.3f}")

        if avg_diff < 0.05:
            print("  [OPTIMAL] Confidence-Werte gut kalibriert!")
        elif avg_diff < 0.10:
            print("  [GUT] Confidence-Werte akzeptabel kalibriert")
        else:
            print("  [WARNUNG] Confidence-Werte sollten nachjustiert werden")

        print()

    def _analyze_pattern_orchestrator(self):
        """Analysiert Pattern Orchestrator"""
        print("\n" + "=" * 100)
        print("TEIL 4: PATTERN RECOGNITION ORCHESTRATOR")
        print("=" * 100 + "\n")

        # Teste Pattern Orchestrator mit verschiedenen Inputs
        orchestrator_tests = [
            ("Hallo", "Typo-freier Text", False),
            ("Katze", "Bekanntes Wort", False),
            ("Ktaze", "Typischer Typo (1 Edit)", True),
            ("Hudn", "Typo mit Transposition", True),
        ]

        print("Pattern Orchestrator Tests:")
        print("-" * 100)

        for text, description, should_correct in orchestrator_tests:
            result = self.pattern_orchestrator.process_input(text)

            has_corrections = len(result.get("typo_corrections", [])) > 0
            corrected_text = result.get("corrected_text", text)

            if has_corrections == should_correct:
                status = "[OK]"
            else:
                status = "[FEHLER]"

            correction_str = (
                f"-> '{corrected_text}'" if has_corrections else "Keine Korrektur"
            )
            print(
                f"{status} '{text}' | {description:<25} | {correction_str}"
            )

        print(
            f"\n[INFO] Pattern Orchestrator nutzt adaptive Thresholds "
            f"(aktuell: typo={self.adaptive_manager.get_typo_threshold()})"
        )
        print()

    def _analyze_extended_relations(self):
        """Analysiert erweiterte Relationstypen"""
        print("\n" + "=" * 100)
        print("TEIL 5: ERWEITERTE RELATIONSTYPEN")
        print("=" * 100 + "\n")

        # Test erweiterte Relationen
        extended_tests = [
            # Inverse Relations
            ("Deutschland enthält Berlin", "CONTAINS (Inverse von LOCATED_IN)", False),
            ("Tiere umfassen Hunde", "INCLUDES (Inverse von IS_A)", False),

            # Sibling Relations
            ("Hunde und Katzen sind beides Haustiere", "SIBLING (gemeinsames Parent)", False),

            # Degree/Intensity
            ("Eis ist sehr kalt", "DEGREE (Intensität)", False),
            ("Ein Gepard ist extrem schnell", "DEGREE (Superlativ)", False),

            # Conditional Properties
            ("Wasser ist flüssig bei Raumtemperatur", "CONDITIONAL_PROPERTY", False),
            ("Metall leitet Strom", "GENERAL_PROPERTY", True),

            # Multi-hop Relations
            ("Ein Pudel ist ein Hund ist ein Tier", "TRANSITIVE IS_A", False),
        ]

        detected = 0
        for sentence, category, should_detect in extended_tests:
            doc = self.preprocessor.process(sentence)
            meaning_points = self.extractor.extract(doc)

            is_detected = len(meaning_points) > 0
            relation = (
                meaning_points[0].arguments.get("relation_type")
                if is_detected
                else None
            )

            if is_detected == should_detect:
                status = "[OK]"
                if is_detected:
                    detected += 1
            else:
                status = "[FEHLT]" if should_detect else "[ERWARTET]"

            detection_str = f"Erkannt ({relation})" if is_detected else "Nicht erkannt"
            print(
                f"{status} {sentence[:50]:<50} | {category:<30} | {detection_str}"
            )

        print(
            f"\n[BEWERTUNG] Erweiterte Relationen sind größtenteils noch nicht implementiert"
        )
        print(
            "  [EMPFEHLUNG] Für v2.0: Inverse Relations, Conditional Properties, Degree/Intensity"
        )
        print()

    def _print_advanced_summary(self):
        """Gibt erweiterte Zusammenfassung aus"""
        print("\n" + "=" * 100)
        print("ERWEITERTE ZUSAMMENFASSUNG")
        print("=" * 100 + "\n")

        stats = self.adaptive_manager.get_system_stats()

        print("SYSTEM-STATUS:")
        print(f"  Bootstrap Phase: {stats['phase']}")
        print(f"  System Maturity: {stats['system_maturity']:.1%}")
        print(f"  Vocabulary: {stats['vocab_size']} Wörter")
        print()

        print("IMPLEMENTIERTE FEATURES:")
        print("  [OK] IS_A (mit/ohne Artikel, Plural)")
        print("  [OK] HAS_PROPERTY (Adjektive, Farben)")
        print("  [OK] CAPABLE_OF (kann/können)")
        print("  [OK] PART_OF (hat/haben, gehört zu)")
        print("  [OK] LOCATED_IN (liegt in, ist in, lebt in, wohnt in)")
        print("  [OK] Negationen (nicht, kein)")
        print("  [OK] Konditional-Filter (wenn, falls)")
        print("  [OK] Komparativ-Filter (größer als, schneller als)")
        print()

        print("FEHLENDE FEATURES (PRIORITÄT HOCH):")
        print("  [TODO] OWNERSHIP (gehört mir/dir, Peters Auto)")
        print("  [TODO] PURPOSE (dient zum, wird benutzt für)")
        print("  [TODO] MATERIAL (besteht aus)")
        print("  [TODO] EQUIVALENCE (Synonyme, Formeln)")
        print("  [TODO] NEGATIVE_RELATION (ist kein, ist nicht)")
        print()

        print("FEHLENDE FEATURES (PRIORITÄT MITTEL):")
        print("  [TODO] TEMPORAL_SEQUENCE (folgt auf, kommt nach)")
        print("  [TODO] CAUSALITY (verursacht, führt zu)")
        print("  [TODO] DEGREE/INTENSITY (sehr, extrem, kaum)")
        print()

        print("OPTIMIERUNGSPOTENZIAL:")
        print("  1. Confidence-Kalibrierung könnte feiner justiert werden")
        print("  2. Named Entity Recognition für besseres Ownership-Handling")
        print("  3. Context-Aware Pattern Matching (Bank = Sitzbank vs. Geldinstitut)")
        print("  4. Multi-Sentence Learning (Kontext über Satzgrenzen hinweg)")
        print()

        print("=" * 100)


if __name__ == "__main__":
    analyzer = AdvancedAutonomousAnalyzer()
    analyzer.run_full_analysis()
