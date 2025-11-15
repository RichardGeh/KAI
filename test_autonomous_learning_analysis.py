# test_autonomous_learning_analysis.py
"""
Systematische Analyse des automatischen Lernsystems von KAI.

Testet verschiedene Szenarien und identifiziert Schwächen:
1. Einfache deklarative Aussagen (sollten funktionieren)
2. Komplexe/verschachtelte Aussagen
3. Negationen
4. Konditionale Aussagen
5. Temporale Aussagen
6. Quantifizierte Aussagen
7. Vergleichende Aussagen
8. Kausale Beziehungen
"""

import io
import sys
from typing import List, Tuple

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
from component_5_linguistik_strukturen import MeaningPointCategory


class AutonomousLearningAnalyzer:
    """Analysiert automatisches Lernsystem und identifiziert Schwächen"""

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

        self.results = []

    def test_sentence(
        self, sentence: str, expected_detection: bool, category: str = "DEFINITION"
    ) -> dict:
        """
        Testet einen einzelnen Satz.

        Args:
            sentence: Zu testender Satz
            expected_detection: True wenn Satz als DEFINITION erkannt werden sollte
            category: Erwartete Kategorie

        Returns:
            Dict mit Testergebnissen
        """
        doc = self.preprocessor.process(sentence)
        meaning_points = self.extractor.extract(doc)

        detected = False
        confidence = 0.0
        relation_type = None
        detected_category = None
        cue = None

        if meaning_points:
            mp = meaning_points[0]
            detected_category = mp.category.name
            detected = mp.category == MeaningPointCategory.DEFINITION
            confidence = mp.confidence
            relation_type = mp.arguments.get("relation_type")
            cue = mp.cue

        success = detected == expected_detection

        result = {
            "sentence": sentence,
            "expected_detection": expected_detection,
            "detected": detected,
            "success": success,
            "confidence": confidence,
            "relation_type": relation_type,
            "category": detected_category,
            "cue": cue,
        }

        self.results.append(result)
        return result

    def run_test_suite(self):
        """Führt vollständige Test-Suite aus"""
        print("\n" + "=" * 100)
        print("SYSTEMATISCHE ANALYSE: AUTOMATISCHES LERNSYSTEM VON KAI")
        print("=" * 100 + "\n")

        # 1. EINFACHE DEKLARATIVE AUSSAGEN (sollten funktionieren)
        print("\n1. EINFACHE DEKLARATIVE AUSSAGEN (BASELINE)")
        print("-" * 100)
        simple_tests = [
            ("Ein Hund ist ein Tier", True),
            ("Ein Vogel kann fliegen", True),
            ("Berlin liegt in Deutschland", True),
            ("Eine Rose ist rot", True),
            ("Ein Auto hat Räder", True),
        ]
        self._run_test_group(simple_tests, "Einfache Aussagen")

        # 2. NEGATIONEN (kritische Schwäche!)
        print("\n2. NEGATIONEN (KRITISCH)")
        print("-" * 100)
        negation_tests = [
            ("Ein Pinguin kann nicht fliegen", True),  # Sollte erkannt werden!
            ("Ein Stein ist kein Lebewesen", True),
            ("Fische können nicht an Land leben", True),
            ("Glas ist nicht transparent", True),  # Falsche Aussage, aber sollte erkannt werden
        ]
        self._run_test_group(negation_tests, "Negationen")

        # 3. KOMPLEXE/VERSCHACHTELTE AUSSAGEN
        print("\n3. KOMPLEXE/VERSCHACHTELTE AUSSAGEN")
        print("-" * 100)
        complex_tests = [
            ("Ein Hund ist ein Tier, das bellen kann", True),
            ("Berlin ist die Hauptstadt von Deutschland", True),
            ("Ein Vogel ist ein Tier mit Flügeln", True),
            ("Eine Rose ist eine Blume, die Dornen hat", True),
        ]
        self._run_test_group(complex_tests, "Komplexe Aussagen")

        # 4. KONDITIONALE AUSSAGEN
        print("\n4. KONDITIONALE AUSSAGEN")
        print("-" * 100)
        conditional_tests = [
            ("Wenn es regnet, wird die Straße nass", False),  # Zu komplex
            ("Falls ein Tier Flügel hat, kann es fliegen", False),
        ]
        self._run_test_group(conditional_tests, "Konditionale")

        # 5. TEMPORALE AUSSAGEN
        print("\n5. TEMPORALE AUSSAGEN")
        print("-" * 100)
        temporal_tests = [
            ("Hunde waren früher Wölfe", False),  # Temporal -> nicht einfach deklarativ
            ("Im Winter ist es kalt", True),  # Könnte erkannt werden
            ("Morgens singt der Vogel", True),
        ]
        self._run_test_group(temporal_tests, "Temporal")

        # 6. QUANTIFIZIERTE AUSSAGEN
        print("\n6. QUANTIFIZIERTE AUSSAGEN")
        print("-" * 100)
        quantified_tests = [
            ("Alle Hunde sind Tiere", True),  # Sollte erkannt werden
            ("Manche Vögel können nicht fliegen", True),
            ("Die meisten Menschen sind freundlich", True),
            ("Einige Fische leben im Meer", True),
        ]
        self._run_test_group(quantified_tests, "Quantifiziert")

        # 7. VERGLEICHENDE AUSSAGEN
        print("\n7. VERGLEICHENDE AUSSAGEN")
        print("-" * 100)
        comparative_tests = [
            ("Ein Elefant ist größer als eine Maus", False),  # Zu komplex für einfache Patterns
            ("Hunde sind schneller als Schildkröten", False),
            ("Gold ist wertvoller als Silber", False),
        ]
        self._run_test_group(comparative_tests, "Vergleiche")

        # 8. KAUSALE BEZIEHUNGEN
        print("\n8. KAUSALE BEZIEHUNGEN")
        print("-" * 100)
        causal_tests = [
            ("Regen verursacht Nässe", False),  # Zu komplex
            ("Wärme lässt Eis schmelzen", False),
        ]
        self._run_test_group(causal_tests, "Kausal")

        # 9. MEHRDEUTIGE AUSSAGEN
        print("\n9. MEHRDEUTIGE AUSSAGEN")
        print("-" * 100)
        ambiguous_tests = [
            ("Die Bank ist hart", True),  # Bank = Sitzbank? Bank = Geldinstitut?
            ("Der Kiefer schmerzt", True),  # Kiefer = Baum? Kiefer = Körperteil?
        ]
        self._run_test_group(ambiguous_tests, "Mehrdeutig")

        # 10. EDGE CASES
        print("\n10. EDGE CASES")
        print("-" * 100)
        edge_cases = [
            ("", False),  # Leerer String
            ("Hund", False),  # Einzelnes Wort
            ("ist ein", False),  # Nur Relation
            ("X ist Y", True),  # Abstrakte Variablen
        ]
        self._run_test_group(edge_cases, "Edge Cases")

        # SUMMARY
        self._print_summary()

    def _run_test_group(self, tests: List[Tuple[str, bool]], group_name: str):
        """Führt eine Test-Gruppe aus und zeigt Ergebnisse"""
        passed = 0
        failed = 0

        for sentence, expected in tests:
            result = self.test_sentence(sentence, expected)

            status_icon = "[OK]" if result["success"] else "[FEHLER]"
            detection_str = (
                f"Erkannt ({result['relation_type']})"
                if result["detected"]
                else "Nicht erkannt"
            )

            print(
                f"{status_icon} {sentence[:60]:<60} | {detection_str:<25} | Conf: {result['confidence']:.2f}"
            )

            if result["success"]:
                passed += 1
            else:
                failed += 1

        print(
            f"\nGruppe '{group_name}': {passed}/{passed+failed} Tests bestanden ({100*passed/(passed+failed):.1f}%)\n"
        )

    def _print_summary(self):
        """Gibt Zusammenfassung aus"""
        print("\n" + "=" * 100)
        print("ZUSAMMENFASSUNG")
        print("=" * 100 + "\n")

        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])
        failed = total - passed

        print(f"Gesamt: {passed}/{total} Tests bestanden ({100*passed/total:.1f}%)")
        print(f"Bestanden: {passed}")
        print(f"Fehlgeschlagen: {failed}\n")

        # Schwächen-Analyse
        print("IDENTIFIZIERTE SCHWÄCHEN:")
        print("-" * 100)

        # Gruppiere Fehler nach Typ
        negation_failures = [
            r
            for r in self.results
            if not r["success"] and "nicht" in r["sentence"].lower()
        ]
        complex_failures = [
            r
            for r in self.results
            if not r["success"] and "," in r["sentence"] and r["expected_detection"]
        ]
        quantified_failures = [
            r
            for r in self.results
            if not r["success"]
            and any(
                q in r["sentence"].lower()
                for q in ["alle", "manche", "einige", "die meisten"]
            )
            and r["expected_detection"]
        ]

        if negation_failures:
            print(f"\n1. NEGATIONEN: {len(negation_failures)} Fehler")
            print("   Problem: 'nicht', 'kein' werden nicht erkannt")
            for r in negation_failures[:3]:
                print(f"   - {r['sentence']}")

        if complex_failures:
            print(f"\n2. KOMPLEXE SÄTZE: {len(complex_failures)} Fehler")
            print(
                "   Problem: Verschachtelte Sätze mit Nebensätzen werden nicht erkannt"
            )
            for r in complex_failures[:3]:
                print(f"   - {r['sentence']}")

        if quantified_failures:
            print(f"\n3. QUANTIFIZIERER: {len(quantified_failures)} Fehler")
            print("   Problem: 'alle', 'manche', 'einige' werden nicht verarbeitet")
            for r in quantified_failures[:3]:
                print(f"   - {r['sentence']}")

        # False Positives
        false_positives = [
            r for r in self.results if r["detected"] and not r["expected_detection"]
        ]
        if false_positives:
            print(f"\n4. FALSE POSITIVES: {len(false_positives)} Fehler")
            print("   Problem: Sätze werden fälschlicherweise als DEFINITION erkannt")
            for r in false_positives[:3]:
                print(f"   - {r['sentence']}")

        print("\n" + "=" * 100)


if __name__ == "__main__":
    analyzer = AutonomousLearningAnalyzer()
    analyzer.run_test_suite()
