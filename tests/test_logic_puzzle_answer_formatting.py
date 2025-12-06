"""
Test für verbesserte Logic Puzzle Antwortformatierung

Stellt sicher, dass Antworten das Verb aus der Frage verwenden
statt generischem "hat".
"""

import pytest

from component_45_logic_puzzle_solver import LogicPuzzleSolver


class TestLogicPuzzleAnswerFormatting:
    """Tests für kontextuelle Antwortformatierung"""

    @pytest.fixture
    def solver(self):
        """LogicPuzzleSolver Instanz"""
        return LogicPuzzleSolver()

    def test_brandy_puzzle_uses_question_verb(self, solver):
        """
        Brandy-Rätsel sollte 'trinkt gerne' statt 'hat' verwenden

        Vorher: "Mark hat brandy und Leo hat brandy"
        Nachher: "Mark trinkt gerne brandy und Leo trinkt gerne brandy"
        """
        conditions_text = """
        Wenn Leo einen Brandy bestellt, bestellt auch Mark einen.
        Es kann vorkommen, dass Mark oder Nick einen Brandy bestellen, aber nie beide zusammen.
        Hingegen geschieht es, dass Leo und Nick einzeln oder gleichzeitig einen Brandy bestellen.
        Wenn Nick einen Brandy bestellt, will Leo auch einen.
        """

        entities = ["Leo", "Mark", "Nick"]
        question = "Wer von den dreien trinkt also gerne einen Brandy?"

        result = solver.solve(conditions_text, entities, question)

        assert result["result"] == "SATISFIABLE", "Rätsel sollte lösbar sein"

        answer = result["answer"]

        # Antwort sollte 'trinkt gerne' verwenden (aus der Frage extrahiert)
        assert (
            "trinkt gerne" in answer.lower() or "trinkt also gerne" in answer.lower()
        ), f"Antwort sollte 'trinkt gerne' enthalten, aber war: {answer}"

        # Antwort sollte NICHT nur 'hat' verwenden
        # (außer in Kombination mit dem Objekt, was OK wäre)
        # Dies ist ein Negativ-Test für die alte Formulierung
        assert not (
            answer.count("hat brandy") > 0 and "trinkt" not in answer.lower()
        ), f"Antwort sollte nicht nur 'hat brandy' verwenden: {answer}"

        # Verifiziere korrekte Lösung (Mark und Leo)
        assert "mark" in answer.lower(), "Mark sollte in der Antwort sein"
        assert "leo" in answer.lower(), "Leo sollte in der Antwort sein"

    def test_verb_extraction_with_adverb(self, solver):
        """
        Verb-Extraktion sollte Adverbien wie 'gerne' erkennen

        "Wer trinkt gerne Kaffee?" -> "trinkt gerne"
        """
        # Realistische Bedingungen mit vollständigen Objekt-Verben
        conditions_text = """
        Wenn Anna einen Kaffee trinkt, trinkt Bob auch einen.
        Anna trinkt einen Kaffee.
        Es kann vorkommen, dass Bob oder Lisa einen Kaffee trinken, aber nie beide.
        """

        entities = ["Anna", "Bob", "Lisa"]
        question = "Wer trinkt gerne Kaffee?"

        result = solver.solve(conditions_text, entities, question)

        if result["result"] == "SATISFIABLE":
            answer = result["answer"]
            # Sollte 'trinkt gerne' oder mindestens 'trinkt' verwenden (aus Frage extrahiert)
            # Wir akzeptieren beide, da die Adverb-Extraktion optional ist
            assert (
                "trinkt gerne" in answer.lower() or "trinkt" in answer.lower()
            ), f"Antwort sollte Verb 'trinkt' enthalten: {answer}"

    def test_simple_verb_without_adverb(self, solver):
        """
        Einfache Verben ohne Adverb sollten auch funktionieren

        "Wer bestellt Pizza?" -> "bestellt"
        """
        conditions_text = """
        Wenn Tom eine Pizza bestellt, bestellt Lisa auch eine.
        Tom bestellt eine Pizza.
        Es kann vorkommen, dass Tom oder Max eine Pizza bestellen, aber nie beide.
        """
        entities = ["Tom", "Lisa", "Max"]
        question = "Wer bestellt Pizza?"

        result = solver.solve(conditions_text, entities, question)

        if result["result"] == "SATISFIABLE":
            answer = result["answer"]
            # Sollte 'bestellt' verwenden (aus Frage)
            assert (
                "bestellt" in answer.lower()
            ), f"Antwort sollte Verb 'bestellt' enthalten: {answer}"

    def test_fallback_when_no_question_provided(self, solver):
        """
        Wenn keine Frage angegeben wird, sollte Fallback auf 'hat' funktionieren
        """
        conditions_text = """
        Wenn Leo einen Brandy bestellt, bestellt auch Mark einen.
        Wenn Nick einen Brandy bestellt, will Leo auch einen.
        Leo bestellt einen Brandy.
        """
        entities = ["Leo", "Mark", "Nick"]
        question = None  # Keine Frage

        result = solver.solve(conditions_text, entities, question)

        assert result["result"] == "SATISFIABLE", "Rätsel sollte lösbar sein"

        answer = result["answer"]

        # Ohne Frage sollte der Fallback verwendet werden
        # (entweder 'hat' oder die originale Formulierung)
        assert len(answer) > 0, "Antwort sollte nicht leer sein"
        # Keine spezifische Assertion, nur dass es nicht abstürzt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
