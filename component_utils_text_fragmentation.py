# component_utils_text_fragmentation.py
"""
Text fragmentation utilities for word usage tracking.

Provides intelligent extraction of context fragments from sentences:
- Dynamic comma-based or window-based context extraction
- N-gram connection extraction for word co-occurrence statistics
- Configurable via kai_config.py
"""

import re
from typing import Any, Dict, List, Tuple

from component_15_logging_config import get_logger
from kai_config import get_config

logger = get_logger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


class WordFragment:
    """
    Represents a context fragment for a single word in a sentence.

    Attributes:
        word: The target word (original form)
        lemma: Normalized form of the word
        fragment: The context fragment text
        word_position: Position of word within fragment (0-indexed)
        fragment_type: "window" or "comma_delimited"
        total_words: Total number of words in fragment
    """

    def __init__(
        self,
        word: str,
        lemma: str,
        fragment: str,
        word_position: int,
        fragment_type: str,
        total_words: int,
    ):
        self.word = word
        self.lemma = lemma
        self.fragment = fragment
        self.word_position = word_position
        self.fragment_type = fragment_type
        self.total_words = total_words

    def __repr__(self):
        return (
            f"WordFragment(word='{self.word}', lemma='{self.lemma}', "
            f"fragment='{self.fragment}', pos={self.word_position})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "word": self.word,
            "lemma": self.lemma,
            "fragment": self.fragment,
            "word_position": self.word_position,
            "fragment_type": self.fragment_type,
            "total_words": self.total_words,
        }


class WordConnection:
    """
    Represents a directional connection between two words (N-gram).

    Attributes:
        word1_lemma: First word (normalized)
        word2_lemma: Second word (normalized)
        distance: Distance between words (1-N)
        direction: "before" (word1 comes before word2) or "after"
    """

    def __init__(
        self, word1_lemma: str, word2_lemma: str, distance: int, direction: str
    ):
        self.word1_lemma = word1_lemma
        self.word2_lemma = word2_lemma
        self.distance = distance
        self.direction = direction

    def __repr__(self):
        arrow = "->" if self.direction == "before" else "<-"
        return f"WordConnection({self.word1_lemma} {arrow}[{self.distance}] {self.word2_lemma})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "word1_lemma": self.word1_lemma,
            "word2_lemma": self.word2_lemma,
            "distance": self.distance,
            "direction": self.direction,
        }


# ============================================================================
# TEXT FRAGMENTATION
# ============================================================================


class TextFragmenter:
    """
    Extracts context fragments and word connections from sentences.

    Uses configuration from kai_config.py for:
    - context_window_size (±N words)
    - max_words_to_comma (threshold for comma-based extraction)
    """

    def __init__(self, linguistic_preprocessor=None):
        """
        Args:
            linguistic_preprocessor: Optional spaCy preprocessor for lemmatization
                                     (component_6_linguistik_engine.LinguisticPreprocessor)
        """
        self.config = get_config()
        self.preprocessor = linguistic_preprocessor

        # Load settings
        self.window_size = self.config.context_window_size
        self.max_words_to_comma = self.config.max_words_to_comma

        logger.debug(
            "TextFragmenter initialisiert",
            extra={
                "window_size": self.window_size,
                "max_words_to_comma": self.max_words_to_comma,
            },
        )

    def extract_fragments_and_connections(
        self, sentence: str
    ) -> Tuple[List[WordFragment], List[WordConnection]]:
        """
        Extrahiert Fragmente und Connections aus einem Satz.

        Args:
            sentence: Input-Satz

        Returns:
            Tupel von (fragments, connections)
        """
        # Tokenize sentence (einfach via split, oder via spaCy wenn verfügbar)
        tokens = self._tokenize(sentence)

        if not tokens:
            return [], []

        # Extrahiere Fragmente
        fragments = self._extract_fragments(tokens, sentence)

        # Extrahiere Connections
        connections = self._extract_connections(tokens)

        logger.debug(
            "Fragmente und Connections extrahiert",
            extra={
                "sentence_preview": sentence[:50],
                "fragment_count": len(fragments),
                "connection_count": len(connections),
            },
        )

        return fragments, connections

    def _tokenize(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Tokenisiert Satz und gibt strukturierte Token zurück.

        Returns:
            Liste von Dicts mit {text, lemma, pos, is_punct}
        """
        # Falls spaCy verfügbar, nutze Preprocessor
        if self.preprocessor:
            try:
                doc = self.preprocessor.process(sentence)
                tokens = []
                for token in doc:
                    tokens.append(
                        {
                            "text": token.text,
                            "lemma": token.lemma_.lower(),
                            "pos": token.pos_,
                            "is_punct": token.is_punct,
                        }
                    )
                return tokens
            except Exception as e:
                logger.warning(
                    "spaCy Tokenization fehlgeschlagen, verwende Fallback",
                    extra={"error": str(e)},
                )

        # Fallback: Einfaches Whitespace-Splitting
        words = sentence.split()
        tokens = []
        for word in words:
            # Entferne Satzzeichen für Lemma
            lemma = re.sub(r"[^\w-]", "", word).lower()
            is_punct = word in [",", ".", "!", "?", ":", ";"]

            tokens.append(
                {"text": word, "lemma": lemma, "pos": "UNKNOWN", "is_punct": is_punct}
            )

        return tokens

    def _extract_fragments(
        self, tokens: List[Dict[str, Any]], original_sentence: str
    ) -> List[WordFragment]:
        """
        Extrahiert Kontext-Fragmente für jedes Wort.

        Strategie:
        - Prüfe Abstand zum nächsten Komma
        - Falls <= max_words_to_comma: Bis Komma
        - Sonst: ±window_size Fenster
        """
        fragments = []

        for i, token in enumerate(tokens):
            # Ignoriere Satzzeichen
            if token["is_punct"] or not token["lemma"]:
                continue

            # Finde nächstes Komma (links und rechts)
            left_comma_dist = self._distance_to_comma(tokens, i, direction="left")
            right_comma_dist = self._distance_to_comma(tokens, i, direction="right")

            # Entscheide: Komma oder Fenster?
            if (
                left_comma_dist <= self.max_words_to_comma
                and right_comma_dist <= self.max_words_to_comma
            ):
                # Bis Komma
                fragment_type = "comma_delimited"
                start_idx = max(0, i - left_comma_dist)
                end_idx = min(len(tokens), i + right_comma_dist + 1)
            else:
                # Fenster-basiert
                fragment_type = "window"
                start_idx = max(0, i - self.window_size)
                end_idx = min(len(tokens), i + self.window_size + 1)

            # Extrahiere Fragment-Text
            fragment_tokens = tokens[start_idx:end_idx]
            fragment_text = " ".join(t["text"] for t in fragment_tokens)

            # Berechne Position des Wortes im Fragment
            word_position = i - start_idx

            # Erstelle Fragment-Objekt
            fragment = WordFragment(
                word=token["text"],
                lemma=token["lemma"],
                fragment=fragment_text,
                word_position=word_position,
                fragment_type=fragment_type,
                total_words=len(fragment_tokens),
            )

            fragments.append(fragment)

        return fragments

    def _extract_connections(
        self, tokens: List[Dict[str, Any]]
    ) -> List[WordConnection]:
        """
        Extrahiert CONNECTION Edges (N-Gramme) für ±window_size.

        Für jedes Wort werden Verbindungen zu Nachbar-Wörtern (distance 1-N) erstellt.
        """
        connections = []

        for i, token in enumerate(tokens):
            # Ignoriere Satzzeichen
            if token["is_punct"] or not token["lemma"]:
                continue

            # Verbindungen zu vorherigen Wörtern (direction="before")
            for dist in range(1, self.window_size + 1):
                prev_idx = i - dist
                if prev_idx >= 0:
                    prev_token = tokens[prev_idx]
                    if not prev_token["is_punct"] and prev_token["lemma"]:
                        conn = WordConnection(
                            word1_lemma=prev_token["lemma"],
                            word2_lemma=token["lemma"],
                            distance=dist,
                            direction="before",
                        )
                        connections.append(conn)

            # Verbindungen zu folgenden Wörtern (direction="after")
            for dist in range(1, self.window_size + 1):
                next_idx = i + dist
                if next_idx < len(tokens):
                    next_token = tokens[next_idx]
                    if not next_token["is_punct"] and next_token["lemma"]:
                        conn = WordConnection(
                            word1_lemma=token["lemma"],
                            word2_lemma=next_token["lemma"],
                            distance=dist,
                            direction="before",  # "before" because token comes before next_token
                        )
                        connections.append(conn)

        return connections

    def _distance_to_comma(
        self, tokens: List[Dict[str, Any]], current_idx: int, direction: str
    ) -> int:
        """
        Berechnet Distanz zum nächsten Komma (in Anzahl Wörtern, nicht Token).

        Args:
            tokens: Token-Liste
            current_idx: Index des aktuellen Wortes
            direction: "left" oder "right"

        Returns:
            Anzahl Wörter bis zum Komma (999 wenn kein Komma gefunden)
        """
        distance = 0
        step = -1 if direction == "left" else 1
        idx = current_idx + step

        while 0 <= idx < len(tokens):
            token = tokens[idx]

            # Komma gefunden?
            if token["text"] == ",":
                return distance

            # Zähle nur nicht-Satzzeichen
            if not token["is_punct"]:
                distance += 1

            # Abbruch bei Satzende
            if token["text"] in [".", "!", "?"]:
                break

            idx += step

        # Kein Komma gefunden
        return 999


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def extract_word_usage_from_sentence(
    sentence: str, linguistic_preprocessor=None
) -> Tuple[List[WordFragment], List[WordConnection]]:
    """
    Standalone-Funktion zum Extrahieren von Fragmenten und Connections.

    Args:
        sentence: Input-Satz
        linguistic_preprocessor: Optional spaCy preprocessor

    Returns:
        Tupel von (fragments, connections)
    """
    fragmenter = TextFragmenter(linguistic_preprocessor)
    return fragmenter.extract_fragments_and_connections(sentence)


if __name__ == "__main__":
    # Test-Code
    print("=== Text Fragmentation Tests ===\n")

    test_sentences = [
        "Das Haus steht im großen Park.",
        "Der Hund, der sehr groß ist, bellt laut.",
        "Katzen können sehr gut klettern und jagen.",
    ]

    for sentence in test_sentences:
        print(f"Satz: {sentence}")
        fragments, connections = extract_word_usage_from_sentence(sentence)

        print(f"  Fragmente ({len(fragments)}):")
        for frag in fragments:
            print(
                f"    - {frag.lemma:15s} -> '{frag.fragment}' (pos={frag.word_position}, type={frag.fragment_type})"
            )

        print(f"  Connections ({len(connections)}):")
        for conn in connections[:10]:  # Nur erste 10 zeigen
            print(f"    - {conn}")

        print()
