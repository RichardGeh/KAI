import logging
from functools import lru_cache
from typing import Optional, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from component_15_logging_config import get_logger
from kai_exceptions import EmbeddingError

logger = get_logger(__name__)


# Backwards compatibility: Alias für alte Exception-Namen
EmbeddingServiceError = EmbeddingError
ModelNotLoadedError = EmbeddingError


class EmbeddingService:
    """
    Ein zentraler Service zur Erzeugung von Text-Embeddings.
    Lädt ein vortrainiertes Modell und stellt eine einfache Schnittstelle bereit.

    Performance-Optimierung: Embeddings werden gecacht mit LRU-Cache (maxsize=1000).
    Dies vermeidet redundante Berechnungen für identische Texte.
    """

    def __init__(
        self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ) -> None:
        self.model: Optional[SentenceTransformer] = None
        self.vector_dimension: int = 0
        self._model_name: str = model_name

        # Cache für Embeddings (maxsize=1000 Texte)
        # Verwendet LRU-Strategie: Least Recently Used werden entfernt
        self._embedding_cache = lru_cache(maxsize=1000)(
            self._compute_embedding_uncached
        )

        # Periodic cache statistics logging
        self._call_count: int = 0
        self._log_interval: int = 100  # Log every 100 calls

        try:
            logger.info("Lade Embedding-Modell", extra={"model_name": model_name})
            self.model = SentenceTransformer(model_name)
            dimension = self.model.get_sentence_embedding_dimension()
            # get_sentence_embedding_dimension kann theoretisch None zurückgeben
            self.vector_dimension = dimension if dimension is not None else 0
            logger.info(
                "Embedding-Modell geladen",
                extra={
                    "model_name": model_name,
                    "vector_dimension": self.vector_dimension,
                },
            )
        except Exception as e:
            logger.critical(
                "Konnte Embedding-Modell nicht laden",
                extra={
                    "model_name": model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Modell bleibt None, is_available() wird False zurückgeben
            # Graceful Degradation: Kein Raise, aber Modell ist None

    def _compute_embedding_uncached(self, text: str) -> tuple[float, ...]:
        """
        Interne Methode zur Berechnung eines Embeddings OHNE Cache.
        Wird vom LRU-Cache gewrappt für automatisches Caching.

        Args:
            text: Der zu vektorisierende Text

        Returns:
            Tuple von Floats (hashbar für LRU-Cache)

        Raises:
            EmbeddingError: Wenn das Modell nicht verfügbar ist

        Note:
            Diese Methode nimmt an, dass Validierung bereits erfolgt ist.
        """
        # Type narrowing: Nach is_available() Check ist self.model garantiert nicht None
        if self.model is None:
            raise EmbeddingError(
                "Modell ist nicht verfügbar für Embedding-Berechnung. "
                "Service wurde möglicherweise nicht korrekt initialisiert."
            )

        embedding = self.model.encode([text])[0]

        # Conditional logging für DEBUG (Performance-kritisch)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Embedding berechnet (uncached)",
                extra={
                    "text_length": len(text),
                    "text_preview": text[:50],
                    "vector_dimension": len(embedding),
                    "vector_norm": float(np.linalg.norm(embedding)),
                },
            )

        # Konvertiere zu Tuple (hashbar für Cache)
        return tuple(float(x) for x in embedding)

    def get_embedding(self, text: str) -> list[float]:
        """
        Erzeugt einen Embedding-Vektor für einen gegebenen Text.

        Performance-Optimierung: Nutzt LRU-Cache für bereits berechnete Texte.
        Cache-Hits vermeiden teure Modell-Inferenz.

        Args:
            text: Der zu vektorisierende Text

        Returns:
            Liste von Floats, die den Vektor repräsentiert

        Raises:
            ModelNotLoadedError: Wenn das Modell nicht verfügbar ist
            ValueError: Wenn der Text leer ist
        """
        if not self.is_available():
            raise EmbeddingError(
                f"Embedding-Modell '{self._model_name}' ist nicht verfügbar. "
                "Service wurde möglicherweise nicht korrekt initialisiert."
            )

        if not text or not text.strip():
            raise ValueError("Leerer Text für Embedding übergeben.")

        try:
            # Verwende gecachte Version (automatisches Caching durch LRU)
            embedding_tuple = self._embedding_cache(text)

            # Increment call counter and log cache statistics periodically
            self._call_count += 1
            if self._call_count % self._log_interval == 0:
                cache_stats = self.get_cache_info()
                logger.info(
                    "Embedding cache statistics",
                    extra={
                        "calls": self._call_count,
                        "hit_rate": cache_stats["hit_rate"],
                        "cache_size": cache_stats["currsize"],
                        "max_size": cache_stats["maxsize"],
                    },
                )

            # Konvertiere zurück zu Liste für API-Kompatibilität
            return list(embedding_tuple)
        except Exception as e:
            logger.error(
                "Embedding-Generierung fehlgeschlagen",
                extra={
                    "text_preview": text[:50],
                    "text_length": len(text),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise EmbeddingError(f"Embedding-Generierung fehlgeschlagen: {e}") from e

    def get_cache_info(self) -> dict:
        """
        Gibt Statistiken über den Embedding-Cache zurück.

        Returns:
            Dict mit Cache-Statistiken (hits, misses, maxsize, currsize)
        """
        cache_info = self._embedding_cache.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_rate": (
                cache_info.hits / (cache_info.hits + cache_info.misses)
                if (cache_info.hits + cache_info.misses) > 0
                else 0.0
            ),
        }

    def get_embeddings_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """
        Erzeugt Embeddings für mehrere Texte auf einmal (effizienter als Einzelaufrufe).

        Args:
            texts: Liste von Texten

        Returns:
            Liste von Embedding-Vektoren (None für leere Texte)

        Raises:
            EmbeddingError: Wenn das Modell nicht verfügbar ist
            ValueError: Wenn die Liste leer ist
        """
        if not self.is_available():
            raise EmbeddingError("Embedding-Modell ist nicht verfügbar.")

        # Type narrowing: Nach is_available() Check ist self.model garantiert nicht None
        if self.model is None:
            raise EmbeddingError(
                "Modell ist nicht verfügbar für Batch-Embedding-Berechnung. "
                "Service wurde möglicherweise nicht korrekt initialisiert."
            )

        if not texts:
            raise ValueError("Leere Textliste für Batch-Embedding übergeben.")

        # Filtere leere Texte und merke Indizes
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            raise ValueError("Alle Texte in der Batch-Liste sind leer.")

        try:
            embeddings = self.model.encode(valid_texts)

            # Conditional logging für DEBUG
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Batch-Embeddings erzeugt",
                    extra={
                        "batch_size": len(valid_texts),
                        "total_size": len(texts),
                        "empty_texts": len(texts) - len(valid_texts),
                    },
                )

            # Erstelle Ergebnisliste mit None für leere Texte
            result: list[Optional[list[float]]] = [None] * len(texts)
            for idx, embedding in zip(valid_indices, embeddings):
                result[idx] = cast(list[float], embedding.tolist())

            return result
        except Exception as e:
            logger.error(
                "Batch-Embedding-Generierung fehlgeschlagen",
                extra={
                    "batch_size": len(valid_texts),
                    "total_size": len(texts),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise EmbeddingError(
                f"Batch-Embedding-Generierung fehlgeschlagen: {e}"
            ) from e

    def is_available(self) -> bool:
        """
        Prüft, ob der Service einsatzbereit ist.

        Returns:
            True, wenn das Modell geladen und verfügbar ist
        """
        return self.model is not None

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Texten.

        Args:
            text1: Erster Text
            text2: Zweiter Text

        Returns:
            Ähnlichkeitswert zwischen -1 und 1 (typischerweise 0 bis 1)

        Raises:
            ModelNotLoadedError: Wenn das Modell nicht verfügbar ist
        """
        vec1 = np.array(self.get_embedding(text1))
        vec2 = np.array(self.get_embedding(text2))

        # Kosinus-Ähnlichkeit
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))
