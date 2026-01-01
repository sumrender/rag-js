"""
Service for managing and using sentence-transformer embedding models.
"""
import functools
import logging
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
from ..core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Manages the lifecycle of sentence-transformer models and provides
    embedding generation functions.
    """
    def __init__(self):
        self._clip_model: Optional[SentenceTransformer] = None
        self._text_model: Optional[SentenceTransformer] = None
        self._device = settings.DEVICE
        self._clip_model_name = settings.CLIP_MODEL_NAME
        self._text_model_name = settings.TEXT_MODEL_NAME
        self._cache_size = settings.EMBEDDING_CACHE_SIZE
        self.__post_init__()

    def __post_init__(self):
        self._cached_text_embed_internal = functools.lru_cache(maxsize=self._cache_size)(self._cached_text_embed_internal)
        self._cached_clip_embed_internal = functools.lru_cache(maxsize=self._cache_size)(self._cached_clip_embed_internal)

    @property
    def clip_model(self) -> SentenceTransformer:
        """Load CLIP model (singleton pattern)"""
        if self._clip_model is None:
            logger.info(f"Loading CLIP model: {self._clip_model_name} on device: {self._device}")
            self._clip_model = SentenceTransformer(self._clip_model_name, device=self._device)
            logger.info("CLIP model loaded successfully")
        return self._clip_model

    @property
    def text_model(self) -> SentenceTransformer:
        """Load text embedding model (singleton pattern)"""
        if self._text_model is None:
            logger.info(f"Loading text model: {self._text_model_name} on device: {self._device}")
            self._text_model = SentenceTransformer(self._text_model_name, device=self._device)
            logger.info("Text model loaded successfully")
        return self._text_model

    def get_text_embedding(self, text: str) -> Tuple:
        """
        Generate and cache text embedding (384D).
        Checks master cache flag before using LRU cache.
        """
        if not settings.ENABLE_CACHE:
            embedding = self.text_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return tuple(embedding.tolist())
        return self._cached_text_embed_internal(text)

    def get_clip_embedding(self, text: str) -> Tuple:
        """
        Generate and cache CLIP text embedding (512D).
        Checks master cache flag before using LRU cache.
        """
        if not settings.ENABLE_CACHE:
            embedding = self.clip_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return tuple(embedding.tolist())
        return self._cached_clip_embed_internal(text)

    def _cached_text_embed_internal(self, text: str) -> tuple:
        """Internal cached text embedding function (384D)"""
        embedding = self.text_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return tuple(embedding.tolist())

    def _cached_clip_embed_internal(self, text: str) -> tuple:
        """Internal cached CLIP text embedding function (512D)"""
        embedding = self.clip_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return tuple(embedding.tolist())
