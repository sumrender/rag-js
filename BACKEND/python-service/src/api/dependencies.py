"""
Dependency injection for the application's services.
"""
from functools import lru_cache
from typing import Optional
from pymongo import MongoClient
from ..core.config import settings
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import FAISSStore
from ..services.cache_manager import RedisCacheManager, SemanticCache
from ..services.file_processing_service import FileProcessingService
from ..services.ingestion_service import IngestionService
from ..services.retrieval_service import RetrievalService

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get an instance of the EmbeddingService."""
    return EmbeddingService()

@lru_cache(maxsize=1)
def get_text_store() -> FAISSStore:
    """Get an instance of the text FAISS store."""
    return FAISSStore(
        store_name=settings.FAISS_TEXT_COLLECTION,
        dimension=384,  # all-MiniLM-L6-v2 dimension
        data_dir=settings.FAISS_DATA_DIR,
    )

@lru_cache(maxsize=1)
def get_image_store() -> FAISSStore:
    """Get an instance of the image FAISS store."""
    return FAISSStore(
        store_name=settings.FAISS_IMAGE_COLLECTION,
        dimension=512,  # clip-ViT-B-32 dimension
        data_dir=settings.FAISS_DATA_DIR,
    )

@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    """Get an instance of the MongoClient."""
    return MongoClient(settings.MONGO_URI)

@lru_cache(maxsize=1)
def get_cache_manager() -> Optional[RedisCacheManager]:
    """Get an instance of the RedisCacheManager."""
    if settings.ENABLE_CACHE:
        return RedisCacheManager()
    return None

@lru_cache(maxsize=1)
def get_semantic_cache() -> Optional[SemanticCache]:
    """Get an instance of the SemanticCache."""
    if settings.ENABLE_CACHE:
        return SemanticCache(
            dimension=384,
            max_size=settings.SEMANTIC_CACHE_MAX_SIZE,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
        )
    return None

@lru_cache(maxsize=1)
def get_file_processing_service() -> FileProcessingService:
    """Get an instance of the FileProcessingService."""
    return FileProcessingService(embedding_service=get_embedding_service())

@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Get an instance of the IngestionService."""
    return IngestionService(
        embedding_service=get_embedding_service(),
        text_store=get_text_store(),
        image_store=get_image_store(),
        mongo_client=get_mongo_client(),
        cache_manager=get_cache_manager(),
        semantic_cache=get_semantic_cache(),
        file_processing_service=get_file_processing_service(),
    )

@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Get an instance of the RetrievalService."""
    return RetrievalService(
        embedding_service=get_embedding_service(),
        text_store=get_text_store(),
        image_store=get_image_store(),
        cache_manager=get_cache_manager(),
    )
