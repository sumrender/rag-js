"""
Service for retrieving text and images from the vector stores.
"""
import logging
from typing import List, Dict, Any, Optional

import numpy as np

from ..core.config import settings
from .embedding_service import EmbeddingService
from .vector_store import FAISSStore
from .cache_manager import RedisCacheManager

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Handles the logic for retrieving text chunks and images from the FAISS stores.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        text_store: FAISSStore,
        image_store: FAISSStore,
        cache_manager: Optional[RedisCacheManager] = None,
    ):
        self.embedding_service = embedding_service
        self.text_store = text_store
        self.image_store = image_store
        self.cache_manager = cache_manager

    def retrieve_text(self, question: str, file_id: Optional[str] = None, k: int = 3) -> Dict[str, Any]:
        """Retrieve text chunks by query."""
        if self.cache_manager:
            cached_result = self.cache_manager.get_query_cache(question, file_id, "text")
            if cached_result:
                logger.info(f"Cache HIT for query results: '{question[:50]}...'")
                return cached_result

        embedding_tuple = self.embedding_service.get_text_embedding(question)
        query_embedding = np.array(embedding_tuple, dtype=np.float32)

        if self.cache_manager:
            faiss_results = self.cache_manager.get_faiss_cache(query_embedding, file_id, k, "text")
            if faiss_results:
                logger.info(f"Cache HIT for FAISS results: '{question[:50]}...'")
        else:
            faiss_results = None

        if faiss_results is None:
            where_filter = {"fileId": {"$eq": file_id}} if file_id else None
            faiss_results = self.text_store.query(
                query_embeddings=[query_embedding.tolist()], n_results=k, where=where_filter
            )
            if self.cache_manager:
                self.cache_manager.set_faiss_cache(query_embedding, file_id, k, "text", faiss_results)

        chunks = self._format_text_results(faiss_results)
        result = {"chunks": chunks, "queryModel": settings.TEXT_MODEL_NAME}

        if self.cache_manager:
            self.cache_manager.set_query_cache(question, file_id, "text", result)

        return result

    def retrieve_images_by_text(self, question: str, file_id: Optional[str] = None, k: int = 3) -> Dict[str, Any]:
        """Retrieve images by text query."""
        if self.cache_manager:
            cached_result = self.cache_manager.get_query_cache(question, file_id, "images")
            if cached_result:
                logger.info(f"Cache HIT for image query results: '{question[:50]}...'")
                return cached_result

        embedding_tuple = self.embedding_service.get_clip_embedding(question)
        query_embedding = np.array(embedding_tuple, dtype=np.float32)

        if self.cache_manager:
            faiss_results = self.cache_manager.get_faiss_cache(query_embedding, file_id, k, "img")
            if faiss_results:
                logger.info(f"Cache HIT for image FAISS results: '{question[:50]}...'")
        else:
            faiss_results = None

        if faiss_results is None:
            where_filter = {"fileId": {"$eq": file_id}} if file_id else None
            faiss_results = self.image_store.query(
                query_embeddings=[query_embedding.tolist()], n_results=k, where=where_filter
            )
            if self.cache_manager:
                self.cache_manager.set_faiss_cache(query_embedding, file_id, k, "img", faiss_results)

        images = self._format_image_results(faiss_results)
        result = {"images": images}

        if self.cache_manager:
            self.cache_manager.set_query_cache(question, file_id, "images", result)

        return result

    def retrieve_images_by_pages(self, file_id: Optional[str] = None, page_numbers: List[int] = None, max_per_page: int = 3) -> Dict[str, Any]:
        """Retrieve images by page numbers."""
        if not page_numbers:
            return {"images": []}

        # This is inefficient, but it's the simplest way to implement this without a more advanced backend.
        # For a production system, you would want a database that can filter by metadata more efficiently.
        all_images = self.image_store.query(
            query_embeddings=[[0.0] * self.image_store.dimension],
            n_results=min(self.image_store.index.ntotal if self.image_store.index else 0, 10000),
            where={"fileId": {"$eq": file_id}} if file_id else None
        )

        images = self._format_image_results(all_images)

        # Filter by page number and apply max_per_page
        result_images = []
        page_counts = {p: 0 for p in page_numbers}
        for img in images:
            if img.get("pageNumber") in page_numbers and page_counts[img["pageNumber"]] < max_per_page:
                result_images.append(img)
                page_counts[img["pageNumber"]] += 1

        return {"images": result_images}

    def _format_text_results(self, faiss_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format FAISS results for text chunks."""
        chunks = []
        if not faiss_results.get("documents") or not faiss_results["documents"][0]:
            return chunks

        for i, doc in enumerate(faiss_results["documents"][0]):
            distance = faiss_results["distances"][0][i]
            cosine_similarity = 1.0 - (distance / 2.0)
            chunks.append({
                "id": faiss_results["ids"][0][i],
                "text": doc,
                "metadata": faiss_results["metadatas"][0][i],
                "score": cosine_similarity,
            })
        return chunks

    def _format_image_results(self, faiss_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format FAISS results for images."""
        images = []
        if not faiss_results.get("documents") or not faiss_results["documents"][0]:
            return images

        for i, doc in enumerate(faiss_results["documents"][0]):
            metadata = faiss_results["metadatas"][0][i]
            distance = faiss_results["distances"][0][i]
            cosine_similarity = 1.0 - (distance / 2.0)
            images.append({
                "imageId": metadata.get("imageId", faiss_results["ids"][0][i]),
                "fileId": metadata.get("fileId"),
                "pageNumber": metadata.get("pageNumber"),
                "imageUrl": metadata.get("imageUrl"),
                "score": cosine_similarity,
                "nearbyText": metadata.get("nearbyText"),
            })
        return images
