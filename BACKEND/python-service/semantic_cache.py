"""
Semantic Cache Manager for query-response caching
Uses FAISS for fast similarity search on query embeddings
"""
import logging
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class SemanticCache:
    """In-memory semantic cache using FAISS for similarity search"""
    
    def __init__(self, dimension: int = 384, max_size: int = 1000, threshold: float = 0.95):
        """
        Initialize semantic cache
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            max_size: Maximum number of cached entries
            threshold: Similarity threshold for cache hits (0.0-1.0)
        """
        self.dimension = dimension
        self.max_size = max_size
        self.threshold = threshold
        
        # FAISS index for cosine similarity (using inner product on normalized vectors)
        self.index: Optional[faiss.Index] = None
        
        # LRU tracking: OrderedDict maintains insertion order
        # Maps FAISS index position -> (query_text, response, file_id, timestamp)
        self.cache_data: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        
        # Track access order for LRU eviction
        self.access_order: OrderedDict[int, float] = OrderedDict()
        
        # Eviction counter for index rebuilding
        self._eviction_count = 0
        
        # Thread lock for concurrent access
        self._lock = Lock()
        
        self._ensure_index()
        logger.info(f"Semantic cache initialized: dimension={dimension}, max_size={max_size}, threshold={threshold}")
    
    def _ensure_index(self):
        """Create FAISS index if it doesn't exist"""
        if self.index is None:
            # IndexFlatIP for inner product (cosine similarity on normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.debug(f"Created FAISS IndexFlatIP with dimension {self.dimension}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector to unit length for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _rebuild_index(self):
        """Rebuild FAISS index to reclaim memory from evicted entries"""
        if not self.cache_data:
            self.index = None
            self._ensure_index()
            return
        
        old_count = self.index.ntotal if self.index else 0
        logger.info(f"Rebuilding FAISS index: {old_count} vectors -> {len(self.cache_data)} vectors")
        
        # Create new index
        new_index = faiss.IndexFlatIP(self.dimension)
        new_cache_data = OrderedDict()
        new_access_order = OrderedDict()
        
        # Re-add all valid entries with their embeddings
        new_idx = 0
        for old_idx, entry in self.cache_data.items():
            embedding = entry.get("embedding")
            if embedding is None:
                # Skip entries without embeddings (shouldn't happen, but be safe)
                continue
            
            # Normalize and add to new index
            embedding_array = np.array([embedding], dtype=np.float32)
            embedding_normalized = self._normalize_embedding(embedding_array[0])
            embedding_normalized = embedding_normalized.reshape(1, -1)
            new_index.add(embedding_normalized)
            
            # Update cache_data with new index
            new_cache_data[new_idx] = entry
            if old_idx in self.access_order:
                new_access_order[new_idx] = self.access_order[old_idx]
            
            new_idx += 1
        
        # Replace old index and cache data
        self.index = new_index
        self.cache_data = new_cache_data
        self.access_order = new_access_order
        
        logger.info(f"FAISS index rebuilt: {old_count} -> {new_idx} vectors")
    
    def _evict_lru(self):
        """Evict least recently used entry when cache is full"""
        if len(self.cache_data) < self.max_size:
            return
        
        # Find least recently accessed entry
        if not self.access_order:
            # Fallback: evict oldest by insertion order
            oldest_idx = next(iter(self.cache_data))
        else:
            oldest_idx = min(self.access_order.items(), key=lambda x: x[1])[0]
        
        # Remove from cache
        if oldest_idx in self.cache_data:
            del self.cache_data[oldest_idx]
        if oldest_idx in self.access_order:
            del self.access_order[oldest_idx]
        
        self._eviction_count += 1
        
        # Rebuild index if too many evictions (threshold: 50% of max_size)
        if self._eviction_count % max(1, self.max_size // 2) == 0:
            self._rebuild_index()
        
        logger.debug(f"Evicted cache entry at index {oldest_idx}")
    
    def search(self, query_embedding: List[float], file_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search for similar cached query
        
        Args:
            query_embedding: Query embedding vector (384D)
            file_id: Optional file ID to scope search
        
        Returns:
            Cached response dict with similarity score, or None if no match
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return None
            
            # Normalize query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            query_normalized = self._normalize_embedding(query_array[0])
            query_normalized = query_normalized.reshape(1, -1)
            
            # Search in FAISS (returns top 1 result)
            k = min(1, self.index.ntotal)
            similarities, indices = self.index.search(query_normalized, k)
            
            if len(indices[0]) == 0 or indices[0][0] == -1:
                return None
            
            # Get best match
            best_idx = int(indices[0][0])
            similarity = float(similarities[0][0])
            
            # Check threshold
            if similarity < self.threshold:
                return None
            
            # Check if entry still exists in cache_data (might have been evicted)
            if best_idx not in self.cache_data:
                return None
            
            cache_entry = self.cache_data[best_idx]
            
            # Strict file_id filtering to prevent data leaks
            if file_id is not None:
                cached_file_id = cache_entry.get("file_id")
                if cached_file_id is not None and cached_file_id != file_id:
                    return None
            
            # Update access order
            self.access_order[best_idx] = time.time()
            
            logger.debug(f"Cache HIT: similarity={similarity:.4f}, index={best_idx}")
            
            return {
                "response": cache_entry["response"],
                "similarity": similarity,
                "query": cache_entry.get("query", ""),
                "file_id": cache_entry.get("file_id"),
                "timestamp": cache_entry.get("timestamp")
            }
    
    def store(self, query_embedding: List[float], response: str, 
              query_text: str = "", file_id: Optional[str] = None) -> bool:
        """
        Store query-response pair in cache
        
        Args:
            query_embedding: Query embedding vector (384D)
            response: LLM response text
            query_text: Original query text (for debugging)
            file_id: Optional file ID
        
        Returns:
            True if stored successfully
        """
        try:
            with self._lock:
                # Evict if needed
                if len(self.cache_data) >= self.max_size:
                    self._evict_lru()
                
                # Normalize embedding
                embedding_array = np.array([query_embedding], dtype=np.float32)
                embedding_normalized = self._normalize_embedding(embedding_array[0])
                embedding_normalized = embedding_normalized.reshape(1, -1)
                
                # Add to FAISS index
                self._ensure_index()
                self.index.add(embedding_normalized)
                
                # Get index position (last added)
                idx = self.index.ntotal - 1
                
                # Store in cache_data (including embedding for rebuilding)
                self.cache_data[idx] = {
                    "query": query_text,
                    "response": response,
                    "file_id": file_id,
                    "timestamp": time.time(),
                    "embedding": query_embedding  # Store for index rebuilding
                }
                
                # Update access order
                self.access_order[idx] = time.time()
                
                logger.debug(f"Stored cache entry at index {idx}, total={len(self.cache_data)}")
                
                return True
            
        except Exception as e:
            logger.error(f"Failed to store in semantic cache: {e}")
            return False
    
    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self.index = None
            self.cache_data.clear()
            self.access_order.clear()
            self._eviction_count = 0
            logger.info("Semantic cache cleared")
    
    def invalidate_file(self, file_id: str) -> int:
        """
        Invalidate all cached entries for a specific file
        
        Args:
            file_id: File ID to invalidate
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            removed = 0
            indices_to_remove = []
            
            for idx, entry in list(self.cache_data.items()):
                if entry.get("file_id") == file_id:
                    indices_to_remove.append(idx)
            
            for idx in indices_to_remove:
                if idx in self.cache_data:
                    del self.cache_data[idx]
                    removed += 1
                if idx in self.access_order:
                    del self.access_order[idx]
            
            if removed > 0:
                logger.info(f"Invalidated {removed} semantic cache entries for file_id: {file_id}")
                # Consider rebuilding index if many entries removed (>20% of cache)
                if removed > self.max_size * 0.2:
                    self._rebuild_index()
            
            return removed

