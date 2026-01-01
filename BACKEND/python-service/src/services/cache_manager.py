"""
Cache management services, including Redis and in-memory semantic cache.
"""
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Optional, Dict, Any, List
import numpy as np
import redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError
import faiss

from ..core.config import settings

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """Manages Redis cache operations for query results and FAISS search results"""

    def __init__(self):
        """Initialize Redis connection with retry logic"""
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = settings.ENABLE_CACHE

        if not self.enabled:
            logger.info("Redis caching is disabled via ENABLE_CACHE flag")
            return

        self._connect()

    def _connect(self):
        """Establish Redis connection with configuration from environment"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,  # We'll handle encoding ourselves for binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}")
        except (ConnectionError, TimeoutError, RedisError) as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
            self.redis_client = None
            self.enabled = False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            self.redis_client = None
            self.enabled = False

    def _is_connected(self) -> bool:
        """Check if Redis is connected and enabled"""
        if not self.enabled or self.redis_client is None:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False

    def _hash_text(self, text: str) -> str:
        """Generate hash for text (question)"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """Generate hash for embedding vector"""
        embedding_bytes = embedding.tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()[:32]

    def _generate_query_cache_key(self, question: str, file_id: Optional[str], cache_type: str) -> str:
        """Generate cache key for query results"""
        question_hash = self._hash_text(question)
        file_part = file_id if file_id else "all"
        return f"qr:{cache_type}:{question_hash}:{file_part}"

    def _generate_faiss_cache_key(self, embedding_hash: str, file_id: Optional[str],
                                   k: int, index_type: str) -> str:
        """Generate cache key for FAISS search results"""
        file_part = file_id if file_id else "all"
        return f"faiss:{index_type}:{embedding_hash}:{file_part}:{k}"

    def get_query_cache(self, question: str, file_id: Optional[str], cache_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached query results"""
        if not self._is_connected():
            return None
        try:
            cache_key = self._generate_query_cache_key(question, file_id, cache_type)
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache HIT for query: {cache_key}")
                return json.loads(cached_data.decode('utf-8'))
            logger.debug(f"Cache MISS for query: {cache_key}")
            return None
        except Exception as e:
            logger.warning(f"Error retrieving query cache: {e}")
            return None

    def set_query_cache(self, question: str, file_id: Optional[str], cache_type: str,
                       data: Dict[str, Any], ttl: int = None) -> bool:
        """Store query results in cache"""
        if not self._is_connected():
            return False
        try:
            if ttl is None:
                ttl = settings.CACHE_TTL_QUERY_RESULTS
            cache_key = self._generate_query_cache_key(question, file_id, cache_type)
            data_json = json.dumps(data)
            self.redis_client.setex(cache_key, ttl, data_json.encode('utf-8'))
            logger.debug(f"Cached query results: {cache_key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Error setting query cache: {e}")
            return False

    def get_faiss_cache(self, embedding: np.ndarray, file_id: Optional[str],
                       k: int, index_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached FAISS search results"""
        if not self._is_connected():
            return None
        try:
            embedding_hash = self._hash_embedding(embedding)
            cache_key = self._generate_faiss_cache_key(embedding_hash, file_id, k, index_type)
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache HIT for FAISS: {cache_key}")
                return json.loads(cached_data.decode('utf-8'))
            logger.debug(f"Cache MISS for FAISS: {cache_key}")
            return None
        except Exception as e:
            logger.warning(f"Error retrieving FAISS cache: {e}")
            return None

    def set_faiss_cache(self, embedding: np.ndarray, file_id: Optional[str],
                       k: int, index_type: str, results: Dict[str, Any],
                       ttl: int = None) -> bool:
        """Store FAISS search results in cache"""
        if not self._is_connected():
            return False
        try:
            if ttl is None:
                ttl = settings.CACHE_TTL_FAISS_RESULTS
            embedding_hash = self._hash_embedding(embedding)
            cache_key = self._generate_faiss_cache_key(embedding_hash, file_id, k, index_type)
            data_json = json.dumps(results)
            self.redis_client.setex(cache_key, ttl, data_json.encode('utf-8'))
            logger.debug(f"Cached FAISS results: {cache_key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Error setting FAISS cache: {e}")
            return False

    def invalidate_file_cache(self, file_id: str) -> int:
        """Invalidate all cache entries for a specific file"""
        if not self._is_connected():
            return 0
        try:
            all_keys = []
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor, count=100)
                if keys:
                    for key in keys:
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        if (f":{file_id}:" in key_str or
                            key_str.endswith(f":{file_id}") or
                            ":all:" in key_str or
                            key_str.endswith(":all")):
                            all_keys.append(key)
                if cursor == 0:
                    break

            deleted_count = 0
            if all_keys:
                batch_size = 100
                for i in range(0, len(all_keys), batch_size):
                    batch = all_keys[i:i + batch_size]
                    deleted = self.redis_client.delete(*batch)
                    deleted_count += deleted
                logger.info(f"Invalidated {deleted_count} cache entries for file_id: {file_id} (including global queries)")
            return deleted_count
        except Exception as e:
            logger.warning(f"Error invalidating file cache: {e}")
            return 0

    def invalidate_all_cache(self) -> bool:
        """Invalidate entire cache (use with caution)"""
        if not self._is_connected():
            return False
        try:
            self.redis_client.flushdb()
            logger.warning("Flushed entire Redis cache database")
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False


class SemanticCache:
    """In-memory semantic cache using FAISS for similarity search"""

    def __init__(self, dimension: int = 384, max_size: int = 1000, threshold: float = 0.95):
        """Initialize semantic cache"""
        self.dimension = dimension
        self.max_size = max_size
        self.threshold = threshold
        self.index: Optional[faiss.Index] = None
        self.cache_data: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self.access_order: OrderedDict[int, float] = OrderedDict()
        self._eviction_count = 0
        self._lock = Lock()
        self._ensure_index()
        logger.info(f"Semantic cache initialized: dimension={dimension}, max_size={max_size}, threshold={threshold}")

    def _ensure_index(self):
        """Create FAISS index if it doesn't exist"""
        if self.index is None:
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

        new_index = faiss.IndexFlatIP(self.dimension)
        new_cache_data = OrderedDict()
        new_access_order = OrderedDict()
        new_idx = 0

        for old_idx, entry in self.cache_data.items():
            embedding = entry.get("embedding")
            if embedding is None:
                continue

            embedding_array = np.array([embedding], dtype=np.float32)
            embedding_normalized = self._normalize_embedding(embedding_array[0]).reshape(1, -1)
            new_index.add(embedding_normalized)

            new_cache_data[new_idx] = entry
            if old_idx in self.access_order:
                new_access_order[new_idx] = self.access_order[old_idx]
            new_idx += 1

        self.index = new_index
        self.cache_data = new_cache_data
        self.access_order = new_access_order
        logger.info(f"FAISS index rebuilt: {old_count} -> {new_idx} vectors")

    def _evict_lru(self):
        """Evict least recently used entry when cache is full"""
        if len(self.cache_data) < self.max_size:
            return

        if not self.access_order:
            oldest_idx = next(iter(self.cache_data))
        else:
            oldest_idx = min(self.access_order.items(), key=lambda x: x[1])[0]

        if oldest_idx in self.cache_data:
            del self.cache_data[oldest_idx]
        if oldest_idx in self.access_order:
            del self.access_order[oldest_idx]
        self._eviction_count += 1

        if self._eviction_count % max(1, self.max_size // 2) == 0:
            self._rebuild_index()
        logger.debug(f"Evicted cache entry at index {oldest_idx}")

    def search(self, query_embedding: List[float], file_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Search for similar cached query"""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return None

            query_array = np.array([query_embedding], dtype=np.float32)
            query_normalized = self._normalize_embedding(query_array[0]).reshape(1, -1)

            k = min(1, self.index.ntotal)
            similarities, indices = self.index.search(query_normalized, k)

            if len(indices[0]) == 0 or indices[0][0] == -1:
                return None

            best_idx = int(indices[0][0])
            similarity = float(similarities[0][0])

            if similarity < self.threshold:
                return None

            if best_idx not in self.cache_data:
                return None

            cache_entry = self.cache_data[best_idx]
            if file_id is not None:
                cached_file_id = cache_entry.get("file_id")
                if cached_file_id is not None and cached_file_id != file_id:
                    return None

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
        """Store query-response pair in cache"""
        try:
            with self._lock:
                if len(self.cache_data) >= self.max_size:
                    self._evict_lru()

                embedding_array = np.array([query_embedding], dtype=np.float32)
                embedding_normalized = self._normalize_embedding(embedding_array[0]).reshape(1, -1)

                self._ensure_index()
                self.index.add(embedding_normalized)
                idx = self.index.ntotal - 1

                self.cache_data[idx] = {
                    "query": query_text,
                    "response": response,
                    "file_id": file_id,
                    "timestamp": time.time(),
                    "embedding": query_embedding
                }
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
        """Invalidate all cached entries for a specific file"""
        with self._lock:
            removed = 0
            indices_to_remove = [
                idx for idx, entry in self.cache_data.items() if entry.get("file_id") == file_id
            ]

            for idx in indices_to_remove:
                if idx in self.cache_data:
                    del self.cache_data[idx]
                    removed += 1
                if idx in self.access_order:
                    del self.access_order[idx]

            if removed > 0:
                logger.info(f"Invalidated {removed} semantic cache entries for file_id: {file_id}")
                if removed > self.max_size * 0.2:
                    self._rebuild_index()
            return removed
