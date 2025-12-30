"""
Redis Cache Manager for distributed caching
Handles query results cache and FAISS search results cache
"""
import hashlib
import json
import logging
import os
from typing import Optional, Dict, Any, List
import numpy as np
import redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """Manages Redis cache operations for query results and FAISS search results"""
    
    def __init__(self):
        """Initialize Redis connection with retry logic"""
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true" or \
                      os.getenv("ENABLE_FAISS_CACHE", "true").lower() == "true"
        
        if not self.enabled:
            logger.info("Redis caching is disabled via environment variables")
            return
        
        self._connect()
    
    def _connect(self):
        """Establish Redis connection with configuration from environment"""
        try:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            password = os.getenv("REDIS_PASSWORD", None)
            
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # We'll handle encoding ourselves for binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}/{db}")
            
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
        # Convert to bytes and hash
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
        """
        Retrieve cached query results
        
        Args:
            question: User question
            file_id: Optional file ID
            cache_type: "text" or "images"
        
        Returns:
            Cached data or None if not found
        """
        if not self._is_connected():
            return None
        
        try:
            cache_key = self._generate_query_cache_key(question, file_id, cache_type)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                logger.debug(f"Cache HIT for query: {cache_key}")
                return json.loads(cached_data.decode('utf-8'))
            else:
                logger.debug(f"Cache MISS for query: {cache_key}")
                return None
                
        except Exception as e:
            logger.warning(f"Error retrieving query cache: {e}")
            return None
    
    def set_query_cache(self, question: str, file_id: Optional[str], cache_type: str, 
                       data: Dict[str, Any], ttl: int = None) -> bool:
        """
        Store query results in cache
        
        Args:
            question: User question
            file_id: Optional file ID
            cache_type: "text" or "images"
            data: Data to cache
            ttl: Time to live in seconds (default from env)
        
        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            if ttl is None:
                ttl = int(os.getenv("CACHE_TTL_QUERY_RESULTS", "3600"))  # Default 1 hour
            
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
        """
        Retrieve cached FAISS search results
        
        Args:
            embedding: Query embedding vector
            file_id: Optional file ID
            k: Number of results requested
            index_type: "text" or "img"
        
        Returns:
            Cached FAISS results or None if not found
        """
        if not self._is_connected():
            return None
        
        try:
            embedding_hash = self._hash_embedding(embedding)
            cache_key = self._generate_faiss_cache_key(embedding_hash, file_id, k, index_type)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                logger.debug(f"Cache HIT for FAISS: {cache_key}")
                return json.loads(cached_data.decode('utf-8'))
            else:
                logger.debug(f"Cache MISS for FAISS: {cache_key}")
                return None
                
        except Exception as e:
            logger.warning(f"Error retrieving FAISS cache: {e}")
            return None
    
    def set_faiss_cache(self, embedding: np.ndarray, file_id: Optional[str], 
                       k: int, index_type: str, results: Dict[str, Any], 
                       ttl: int = None) -> bool:
        """
        Store FAISS search results in cache
        
        Args:
            embedding: Query embedding vector
            file_id: Optional file ID
            k: Number of results
            index_type: "text" or "img"
            results: FAISS results to cache
            ttl: Time to live in seconds (default from env)
        
        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            if ttl is None:
                ttl = int(os.getenv("CACHE_TTL_FAISS_RESULTS", "7200"))  # Default 2 hours
            
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
        """
        Invalidate all cache entries for a specific file
        
        Args:
            file_id: File ID to invalidate
        
        Returns:
            Number of keys deleted
        """
        if not self._is_connected():
            return 0
        
        try:
            # Scan all keys and filter for those containing the file_id
            # This is more reliable than pattern matching with wildcards
            all_keys = []
            cursor = 0
            
            while True:
                cursor, keys = self.redis_client.scan(cursor, count=100)
                if keys:
                    # Decode keys if needed and filter for file_id
                    for key in keys:
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        # Check if key contains the file_id in the expected positions
                        # Query results: qr:text:hash:file_id or qr:images:hash:file_id
                        # FAISS results: faiss:text:hash:file_id:k or faiss:img:hash:file_id:k
                        # Also invalidate global queries (containing :all: or ending with :all)
                        if (f":{file_id}:" in key_str or 
                            key_str.endswith(f":{file_id}") or
                            ":all:" in key_str or 
                            key_str.endswith(":all")):
                            all_keys.append(key)
                
                if cursor == 0:
                    break
            
            # Delete all collected keys in batches (Redis DELETE can handle multiple keys)
            deleted_count = 0
            if all_keys:
                # Delete in batches of 100 to avoid overwhelming Redis
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
        """
        Invalidate entire cache (use with caution)
        
        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("Flushed entire Redis cache database")
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    
