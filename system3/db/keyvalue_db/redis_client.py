"""
Redis client for RAGShelf caching operations.
"""
import redis
import json
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Union
from .models import UserSession, QueryCache, EmbeddingCache, AnalyticsCache, MetricsCache


class RedisClient:
    """Redis client for key-value store operations with model support."""
    
    def __init__(self, host: str, port: int, db: int = 0, password: Optional[str] = None):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Support for binary data
        )
        
    # Session operations
    def set_user_session(self, session_id: str, session: UserSession) -> bool:
        """Set user session using UserSession model."""
        try:
            key = UserSession.get_key_pattern(session_id)
            ttl = UserSession.get_ttl()
            
            # Use Redis hash for session data
            hash_data = session.to_redis_hash()
            pipe = self.client.pipeline()
            pipe.hmset(key, hash_data)
            pipe.expire(key, ttl)
            pipe.execute()
            return True
        except Exception as e:
            print(f"Error setting user session: {e}")
            return False
    
    def get_user_session(self, session_id: str) -> Optional[UserSession]:
        """Get user session as UserSession model."""
        try:
            key = UserSession.get_key_pattern(session_id)
            hash_data = self.client.hgetall(key)
            
            if hash_data:
                # Convert bytes to strings
                decoded_data = {k.decode(): v.decode() for k, v in hash_data.items()}
                return UserSession.from_redis_hash(decoded_data)
            return None
        except Exception as e:
            print(f"Error getting user session: {e}")
            return None
    
    def delete_user_session(self, session_id: str) -> bool:
        """Delete user session."""
        try:
            key = UserSession.get_key_pattern(session_id)
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Error deleting user session: {e}")
            return False
    
    # Query cache operations
    def set_query_cache(self, query_text: str, cache: QueryCache) -> bool:
        """Set query cache using QueryCache model."""
        try:
            query_hash = hashlib.sha256(query_text.encode()).hexdigest()
            key = QueryCache.get_key_pattern(query_hash)
            ttl = QueryCache.get_ttl()
            
            cache_data = cache.to_redis_string()
            return self.client.setex(key, ttl, cache_data)
        except Exception as e:
            print(f"Error setting query cache: {e}")
            return False
    
    def get_query_cache(self, query_text: str) -> Optional[QueryCache]:
        """Get query cache as QueryCache model."""
        try:
            query_hash = hashlib.sha256(query_text.encode()).hexdigest()
            key = QueryCache.get_key_pattern(query_hash)
            
            cached_data = self.client.get(key)
            if cached_data:
                return QueryCache.from_redis_string(cached_data.decode())
            return None
        except Exception as e:
            print(f"Error getting query cache: {e}")
            return None
    
    # Embedding cache operations  
    def set_embedding_cache(self, document_id: str, chunk_id: str, cache: EmbeddingCache) -> bool:
        """Set embedding cache using EmbeddingCache model."""
        try:
            key = EmbeddingCache.get_key_pattern(document_id, chunk_id)
            ttl = EmbeddingCache.get_ttl()
            
            cache_data = cache.to_redis_string()
            return self.client.setex(key, ttl, cache_data)
        except Exception as e:
            print(f"Error setting embedding cache: {e}")
            return False
    
    def get_embedding_cache(self, document_id: str, chunk_id: str) -> Optional[EmbeddingCache]:
        """Get embedding cache as EmbeddingCache model."""
        try:
            key = EmbeddingCache.get_key_pattern(document_id, chunk_id)
            
            cached_data = self.client.get(key)
            if cached_data:
                return EmbeddingCache.from_redis_string(cached_data.decode())
            return None
        except Exception as e:
            print(f"Error getting embedding cache: {e}")
            return None
    
    def delete_document_embeddings(self, document_id: str) -> int:
        """Delete all embedding caches for a document."""
        try:
            pattern = f"embedding:{document_id}:*"
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Error deleting document embeddings: {e}")
            return 0
    
    # Analytics cache operations
    def set_analytics_cache(self, period: str, analytics: AnalyticsCache) -> bool:
        """Set analytics cache using AnalyticsCache model."""
        try:
            key = AnalyticsCache.get_key_pattern(period)
            ttl = AnalyticsCache.get_ttl()
            
            cache_data = analytics.to_redis_string()
            return self.client.setex(key, ttl, cache_data)
        except Exception as e:
            print(f"Error setting analytics cache: {e}")
            return False
    
    def get_analytics_cache(self, period: str) -> Optional[AnalyticsCache]:
        """Get analytics cache as AnalyticsCache model."""
        try:
            key = AnalyticsCache.get_key_pattern(period)
            
            cached_data = self.client.get(key)
            if cached_data:
                return AnalyticsCache.from_redis_string(cached_data.decode())
            return None
        except Exception as e:
            print(f"Error getting analytics cache: {e}")
            return None
    
    # Metrics cache operations
    def set_metrics_cache(self, metric_name: str, timestamp: str, metrics: MetricsCache) -> bool:
        """Set metrics cache using MetricsCache model."""
        try:
            key = MetricsCache.get_key_pattern(metric_name, timestamp)
            ttl = MetricsCache.get_ttl()
            
            # Use Redis hash for metrics data
            hash_data = metrics.to_redis_hash()
            pipe = self.client.pipeline()
            pipe.hmset(key, hash_data)
            pipe.expire(key, ttl)
            pipe.execute()
            return True
        except Exception as e:
            print(f"Error setting metrics cache: {e}")
            return False
    
    def get_metrics_cache(self, metric_name: str, timestamp: str) -> Optional[MetricsCache]:
        """Get metrics cache as MetricsCache model."""
        try:
            key = MetricsCache.get_key_pattern(metric_name, timestamp)
            hash_data = self.client.hgetall(key)
            
            if hash_data:
                # Convert bytes to strings
                decoded_data = {k.decode(): v.decode() for k, v in hash_data.items()}
                return MetricsCache.from_redis_hash(decoded_data)
            return None
        except Exception as e:
            print(f"Error getting metrics cache: {e}")
            return None
    
    def get_metrics_by_pattern(self, metric_name: str, limit: int = 100) -> List[MetricsCache]:
        """Get metrics by pattern."""
        try:
            pattern = f"metrics:{metric_name}:*"
            keys = self.client.keys(pattern)[:limit]
            
            metrics = []
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                timestamp = key_str.split(':')[-1]
                metric = self.get_metrics_cache(metric_name, timestamp)
                if metric:
                    metrics.append(metric)
            
            return metrics
        except Exception as e:
            print(f"Error getting metrics by pattern: {e}")
            return []
        
    # Legacy cache operations (for backward compatibility)
    def set_cache(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """Set cache data using pickle serialization."""
        try:
            serialized_value = pickle.dumps(value)
            return self.client.set(key, serialized_value, ex=expire_seconds)
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cache data using pickle deserialization."""
        try:
            cached_data = self.client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
            return None
        except Exception as e:
            print(f"Error getting cache: {e}")
            return None
    
    def delete_cache(self, key: str) -> bool:
        """Delete cache entry."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Error deleting cache: {e}")
            return False
    
    # Processing queue operations
    def add_to_processing_queue(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """Add task to processing queue."""
        try:
            queue_key = "processing_queue"
            task_item = {
                "task_id": task_id,
                "data": task_data,
                "created_at": str(self.client.time()[0])
            }
            return bool(self.client.lpush(queue_key, pickle.dumps(task_item)))
        except Exception as e:
            print(f"Error adding to queue: {e}")
            return False
    
    def get_from_processing_queue(self, timeout: int = 1) -> Optional[Dict[str, Any]]:
        """Get task from processing queue."""
        try:
            queue_key = "processing_queue"
            task_data = self.client.brpop(queue_key, timeout=timeout)
            if task_data:
                return pickle.loads(task_data[1])
            return None
        except Exception as e:
            print(f"Error getting from queue: {e}")
            return None
    
    def get_queue_length(self) -> int:
        """Get processing queue length."""
        try:
            return self.client.llen("processing_queue")
        except Exception as e:
            print(f"Error getting queue length: {e}")
            return 0
    
    # Utility operations
    def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        try:
            keys = self.client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            print(f"Error getting keys by pattern: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics by key patterns."""
        patterns = {
            "sessions": "session:*",
            "queries": "query:*", 
            "embeddings": "embedding:*",
            "analytics": "analytics:*",
            "metrics": "metrics:*"
        }
        
        stats = {}
        try:
            info = self.client.info()
            stats["redis_info"] = {
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed")
            }
            
            for cache_type, pattern in patterns.items():
                keys = self.client.keys(pattern)
                stats[cache_type] = len(keys)
                
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            
        return stats
    
    def expire_key(self, key: str, seconds: int) -> bool:
        """Set expiration for a key."""
        try:
            return bool(self.client.expire(key, seconds))
        except Exception as e:
            print(f"Error setting expiration: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get time to live for a key."""
        try:
            return self.client.ttl(key)
        except Exception as e:
            print(f"Error getting TTL: {e}")
            return -1
    
    def flush_all(self) -> bool:
        """Delete all data (use with caution)."""
        try:
            return self.client.flushall()
        except Exception as e:
            print(f"Error flushing all data: {e}")
            return False
    
    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            return self.client.ping()
        except Exception as e:
            print(f"Error pinging Redis: {e}")
            return False
    
    def close(self):
        """Close the Redis connection."""
        try:
            self.client.close()
        except Exception as e:
            print(f"Error closing connection: {e}")