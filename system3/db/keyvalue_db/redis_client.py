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

    
    def get_all_keys(self) -> List[str]:
        """Get all keys in the database."""
        try:
            keys = self.client.keys("*")
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            print(f"Error getting all keys: {e}")
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