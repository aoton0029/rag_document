"""
Embedding cache model for Redis storage.
"""
import json
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EmbeddingCache(BaseModel):
    """Embedding cache model for Redis storage."""
    
    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used for embedding")
    content_hash: str = Field(..., description="Hash of original content")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Cache creation timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "content_hash": "abc123def456...",
                "created_at": "2023-01-01T00:00:00Z"
            }
        }

    @classmethod
    def get_key_pattern(cls, document_id: str, chunk_id: str) -> str:
        """Get Redis key pattern for embedding cache."""
        return f"embedding:{document_id}:{chunk_id}"

    @classmethod
    def get_ttl(cls) -> int:
        """Get TTL for embedding cache (24 hours)."""
        return 86400

    def to_redis_string(self) -> str:
        """Convert to Redis string format (JSON)."""
        return self.json()

    @classmethod
    def from_redis_string(cls, data: str) -> "EmbeddingCache":
        """Create instance from Redis string data."""
        return cls.parse_raw(data)