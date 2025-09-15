"""
Query cache model for Redis storage.
"""
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class QueryResult(BaseModel):
    """Individual query result."""
    
    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    relevance_score: float = Field(..., description="Relevance score")
    snippet: str = Field(..., description="Content snippet")


class QueryCache(BaseModel):
    """Query cache model for Redis storage."""
    
    query_text: str = Field(..., description="Original query text")
    results: List[QueryResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results")
    response_time: float = Field(..., description="Response time in seconds")
    cached_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Cache timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "query_text": "machine learning",
                "results": [
                    {
                        "document_id": "doc_123",
                        "title": "Introduction to ML",
                        "relevance_score": 0.95,
                        "snippet": "Machine learning is a subset..."
                    }
                ],
                "total_results": 10,
                "response_time": 0.25,
                "cached_at": "2023-01-01T00:00:00Z"
            }
        }

    @classmethod
    def get_key_pattern(cls, query_hash: str) -> str:
        """Get Redis key pattern for query cache."""
        return f"query:{query_hash}"

    @classmethod
    def get_ttl(cls) -> int:
        """Get TTL for query cache (30 minutes)."""
        return 1800

    def to_redis_string(self) -> str:
        """Convert to Redis string format (JSON)."""
        return self.json()

    @classmethod
    def from_redis_string(cls, data: str) -> "QueryCache":
        """Create instance from Redis string data."""
        return cls.parse_raw(data)