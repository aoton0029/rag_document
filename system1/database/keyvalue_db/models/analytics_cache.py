"""
Analytics cache model for Redis storage.
"""
import json
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class PopularQuery(BaseModel):
    """Popular query item."""
    
    query: str = Field(..., description="Query text")
    score: float = Field(..., description="Popularity score")
    count: int = Field(..., description="Query frequency")


class AnalyticsCache(BaseModel):
    """Analytics cache model for Redis storage."""
    
    period: str = Field(..., description="Analytics period (daily, weekly, monthly)")
    popular_queries: List[PopularQuery] = Field(
        default_factory=list,
        description="Popular queries list"
    )
    total_queries: int = Field(0, description="Total number of queries")
    unique_users: int = Field(0, description="Number of unique users")
    avg_response_time: float = Field(0.0, description="Average response time")
    updated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Last update timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "period": "daily",
                "popular_queries": [
                    {
                        "query": "machine learning",
                        "score": 0.95,
                        "count": 150
                    }
                ],
                "total_queries": 1000,
                "unique_users": 50,
                "avg_response_time": 0.25,
                "updated_at": "2023-01-01T00:00:00Z"
            }
        }

    @classmethod
    def get_key_pattern(cls, period: str) -> str:
        """Get Redis key pattern for analytics cache."""
        return f"analytics:{period}"

    @classmethod
    def get_ttl(cls) -> int:
        """Get TTL for analytics cache (6 hours)."""
        return 21600

    def to_redis_string(self) -> str:
        """Convert to Redis string format (JSON)."""
        return self.json()

    @classmethod
    def from_redis_string(cls, data: str) -> "AnalyticsCache":
        """Create instance from Redis string data."""
        return cls.parse_raw(data)