"""
System metrics cache model for Redis storage.
"""
from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MetricsCache(BaseModel):
    """System metrics cache model for Redis storage."""
    
    value: str = Field(..., description="Metric value")
    unit: str = Field(..., description="Metric unit")
    tags: Optional[str] = Field(None, description="JSON encoded tags")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Metric timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "value": "0.85",
                "unit": "percentage",
                "tags": '{"service": "api", "endpoint": "/search"}',
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }

    @classmethod
    def get_key_pattern(cls, metric_name: str, timestamp: str) -> str:
        """Get Redis key pattern for metrics cache."""
        return f"metrics:{metric_name}:{timestamp}"

    @classmethod
    def get_ttl(cls) -> int:
        """Get TTL for metrics cache (7 days)."""
        return 604800

    def to_redis_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format."""
        return {
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags or "",
            "timestamp": self.timestamp
        }

    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> "MetricsCache":
        """Create instance from Redis hash data."""
        return cls(
            value=data["value"],
            unit=data["unit"],
            tags=data.get("tags") or None,
            timestamp=data["timestamp"]
        )