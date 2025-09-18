"""
User session cache model for Redis storage.
"""
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class UserSession(BaseModel):
    """User session cache model for Redis storage."""
    
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    role: str = Field(..., description="User role")
    preferences: Optional[str] = Field(None, description="JSON encoded preferences")
    last_activity: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of last activity"
    )

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "john_doe",
                "role": "user",
                "preferences": '{"language": "en", "theme": "dark"}',
                "last_activity": "2023-01-01T00:00:00Z"
            }
        }

    @classmethod
    def get_key_pattern(cls, session_id: str) -> str:
        """Get Redis key pattern for session."""
        return f"session:{session_id}"

    @classmethod
    def get_ttl(cls) -> int:
        """Get TTL for session cache (1 hour)."""
        return 3600

    def to_redis_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "preferences": self.preferences or "",
            "last_activity": self.last_activity
        }

    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> "UserSession":
        """Create instance from Redis hash data."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            role=data["role"],
            preferences=data.get("preferences") or None,
            last_activity=data["last_activity"]
        )