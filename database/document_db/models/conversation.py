"""
Conversation model for MongoDB storage.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId class for Pydantic compatibility."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class MessageMetadata(BaseModel):
    """Metadata for conversation messages."""
    
    model_used: Optional[str] = Field(None, description="AI model used for response")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    sources_used: Optional[List[str]] = Field(
        default_factory=list,
        description="Document sources used in response"
    )


class Message(BaseModel):
    """Individual message within a conversation."""
    
    message_id: str = Field(..., description="Unique message identifier")
    role: str = Field(
        ...,
        pattern="^(user|assistant|system)$",
        description="Message role"
    )
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: Optional[MessageMetadata] = Field(None, description="Message metadata")

    class Config:
        schema_extra = {
            "example": {
                "message_id": "msg_123",
                "role": "user",
                "content": "What is the capital of France?",
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }


class Conversation(BaseModel):
    """Conversation model for MongoDB storage."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    title: Optional[str] = Field(None, description="Conversation title")
    messages: List[Message] = Field(default_factory=list, description="List of messages")
    status: str = Field(
        "active",
        pattern="^(active|completed|archived)$",
        description="Conversation status"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "user_id": "user_123",
                "title": "Q&A Session",
                "status": "active",
                "messages": [
                    {
                        "message_id": "msg_123",
                        "role": "user",
                        "content": "Hello",
                        "timestamp": "2023-01-01T00:00:00Z"
                    }
                ],
                "created_at": "2023-01-01T00:00:00Z"
            }
        }