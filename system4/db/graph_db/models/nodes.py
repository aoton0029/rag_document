"""
Neo4j node models for RAGShelf graph database.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr


class Document(BaseModel):
    """Document node for Neo4j graph database."""
    
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content_type: Optional[str] = Field(None, description="MIME type of document")
    language: Optional[str] = Field(None, description="Document language")
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Document source")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    embedding_count: Optional[int] = Field(0, ge=0, description="Number of embeddings")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc_123",
                "title": "Sample Document",
                "content_type": "text/plain",
                "language": "en",
                "author": "John Doe",
                "source": "example.com",
                "embedding_count": 5
            }
        }


class Concept(BaseModel):
    """Concept node for Neo4j graph database."""
    
    concept_id: str = Field(..., description="Unique concept identifier")
    name: str = Field(..., description="Concept name")
    type: str = Field(
        ...,
        pattern="^(entity|topic|keyword|category)$",
        description="Type of concept"
    )
    definition: Optional[str] = Field(None, description="Concept definition")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence"
    )
    frequency: Optional[int] = Field(1, ge=1, description="Occurrence frequency")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "concept_id": "concept_123",
                "name": "Machine Learning",
                "type": "topic",
                "definition": "A subset of AI that involves algorithms learning from data",
                "confidence_score": 0.95,
                "frequency": 10
            }
        }


class UserPreferences(BaseModel):
    """User preferences sub-model."""
    
    language: Optional[str] = Field(None, description="Preferred language")
    domains: Optional[List[str]] = Field(default_factory=list, description="Preferred domains")


class User(BaseModel):
    """User node for Neo4j graph database."""
    
    user_id: str = Field(..., description="Unique user identifier")
    username: Optional[str] = Field(None, description="User display name")
    email: Optional[EmailStr] = Field(None, description="User email address")
    preferences: Optional[UserPreferences] = Field(None, description="User preferences")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "john_doe",
                "email": "john@example.com",
                "preferences": {
                    "language": "en",
                    "domains": ["technology", "science"]
                }
            }
        }


class Query(BaseModel):
    """Query node for Neo4j graph database."""
    
    query_id: str = Field(..., description="Unique query identifier")
    text: str = Field(..., description="Query text")
    normalized_text: Optional[str] = Field(None, description="Normalized query text")
    intent: str = Field(
        ...,
        pattern="^(search|question|analysis|comparison)$",
        description="Query intent"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")
    response_time: Optional[float] = Field(None, description="Response time in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "query_id": "query_123",
                "text": "What is machine learning?",
                "normalized_text": "machine learning definition",
                "intent": "question",
                "response_time": 150.5
            }
        }