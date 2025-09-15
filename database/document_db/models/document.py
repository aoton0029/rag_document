"""
Document model for MongoDB storage.
"""
from datetime import datetime
from typing import List, Optional
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


class Document(BaseModel):
    """Document model for MongoDB storage."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., max_length=500, description="Document title")
    content: str = Field(..., description="Full document content")
    content_type: str = Field(
        ..., 
        pattern="^(text/plain|text/markdown|text/html|application/pdf|application/json)$",
        description="MIME type of the document"
    )
    language: Optional[str] = Field(
        None,
        pattern="^[a-z]{2}(-[A-Z]{2})?$",
        description="Document language code (ISO 639-1)"
    )
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Document source or origin")
    url: Optional[str] = Field(None, description="Original URL if applicable")
    file_path: Optional[str] = Field(None, description="Local file path if applicable")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    checksum: Optional[str] = Field(
        None,
        pattern="^[a-f0-9]{64}$",
        description="SHA-256 checksum of content"
    )
    meta_keywords: Optional[List[str]] = Field(
        default_factory=list,
        description="Document keywords for search"
    )
    meta_tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Document tags"
    )
    meta_category: Optional[str] = Field(None, description="Document category")
    processing_status: str = Field(
        "pending",
        pattern="^(pending|processing|completed|failed)$",
        description="Document processing status"
    )
    chunk_count: Optional[int] = Field(
        0, ge=0, description="Number of chunks created from this document"
    )
    embedding_model: Optional[str] = Field(None, description="Model used for embeddings")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    indexed_at: Optional[datetime] = Field(None, description="Last indexing timestamp")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "document_id": "doc_123",
                "title": "Sample Document",
                "content": "This is a sample document content.",
                "content_type": "text/plain",
                "language": "en",
                "author": "John Doe",
                "processing_status": "completed",
                "created_at": "2023-01-01T00:00:00Z"
            }
        }