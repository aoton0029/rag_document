"""
Chunk model for MongoDB storage.
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


class Chunk(BaseModel):
    """Chunk model for MongoDB storage."""
    
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Reference to parent document")
    chunk_index: int = Field(..., ge=0, description="Order of chunk within document")
    content: str = Field(..., description="Chunk content")
    content_length: Optional[int] = Field(None, ge=0, description="Character count of chunk")
    start_position: Optional[int] = Field(None, ge=0, description="Start position in original document")
    end_position: Optional[int] = Field(None, ge=0, description="End position in original document")
    overlap_with_prev: Optional[int] = Field(None, ge=0, description="Overlap characters with previous chunk")
    overlap_with_next: Optional[int] = Field(None, ge=0, description="Overlap characters with next chunk")
    embedding_vector_id: Optional[str] = Field(None, description="Reference to embedding vector in Milvus")
    computed_summary: Optional[str] = Field(None, description="AI-generated summary of chunk")
    computed_keywords: Optional[List[str]] = Field(
        default_factory=list,
        description="Extracted keywords from chunk"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Chunk creation timestamp")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "chunk_id": "chunk_123",
                "document_id": "doc_123",
                "chunk_index": 0,
                "content": "This is the first chunk of the document.",
                "content_length": 42,
                "start_position": 0,
                "end_position": 42,
                "created_at": "2023-01-01T00:00:00Z"
            }
        }