"""
Query embedding model for Milvus vector storage.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryMetadata(BaseModel):
    """Metadata for query embeddings."""
    
    language: Optional[str] = Field(None, description="Query language")
    intent: Optional[str] = Field(
        None,
        pattern="^(informational|navigational|transactional)$",
        description="Query intent type"
    )
    complexity: Optional[str] = Field(
        None,
        pattern="^(simple|medium|complex)$",
        description="Query complexity level"
    )
    domain: Optional[str] = Field(None, description="Query domain")


class QueryEmbedding(BaseModel):
    """Query embedding model for Milvus storage."""
    
    query_id: str = Field(..., description="Unique query identifier")
    query_text: str = Field(..., description="Original query text")
    query_hash: str = Field(
        ...,
        pattern="^[a-f0-9]{64}$",
        description="Hash of normalized query"
    )
    embedding_vector: List[float] = Field(..., description="Query embedding vector")
    embedding_model: str = Field(..., description="Model used for query embedding")
    query_metadata: Optional[QueryMetadata] = Field(None, description="Query metadata")
    usage_count: int = Field(
        1,
        ge=1,
        description="Number of times this query was used"
    )
    avg_relevance_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Average relevance score of results"
    )
    created_at: int = Field(..., description="Unix timestamp of creation")
    last_used_at: int = Field(..., description="Unix timestamp of last usage")

    class Config:
        schema_extra = {
            "example": {
                "query_id": "query_123",
                "query_text": "What is machine learning?",
                "query_hash": "abc123...",
                "embedding_vector": [0.1, 0.2, 0.3],  # Truncated for example
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "usage_count": 5,
                "avg_relevance_score": 0.85,
                "created_at": 1640995200,
                "last_used_at": 1640995200
            }
        }

    def increment_usage(self):
        """Increment the usage count and update last used timestamp."""
        import time
        self.usage_count += 1
        self.last_used_at = int(time.time())
        return self

    def update_relevance_score(self, new_score: float):
        """Update the average relevance score with a new score."""
        if self.avg_relevance_score is None:
            self.avg_relevance_score = new_score
        else:
            # Simple moving average
            self.avg_relevance_score = (
                (self.avg_relevance_score * (self.usage_count - 1) + new_score) / 
                self.usage_count
            )
        return self