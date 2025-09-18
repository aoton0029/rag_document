"""
Document embedding model for Milvus vector storage.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentEmbeddingMetadata(BaseModel):
    """Metadata for document embeddings."""
    
    content_type: Optional[str] = Field(None, description="Content type of the document")
    language: Optional[str] = Field(None, description="Language of the content")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Keywords from the content")
    category: Optional[str] = Field(None, description="Content category")


class DocumentEmbedding(BaseModel):
    """Document embedding model for Milvus storage."""
    
    vector_id: str = Field(..., description="Unique vector identifier")
    chunk_id: str = Field(..., description="Reference to chunk in MongoDB")
    document_id: str = Field(..., description="Reference to parent document")
    embedding_vector: List[float] = Field(
        ...,
        min_items=384,
        max_items=1536,
        description="Dense vector embedding"
    )
    embedding_model: str = Field(..., description="Model used to generate embedding")
    vector_dimension: int = Field(
        ...,
        ge=384,
        le=1536,
        description="Dimension of the embedding vector"
    )
    content_hash: Optional[str] = Field(
        None,
        pattern="^[a-f0-9]{64}$",
        description="SHA-256 hash of the original content"
    )
    metadata: Optional[DocumentEmbeddingMetadata] = Field(None, description="Embedding metadata")
    created_at: int = Field(..., description="Unix timestamp of creation")

    class Config:
        schema_extra = {
            "example": {
                "vector_id": "vec_123",
                "chunk_id": "chunk_123",
                "document_id": "doc_123",
                "embedding_vector": [0.1, 0.2, 0.3],  # Truncated for example
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_dimension": 384,
                "content_hash": "abc123...",
                "created_at": 1640995200
            }
        }

    def validate_vector_dimension(self):
        """Validate that vector dimension matches the actual vector length."""
        if len(self.embedding_vector) != self.vector_dimension:
            raise ValueError(
                f"Vector dimension {self.vector_dimension} does not match "
                f"actual vector length {len(self.embedding_vector)}"
            )
        return self