from typing import List, Optional, ClassVar
import time
from pydantic import BaseModel, Field
from pymilvus import FieldSchema, DataType
from utils.file import pydantic_field_info
from abc import ABC, abstractmethod

class BaseCollection(BaseModel, ABC):
    collection_name: ClassVar[str] = None
    vector_dim: ClassVar[int] = None

    id: Optional[int] = Field(None, alias='id')
    embedding_vector: List[float] = Field(..., alias='embedding_vector')
    embedding_model: Optional[str] = Field(None, max_length=100, alias='embedding_model')
    created_at: int = Field(default_factory=lambda: int(time.time()), alias='created_at')

    class Config:
        allow_population_by_field_name = True
        orm_mode = True

    @abstractmethod
    @classmethod
    def to_fields(cls) -> List[FieldSchema]:
        return [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="primary id"),
            FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=cls.vector_dim, description="embedding vector"),
            FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100, description="embedding model"),
            FieldSchema(name="created_at", dtype=DataType.INT64, description="created timestamp"),
        ]


class DocumentMetadata(BaseModel):
    content_type: Optional[str] = Field(..., alias='content_type')
    language: Optional[str] = Field(..., alias='language')

    class Config:
        allow_population_by_field_name = True


class DocumentCollection(BaseCollection):
    collection_name: ClassVar[str] = "document"
    vector_dim: ClassVar[int] = 3072

    metadata: Optional[DocumentMetadata] = Field(None, alias='metadata')
    vector_id: str = Field(..., max_length=100, alias='vector_id')
    chunk_id: Optional[str] = Field(None, max_length=100, alias='chunk_id')
    document_id: str = Field(..., max_length=100, alias='document_id')
    content_hash: Optional[str] = Field(None, max_length=64, alias='content_hash')

    class Config:
        allow_population_by_field_name = True
        orm_mode = True
    
    @classmethod
    def to_fields(cls):
        schemas = super().to_fields()
        schemas.extend([
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024, description="json metadata"),
            FieldSchema(name="vector_id", dtype=DataType.VARCHAR, max_length=100, description="vector id"),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100, description="chunk id"),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100, description="document id"),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64, description="content hash"),
        ])
        return schemas


class QueryMetadata(BaseModel):
    language: Optional[str] = Field(None, max_length=10, alias='language')
    intent: Optional[str] = Field(None, max_length=20, alias='intent')
    complexity: Optional[str] = Field(None, max_length=20, alias='complexity')
    
    class Config:
        allow_population_by_field_name = True
    

class QueryCollection(BaseCollection):
    collection_name: ClassVar[str] = "query"
    vector_dim: ClassVar[int] = 3072

    metadata: Optional[QueryMetadata] = Field(None, alias='metadata')
    query_id: str = Field(..., max_length=100, alias='query_id')
    query_text: str = Field(..., max_length=1000, alias='query_text')
    query_hash: str = Field(..., max_length=64, alias='query_hash')
    usage_count: int = Field(0, ge=0, alias='usage_count')
    avg_relevance_score: float = Field(0.0, ge=0.0, alias='avg_relevance_score')
    last_used_at: Optional[int] = Field(None, alias='last_used_at')
    
    class Config:
        allow_population_by_field_name = True
        orm_mode = True
    
    @classmethod
    def to_fields(cls):
        schemas = super().to_fields()
        schemas.extend([
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024, description="json metadata"),
            FieldSchema(name="query_id", dtype=DataType.VARCHAR, max_length=100, description="query id"),
            FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=1000, description="query text"),
            FieldSchema(name="query_hash", dtype=DataType.VARCHAR, max_length=64, description="query hash"),
            FieldSchema(name="usage_count", dtype=DataType.INT64, description="usage count"),
            FieldSchema(name="avg_relevance_score", dtype=DataType.FLOAT, description="avg relevance"),
            FieldSchema(name="last_used_at", dtype=DataType.INT64, description="last used timestamp"),
        ])
        return schemas
