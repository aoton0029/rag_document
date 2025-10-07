from typing import List, Optional, ClassVar
import time
from pydantic import BaseModel, Field
from pymilvus import FieldSchema, DataType
from utils.file import pydantic_field_info


class BaseCollection(BaseModel):
    embedding_model: Optional[str] = Field(None, max_length=100, alias='embedding_model')
    created_at: int = Field(default_factory=lambda: int(time.time()), alias='created_at')

    class Config:
        validate_by_name = True
        from_attributes = True

    @classmethod
    def to_fields(cls, vector_dim:int) -> List[FieldSchema]:
        return [
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="primary id"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim, description="embedding vector"),
            FieldSchema(name="page_label", dtype=DataType.VARCHAR, max_length=100, description="page id"),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255, description="file name"),
            FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100, description="embedding model"),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100, description="document id"),
            # FieldSchema(name="source_document_id", dtype=DataType.VARCHAR, max_length=100, description="source document id"),
            # FieldSchema(name="source_document_metadata", dtype=DataType.JSON, description="source document metadata"),
            # FieldSchema(name="chunk_method", dtype=DataType.VARCHAR, max_length=100, description="chunk method"),
            # FieldSchema(name="chunk_size", dtype=DataType.INT64, description="chunk size"),
            # FieldSchema(name="chunk_overlap", dtype=DataType.INT64, description="chunk overlap"),
            FieldSchema(name="_node_content", dtype=DataType.VARCHAR, max_length=8192, description="node content"),
            FieldSchema(name="_node_type", dtype=DataType.VARCHAR, max_length=1024, description="node type"),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=1024, description="doc id"),
            FieldSchema(name="ref_doc_id", dtype=DataType.VARCHAR, max_length=1024, description="reference document id"),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192, description="text"),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, description="id"),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=30, description="created timestamp"),
        ]

