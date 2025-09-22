from pydantic import BaseModel
from typing import Optional, List

class UnifiedIDModel(BaseModel):
    unified_id: str
    global_sequence: int
    correlation_id: Optional[str] = None

class DocumentModel(BaseModel):
    title: str
    author: str
    created_at: str
    language: str
    original_format: str
    checksum: str
    version: str
    file_size: int
    metadata: dict
    processing_status: str
    raw_data: str

class ChunkModel(BaseModel):
    unified_chunk_id: str
    doc_unified_id: str
    text: str
    chunk_index: int
    offset_start: int
    offset_end: int
    metadata: dict

class IndexStatusModel(BaseModel):
    mongodb: str
    milvus: str
    neo4j: str
    redis: str

class DistributedIndexRegistryModel(BaseModel):
    unified_id: str
    entity_type: str
    indexes: dict
    overall_status: str
    created_at: str
    updated_at: str