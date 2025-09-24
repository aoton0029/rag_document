import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from datetime import datetime
from dataclasses import dataclass, asdict

from llama_index.core import Document

@dataclass
class ProcessingConfig:
    """ドキュメント処理設定"""
    # OCR設定
    ocr_enabled: bool = True
    ocr_confidence_threshold: float = 0.5
    ocr_languages: List[str] = None
    
    # チャンク化設定
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_method: str = "sentence"  # "sentence", "token", "simple"
    
    # 前処理設定
    clean_text: bool = True
    extract_entities: bool = True
    detect_language: bool = True
    
    # メタデータ設定
    include_file_metadata: bool = True
    include_processing_metadata: bool = True
    
    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ["ja", "en"]

@dataclass
class IngestionResult:
    """取り込み結果"""
    success: bool
    documents: List[Document] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.documents is None:
            self.documents = []
        if self.metadata is None:
            self.metadata = {}
          
@dataclass
class EntityRelation:
    """エンティティ間の関係"""
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float = 0.0
    context: Optional[str] = None

@dataclass
class ChunkData:
    """MongoDB chunksコレクション用のチャンクデータ"""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    offset_start: int
    offset_end: int
    metadata: Dict[str, Any]
    embedding_generated_at: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Neo4jData:
    """Neo4j用のエンティティ・関係性データ"""
    entities: List[str]
    relations: List[EntityRelation]
    chunk_id: str
    doc_id: str
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relations is None:
            self.relations = []

@dataclass
class ChunkingResult:
    """チャンク化結果"""
    success: bool
    documents: List[Document] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    # MongoDB chunks コレクション用データ
    mongodb_chunks: List[ChunkData] = None
    # Neo4j エンティティ・関係性用データ
    neo4j_data: List[Neo4jData] = None
    
    def __post_init__(self):
        if self.documents is None:
            self.documents = []
        if self.metadata is None:
            self.metadata = {}
        if self.mongodb_chunks is None:
            self.mongodb_chunks = []
        if self.neo4j_data is None:
            self.neo4j_data = []


@dataclass
class PreprocessingResult:
    """前処理結果"""
    success: bool
    documents: List[Document] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.documents is None:
            self.documents = []
        if self.metadata is None:
            self.metadata = {}