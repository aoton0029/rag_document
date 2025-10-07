import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from datetime import datetime
from llama_index.core import Document
from pydantic import BaseModel
from llama_index.core.schema import BaseNode, TextNode

class IngestionResult(BaseModel):
    """取り込み結果"""
    success: bool
    documents: List[Document] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None


class EntityRelation(BaseModel):
    """エンティティ間の関係"""
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float = 0.0
    context: Optional[str] = None

class ChunkingResult(BaseModel):
    """チャンク化結果"""
    success: bool
    nodes: List[BaseNode] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None


class PreprocessingResult(BaseModel):
    """前処理結果"""
    success: bool
    documents: List[Document] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
