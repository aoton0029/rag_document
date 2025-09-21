import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pathlib import Path
from datetime import datetime
from ..llm.ollama_connector import OllamaConnector
from chunk_service import DocumentChunker
from ingest_service import DocumentIngestionService
from preprocessor_service import DocumentPreprocessor
from models import ProcessingConfig, PreprocessingResult
from ..db.database_manager import db_manager

class IntegratedDocumentProcessor:
    """
    統合ドキュメント処理サービス
    取り込み→前処理→チャンク化を一貫して実行
    """
    
    def __init__(self, 
                 config: Optional[ProcessingConfig],
                 ):
        self.config = config or ProcessingConfig()
        self.ingestion_service = DocumentIngestionService(self.config)
        self.preprocessor = DocumentPreprocessor(self.config)
        self.chunker = DocumentChunker(self.config)


    def process_file(self, file_path:str):
        ingestion_result = self.ingestion_service.ingest_from_file_path(file_path=file_path)
        if not ingestion_result.success:
            logging.error(f"Ingestion failed: {ingestion_result.error_message}")
            return None
        
        preprocessing_result = self.preprocessor.preprocess(ingestion_result.documents)
        if not preprocessing_result.success:
            logging.error(f"Preprocessing failed: {preprocessing_result.error_message}")
            return None
        
        chunking_result = self.chunker.chunk_documents(preprocessing_result.documents)
        if not chunking_result.success:
            logging.error(f"Chunking failed: {chunking_result.error_message}")
            return None
        
        # MongoDBとNeo4jに保存
        
        
        return {
            "ingestion": ingestion_result,
            "preprocessing": preprocessing_result,
            "chunking": chunking_result
        }