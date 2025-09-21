import uuid
import hashlib
import logging
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile
from huggingface_hub import upload_file
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors import TitleExtractor, KeywordExtractor, SummaryExtractor
from llama_index.extractors.entity import EntityExtractor

from db.database_manager import db_manager
from config.settings import settings
from models import ProcessingConfig, IngestionResult
from loader import DocumentLoader
from ..ocr.yomitoku_ocr import YomitokuOCR, OCRConfig, OCRResult
from ..utils.file import temp_file

logger = logging.getLogger(__name__)

# class DocumentIngestService:
#     def __init__(self,
#                 ollama_connector: OllamaConnector):
#         self.db_manager = db_manager
#         self.ollama_connector = ollama_connector

#         # Node parser for chunking
#         self.node_parser = SentenceSplitter(
#             chunk_size=settings.chunk_size,
#             chunk_overlap=settings.chunk_overlap,
#             separator=" "
#         )
        
#         # Extractors for metadata enhancement
#         self.extractors = [
#             TitleExtractor(llm=self.ollama_connector.llm),
#             KeywordExtractor(llm=self.ollama_connector.llm),
#             SummaryExtractor(llm=self.ollama_connector.llm),
#             EntityExtractor(llm=self.ollama_connector.llm)
#         ]
    

class DocumentIngestionService:
    """
    ドキュメント取り込みサービス
    様々なソースからドキュメントを取り込み、統一的なDocument形式に変換
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.document_loader = DocumentLoader()
        self.ocr_service = YomitokuOCR(
            OCRConfig(
                confidence_threshold=self.config.ocr_confidence_threshold,
                languages=self.config.ocr_languages
            )
        ) if self.config.ocr_enabled else None
        
        logger.info(f"DocumentIngestionService initialized with config: {self.config}")

    def ingest_from_file_path(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> IngestionResult:
        """ファイルパスからドキュメントを取り込み"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            documents = []
            with temp_file(file_path=path, suffix=path.suffix) as tmp_path:
                documents = self.document_loader.load(tmp_path)
            
            return IngestionResult(
                success=True,
                documents=documents,
                metadata={'filename': path.name, 'file_path': str(path)}
            )
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return IngestionResult(
                success=False,
                documents=[],
                error_message=str(e),
                metadata={'filename': path.name, 'file_path': str(path)}
            )
    
    def ingest_from_directory(self, directory_path: str, metadata: Optional[Dict[str, Any]] = None) -> IngestionResult:
        """ディレクトリ内の全ファイルを取り込み"""
        try:
            path = Path(directory_path)
            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {directory_path}")

            documents = self.document_loader.load(directory_path, "directory")

            return IngestionResult(
                success=True,
                documents=documents,
                metadata={'directory_path': directory_path, 'file_count': len(documents)}
            )
        
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return IngestionResult(
                success=False,
                documents=[],
                error_message=str(e),
                metadata={'directory_path': directory_path}
            )
    
    def ingest_with_ocr(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> IngestionResult:
        """OCRを用いて画像/PDFからテキストを抽出し、ドキュメントを取り込み"""
        if not self.ocr_service:
            raise RuntimeError("OCR service is not enabled in the configuration.")
        
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            ocr_result: OCRResult = self.ocr_service.perform_ocr(str(path))
            if not ocr_result.success:
                raise RuntimeError(f"OCR failed for {file_path}: {ocr_result.error_message}")
            
            documents = self.document_loader.load_from_text(ocr_result.extracted_text)

            return IngestionResult(
                success=True,
                documents=documents,
                metadata={'filename': path.name, 'file_path': str(path), 'ocr_pages': ocr_result.page_count}
            )
        
        except Exception as e:
            logger.error(f"Error processing file with OCR {file_path}: {str(e)}")
            return IngestionResult(
                success=False,
                documents=[],
                error_message=str(e),
                metadata={'filename': path.name, 'file_path': str(path)}
            )