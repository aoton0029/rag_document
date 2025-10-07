import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode
from db.database_manager import db_manager
from services.loader import DocumentLoader
from services.models import IngestionResult
from utils.file import temp_file
from configs import ProcessingConfig

logger = logging.getLogger(__name__)

class DocumentIngestionService:
    """
    ドキュメント取り込みサービス
    様々なソースからドキュメントを取り込み、統一的なDocument形式に変換
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.document_loader = DocumentLoader()
        
        logger.info(f"DocumentIngestionService initialized with config: {self.config}")

    def ingest_from_file_path(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> IngestionResult:
        """ファイルパスからドキュメントを取り込み"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # documents = []
            # with temp_file(file_path=path, suffix=path.suffix) as tmp_path:
            #     documents = self.document_loader.load(tmp_path)
            documents = self.document_loader.load(path)
            
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
   