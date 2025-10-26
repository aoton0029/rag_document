"""
Document Loader Module
PDF読み取り、メタデータ抽出など
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from llama_index.core.schema import Document
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.readers.file import (
    PDFReader, 
    DocxReader, 
    EpubReader, 
    MarkdownReader,
    PandasExcelReader
)

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """
    ドキュメントローダー基底クラス
    """
    
    def __init__(self, **kwargs):
        """
        DocumentLoaderの初期化
        
        Args:
            **kwargs: 追加パラメータ
        """
        self.kwargs = kwargs
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        ドキュメントを読み込む
        
        Args:
            file_path: ファイルパス
            
        Returns:
            Documentのリスト
        """
        pass
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        ディレクトリ内の全ドキュメントを読み込む
        
        Args:
            directory_path: ディレクトリパス
            
        Returns:
            Documentのリスト
        """
        try:
            reader = SimpleDirectoryReader(
                input_dir=directory_path,
                recursive=True,
                **self.kwargs
            )
            documents = reader.load_data()
            logger.info(f"{directory_path}から{len(documents)}ドキュメントを読み込み")
            return documents
        except Exception as e:
            logger.error(f"ディレクトリ読み込みエラー: {e}")
            raise


class PDFLoader(DocumentLoader):
    """
    PDFローダー
    """
    
    def __init__(self, use_pymupdf: bool = True, **kwargs):
        """
        PDFLoaderの初期化
        
        Args:
            use_pymupdf: pymupdf4llmを使用するか
            **kwargs: 追加パラメータ
        """
        super().__init__(**kwargs)
        self.use_pymupdf = use_pymupdf
        self._reader = None
    
    def _get_reader(self) -> PDFReader:
        """PDFReaderを取得"""
        if self._reader is None:
            self._reader = PDFReader()
        return self._reader
    
    def load(self, file_path: str) -> List[Document]:
        """
        PDFファイルを読み込む
        
        Args:
            file_path: PDFファイルパス
            
        Returns:
            Documentのリスト
        """
        try:
            if self.use_pymupdf:
                # pymupdf4llmを使用
                return self._load_with_pymupdf(file_path)
            else:
                # llama_index PDFReaderを使用
                reader = self._get_reader()
                documents = reader.load_data(file=Path(file_path))
                logger.info(f"PDFファイル読み込み: {file_path} ({len(documents)}ドキュメント)")
                return documents
        except Exception as e:
            logger.error(f"PDF読み込みエラー: {file_path} - {e}")
            raise
    
    def _load_with_pymupdf(self, file_path: str) -> List[Document]:
        """
        pymupdf4llmでPDFを読み込む
        
        Args:
            file_path: PDFファイルパス
            
        Returns:
            Documentのリスト
        """
        try:
            import pymupdf4llm
            
            # PDFをMarkdown形式で抽出
            md_text = pymupdf4llm.to_markdown(file_path)
            
            # Documentを作成
            doc = Document(
                text=md_text,
                metadata={
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "file_type": "pdf",
                    "loader": "pymupdf4llm"
                }
            )
            
            logger.info(f"pymupdf4llmでPDFファイル読み込み: {file_path}")
            return [doc]
        except ImportError:
            logger.warning("pymupdf4llmが利用不可、通常のPDFReaderにフォールバック")
            reader = self._get_reader()
            return reader.load_data(file=Path(file_path))
        except Exception as e:
            logger.error(f"pymupdf4llm読み込みエラー: {e}")
            raise


class DocxLoader(DocumentLoader):
    """
    Docxローダー
    """
    
    def __init__(self, **kwargs):
        """
        DocxLoaderの初期化
        
        Args:
            **kwargs: 追加パラメータ
        """
        super().__init__(**kwargs)
        self._reader = None
    
    def _get_reader(self) -> DocxReader:
        """DocxReaderを取得"""
        if self._reader is None:
            self._reader = DocxReader()
        return self._reader
    
    def load(self, file_path: str) -> List[Document]:
        """
        Docxファイルを読み込む
        
        Args:
            file_path: Docxファイルパス
            
        Returns:
            Documentのリスト
        """
        try:
            reader = self._get_reader()
            documents = reader.load_data(file=Path(file_path))
            logger.info(f"Docxファイル読み込み: {file_path} ({len(documents)}ドキュメント)")
            return documents
        except Exception as e:
            logger.error(f"Docx読み込みエラー: {file_path} - {e}")
            raise


class MarkdownLoader(DocumentLoader):
    """
    Markdownローダー
    """
    
    def __init__(self, **kwargs):
        """
        MarkdownLoaderの初期化
        
        Args:
            **kwargs: 追加パラメータ
        """
        super().__init__(**kwargs)
        self._reader = None
    
    def _get_reader(self) -> MarkdownReader:
        """MarkdownReaderを取得"""
        if self._reader is None:
            self._reader = MarkdownReader()
        return self._reader
    
    def load(self, file_path: str) -> List[Document]:
        """
        Markdownファイルを読み込む
        
        Args:
            file_path: Markdownファイルパス
            
        Returns:
            Documentのリスト
        """
        try:
            reader = self._get_reader()
            documents = reader.load_data(file=Path(file_path))
            logger.info(f"Markdownファイル読み込み: {file_path} ({len(documents)}ドキュメント)")
            return documents
        except Exception as e:
            logger.error(f"Markdown読み込みエラー: {file_path} - {e}")
            raise


class MultiModalLoader(DocumentLoader):
    """
    マルチモーダルローダー
    複数のファイル形式に対応
    """
    
    def __init__(self, **kwargs):
        """
        MultiModalLoaderの初期化
        
        Args:
            **kwargs: 追加パラメータ
        """
        super().__init__(**kwargs)
        self.loaders = {
            ".pdf": PDFLoader(**kwargs),
            ".docx": DocxLoader(**kwargs),
            ".md": MarkdownLoader(**kwargs),
        }
    
    def load(self, file_path: str) -> List[Document]:
        """
        ファイルを読み込む（拡張子に応じて適切なローダーを使用）
        
        Args:
            file_path: ファイルパス
            
        Returns:
            Documentのリスト
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext in self.loaders:
            loader = self.loaders[ext]
            return loader.load(file_path)
        else:
            logger.warning(f"未対応の拡張子: {ext}、SimpleDirectoryReaderを使用")
            reader = SimpleDirectoryReader(input_files=[file_path])
            return reader.load_data()
