"""
Document processing and indexing functionality
ドキュメントの読み込み、チャンク化、インデックス作成機能
"""
import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.readers.file import PyMuPDFReader, DocxReader
from llama_index.core.schema import BaseNode, MetadataMode

from config.config import config
from src.rag_engine import rag_engine
from loguru import logger
import asyncio

class DocumentProcessor:
    """
    ドキュメントの処理とインデックス化を担当するクラス
    """
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': PyMuPDFReader(),
            '.docx': DocxReader(),
            '.doc': DocxReader(),
            '.txt': None,  # テキストファイルは直接処理
            '.md': None,   # Markdownファイルは直接処理
        }
        
        # Node Parser の設定
        self.sentence_splitter = SentenceSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?"
        )
        
        # セマンティック分割器（より高度なチャンク分割）
        try:
            self.semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=None  # 後で設定
            )
        except Exception as e:
            logger.warning(f"Semantic splitter not available: {str(e)}")
            self.semantic_splitter = None
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """単一ファイルからドキュメントを読み込み"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Loading document: {file_path.name}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # メタデータの準備
            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_type": file_extension,
                "created_date": file_path.stat().st_ctime,
                "modified_date": file_path.stat().st_mtime
            }
            
            # ファイルタイプに応じた読み込み
            if file_extension in ['.txt', '.md']:
                # テキストファイルの直接読み込み
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                documents = [Document(text=content, metadata=metadata)]
            else:
                # 専用リーダーを使用
                reader = self.supported_formats[file_extension]
                documents = reader.load_data(file_path)
                
                # メタデータを各ドキュメントに追加
                for doc in documents:
                    doc.metadata.update(metadata)
            
            logger.success(f"Loaded {len(documents)} document(s) from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {str(e)}")
            raise
    
    def load_documents_from_directory(self, 
                                    directory_path: Union[str, Path],
                                    recursive: bool = True) -> List[Document]:
        """ディレクトリから複数のドキュメントを読み込み"""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists() or not directory_path.is_dir():
                raise ValueError(f"Invalid directory path: {directory_path}")
            
            logger.info(f"Loading documents from directory: {directory_path}")
            
            all_documents = []
            
            # ファイルパターンの作成
            if recursive:
                file_patterns = [f"**/*{ext}" for ext in self.supported_formats.keys()]
            else:
                file_patterns = [f"*{ext}" for ext in self.supported_formats.keys()]
            
            # 各パターンでファイルを検索
            for pattern in file_patterns:
                for file_path in directory_path.glob(pattern):
                    if file_path.is_file():
                        try:
                            documents = self.load_document(file_path)
                            all_documents.extend(documents)
                        except Exception as e:
                            logger.warning(f"Failed to load {file_path}: {str(e)}")
                            continue
            
            logger.success(f"Loaded {len(all_documents)} document(s) from {directory_path}")
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from directory: {str(e)}")
            raise
    
    def chunk_documents(self, 
                       documents: List[Document], 
                       use_semantic_chunking: bool = False) -> List[BaseNode]:
        """ドキュメントをチャンクに分割"""
        try:
            logger.info(f"Chunking {len(documents)} document(s)...")
            
            if use_semantic_chunking and self.semantic_splitter:
                # セマンティック分割を使用
                logger.info("Using semantic chunking")
                nodes = self.semantic_splitter.get_nodes_from_documents(documents)
            else:
                # 文章ベース分割を使用
                logger.info("Using sentence-based chunking")
                nodes = self.sentence_splitter.get_nodes_from_documents(documents)
            
            # チャンクの統計情報をログ出力
            chunk_sizes = [len(node.text) for node in nodes]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            logger.info(f"Created {len(nodes)} chunks")
            logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
            logger.info(f"Min/Max chunk size: {min(chunk_sizes) if chunk_sizes else 0}/{max(chunk_sizes) if chunk_sizes else 0}")
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to chunk documents: {str(e)}")
            raise
    
    def process_and_index(self, 
                         file_paths: Union[str, Path, List[Union[str, Path]]], 
                         use_semantic_chunking: bool = False,
                         is_directory: bool = False) -> Dict[str, Any]:
        """ドキュメントの処理とインデックス化を一括実行"""
        try:
            logger.info("Starting document processing and indexing...")
            
            # ファイルパスの正規化
            if isinstance(file_paths, (str, Path)):
                file_paths = [file_paths]
            
            all_documents = []
            
            # ドキュメントの読み込み
            for file_path in file_paths:
                if is_directory or Path(file_path).is_dir():
                    documents = self.load_documents_from_directory(file_path)
                else:
                    documents = self.load_document(file_path)
                all_documents.extend(documents)
            
            if not all_documents:
                raise ValueError("No documents were loaded")
            
            # ドキュメントのチャンク化
            nodes = self.chunk_documents(all_documents, use_semantic_chunking)
            
            # RAGエンジンでインデックス化
            rag_engine.initialize(all_documents)
            
            # 処理結果の統計
            result = {
                "total_documents": len(all_documents),
                "total_chunks": len(nodes),
                "processed_files": [doc.metadata.get("file_name", "unknown") for doc in all_documents],
                "average_chunk_size": sum(len(node.text) for node in nodes) / len(nodes) if nodes else 0,
                "indexing_completed": True
            }
            
            logger.success(f"Document processing completed: {result['total_documents']} documents, {result['total_chunks']} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process and index documents: {str(e)}")
            raise
    
    def get_document_info(self, documents: List[Document]) -> Dict[str, Any]:
        """ドキュメントの詳細情報を取得"""
        try:
            total_chars = sum(len(doc.text) for doc in documents)
            file_types = {}
            file_sizes = []
            
            for doc in documents:
                file_type = doc.metadata.get("file_type", "unknown")
                file_types[file_type] = file_types.get(file_type, 0) + 1
                file_sizes.append(doc.metadata.get("file_size", 0))
            
            return {
                "document_count": len(documents),
                "total_characters": total_chars,
                "average_document_size": total_chars / len(documents) if documents else 0,
                "file_types": file_types,
                "total_file_size": sum(file_sizes),
                "average_file_size": sum(file_sizes) / len(file_sizes) if file_sizes else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get document info: {str(e)}")
            return {}
    
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """ファイルから詳細なメタデータを抽出"""
        try:
            file_path = Path(file_path)
            
            # 基本的なファイル情報
            stat_info = file_path.stat()
            
            # MIME typeの判定
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            metadata = {
                "file_name": file_path.name,
                "file_stem": file_path.stem,
                "file_extension": file_path.suffix,
                "file_path": str(file_path.absolute()),
                "file_size": stat_info.st_size,
                "created_time": stat_info.st_ctime,
                "modified_time": stat_info.st_mtime,
                "mime_type": mime_type,
                "is_readable": os.access(file_path, os.R_OK),
                "parent_directory": str(file_path.parent)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {str(e)}")
            return {}

# グローバルドキュメントプロセッサインスタンス
document_processor = DocumentProcessor()