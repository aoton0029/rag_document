import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from .embedding_service import EmbeddingService, EmbeddingConfig
from ..ocrs.yomitoku_ocr import YomitokuOCR, OCRConfig, OCRResult
from ..utils.file_handler import FileHandler, ProcessingResult
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from ..database.vector_db.milvus_client import MilvusClient
from ..database.graph_db.neo4j_client import Neo4jClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.document_db.mongo_client import MongoClient


logger = logging.getLogger(__name__)


@dataclass
class OCREmbeddingConfig(EmbeddingConfig):
    """OCR対応埋め込み設定"""
    # OCR specific settings
    min_ocr_confidence: float = 0.5
    ocr_text_preprocessing: bool = True
    confidence_weighting: bool = True
    
    # Text processing for OCR content
    clean_ocr_artifacts: bool = True
    merge_text_chunks: bool = True
    preserve_layout_info: bool = True
    
    # Performance settings
    ocr_batch_size: int = 4
    parallel_processing: bool = True


@dataclass
class OCRTextChunk:
    """OCRで抽出されたテキストチャンク"""
    text: str
    confidence: float
    page_number: int = 0
    bbox: Optional[Dict[str, float]] = None
    language: str = "unknown"
    chunk_type: str = "text"  # "text", "title", "header", "caption"
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """高信頼度チャンクかどうか"""
        return self.confidence >= threshold


class OCRAwareEmbeddingService(EmbeddingService):
    """
    OCR対応埋め込みサービス
    OCRで抽出されたテキストを適切に処理して埋め込みを生成
    """
    
    def __init__(
        self,
        config: OCREmbeddingConfig,
        ocr_service: Optional[YomitokuOCR] = None,
        file_handler: Optional[FileHandler] = None,
        **kwargs
    ):
        # 基底クラス初期化
        super().__init__(config, **kwargs)
        
        self.ocr_config = config
        self.ocr_service = ocr_service or YomitokuOCR()
        self.file_handler = file_handler or FileHandler(enable_ocr=True)
        
        logger.info("OCRAwareEmbeddingService initialized")
    
    def preprocess_ocr_text(self, text: str, confidence: float = 1.0) -> str:
        """
        OCRテキストの前処理
        
        Args:
            text: OCRで抽出されたテキスト
            confidence: OCR信頼度
            
        Returns:
            前処理されたテキスト
        """
        if not self.ocr_config.ocr_text_preprocessing:
            return text
        
        processed_text = text
        
        if self.ocr_config.clean_ocr_artifacts:
            # OCRアーティファクトの除去
            processed_text = self._clean_ocr_artifacts(processed_text)
        
        # 低信頼度テキストのマーキング
        if confidence < self.ocr_config.min_ocr_confidence:
            processed_text = f"[低信頼度: {confidence:.2f}] {processed_text}"
        
        return processed_text
    
    def _clean_ocr_artifacts(self, text: str) -> str:
        """OCRアーティファクトの除去"""
        import re
        
        # 連続したスペースを単一化
        text = re.sub(r'\s+', ' ', text)
        
        # 不適切な改行の修正
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        text = re.sub(r'([ひ-ん])\n([ひ-ん])', r'\1\2', text)
        
        # 特殊文字の除去（OCRエラーによる）
        text = re.sub(r'[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u002D\u0021-\u007E]', '', text)
        
        # 空行の除去
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def create_ocr_chunks(self, ocr_result: OCRResult) -> List[OCRTextChunk]:
        """
        OCR結果からチャンクを作成
        
        Args:
            ocr_result: OCR処理結果
            
        Returns:
            OCRテキストチャンクのリスト
        """
        chunks = []
        
        if not ocr_result.success or not ocr_result.text:
            return chunks
        
        # バウンディングボックス情報がある場合
        if ocr_result.bounding_boxes:
            for bbox_info in ocr_result.bounding_boxes:
                chunk = OCRTextChunk(
                    text=bbox_info.get("text", ""),
                    confidence=bbox_info.get("confidence", ocr_result.confidence),
                    bbox=bbox_info.get("bbox"),
                    language=ocr_result.language_detected
                )
                
                if chunk.text.strip() and chunk.confidence >= self.ocr_config.min_ocr_confidence:
                    chunks.append(chunk)
        
        # バウンディングボックス情報がない場合は全体テキストを使用
        else:
            # テキストを段落で分割
            paragraphs = ocr_result.text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    chunk = OCRTextChunk(
                        text=para.strip(),
                        confidence=ocr_result.confidence,
                        language=ocr_result.language_detected
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def merge_high_confidence_chunks(self, chunks: List[OCRTextChunk]) -> List[OCRTextChunk]:
        """
        高信頼度チャンクをマージ
        
        Args:
            chunks: OCRチャンクのリスト
            
        Returns:
            マージされたチャンクのリスト
        """
        if not self.ocr_config.merge_text_chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if chunk.is_high_confidence(self.ocr_config.min_ocr_confidence):
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # 同じページで言語が同じ場合はマージ
                    if (current_chunk.page_number == chunk.page_number and 
                        current_chunk.language == chunk.language):
                        current_chunk.text += " " + chunk.text
                        current_chunk.confidence = min(current_chunk.confidence, chunk.confidence)
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                merged_chunks.append(chunk)
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    async def process_document_with_ocr(
        self,
        file_path: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        OCRを使用してドキュメントを処理し埋め込みを生成
        
        Args:
            file_path: ファイルパス
            document_id: ドキュメントID
            
        Returns:
            処理結果辞書
        """
        try:
            # ファイル処理
            processing_result = await self.file_handler.process_file_async(file_path)
            
            if not processing_result.success:
                return {
                    "success": False,
                    "error": processing_result.error_message,
                    "document_id": document_id
                }
            
            # OCR結果の確認
            ocr_used = processing_result.metadata.custom_metadata.get('ocr_used', False)
            
            if ocr_used:
                # OCRで抽出されたテキストの特別処理
                return await self._process_ocr_document(
                    processing_result, 
                    file_path, 
                    document_id
                )
            else:
                # 通常のテキスト処理
                return await self._process_regular_document(
                    processing_result, 
                    document_id
                )
        
        except Exception as e:
            logger.error(f"Error processing document with OCR: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    async def _process_ocr_document(
        self,
        processing_result: ProcessingResult,
        file_path: str,
        document_id: Optional[str]
    ) -> Dict[str, Any]:
        """OCRドキュメントの特別処理"""
        
        # OCR結果を取得
        if processing_result.metadata.file_extension == '.pdf':
            ocr_result = self.ocr_service.extract_text_from_pdf(file_path)
        else:
            ocr_result = self.ocr_service.extract_text_from_image(file_path)
        
        if not ocr_result.success:
            return {
                "success": False,
                "error": f"OCR extraction failed: {ocr_result.error_message}",
                "document_id": document_id
            }
        
        # OCRチャンクを作成
        ocr_chunks = self.create_ocr_chunks(ocr_result)
        
        # 高信頼度チャンクをマージ
        merged_chunks = self.merge_high_confidence_chunks(ocr_chunks)
        
        # LlamaIndex Documentを作成
        documents = []
        for i, chunk in enumerate(merged_chunks):
            # テキスト前処理
            processed_text = self.preprocess_ocr_text(chunk.text, chunk.confidence)
            
            # メタデータ作成
            chunk_metadata = {
                "document_id": document_id,
                "chunk_id": f"{document_id}_ocr_{i}" if document_id else f"ocr_{i}",
                "source_file": file_path,
                "ocr_confidence": chunk.confidence,
                "ocr_language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "processing_method": "ocr"
            }
            
            if chunk.bbox:
                chunk_metadata["bounding_box"] = chunk.bbox
            
            # Document作成
            doc = Document(
                text=processed_text,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        # 埋め込み生成
        return await self._generate_embeddings_for_documents(documents, document_id)
    
    async def _process_regular_document(
        self,
        processing_result: ProcessingResult,
        document_id: Optional[str]
    ) -> Dict[str, Any]:
        """通常ドキュメントの処理"""
        
        # LlamaIndex Documentを作成
        doc_metadata = {
            "document_id": document_id,
            "source_file": processing_result.metadata.file_path,
            "file_type": processing_result.metadata.file_extension,
            "processing_method": "regular",
            "word_count": processing_result.metadata.word_count,
            "page_count": processing_result.metadata.page_count
        }
        
        doc = Document(
            text=processing_result.content,
            metadata=doc_metadata
        )
        
        # 埋め込み生成
        return await self._generate_embeddings_for_documents([doc], document_id)
    
    async def _generate_embeddings_for_documents(
        self,
        documents: List[Document],
        document_id: Optional[str]
    ) -> Dict[str, Any]:
        """ドキュメントの埋め込み生成"""
        
        try:
            # ノード分割
            nodes = []
            for doc in documents:
                doc_nodes = self.node_parser.get_nodes_from_documents([doc])
                nodes.extend(doc_nodes)
            
            # 埋め込み生成
            embeddings = []
            for node in nodes:
                embedding = await self.generate_embedding_async(node.text)
                embeddings.append({
                    "node_id": node.node_id,
                    "text": node.text,
                    "embedding": embedding,
                    "metadata": node.metadata
                })
            
            # ベクトルストアに保存
            if self.vector_store:
                await self._store_embeddings(embeddings, document_id)
            
            return {
                "success": True,
                "document_id": document_id,
                "nodes_count": len(nodes),
                "embeddings_count": len(embeddings),
                "ocr_chunks": len([e for e in embeddings if e["metadata"].get("processing_method") == "ocr"])
            }
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    async def _store_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        document_id: Optional[str]
    ):
        """埋め込みをベクトルストアに保存"""
        
        # Milvusに保存
        vectors = []
        metadatas = []
        
        for emb in embeddings:
            vectors.append(emb["embedding"])
            metadatas.append({
                "node_id": emb["node_id"],
                "document_id": document_id,
                "text": emb["text"][:1000],  # テキストは1000文字まで
                **emb["metadata"]
            })
        
        # 非同期でベクトルストアに保存
        if vectors:
            await self.store_embeddings_async(vectors, metadatas, document_id)
    
    async def batch_process_documents(
        self,
        file_paths: List[str],
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        複数ドキュメントのバッチ処理
        
        Args:
            file_paths: ファイルパスのリスト
            document_ids: ドキュメントIDのリスト
            
        Returns:
            処理結果のリスト
        """
        if document_ids is None:
            document_ids = [None] * len(file_paths)
        
        if len(file_paths) != len(document_ids):
            raise ValueError("file_paths and document_ids must have the same length")
        
        # 並列処理
        tasks = []
        for file_path, doc_id in zip(file_paths, document_ids):
            task = self.process_document_with_ocr(file_path, doc_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 例外処理
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "document_id": document_ids[i],
                    "file_path": file_paths[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_ocr_statistics(self) -> Dict[str, Any]:
        """OCR統計情報を取得"""
        # Redis からOCR統計を取得
        stats = {
            "total_ocr_processed": 0,
            "average_confidence": 0.0,
            "language_distribution": {},
            "processing_times": [],
            "error_count": 0
        }
        
        try:
            # OCR関連のキャッシュ統計を取得
            if self.redis_client:
                cache_stats = self.redis_client.get_cache_stats()
                stats.update(cache_stats)
        except Exception as e:
            logger.warning(f"Failed to get OCR statistics: {e}")
        
        return stats


# ファクトリ関数
def create_ocr_embedding_service(
    config: Optional[OCREmbeddingConfig] = None,
    ocr_config: Optional[OCRConfig] = None,
    **kwargs
) -> OCRAwareEmbeddingService:
    """OCR対応埋め込みサービスのファクトリ関数"""
    
    if config is None:
        config = OCREmbeddingConfig()
    
    # OCRサービス作成
    ocr_service = YomitokuOCR(ocr_config) if ocr_config else None
    
    # ファイルハンドラー作成
    file_handler = FileHandler(enable_ocr=True, ocr_config=ocr_config)
    
    return OCRAwareEmbeddingService(
        config=config,
        ocr_service=ocr_service,
        file_handler=file_handler,
        **kwargs
    )


# テスト用関数
async def test_ocr_embedding_service():
    """OCR埋め込みサービスのテスト"""
    
    config = OCREmbeddingConfig(
        model_name="nomic-embed-text",
        min_ocr_confidence=0.5,
        ocr_text_preprocessing=True,
        confidence_weighting=True
    )
    
    service = create_ocr_embedding_service(config)
    
    # テストファイル
    test_files = [
        "test_image.png",
        "test_document.pdf",
        "test_scanned.pdf"
    ]
    
    for test_file in test_files:
        from pathlib import Path
        if Path(test_file).exists():
            result = await service.process_document_with_ocr(test_file, f"test_{test_file}")
            print(f"File: {test_file}")
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"Nodes: {result['nodes_count']}")
                print(f"OCR chunks: {result['ocr_chunks']}")
            else:
                print(f"Error: {result['error']}")
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(test_ocr_embedding_service())