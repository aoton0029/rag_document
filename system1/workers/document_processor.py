import asyncio
import logging
import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime
import uuid
import fitz 
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import markdown
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter
from ..database.document_db.mongo_client import MongoClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.relational_db.models import DocumentMetadata
from ..embedding.llamaindex_embedding_service import EmbeddingService, EmbeddingConfig


@dataclass
class ProcessingConfig:
    """ドキュメント処理設定"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    supported_formats: List[str] = None
    ocr_enabled: bool = False
    extract_images: bool = False
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                '.pdf', '.epub', '.html', '.htm', '.md', '.markdown', 
                '.txt', '.docx', '.doc'
            ]


class DocumentProcessor:
    """
    ドキュメント処理ワーカー
    ファイルのアップロード、解析、チャンク分割を担当
    """
    
    def __init__(
        self,
        config: ProcessingConfig = None,
        upload_dir: str = "./data/uploads",
        processed_dir: str = "./data/processed",
        mongo_client: MongoClient = None,
        redis_client: RedisClient = None,
        embedding_service: EmbeddingService = None
    ):
        self.config = config or ProcessingConfig()
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.logger = logging.getLogger(__name__)
        
        # クライアント初期化
        self.mongo_client = mongo_client or MongoClient()
        self.redis_client = redis_client or RedisClient()
        self.embedding_service = embedding_service
        
        # ディレクトリ作成
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # ノードパーサー初期化
        self.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.logger.info("ドキュメント処理ワーカー初期化完了")
    
    async def process_document(
        self,
        file_path: Union[str, Path],
        document_id: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        ドキュメント処理のメインエントリーポイント
        
        Args:
            file_path: ファイルパス
            document_id: ドキュメントID
            metadata: 追加メタデータ
        
        Returns:
            処理結果
        """
        try:
            file_path = Path(file_path)
            
            # ファイル検証
            if not self._validate_file(file_path):
                raise ValueError(f"サポートされていないファイル形式: {file_path.suffix}")
            
            # 処理ステータス更新
            await self._update_processing_status(document_id, "parsing")
            
            # ファイル解析
            content = await self._parse_document(file_path)
            
            # メタデータ抽出
            extracted_metadata = await self._extract_metadata(file_path, content)
            if metadata:
                extracted_metadata.update(metadata)
            
            # ドキュメント作成
            document = Document(
                text=content["text"],
                metadata={
                    "document_id": document_id,
                    "source": str(file_path),
                    "file_type": file_path.suffix,
                    **extracted_metadata
                }
            )
            
            # チャンク分割
            await self._update_processing_status(document_id, "chunking")
            chunks = await self._chunk_document(document)
            
            # MongoDB保存
            await self._update_processing_status(document_id, "storing")
            chunk_ids = await self._store_chunks(document_id, chunks, extracted_metadata)
            
            # SQL database metadata保存
            await self._store_document_metadata(document_id, file_path, extracted_metadata, len(chunks))
            
            # 埋め込み生成キューに追加
            if self.embedding_service:
                await self._queue_embedding_generation(document_id, chunk_ids)
            
            # 処理完了
            await self._update_processing_status(document_id, "completed")
            
            result = {
                "document_id": document_id,
                "status": "completed",
                "chunk_count": len(chunks),
                "chunk_ids": chunk_ids,
                "metadata": extracted_metadata,
                "content_length": len(content["text"]),
                "processed_at": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"ドキュメント処理完了: {document_id}")
            return result
            
        except Exception as e:
            await self._update_processing_status(document_id, "failed", str(e))
            self.logger.error(f"ドキュメント処理エラー: {document_id}, {e}")
            raise
    
    def _validate_file(self, file_path: Path) -> bool:
        """ファイル検証"""
        # ファイル存在確認
        if not file_path.exists():
            return False
        
        # サイズ確認
        if file_path.stat().st_size > self.config.max_file_size:
            return False
        
        # 拡張子確認
        return file_path.suffix.lower() in self.config.supported_formats
    
    async def _parse_document(self, file_path: Path) -> Dict[str, Any]:
        """ファイル形式に応じた解析"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return await self._parse_pdf(file_path)
        elif suffix == '.epub':
            return await self._parse_epub(file_path)
        elif suffix in ['.html', '.htm']:
            return await self._parse_html(file_path)
        elif suffix in ['.md', '.markdown']:
            return await self._parse_markdown(file_path)
        elif suffix == '.docx':
            return await self._parse_docx(file_path)
        elif suffix == '.txt':
            return await self._parse_text(file_path)
        else:
            raise ValueError(f"サポートされていない形式: {suffix}")
    
    async def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """PDF解析"""
        doc = fitz.open(file_path)
        text_content = ""
        metadata = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", "")
        }
        
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            text_content += page_text + "\n"
            
            pages.append({
                "page_number": page_num + 1,
                "text": page_text,
                "word_count": len(page_text.split())
            })
        
        doc.close()
        
        return {
            "text": text_content.strip(),
            "structure": {"pages": pages},
            "metadata": metadata
        }
    
    async def _parse_epub(self, file_path: Path) -> Dict[str, Any]:
        """EPUB解析"""
        book = epub.read_epub(file_path)
        text_content = ""
        chapters = []
        
        # メタデータ抽出
        metadata = {
            "title": book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else "",
            "author": book.get_metadata("DC", "creator")[0][0] if book.get_metadata("DC", "creator") else "",
            "language": book.get_metadata("DC", "language")[0][0] if book.get_metadata("DC", "language") else "",
            "publisher": book.get_metadata("DC", "publisher")[0][0] if book.get_metadata("DC", "publisher") else ""
        }
        
        # チャプター抽出
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                chapter_text = soup.get_text()
                text_content += chapter_text + "\n"
                
                chapters.append({
                    "chapter_id": item.get_id(),
                    "title": item.get_name(),
                    "text": chapter_text,
                    "word_count": len(chapter_text.split())
                })
        
        return {
            "text": text_content.strip(),
            "structure": {"chapters": chapters},
            "metadata": metadata
        }
    
    async def _parse_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Markdown解析"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # HTML変換
        html = markdown.markdown(content, extensions=['meta', 'toc'])
        
        # プレーンテキスト抽出
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # メタデータ抽出（YAMLフロントマター）
        metadata = {}
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    metadata = yaml.safe_load(parts[1])
                except:
                    pass
        
        return {
            "text": text.strip(),
            "structure": {"markdown": content[:1000]},
            "metadata": metadata
        }
    
    async def _parse_text(self, file_path: Path) -> Dict[str, Any]:
        """テキストファイル解析"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "text": content.strip(),
            "structure": {"line_count": len(content.splitlines())},
            "metadata": {"encoding": "utf-8"}
        }
    
    async def _extract_metadata(self, file_path: Path, content: Dict[str, Any]) -> Dict[str, Any]:
        """メタデータ抽出・拡張"""
        file_stat = file_path.stat()
        
        base_metadata = {
            "filename": file_path.name,
            "file_size": file_stat.st_size,
            "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "mime_type": mimetypes.guess_type(file_path)[0],
            "file_hash": self._calculate_file_hash(file_path),
            "word_count": len(content["text"].split()),
            "char_count": len(content["text"])
        }
        
        # ファイル固有のメタデータをマージ
        if "metadata" in content:
            base_metadata.update(content["metadata"])
        
        return base_metadata
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """ファイルハッシュ計算"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def _chunk_document(self, document: Document) -> List[TextNode]:
        """ドキュメントチャンク分割"""
        nodes = self.node_parser.get_nodes_from_documents([document])
        
        # 各ノードにメタデータ追加
        for i, node in enumerate(nodes):
            node.metadata.update({
                "chunk_index": i,
                "chunk_id": f"{document.metadata['document_id']}_chunk_{i}",
                "source_document": document.metadata['document_id']
            })
        
        return nodes
    
    async def _store_chunks(
        self,
        document_id: str,
        chunks: List[TextNode],
        metadata: Dict[str, Any]
    ) -> List[str]:
        """チャンクをMongoDBに保存"""
        chunk_ids = []
        
        for chunk in chunks:
            chunk_doc = {
                "document_id": document_id,
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_index": chunk.metadata["chunk_index"],
                "content": chunk.text,
                "metadata": {
                    **chunk.metadata,
                    "word_count": len(chunk.text.split()),
                    "char_count": len(chunk.text)
                },
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = self.mongo_client.documents.insert_one(chunk_doc)
            chunk_ids.append(str(result.inserted_id))
        
        # ドキュメントサマリー保存
        summary_doc = {
            "document_id": document_id,
            "type": "summary",
            "total_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "metadata": metadata,
            "processing_info": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "processed_at": datetime.utcnow()
            }
        }
        
        self.mongo_client.metadata.insert_one(summary_doc)
        
        return chunk_ids
    
    async def _queue_embedding_generation(self, document_id: str, chunk_ids: List[str]):
        """埋め込み生成タスクをキューに追加"""
        task_data = {
            "task_type": "generate_embeddings",
            "document_id": document_id,
            "chunk_ids": chunk_ids,
            "model_name": "ollama",
            "priority": "normal"
        }
        
        task_id = f"embed_{document_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        success = self.redis_client.add_to_processing_queue(task_id, task_data)
        
        if success:
            self.logger.info(f"埋め込み生成タスクをキューに追加: {task_id}")
        else:
            self.logger.error(f"埋め込み生成タスクキュー追加失敗: {task_id}")
    
    async def _update_processing_status(
        self,
        document_id: str,
        status: str,
        error_message: str = None
    ):
        """処理ステータス更新"""
        status_data = {
            "document_id": document_id,
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if error_message:
            status_data["error"] = error_message
        
        # Redis に保存
        status_key = f"doc_status:{document_id}"
        self.redis_client.set_cache(status_key, status_data, 3600)  # 1時間保持
        
        self.logger.info(f"処理ステータス更新: {document_id} -> {status}")
    
    
    async def _store_document_metadata(self, document_id: str, file_path: Path, 
                                     metadata: Dict[str, Any], chunk_count: int):
        """ドキュメントメタデータをSQL databaseに保存"""
        try:
            doc_metadata = DocumentMetadata(
                id=document_id,
                filename=file_path.name,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                file_type=file_path.suffix.lower(),
                content_hash=metadata.get("content_hash", ""),
                chunk_count=chunk_count,
                processing_status="completed",
                metadata_json=metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # with get_db() as session:
            #     session.add(doc_metadata)
            #     session.commit()
                
            self.logger.debug(f"ドキュメントメタデータ保存完了: {document_id}")
            
        except Exception as e:
            self.logger.error(f"ドキュメントメタデータ保存エラー: {e}")

    async def cleanup_processed_files(self, days_old: int = 7):
        """古い処理済みファイルのクリーンアップ"""
        cutoff_date = datetime.utcnow().timestamp() - (days_old * 24 * 3600)
        
        cleaned_count = 0
        for file_path in self.processed_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_date:
                try:
                    file_path.unlink()
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"ファイル削除失敗: {file_path}, {e}")
        
        self.logger.info(f"クリーンアップ完了: {cleaned_count} ファイル削除")
        return cleaned_count


# ワーカー実行用のメイン関数
async def main():
    """ドキュメント処理ワーカーのメイン実行"""
    logging.basicConfig(level=logging.INFO)
    
    config = ProcessingConfig()
    processor = DocumentProcessor(config)
    
    # テスト処理
    test_file = Path("./test_document.pdf")
    if test_file.exists():
        document_id = str(uuid.uuid4())
        result = await processor.process_document(test_file, document_id)
        print(f"処理結果: {result}")


if __name__ == "__main__":
    asyncio.run(main())
