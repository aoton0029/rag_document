from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import SimpleDirectoryReader
from llama_index.extractors.entity import EntityExtractor
from core.database import db_manager
from config.settings import settings
from typing import List, Dict, Any
import uuid
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentIngestService:
    def __init__(self):
        self.embedding_model = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
        self.node_parser = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.entity_extractor = EntityExtractor()
    
    async def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """ドキュメント処理のメインフロー"""
        doc_id = str(uuid.uuid4())
        
        try:
            # 1. ファイル受け取りと初期レコード作成
            documents = await self._load_documents(file_path, doc_id, metadata)
            
            # 2. 前処理
            processed_docs = await self._preprocess_documents(documents)
            
            # 3. チャンク化
            chunks = await self._chunk_documents(processed_docs)
            
            # 4. エンティティ抽出（Neo4j用）
            await self._extract_entities(chunks)
            
            # 5. 埋め込み生成とベクトルDB登録
            await self._generate_embeddings_and_store(chunks, doc_id)
            
            # 6. インデックス更新
            await self._update_index(chunks)
            
            logger.info(f"Document {doc_id} processed successfully")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            await self._update_processing_status(doc_id, "failed")
            raise
    
    async def _load_documents(self, file_path: str, doc_id: str, metadata: Dict[str, Any]) -> List[Document]:
        """ファイル読み込みと初期メタデータ設定"""
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        
        # ドキュメントにメタデータを付与
        for doc in documents:
            doc.metadata.update({
                "doc_id": doc_id,
                "source_uri": file_path,
                "processing_status": "processing",
                "created_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            })
            
            # チェックサムを計算
            content_hash = hashlib.sha256(doc.text.encode()).hexdigest()
            doc.metadata["checksum"] = content_hash
        
        return documents
    
    async def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """前処理（言語検出、ノイズ除去等）"""
        processed_docs = []
        
        for doc in documents:
            # 基本的なテキストクリーニング
            cleaned_text = doc.text.strip()
            
            # 言語検出（簡易版）
            if any(ord(char) > 127 for char in cleaned_text[:100]):
                doc.metadata["language"] = "ja"
            else:
                doc.metadata["language"] = "en"
            
            doc.text = cleaned_text
            processed_docs.append(doc)
        
        return processed_docs
    
    async def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """ドキュメントのチャンク化"""
        chunks = []
        
        for doc in documents:
            doc_chunks = self.node_parser.get_nodes_from_documents([doc])
            
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "doc_id": doc.metadata["doc_id"]
                })
                chunks.append(chunk)
        
        return chunks
    
    async def _extract_entities(self, chunks: List[Document]):
        """エンティティ抽出（Neo4j連携用）"""
        try:
            for chunk in chunks:
                entities = await self.entity_extractor.aextract([chunk])
                chunk.metadata["entities"] = [entity.label for entity in entities]
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
    
    async def _generate_embeddings_and_store(self, chunks: List[Document], doc_id: str):
        """埋め込み生成とベクトルストア保存"""
        storage_context = db_manager.get_storage_context()
        
        # VectorStoreIndexを作成してチャンクを保存
        index = VectorStoreIndex.from_documents(
            chunks,
            storage_context=storage_context,
            embed_model=self.embedding_model
        )
        
        # 処理状況を更新
        await self._update_processing_status(doc_id, "completed")
    
    async def _update_index(self, chunks: List[Document]):
        """インデックス情報の更新"""
        # Redis Index Storeに情報をキャッシュ
        storage_context = db_manager.get_storage_context()
        index_store = storage_context.index_store
        
        for chunk in chunks:
            cache_key = f"chunk:{chunk.metadata['chunk_id']}"
            await index_store.aput(cache_key, chunk.metadata)
    
    async def _update_processing_status(self, doc_id: str, status: str):
        """処理状況の更新"""
        try:
            storage_context = db_manager.get_storage_context()
            kvstore = storage_context.kvstore
            
            status_data = {
                "doc_id": doc_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            await kvstore.aput(f"doc_status:{doc_id}", status_data)
        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")
