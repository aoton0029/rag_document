"""
インデクシング基底クラス
Milvus, MongoDB, Redis, Neo4j を使用したStorageContext設定とインデックス作成
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex, KnowledgeGraphIndex, DocumentSummaryIndex
from llama_index.core import Document, StorageContext, ServiceContext
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.docstores.mongodb_docstore import MongoDBDocumentStore
from llama_index.index_stores.redis_index_store import RedisIndexStore
from llama_index.graph_stores.neo4j_graph_store import Neo4jGraphStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

from ..utils import get_logger, performance_monitor
from ..embedding import create_embedder
from ..chunking import create_chunker


class IndexType(Enum):
    """インデックス種別"""
    VECTOR = "vector"
    SUMMARY = "summary" 
    TREE = "tree"
    KEYWORD = "keyword"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DOCUMENT_SUMMARY = "document_summary"


@dataclass
class IndexingResult:
    """インデクシング結果"""
    index: Any  # LlamaIndex index object
    index_type: IndexType
    document_count: int
    node_count: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        self.metadata.update({
            "index_type": self.index_type.value,
            "document_count": self.document_count,
            "node_count": self.node_count
        })


class StorageConfig:
    """ストレージ設定クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Vector Store設定 (Milvus)
        self.vector_store_config = config.get("vector_store", {})
        self.use_milvus = self.vector_store_config.get("use_milvus", True)
        
        # Document Store設定 (MongoDB)  
        self.docstore_config = config.get("docstore", {})
        self.use_mongodb = self.docstore_config.get("use_mongodb", True)
        
        # Index Store設定 (Redis)
        self.index_store_config = config.get("index_store", {})
        self.use_redis = self.index_store_config.get("use_redis", True)
        
        # Graph Store設定 (Neo4j)
        self.graph_store_config = config.get("graph_store", {})
        self.use_neo4j = self.graph_store_config.get("use_neo4j", False)


class BaseIndexer(ABC):
    """インデクサー基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"indexer_{self.__class__.__name__}")
        
        # ストレージ設定
        self.storage_config = StorageConfig(config.get("storage", {}))
        
        # 埋め込み設定
        self.embedding_config = config.get("embedding", {
            "provider": "ollama",
            "model_name": "qwen3-embedding:8b"
        })
        
        # チャンキング設定
        self.chunking_config = config.get("chunking", {
            "strategy": "semantic",
            "chunk_size": 1024
        })
        
        # StorageContext初期化
        self.storage_context = self._create_storage_context()
        
        # 埋め込みモデル初期化
        try:
            self.embedder = create_embedder(
                self.embedding_config["provider"],
                self.embedding_config
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embedder: {e}")
            self.embedder = None
        
        # チャンカー初期化
        try:
            self.chunker = create_chunker(
                self.chunking_config["strategy"],
                self.chunking_config
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize chunker: {e}")
            self.chunker = None
    
    def _create_storage_context(self) -> StorageContext:
        """StorageContextを作成"""
        
        # Vector Store
        vector_store = self._create_vector_store()
        
        # Document Store
        docstore = self._create_document_store()
        
        # Index Store
        index_store = self._create_index_store()
        
        # Graph Store (オプション)
        graph_store = None
        if self.storage_config.use_neo4j:
            graph_store = self._create_graph_store()
        
        # StorageContext作成
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
            index_store=index_store,
            graph_store=graph_store
        )
        
        return storage_context
    
    def _create_vector_store(self):
        """Vector Store (Milvus)を作成"""
        if self.storage_config.use_milvus:
            try:
                milvus_config = self.storage_config.vector_store_config
                
                vector_store = MilvusVectorStore(
                    uri=milvus_config.get("uri", "http://localhost:19530"),
                    collection_name=milvus_config.get("collection_name", "rag_collection"),
                    dim=milvus_config.get("dim", 1536),
                    overwrite=milvus_config.get("overwrite", False)
                )
                
                self.logger.info("Milvus vector store created",
                               collection=milvus_config.get("collection_name"))
                return vector_store
                
            except Exception as e:
                self.logger.error(f"Failed to create Milvus vector store: {e}")
                self.logger.warning("Falling back to SimpleVectorStore")
        
        # フォールバック
        return SimpleVectorStore()
    
    def _create_document_store(self):
        """Document Store (MongoDB)を作成"""
        if self.storage_config.use_mongodb:
            try:
                mongodb_config = self.storage_config.docstore_config
                
                docstore = MongoDBDocumentStore.from_uri(
                    uri=mongodb_config.get("uri", "mongodb://localhost:27017"),
                    db_name=mongodb_config.get("db_name", "rag_docstore"),
                    namespace=mongodb_config.get("namespace", "documents")
                )
                
                self.logger.info("MongoDB document store created",
                               db_name=mongodb_config.get("db_name"))
                return docstore
                
            except Exception as e:
                self.logger.error(f"Failed to create MongoDB document store: {e}")
                self.logger.warning("Falling back to SimpleDocumentStore")
        
        # フォールバック
        return SimpleDocumentStore()
    
    def _create_index_store(self):
        """Index Store (Redis)を作成"""
        if self.storage_config.use_redis:
            try:
                redis_config = self.storage_config.index_store_config
                
                index_store = RedisIndexStore.from_host_and_port(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    namespace=redis_config.get("namespace", "rag_index")
                )
                
                self.logger.info("Redis index store created",
                               host=redis_config.get("host"),
                               port=redis_config.get("port"))
                return index_store
                
            except Exception as e:
                self.logger.error(f"Failed to create Redis index store: {e}")
                self.logger.warning("Falling back to SimpleIndexStore")
        
        # フォールバック
        return SimpleIndexStore()
    
    def _create_graph_store(self):
        """Graph Store (Neo4j)を作成"""
        try:
            neo4j_config = self.storage_config.graph_store_config
            
            graph_store = Neo4jGraphStore(
                username=neo4j_config.get("username", "neo4j"),
                password=neo4j_config.get("password", "password"),
                url=neo4j_config.get("url", "bolt://localhost:7687"),
                database=neo4j_config.get("database", "neo4j")
            )
            
            self.logger.info("Neo4j graph store created",
                           url=neo4j_config.get("url"))
            return graph_store
            
        except Exception as e:
            self.logger.error(f"Failed to create Neo4j graph store: {e}")
            return None
    
    @abstractmethod
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> Any:
        """内部的なインデックス作成処理（実装必須）"""
        pass
    
    def create_index_from_documents(self, 
                                  documents: List[Document],
                                  index_id: Optional[str] = None,
                                  **kwargs) -> IndexingResult:
        """ドキュメントからインデックスを作成"""
        
        self.logger.info("Starting index creation from documents",
                        document_count=len(documents))
        
        with performance_monitor("document_indexing"):
            # ドキュメントをチャンク（ノード）に変換
            nodes = self._documents_to_nodes(documents)
            
            # インデックス作成
            index = self._create_index_internal(nodes, index_id=index_id, **kwargs)
            
            # 結果作成
            result = IndexingResult(
                index=index,
                index_type=self.get_index_type(),
                document_count=len(documents),
                node_count=len(nodes),
                metadata={
                    "chunking_strategy": self.chunking_config.get("strategy"),
                    "embedding_model": self.embedding_config.get("model_name"),
                    "index_id": index_id
                }
            )
        
        self.logger.info("Index creation completed",
                        index_type=self.get_index_type().value,
                        node_count=len(nodes))
        
        return result
    
    def create_index_from_files(self,
                               file_paths: List[Union[str, Path]],
                               index_id: Optional[str] = None,
                               **kwargs) -> IndexingResult:
        """ファイルからインデックスを作成"""
        
        # ファイルを読み込んでドキュメント化
        documents = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    document = Document(
                        text=content,
                        metadata={
                            "file_path": str(path),
                            "file_name": path.name,
                            "file_size": len(content)
                        }
                    )
                    documents.append(document)
                    
                except Exception as e:
                    self.logger.error(f"Failed to read file {path}: {e}")
            else:
                self.logger.warning(f"File not found: {path}")
        
        return self.create_index_from_documents(documents, index_id, **kwargs)
    
    def _documents_to_nodes(self, documents: List[Document]) -> List[BaseNode]:
        """ドキュメントをノードに変換"""
        all_nodes = []
        
        for doc in documents:
            if self.chunker:
                # チャンキング実行
                chunking_result = self.chunker.chunk_text(doc.text, doc.metadata)
                
                # チャンクをTextNodeに変換
                for i, chunk in enumerate(chunking_result.chunks):
                    node = TextNode(
                        text=chunk.text,
                        metadata={
                            **doc.metadata,
                            **chunk.metadata,
                            "doc_id": doc.doc_id,
                            "chunk_id": i,
                            "start_char_idx": chunk.start_idx,
                            "end_char_idx": chunk.end_idx
                        }
                    )
                    all_nodes.append(node)
            else:
                # チャンキングしない場合はそのままノード化
                node = TextNode(
                    text=doc.text,
                    metadata=doc.metadata or {}
                )
                all_nodes.append(node)
        
        return all_nodes
    
    def add_documents_to_index(self, 
                             index: Any,
                             documents: List[Document]) -> IndexingResult:
        """既存インデックスにドキュメントを追加"""
        
        nodes = self._documents_to_nodes(documents)
        
        try:
            # インデックスに追加
            for node in nodes:
                index.insert(node)
            
            self.logger.info("Documents added to index",
                           added_documents=len(documents),
                           added_nodes=len(nodes))
            
            return IndexingResult(
                index=index,
                index_type=self.get_index_type(),
                document_count=len(documents),
                node_count=len(nodes),
                metadata={"operation": "add_documents"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to index: {e}")
            raise
    
    def update_index(self, 
                    index: Any,
                    updated_documents: List[Document]) -> IndexingResult:
        """インデックスを更新"""
        
        # 既存ノードを削除してから新しいノードを追加
        # 実際の実装ではドキュメントIDベースの削除が必要
        
        nodes = self._documents_to_nodes(updated_documents)
        
        try:
            # 更新処理（簡易実装）
            for node in nodes:
                index.delete_ref_doc(node.metadata.get("doc_id", ""))
                index.insert(node)
            
            return IndexingResult(
                index=index,
                index_type=self.get_index_type(),
                document_count=len(updated_documents),
                node_count=len(nodes),
                metadata={"operation": "update_documents"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update index: {e}")
            raise
    
    def delete_from_index(self, 
                         index: Any,
                         document_ids: List[str]) -> bool:
        """インデックスからドキュメントを削除"""
        
        try:
            for doc_id in document_ids:
                index.delete_ref_doc(doc_id)
            
            self.logger.info("Documents deleted from index",
                           deleted_count=len(document_ids))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from index: {e}")
            return False
    
    def save_index(self, index: Any, index_id: str) -> bool:
        """インデックスを保存"""
        try:
            # StorageContextを使用してインデックスを永続化
            index.storage_context.persist(persist_dir=f"./storage/{index_id}")
            
            self.logger.info("Index saved", index_id=index_id)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index {index_id}: {e}")
            return False
    
    def load_index(self, index_id: str) -> Optional[Any]:
        """保存されたインデックスを読み込み"""
        try:
            # StorageContextを使用してインデックスを読み込み
            storage_context = StorageContext.from_defaults(
                persist_dir=f"./storage/{index_id}"
            )
            
            # インデックスタイプに応じた読み込み
            if self.get_index_type() == IndexType.VECTOR:
                index = VectorStoreIndex.from_documents([], storage_context=storage_context)
            else:
                # 他のタイプの実装
                pass
            
            self.logger.info("Index loaded", index_id=index_id)
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to load index {index_id}: {e}")
            return None
    
    @abstractmethod
    def get_index_type(self) -> IndexType:
        """インデックスタイプを取得（実装必須）"""
        pass
    
    def get_index_stats(self, index: Any) -> Dict[str, Any]:
        """インデックス統計を取得"""
        try:
            # 基本統計
            stats = {
                "index_type": self.get_index_type().value,
                "storage_context_available": self.storage_context is not None
            }
            
            # ベクトルストア統計
            if hasattr(index, 'vector_store'):
                vector_store = index.vector_store
                if hasattr(vector_store, 'get_collection_name'):
                    stats["vector_collection"] = vector_store.get_collection_name()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {}
    
    async def create_index_async(self, 
                               documents: List[Document],
                               index_id: Optional[str] = None,
                               **kwargs) -> IndexingResult:
        """非同期でインデックスを作成"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.create_index_from_documents,
            documents,
            index_id,
            **kwargs
        )


class VectorIndexer(BaseIndexer):
    """ベクトルインデクサー"""
    
    def get_index_type(self) -> IndexType:
        return IndexType.VECTOR
    
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> VectorStoreIndex:
        """ベクトルインデックスを作成"""
        
        # 埋め込み設定
        if self.embedder:
            # カスタム埋め込みを使用する場合の設定
            pass
        
        # VectorStoreIndexを作成
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            **kwargs
        )
        
        return index


class SummaryIndexer(BaseIndexer):
    """サマリーインデクサー"""
    
    def get_index_type(self) -> IndexType:
        return IndexType.SUMMARY
    
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> SummaryIndex:
        """サマリーインデックスを作成"""
        
        index = SummaryIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            **kwargs
        )
        
        return index


class TreeIndexer(BaseIndexer):
    """ツリーインデクサー"""
    
    def get_index_type(self) -> IndexType:
        return IndexType.TREE
    
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> TreeIndex:
        """ツリーインデックスを作成"""
        
        index = TreeIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            **kwargs
        )
        
        return index


class KeywordIndexer(BaseIndexer):
    """キーワードインデクサー"""
    
    def get_index_type(self) -> IndexType:
        return IndexType.KEYWORD
    
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> KeywordTableIndex:
        """キーワードインデックスを作成"""
        
        index = KeywordTableIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            **kwargs
        )
        
        return index


class KnowledgeGraphIndexer(BaseIndexer):
    """ナレッジグラフインデクサー"""
    
    def __init__(self, config: Dict[str, Any]):
        # Neo4j必須
        if not config.get("storage", {}).get("graph_store", {}).get("use_neo4j", False):
            config.setdefault("storage", {}).setdefault("graph_store", {})["use_neo4j"] = True
        
        super().__init__(config)
    
    def get_index_type(self) -> IndexType:
        return IndexType.KNOWLEDGE_GRAPH
    
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> KnowledgeGraphIndex:
        """ナレッジグラフインデックスを作成"""
        
        # Neo4j必須チェック
        if not self.storage_context.graph_store:
            raise ValueError("Knowledge Graph Index requires Neo4j graph store")
        
        index = KnowledgeGraphIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            **kwargs
        )
        
        return index


class DocumentSummaryIndexer(BaseIndexer):
    """ドキュメントサマリーインデクサー"""
    
    def get_index_type(self) -> IndexType:
        return IndexType.DOCUMENT_SUMMARY
    
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> DocumentSummaryIndex:
        """ドキュメントサマリーインデックスを作成"""
        
        index = DocumentSummaryIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            **kwargs
        )
        
        return index


class HybridIndexer(BaseIndexer):
    """ハイブリッドインデクサー（複数インデックスの組み合わせ）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 使用するインデックスタイプ
        self.index_types = config.get("index_types", ["vector", "keyword"])
        self.indexes = {}
    
    def get_index_type(self) -> IndexType:
        return IndexType.VECTOR  # プライマリタイプ
    
    def _create_index_internal(self, nodes: List[BaseNode], **kwargs) -> Dict[str, Any]:
        """複数のインデックスを作成"""
        
        indexes = {}
        
        for index_type in self.index_types:
            try:
                if index_type == "vector":
                    indexes["vector"] = VectorStoreIndex(
                        nodes=nodes,
                        storage_context=self.storage_context
                    )
                elif index_type == "summary":
                    indexes["summary"] = SummaryIndex(
                        nodes=nodes,
                        storage_context=self.storage_context
                    )
                elif index_type == "keyword":
                    indexes["keyword"] = KeywordTableIndex(
                        nodes=nodes,
                        storage_context=self.storage_context
                    )
                elif index_type == "tree":
                    indexes["tree"] = TreeIndex(
                        nodes=nodes,
                        storage_context=self.storage_context
                    )
                
                self.logger.info(f"Created {index_type} index")
                
            except Exception as e:
                self.logger.error(f"Failed to create {index_type} index: {e}")
        
        self.indexes = indexes
        return indexes
    
    def create_index_from_documents(self, 
                                  documents: List[Document],
                                  index_id: Optional[str] = None,
                                  **kwargs) -> Dict[str, IndexingResult]:
        """複数のインデックスを作成して結果を返す"""
        
        nodes = self._documents_to_nodes(documents)
        indexes = self._create_index_internal(nodes, **kwargs)
        
        results = {}
        for index_type, index in indexes.items():
            result = IndexingResult(
                index=index,
                index_type=IndexType(index_type) if index_type != "vector" else IndexType.VECTOR,
                document_count=len(documents),
                node_count=len(nodes),
                metadata={
                    "hybrid_indexer": True,
                    "index_type": index_type
                }
            )
            results[index_type] = result
        
        return results
