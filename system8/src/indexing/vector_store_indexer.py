"""
ベクターストア実装（Milvus, Qdrant, Chroma等）
"""
from typing import List, Dict, Any, Optional
from . import BaseIndexer, IndexConfig

class VectorStoreIndexer(BaseIndexer):
    """ベクターストアを使用したインデクサー"""
    
    def __init__(self, config: IndexConfig):
        super().__init__(config)
        self.vector_store = None
        self._setup_vector_store()
        
    def _setup_vector_store(self):
        """ベクターストアをセットアップ"""
        vector_config = self.config.vector_store_config
        store_type = vector_config.get('type', 'chroma')
        
        if store_type == 'milvus':
            self._setup_milvus(vector_config)
        elif store_type == 'qdrant':
            self._setup_qdrant(vector_config)
        elif store_type == 'chroma':
            self._setup_chroma(vector_config)
        else:
            self._setup_simple_vector_store(vector_config)
    
    def _setup_milvus(self, config: Dict[str, Any]):
        """Milvusベクターストアをセットアップ"""
        try:
            from llama_index.vector_stores.milvus import MilvusVectorStore
            
            self.vector_store = MilvusVectorStore(
                uri=config.get('uri', 'http://localhost:19530'),
                collection_name=config.get('collection_name', 'rag_documents'),
                dim=config.get('dimension', 768),
                overwrite=config.get('overwrite', False)
            )
            print("Milvus vector store configured")
            
        except Exception as e:
            print(f"Failed to setup Milvus: {e}")
            self._setup_simple_vector_store(config)
    
    def _setup_qdrant(self, config: Dict[str, Any]):
        """Qdrantベクターストアをセットアップ"""
        try:
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            import qdrant_client
            
            client = qdrant_client.QdrantClient(
                url=config.get('url', 'http://localhost:6333'),
                api_key=config.get('api_key')
            )
            
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name=config.get('collection_name', 'rag_documents')
            )
            print("Qdrant vector store configured")
            
        except Exception as e:
            print(f"Failed to setup Qdrant: {e}")
            self._setup_simple_vector_store(config)
    
    def _setup_chroma(self, config: Dict[str, Any]):
        """Chromaベクターストアをセットアップ"""
        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb
            
            chroma_client = chromadb.PersistentClient(
                path=config.get('persist_dir', './chroma_db')
            )
            
            chroma_collection = chroma_client.get_or_create_collection(
                config.get('collection_name', 'rag_documents')
            )
            
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            print("Chroma vector store configured")
            
        except Exception as e:
            print(f"Failed to setup Chroma: {e}")
            self._setup_simple_vector_store(config)
    
    def _setup_simple_vector_store(self, config: Dict[str, Any]):
        """シンプルベクターストアをセットアップ"""
        try:
            from llama_index.vector_stores.simple import SimpleVectorStore
            
            self.vector_store = SimpleVectorStore()
            print("Simple vector store configured")
            
        except Exception as e:
            print(f"Failed to setup simple vector store: {e}")
            self.vector_store = None
    
    def create_index(self, documents: List[Any]) -> Any:
        """ベクターインデックスを作成"""
        if self.vector_store is None:
            print("Vector store not configured")
            return None
            
        try:
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.core import Document
            
            # ストレージコンテキストを作成
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # ドキュメントの変換
            llama_docs = []
            for doc in documents:
                if isinstance(doc, dict):
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    llama_docs.append(Document(text=content, metadata=metadata))
                elif hasattr(doc, 'content'):
                    llama_docs.append(Document(text=doc.content, metadata=doc.metadata))
                else:
                    llama_docs.append(Document(text=str(doc)))
            
            # インデックス作成
            self.index = VectorStoreIndex.from_documents(
                llama_docs,
                storage_context=storage_context
            )
            
            print(f"Created vector store index with {len(llama_docs)} documents")
            return self.index
            
        except Exception as e:
            print(f"Failed to create vector store index: {e}")
            return None
    
    def load_index(self, storage_path: str) -> Any:
        """既存のインデックスをロード"""
        try:
            from llama_index.core import VectorStoreIndex, StorageContext
            
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=storage_path
            )
            
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context
            )
            
            print(f"Loaded vector store index from {storage_path}")
            return self.index
            
        except Exception as e:
            print(f"Failed to load vector store index: {e}")
            return None
    
    def save_index(self, storage_path: str) -> bool:
        """インデックスを保存"""
        if self.index is None:
            print("No index to save")
            return False
            
        try:
            import os
            os.makedirs(storage_path, exist_ok=True)
            self.index.storage_context.persist(persist_dir=storage_path)
            print(f"Vector store index saved to {storage_path}")
            return True
            
        except Exception as e:
            print(f"Failed to save vector store index: {e}")
            return False
    
    def add_documents(self, documents: List[Any]) -> bool:
        """文書をインデックスに追加"""
        if self.index is None:
            print("No index available")
            return False
            
        try:
            from llama_index.core import Document
            
            # ドキュメントの変換
            llama_docs = []
            for doc in documents:
                if isinstance(doc, dict):
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    llama_docs.append(Document(text=content, metadata=metadata))
                elif hasattr(doc, 'content'):
                    llama_docs.append(Document(text=doc.content, metadata=doc.metadata))
                else:
                    llama_docs.append(Document(text=str(doc)))
            
            # 文書を追加
            for doc in llama_docs:
                self.index.insert(doc)
            
            print(f"Added {len(llama_docs)} documents to index")
            return True
            
        except Exception as e:
            print(f"Failed to add documents: {e}")
            return False

class MultiVectorStoreIndexer:
    """複数のベクターストアを統合したインデクサー"""
    
    def __init__(self, configs: List[IndexConfig]):
        self.indexers = []
        for config in configs:
            indexer = VectorStoreIndexer(config)
            self.indexers.append(indexer)
    
    def create_indices(self, documents: List[Any]) -> List[Any]:
        """複数のインデックスを作成"""
        indices = []
        for indexer in self.indexers:
            index = indexer.create_index(documents)
            indices.append(index)
        return indices
    
    def search_all(self, query: str, top_k: int = 10) -> Dict[str, List[Any]]:
        """全てのインデックスから検索"""
        results = {}
        for i, indexer in enumerate(self.indexers):
            if indexer.index is not None:
                try:
                    query_engine = indexer.index.as_query_engine(similarity_top_k=top_k)
                    response = query_engine.query(query)
                    results[f"index_{i}"] = response
                except Exception as e:
                    print(f"Search failed for index {i}: {e}")
                    results[f"index_{i}"] = None
        return results