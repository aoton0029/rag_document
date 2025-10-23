"""
インデキシングモジュールの基底クラスとLlamaIndex連携
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os

@dataclass
class IndexConfig:
    """インデックス設定のデータクラス"""
    index_type: str
    storage_path: str
    vector_store_config: Dict[str, Any]
    embedding_config: Dict[str, Any]
    chunk_config: Dict[str, Any]
    metadata: Dict[str, Any] = None

class BaseIndexer(ABC):
    """インデクサーの基底クラス"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.index = None
        
    @abstractmethod
    def create_index(self, documents: List[Any]) -> Any:
        """インデックスを作成"""
        pass
    
    @abstractmethod
    def load_index(self, storage_path: str) -> Any:
        """既存のインデックスをロード"""
        pass
    
    @abstractmethod
    def save_index(self, storage_path: str) -> bool:
        """インデックスを保存"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Any]) -> bool:
        """文書をインデックスに追加"""
        pass

class LlamaIndexManager:
    """LlamaIndexを使用したインデックス管理"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.index = None
        self.service_context = None
        self._setup_service_context()
        
    def _setup_service_context(self):
        """ServiceContextをセットアップ"""
        try:
            from llama_index.core import Settings
            from llama_index.core.node_parser import SentenceSplitter
            
            # 埋め込みモデルの設定
            embedding_config = self.config.embedding_config
            embedding_model = self._create_embedding_model(embedding_config)
            
            # LLMの設定
            llm = self._create_llm()
            
            # チャンクパーサーの設定
            chunk_config = self.config.chunk_config
            node_parser = SentenceSplitter(
                chunk_size=chunk_config.get('chunk_size', 1024),
                chunk_overlap=chunk_config.get('chunk_overlap', 50)
            )
            
            # グローバル設定
            Settings.llm = llm
            Settings.embed_model = embedding_model
            Settings.node_parser = node_parser
            
            print("LlamaIndex ServiceContext configured successfully")
            
        except Exception as e:
            print(f"Failed to setup ServiceContext: {e}")
    
    def _create_embedding_model(self, embedding_config: Dict[str, Any]):
        """埋め込みモデルを作成"""
        provider = embedding_config.get('provider', 'huggingface')
        model_name = embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        
        if provider == 'openai':
            from llama_index.embeddings.openai import OpenAIEmbedding
            api_key = os.getenv(embedding_config.get('api_key_env', 'OPENAI_API_KEY'))
            return OpenAIEmbedding(api_key=api_key, model=model_name)
            
        elif provider == 'huggingface':
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            return HuggingFaceEmbedding(model_name=model_name)
            
        elif provider == 'ollama':
            from llama_index.embeddings.ollama import OllamaEmbedding
            base_url = embedding_config.get('base_url', 'http://localhost:11434')
            return OllamaEmbedding(model_name=model_name, base_url=base_url)
            
        else:
            # デフォルトのHuggingFace
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            return HuggingFaceEmbedding()
    
    def _create_llm(self):
        """LLMを作成（クエリ処理用）"""
        try:
            from llama_index.llms.ollama import Ollama
            return Ollama(model="llama2", request_timeout=60.0)
        except:
            # フォールバック
            from llama_index.llms.mock import MockLLM
            return MockLLM()
    
    def create_vector_index(self, documents: List[Any]) -> Any:
        """ベクターインデックスを作成"""
        try:
            from llama_index.core import VectorStoreIndex
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
            
            # インデックス作成
            self.index = VectorStoreIndex.from_documents(llama_docs)
            print(f"Created vector index with {len(llama_docs)} documents")
            return self.index
            
        except Exception as e:
            print(f"Failed to create vector index: {e}")
            return None
    
    def create_graph_index(self, documents: List[Any]) -> Any:
        """グラフインデックスを作成"""
        try:
            from llama_index.core import KnowledgeGraphIndex
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
            
            # グラフインデックス作成
            self.index = KnowledgeGraphIndex.from_documents(llama_docs)
            print(f"Created knowledge graph index with {len(llama_docs)} documents")
            return self.index
            
        except Exception as e:
            print(f"Failed to create graph index: {e}")
            return None
    
    def create_tree_index(self, documents: List[Any]) -> Any:
        """ツリーインデックスを作成"""
        try:
            from llama_index.core import TreeIndex
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
            
            # ツリーインデックス作成
            self.index = TreeIndex.from_documents(llama_docs)
            print(f"Created tree index with {len(llama_docs)} documents")
            return self.index
            
        except Exception as e:
            print(f"Failed to create tree index: {e}")
            return None
    
    def save_index(self, storage_path: str) -> bool:
        """インデックスを保存"""
        if self.index is None:
            print("No index to save")
            return False
            
        try:
            os.makedirs(storage_path, exist_ok=True)
            self.index.storage_context.persist(persist_dir=storage_path)
            print(f"Index saved to {storage_path}")
            return True
        except Exception as e:
            print(f"Failed to save index: {e}")
            return False
    
    def load_index(self, storage_path: str, index_type: str = "vector") -> Any:
        """インデックスをロード"""
        try:
            from llama_index.core import StorageContext, load_index_from_storage
            
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            self.index = load_index_from_storage(storage_context)
            print(f"Index loaded from {storage_path}")
            return self.index
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            return None