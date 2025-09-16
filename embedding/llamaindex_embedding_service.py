import asyncio
import torch
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import BaseNode, TextNode, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.tools.neo4j import Neo4jQueryToolSpec

from ..llms.ollama_connector import OllamaConnector
from ..database.vector_db.milvus_client import MilvusClient
from ..database.graph_db.neo4j_client import Neo4jClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.document_db.mongo_client import MongoClient


@dataclass
class EmbeddingConfig:
    """埋め込み設定クラス"""
    model_name: str = "nomic-embed-text"
    embedding_dim: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32


class EmbeddingService:
    """
    統合埋め込みサービス
    """
    
    def __init__(
        self,
        config: EmbeddingConfig,
        ollama_connector: OllamaConnector = None
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # モデル管理
        self.embedding_models: Dict[str, BaseEmbedding] = {}

    def add_embedding_model(self, model_name: str) -> BaseEmbedding:
        """新しい埋め込みモデルを追加"""
        try:
            embedding_model = self.ollama_connector.initialize_embedding(model_name)
            self.embedding_models[model_name] = embedding_model
            self.logger.info(f"埋め込みモデル '{model_name}' を追加しました")
            return embedding_model
        except Exception as e:
            self.logger.error(f"埋め込みモデル追加エラー: {e}")
            raise
        
    def embed_text(self, texts: Union[str, List[str]], model_name: Optional[str] = None) -> Union[List[float], List[List[float]]]:
        """テキストの埋め込みを取得"""
        model_name = model_name or self.config.model_name
        
        if model_name not in self.embedding_models:
            self.add_embedding_model(model_name)
        
        embedding_model = self.embedding_models[model_name]
        
        try:
            if isinstance(texts, str):
                # 単一テキストの場合
                embeddings = embedding_model.get_text_embedding(texts)
                return embeddings
            else:
                # 複数テキストの場合
                embeddings = []
                for text in texts:
                    embedding = embedding_model.get_text_embedding(text)
                    embeddings.append(embedding)
                return embeddings
        except Exception as e:
            self.logger.error(f"テキスト埋め込み生成エラー: {e}")
            raise
    
    async def embed_text_async(self, texts: Union[str, List[str]], model_name: Optional[str] = None) -> Union[List[float], List[List[float]]]:
        """非同期でテキストの埋め込みを取得"""
        return await asyncio.to_thread(self.embed_text, texts, model_name)
    
    def create_nodes_from_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextNode]:
        """テキストからノードを作成"""
        try:
            # ドキュメント作成
            document = Document(text=text, metadata=metadata or {})
            
            # ノードに分割
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            self.logger.info(f"テキストから {len(nodes)} 個のノードを作成しました")
            return nodes
        except Exception as e:
            self.logger.error(f"ノード作成エラー: {e}")
            raise
    
    def create_nodes_from_documents(self, documents: List[Document]) -> List[TextNode]:
        """ドキュメントリストからノードを作成"""
        try:
            nodes = self.node_parser.get_nodes_from_documents(documents)
            self.logger.info(f"{len(documents)} 個のドキュメントから {len(nodes)} 個のノードを作成しました")
            return nodes
        except Exception as e:
            self.logger.error(f"ドキュメントからのノード作成エラー: {e}")
            raise
    
    def embed_nodes(self, nodes: List[BaseNode], model_name: Optional[str] = None) -> List[BaseNode]:
        """ノードに埋め込みベクトルを追加"""
        model_name = model_name or self.config.model_name
        
        if model_name not in self.embedding_models:
            self.add_embedding_model(model_name)
        
        embedding_model = self.embedding_models[model_name]
        
        try:
            # バッチ処理で埋め込み生成
            for i in range(0, len(nodes), self.config.batch_size):
                batch_nodes = nodes[i:i + self.config.batch_size]
                
                for node in batch_nodes:
                    if node.text:
                        embedding = embedding_model.get_text_embedding(node.text)
                        node.embedding = embedding
            
            self.logger.info(f"{len(nodes)} 個のノードに埋め込みを追加しました")
            return nodes
        except Exception as e:
            self.logger.error(f"ノード埋め込み生成エラー: {e}")
            raise
    
    async def embed_nodes_async(self, nodes: List[BaseNode], model_name: Optional[str] = None) -> List[BaseNode]:
        """非同期でノードに埋め込みベクトルを追加"""
        return await asyncio.to_thread(self.embed_nodes, nodes, model_name)
    
    def process_document(self, text: str, metadata: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None) -> List[BaseNode]:
        """ドキュメントを処理してノードと埋め込みを生成"""
        try:
            # ノード作成
            nodes = self.create_nodes_from_text(text, metadata)
            
            # 埋め込み生成
            embedded_nodes = self.embed_nodes(nodes, model_name)
            
            self.logger.info(f"ドキュメント処理完了: {len(embedded_nodes)} 個のノードを生成")
            return embedded_nodes
        except Exception as e:
            self.logger.error(f"ドキュメント処理エラー: {e}")
            raise
    
    async def process_document_async(self, text: str, metadata: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None) -> List[BaseNode]:
        """非同期でドキュメントを処理"""
        return await asyncio.to_thread(self.process_document, text, metadata, model_name)
    
    def get_embedding_dimension(self, model_name: Optional[str] = None) -> int:
        """埋め込み次元数を取得"""
        model_name = model_name or self.config.model_name
        
        if model_name not in self.embedding_models:
            self.add_embedding_model(model_name)
        
        # サンプルテキストで次元数を確認
        sample_embedding = self.embed_text("sample", model_name)
        return len(sample_embedding)
    
    def get_available_models(self) -> List[str]:
        """利用可能な埋め込みモデル一覧を取得"""
        try:
            return self.ollama_connector.get_available_models()
        except Exception as e:
            self.logger.error(f"モデル一覧取得エラー: {e}")
            return list(self.embedding_models.keys())
    
    def check_model_availability(self, model_name: str) -> bool:
        """モデルの利用可能性を確認"""
        available_models = self.get_available_models()
        return model_name in available_models
    
    def similarity_search(self, query: str, nodes: List[BaseNode], top_k: int = 5, model_name: Optional[str] = None) -> List[BaseNode]:
        """類似度検索を実行"""
        model_name = model_name or self.config.model_name
        
        try:
            # クエリの埋め込み生成
            query_embedding = self.embed_text(query, model_name)
            
            # 各ノードとの類似度計算
            similarities = []
            for node in nodes:
                if hasattr(node, 'embedding') and node.embedding:
                    similarity = self._cosine_similarity(query_embedding, node.embedding)
                    similarities.append((node, similarity))
            
            # 類似度でソート
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # トップk個を返す
            return [node for node, _ in similarities[:top_k]]
        except Exception as e:
            self.logger.error(f"類似度検索エラー: {e}")
            raise
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """コサイン類似度を計算"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            "loaded_models": list(self.embedding_models.keys()),
            "default_model": self.config.model_name,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "batch_size": self.config.batch_size,
            "embedding_dimension": self.config.embedding_dim
        }      