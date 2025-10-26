"""
データベース接続管理クラス
各種データベースクライアントを統合管理
"""

import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.graph_stores.types import GraphStore

from .mongodb_client import MongoDBClient
from .redis_client import RedisClient
from .milvus_client import MilvusClient
from .neo4j_client import Neo4jClient


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """データベース設定クラス"""
    # MongoDB設定
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_database: str = "rag_system"
    mongodb_username: Optional[str] = None
    mongodb_password: Optional[str] = None
    
    # Redis設定
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Milvus設定
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    milvus_collection: str = "rag_vectors"
    milvus_dim: int = 1536
    
    # Neo4j設定
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"


class DatabaseManager:
    """
    データベース管理クラス
    MongoDB、Redis、Milvus、Neo4jクライアントを統合管理
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        DatabaseManagerの初期化
        
        Args:
            config: データベース設定
        """
        self.config = config
        
        # クライアントインスタンス
        self._mongodb_client: Optional[MongoDBClient] = None
        self._redis_client: Optional[RedisClient] = None
        self._milvus_client: Optional[MilvusClient] = None
        self._neo4j_client: Optional[Neo4jClient] = None
        
        # ストアインスタンス
        self._docstore: Optional[BaseDocumentStore] = None
        self._index_store: Optional[BaseIndexStore] = None
        self._vector_store: Optional[VectorStore] = None
        self._graph_store: Optional[GraphStore] = None
        self._storage_context: Optional[StorageContext] = None
    
    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> 'DatabaseManager':
        """
        設定辞書からDatabaseManagerを作成
        
        Args:
            config_dict: 設定辞書
            
        Returns:
            DatabaseManager インスタンス
        """
        config = DatabaseConfig(**config_dict)
        return cls(config)
    
    def get_mongodb_client(self) -> MongoDBClient:
        """MongoDBクライアントを取得"""
        if self._mongodb_client is None:
            self._mongodb_client = MongoDBClient(
                host=self.config.mongodb_host,
                port=self.config.mongodb_port,
                database_name=self.config.mongodb_database,
                username=self.config.mongodb_username,
                password=self.config.mongodb_password
            )
        return self._mongodb_client
    
    def get_redis_client(self) -> RedisClient:
        """Redisクライアントを取得"""
        if self._redis_client is None:
            self._redis_client = RedisClient(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db
            )
        return self._redis_client
    
    def get_milvus_client(self) -> MilvusClient:
        """Milvusクライアントを取得"""
        if self._milvus_client is None:
            self._milvus_client = MilvusClient(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                user=self.config.milvus_user,
                password=self.config.milvus_password,
                collection_name=self.config.milvus_collection,
                dim=self.config.milvus_dim
            )
        return self._milvus_client
    
    def get_neo4j_client(self) -> Neo4jClient:
        """Neo4jクライアントを取得"""
        if self._neo4j_client is None:
            self._neo4j_client = Neo4jClient(
                uri=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
        return self._neo4j_client
    
    def get_docstore(
        self, 
        namespace: str = "default",
        collection_name: str = "documents"
    ) -> BaseDocumentStore:
        """
        ドキュメントストア（MongoDB）を取得
        
        Args:
            namespace: ドキュメントの名前空間
            collection_name: コレクション名
            
        Returns:
            BaseDocumentStore インスタンス
        """
        if self._docstore is None:
            mongodb_client = self.get_mongodb_client()
            self._docstore = mongodb_client.get_docstore(
                namespace=namespace,
                collection_name=collection_name
            )
        return self._docstore
    
    def get_index_store(
        self,
        namespace: str = "default",
        collection_suffix: str = "index_store"
    ) -> BaseIndexStore:
        """
        インデックスストア（Redis）を取得
        
        Args:
            namespace: インデックスの名前空間
            collection_suffix: コレクションサフィックス
            
        Returns:
            BaseIndexStore インスタンス
        """
        if self._index_store is None:
            redis_client = self.get_redis_client()
            self._index_store = redis_client.get_index_store(
                namespace=namespace,
                collection_suffix=collection_suffix
            )
        return self._index_store
    
    def get_vector_store(
        self,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> VectorStore:
        """
        ベクトルストア（Milvus）を取得
        
        Args:
            collection_name: コレクション名
            **kwargs: MilvusVectorStoreの追加パラメータ
            
        Returns:
            VectorStore インスタンス
        """
        if self._vector_store is None:
            milvus_client = self.get_milvus_client()
            self._vector_store = milvus_client.get_vector_store(
                collection_name=collection_name,
                **kwargs
            )
        return self._vector_store
    
    def get_graph_store(
        self,
        node_label: str = "Entity",
        rel_type: str = "RELATED",
        **kwargs
    ) -> GraphStore:
        """
        グラフストア（Neo4j）を取得
        
        Args:
            node_label: ノードラベル
            rel_type: リレーションシップタイプ
            **kwargs: Neo4jGraphStoreの追加パラメータ
            
        Returns:
            GraphStore インスタンス
        """
        if self._graph_store is None:
            neo4j_client = self.get_neo4j_client()
            self._graph_store = neo4j_client.get_graph_store(
                node_label=node_label,
                rel_type=rel_type,
                **kwargs
            )
        return self._graph_store
    
    def get_storage_context(
        self,
        docstore_namespace: str = "default",
        index_namespace: str = "default",
        vector_collection: Optional[str] = None,
        graph_node_label: str = "Entity",
        graph_rel_type: str = "RELATED"
    ) -> StorageContext:
        """
        StorageContextを構築
        
        Args:
            docstore_namespace: ドキュメントストアの名前空間
            index_namespace: インデックスストアの名前空間
            vector_collection: ベクトルストアのコレクション名
            graph_node_label: グラフストアのノードラベル
            graph_rel_type: グラフストアのリレーションシップタイプ
            
        Returns:
            StorageContext インスタンス
        """
        if self._storage_context is None:
            self._storage_context = StorageContext.from_defaults(
                docstore=self.get_docstore(namespace=docstore_namespace),
                index_store=self.get_index_store(namespace=index_namespace),
                vector_store=self.get_vector_store(collection_name=vector_collection),
                graph_store=self.get_graph_store(
                    node_label=graph_node_label,
                    rel_type=graph_rel_type
                )
            )
        return self._storage_context
    
    def connect_all(self) -> None:
        """全データベースに接続"""
        try:
            self.get_mongodb_client().connect()
            logger.info("MongoDB接続完了")
        except Exception as e:
            logger.error(f"MongoDB接続失敗: {e}")
        
        try:
            self.get_redis_client().connect()
            logger.info("Redis接続完了")
        except Exception as e:
            logger.error(f"Redis接続失敗: {e}")
        
        try:
            self.get_milvus_client().connect()
            logger.info("Milvus接続完了")
        except Exception as e:
            logger.error(f"Milvus接続失敗: {e}")
        
        try:
            self.get_neo4j_client().connect()
            logger.info("Neo4j接続完了")
        except Exception as e:
            logger.error(f"Neo4j接続失敗: {e}")
    
    def disconnect_all(self) -> None:
        """全データベースから切断"""
        if self._mongodb_client:
            self._mongodb_client.disconnect()
        if self._redis_client:
            self._redis_client.disconnect()
        if self._milvus_client:
            self._milvus_client.disconnect()
        if self._neo4j_client:
            self._neo4j_client.disconnect()
        
        # ストアインスタンスもクリア
        self._docstore = None
        self._index_store = None
        self._vector_store = None
        self._graph_store = None
        self._storage_context = None
        
        logger.info("全データベース接続を切断しました")
    
    def health_check_all(self) -> Dict[str, bool]:
        """全データベースのヘルスチェック"""
        health_status = {}
        
        try:
            health_status["mongodb"] = self.get_mongodb_client().health_check()
        except Exception as e:
            logger.error(f"MongoDB ヘルスチェックエラー: {e}")
            health_status["mongodb"] = False
        
        try:
            health_status["redis"] = self.get_redis_client().health_check()
        except Exception as e:
            logger.error(f"Redis ヘルスチェックエラー: {e}")
            health_status["redis"] = False
        
        try:
            health_status["milvus"] = self.get_milvus_client().health_check()
        except Exception as e:
            logger.error(f"Milvus ヘルスチェックエラー: {e}")
            health_status["milvus"] = False
        
        try:
            health_status["neo4j"] = self.get_neo4j_client().health_check()
        except Exception as e:
            logger.error(f"Neo4j ヘルスチェックエラー: {e}")
            health_status["neo4j"] = False
        
        return health_status
    
    def clear_all_data(self) -> None:
        """全データベースのデータをクリア（注意: 破壊的操作）"""
        logger.warning("全データベースのデータクリアを開始します")
        
        # MongoDB - コレクションを削除
        try:
            mongodb_client = self.get_mongodb_client()
            database = mongodb_client.get_database()
            for collection_name in database.list_collection_names():
                mongodb_client.drop_collection(collection_name)
            logger.info("MongoDBデータクリア完了")
        except Exception as e:
            logger.error(f"MongoDBデータクリアエラー: {e}")
        
        # Redis - 全キーを削除
        try:
            redis_client = self.get_redis_client()
            redis_client.flush_db()
            logger.info("Redisデータクリア完了")
        except Exception as e:
            logger.error(f"Redisデータクリアエラー: {e}")
        
        # Milvus - コレクションを削除
        try:
            milvus_client = self.get_milvus_client()
            milvus_client.drop_collection()
            logger.info("Milvusデータクリア完了")
        except Exception as e:
            logger.error(f"Milvusデータクリアエラー: {e}")
        
        # Neo4j - 全ノード・リレーションシップを削除
        try:
            neo4j_client = self.get_neo4j_client()
            neo4j_client.clear_database()
            logger.info("Neo4jデータクリア完了")
        except Exception as e:
            logger.error(f"Neo4jデータクリアエラー: {e}")
    
    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.connect_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.disconnect_all()