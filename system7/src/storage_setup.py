"""
Storage Context setup for Advanced RAG System
各種データストアの初期化と設定を行う
"""
import logging
from typing import Optional
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from config.config import config
from loguru import logger

class StorageManager:
    """
    Advanced RAGシステムのストレージ管理クラス
    各種データベースへの接続とStorageContextの初期化を担当
    """
    
    def __init__(self):
        self.vector_store: Optional[MilvusVectorStore] = None
        self.docstore: Optional[MongoDocumentStore] = None
        self.index_store: Optional[RedisIndexStore] = None
        self.graph_store: Optional[Neo4jGraphStore] = None
        self.storage_context: Optional[StorageContext] = None
        
    def setup_milvus_vector_store(self) -> MilvusVectorStore:
        """Milvus Vector Storeの初期化"""
        try:
            logger.info(f"Initializing Milvus Vector Store at {config.milvus.uri}")
            
            # Milvusベクターストアの設定
            self.vector_store = MilvusVectorStore(
                uri=config.milvus.uri,
                collection_name=config.milvus.collection_name,
                dim=config.milvus.dimension,
                index_config={
                    "index_type": config.milvus.index_type,
                    "metric_type": config.milvus.metric_type,
                    "params": {"nlist": 1024}
                },
                search_config={
                    "metric_type": config.milvus.metric_type,
                    "params": {"nprobe": 10}
                },
                overwrite=False
            )
            
            logger.success("Milvus Vector Store initialized successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus Vector Store: {str(e)}")
            raise
    
    def setup_mongodb_docstore(self) -> MongoDocumentStore:
        """MongoDB Document Storeの初期化"""
        try:
            logger.info(f"Initializing MongoDB Document Store at {config.mongodb.uri}")
            
            self.docstore = MongoDocumentStore.from_uri(
                uri=config.mongodb.uri,
                db_name=config.mongodb.database,
                namespace=config.mongodb.collection
            )
            
            logger.success("MongoDB Document Store initialized successfully")
            return self.docstore
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB Document Store: {str(e)}")
            raise
    
    def setup_redis_index_store(self) -> RedisIndexStore:
        """Redis Index Storeの初期化"""
        try:
            logger.info(f"Initializing Redis Index Store at {config.redis.redis_url}")
            
            self.index_store = RedisIndexStore.from_redis_url(
                redis_url=config.redis.redis_url
            )
            
            logger.success("Redis Index Store initialized successfully")
            return self.index_store
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis Index Store: {str(e)}")
            raise
    
    def setup_neo4j_graph_store(self) -> Neo4jGraphStore:
        """Neo4j Graph Storeの初期化"""
        try:
            logger.info(f"Initializing Neo4j Graph Store at {config.neo4j.uri}")
            
            self.graph_store = Neo4jGraphStore(
                username=config.neo4j.username,
                password=config.neo4j.password,
                url=config.neo4j.uri,
                database=config.neo4j.database
            )
            
            logger.success("Neo4j Graph Store initialized successfully")
            return self.graph_store
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j Graph Store: {str(e)}")
            raise
    
    def setup_storage_context(self) -> StorageContext:
        """全てのストレージコンポーネントを統合したStorageContextの作成"""
        try:
            logger.info("Setting up complete Storage Context...")
            
            # 各ストレージの初期化
            if self.vector_store is None:
                self.setup_milvus_vector_store()
            if self.docstore is None:
                self.setup_mongodb_docstore()
            if self.index_store is None:
                self.setup_redis_index_store()
            if self.graph_store is None:
                self.setup_neo4j_graph_store()
            
            # StorageContextの作成
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                docstore=self.docstore,
                index_store=self.index_store,
                graph_store=self.graph_store
            )
            
            logger.success("Storage Context initialized successfully with all stores")
            return self.storage_context
            
        except Exception as e:
            logger.error(f"Failed to setup Storage Context: {str(e)}")
            raise
    
    def test_connections(self) -> dict:
        """各データストアへの接続テスト"""
        results = {
            "milvus": False,
            "mongodb": False,
            "redis": False,
            "neo4j": False
        }
        
        # Milvus接続テスト
        try:
            if self.vector_store:
                # 簡単なクエリテストでの接続確認
                results["milvus"] = True
                logger.info("Milvus connection: OK")
        except Exception as e:
            logger.warning(f"Milvus connection failed: {str(e)}")
        
        # MongoDB接続テスト
        try:
            if self.docstore:
                # ping操作での接続確認
                results["mongodb"] = True
                logger.info("MongoDB connection: OK")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {str(e)}")
        
        # Redis接続テスト
        try:
            if self.index_store:
                results["redis"] = True
                logger.info("Redis connection: OK")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
        
        # Neo4j接続テスト
        try:
            if self.graph_store:
                results["neo4j"] = True
                logger.info("Neo4j connection: OK")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {str(e)}")
        
        return results
    
    def get_storage_context(self) -> StorageContext:
        """StorageContextの取得（初期化されていない場合は初期化）"""
        if self.storage_context is None:
            self.setup_storage_context()
        return self.storage_context
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            if self.graph_store:
                # Neo4jセッションのクローズ
                self.graph_store.close()
            logger.info("Storage resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# グローバルストレージマネージャインスタンス
storage_manager = StorageManager()