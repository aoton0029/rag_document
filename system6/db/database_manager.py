from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from configs.processing_config import ProcessingConfig
from configs.db_settings import settings
import logging
from db.document_db.mongo_client import MongoClient
from db.graph_db.neo4j_client import Neo4jClient
from db.keyvalue_db.redis_client import RedisClient
from db.vector_db.milvus_client import MilvusClient
from schemas.base_collection import BaseCollection
from llm.ollama_connector import OllamaConnector

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.vector_store: MilvusVectorStore = None
        self.docstore: MongoDocumentStore = None
        self.kvstore: RedisKVStore = None
        self.index_store: RedisIndexStore = None
        self.graph_store: Neo4jGraphStore = None
        self.storage_context: StorageContext = None
        self.mongo: MongoClient = None
        self.redis: RedisClient = None
        self.neo4j: Neo4jClient = None
        self.milvus: MilvusClient = None

    def initialize_connections(self, config:ProcessingConfig, ollama:OllamaConnector, reset:bool=False):
        """すべてのデータベース接続を初期化"""
        try:
            # Milvus Vector Store
            self.vector_store = MilvusVectorStore(
                uri=settings.milvus_url,
                dim=ollama.embedding_model.vector_dim,
                embedding_field='embedding',
                collection_name=config.milvus_collection_name,
                index_config={
                    "index_type": config.milvus_index_type,
                    "metric_type": config.milvus_metric_type
                }
            )

            self.milvus = MilvusClient(
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            
            # MongoDB Document Store
            self.docstore = MongoDocumentStore.from_uri(
                uri=settings.mongodb_url,
                db_name=config.mongodb_database
            )

            self.mongo = MongoClient(
                connection_string=settings.mongodb_url,
                database_name=config.mongodb_database
            )
            
            # Redis Client
            self.redis = RedisClient(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password
            )

            # Redis KV Store
            self.kvstore = RedisKVStore(redis_client=self.redis.client)

            # Redis Index Store
            self.index_store = RedisIndexStore(redis_kvstore=self.kvstore)
            
            # Neo4j Client
            self.neo4j = Neo4jClient(
                uri=settings.neo4j_url,
                user=settings.neo4j_username,
                password=settings.neo4j_password
            )

            # Neo4j Graph Store
            self.graph_store = Neo4jGraphStore(
                url=settings.neo4j_url,
                username=settings.neo4j_username,
                password=settings.neo4j_password
            )
            
            # リセット
            if reset:
                logger.info("Resetting all databases...")
                self.milvus.reset_collection(
                    config.milvus_collection_name, 
                    'embedding', 
                    BaseCollection.to_fields(ollama.embedding_model.vector_dim))
                self.mongo.clear_database()
                self.redis.clear_database()
                self.neo4j.clear_database()
                logger.info("All databases have been reset.")

            # Storage Context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                docstore=self.docstore,
                index_store=self.index_store,
                graph_store=self.graph_store
            )
            
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    def get_storage_context(self) -> StorageContext:
        if self.storage_context is None:
            raise ValueError("Storage context is not initialized. Call initialize_connections() first.")
        return self.storage_context
    
# Global instance
db_manager = DatabaseManager()
