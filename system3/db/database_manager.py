from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from config.settings import settings
import logging
from document_db.mongo_client import MongoClient
from graph_db.neo4j_client import Neo4jClient
from keyvalue_db.redis_client import RedisClient
from vector_db.milvus_client import MilvusClient


logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.vector_store = None
        self.docstore = None
        self.kvstore = None
        self.index_store = None
        self.graph_store = None
        self.storage_context = None
        self.mongo = None
        self.redis = None
        self.neo4j = None
        self.milvus = None
    
    def initialize_connections(self):
        """すべてのデータベース接続を初期化"""
        try:
            # Milvus Vector Store
            self.vector_store = MilvusVectorStore(
                host=settings.milvus_host,
                port=settings.milvus_port,
                collection_name=settings.milvus_collection_name,
                index_config={
                    "index_type": settings.milvus_index_type,
                    "metric_type": settings.milvus_metric_type
                }
            )

            self.milvus = MilvusClient(
                host=settings.milvus_host,
                port=settings.milvus_port,
                collection_name=settings.milvus_collection_name
            )
            
            # MongoDB Document Store
            self.docstore = MongoDocumentStore.from_uri(
                uri=settings.mongodb_url,
                db_name=settings.mongodb_database
            )

            self.mongo = MongoClient(
                connection_string=settings.mongodb_url,
                database_name=settings.mongodb_database
            )
            
            # Redis Client
            self.redis = RedisClient(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db
            )

            # Redis KV Store
            self.kvstore = RedisKVStore(redis=self.redis.client)

            # Redis Index Store
            self.index_store = RedisIndexStore(redis=self.kvstore)
            
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
            self.initialize_connections()
        return self.storage_context

# Global instance
db_manager = DatabaseManager()
