from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from settings import settings
import logging
from document_db.mongo_client import MongoClient
from graph_db.neo4j_client import Neo4jClient
from keyvalue_db.redis_client import RedisClient
from vector_db.milvus_client import MilvusClient


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
                port=settings.milvus_port
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
                db=settings.redis_db,
                password=settings.redis_password
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
    
    def verify_stored_data(self):
        """各データベースに保存されたデータを確認"""
        logger.info("=== データベース確認開始 ===")
        
        # Milvus (Vector Store) の確認
        self._verify_milvus_data()
        
        # MongoDB (Document Store) の確認
        self._verify_mongodb_data()
        
        # Redis (KV Store & Index Store) の確認
        self._verify_redis_data()
        
        # Neo4j (Graph Store) の確認
        self._verify_neo4j_data()
    
    def _verify_milvus_data(self):
        """Milvusのデータ確認"""
        try:
            logger.info("--- Milvus Vector Store 確認 ---")
            if self.milvus:
                collection_info = self.milvus.get_collection_info()
                logger.info(f"Collection: {collection_info}")
                
                # ベクトル数を確認
                vector_count = self.milvus.count_vectors()
                logger.info(f"保存されたベクトル数: {vector_count}")
        except Exception as e:
            logger.error(f"Milvus確認エラー: {e}")
    
    def _verify_mongodb_data(self):
        """MongoDBのデータ確認"""
        try:
            logger.info("--- MongoDB Document Store 確認 ---")
            if self.mongo:
                collections = self.mongo.list_collections()
                logger.info(f"Collections: {collections}")
                
                for collection_name in collections:
                    count = self.mongo.count_documents(collection_name)
                    logger.info(f"{collection_name}: {count} documents")
                    
                    # サンプルドキュメントを表示
                    if count > 0:
                        sample = self.mongo.find_documents(collection_name, limit=1)
                        if sample:
                            logger.info(f"Sample document keys: {list(sample[0].keys())}")
        except Exception as e:
            logger.error(f"MongoDB確認エラー: {e}")
    
    def _verify_redis_data(self):
        """Redisのデータ確認"""
        try:
            logger.info("--- Redis KV/Index Store 確認 ---")
            if self.redis:
                all_keys = self.redis.get_all_keys()
                logger.info(f"Redis keys 総数: {len(all_keys)}")
                
                # キーの種類別に分類
                index_keys = [k for k in all_keys if 'index' in k.lower()]
                other_keys = [k for k in all_keys if 'index' not in k.lower()]
                
                logger.info(f"Index関連キー: {len(index_keys)}")
                logger.info(f"その他のキー: {len(other_keys)}")
                
                # サンプルキーを表示
                if all_keys:
                    sample_keys = all_keys[:5]
                    logger.info(f"Sample keys: {sample_keys}")
        except Exception as e:
            logger.error(f"Redis確認エラー: {e}")
    
    def _verify_neo4j_data(self):
        """Neo4jのデータ確認"""
        try:
            logger.info("--- Neo4j Graph Store 確認 ---")
            if self.neo4j:
                node_count = self.neo4j.count_nodes()
                relationship_count = self.neo4j.count_relationships()
                
                logger.info(f"ノード数: {node_count}")
                logger.info(f"リレーション数: {relationship_count}")
                
                # ノードタイプを確認
                node_types = self.neo4j.get_node_labels()
                logger.info(f"ノードタイプ: {node_types}")
        except Exception as e:
            logger.error(f"Neo4j確認エラー: {e}")

# Global instance
db_manager = DatabaseManager()
