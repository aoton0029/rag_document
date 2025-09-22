from core.unified_id import UnifiedID
from core.correlation_id import CorrelationID
from core.global_sequence import GlobalSequence
from db.mongo_client import MongoClient
from db.neo4j_client import Neo4jClient
from db.redis_client import RedisClient
from db.milvus_client import MilvusClient
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.mongo_client = MongoClient()
        self.neo4j_client = Neo4jClient()
        self.redis_client = RedisClient()
        self.milvus_client = MilvusClient()
    
    def initialize_connections(self):
        try:
            self.mongo_client.connect()
            self.neo4j_client.connect()
            self.redis_client.connect()
            self.milvus_client.connect()
            logger.info("All database connections initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise

    def get_mongo_client(self):
        return self.mongo_client

    def get_neo4j_client(self):
        return self.neo4j_client

    def get_redis_client(self):
        return self.redis_client

    def get_milvus_client(self):
        return self.milvus_client

    def close_connections(self):
        self.mongo_client.close()
        self.neo4j_client.close()
        self.redis_client.close()
        self.milvus_client.close()
        logger.info("All database connections closed")