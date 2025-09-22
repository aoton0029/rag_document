# File: /unified-id-indexing-system/unified-id-indexing-system/src/config/settings.py

import os

class Settings:
    def __init__(self):
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.mongodb_database = os.getenv("MONGODB_DATABASE", "unified_id_db")
        self.neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = os.getenv("REDIS_PORT", 6379)
        self.redis_db = os.getenv("REDIS_DB", 0)
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = os.getenv("MILVUS_PORT", 19530)
        self.milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME", "unified_id_collection")
        self.milvus_index_type = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
        self.milvus_metric_type = os.getenv("MILVUS_METRIC_TYPE", "L2")

settings = Settings()