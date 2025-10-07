from pydantic import BaseModel
from typing import Dict, Any

class DbSettings(BaseModel):
    # Milvus Settings
    milvus_url: str = "http://milvus:19530"
    milvus_host: str = "milvus"
    milvus_port: int = 19530

    # MongoDB Settings
    mongodb_url: str = "mongodb://admin:pdntsPa0@mongodb:27017"
    mongodb_username: str = 'admin'
    mongodb_password: str = 'pdntsPa0'

    # Neo4j Settings
    neo4j_url: str = "neo4j://neo4j:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "p@ssw0rd"
    
    # Redis Settings
    redis_url:str = "redis://:pdntsPa0@redis:6379/0"
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = "pdntsPa0"
    redis_cache_ttl: int = 3600
    
    # LLM Settings
    ollama_base_url: str = "http://ollama:11434"



settings = DbSettings()
