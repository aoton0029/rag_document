from pydantic import BaseSettings
from typing import Dict, Any
import os

class Settings(BaseSettings):
    # Milvus Settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "document_vectors"
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "COSINE"
    
    # MongoDB Settings
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "rag_system"
    
    # Neo4j Settings
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    
    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_cache_ttl: int = 3600
    
    # LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama2"
    embedding_model: str = "nomic-embed-text"
    
    # Chunking Settings
    chunk_size: int = 800
    chunk_overlap: int = 50
    
    # Search Settings
    top_k: int = 10
    similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()
