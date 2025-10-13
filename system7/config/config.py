"""
Configuration settings for Advanced RAG System
各データベースとOllamaの接続設定
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class OllamaConfig:
    """Ollama LLM and Embedding configuration"""
    base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"  # または利用可能な任意のモデル
    embed_model: str = "nomic-embed-text"  # または mxbai-embed-large
    temperature: float = 0.1
    request_timeout: float = 300.0

@dataclass
class MilvusConfig:
    """Milvus Vector Store configuration"""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "advanced_rag_collection"
    dimension: int = 768  # embedding dimension
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    
    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"

@dataclass
class RedisConfig:
    """Redis Index Store configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    @property
    def redis_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class MongoDBConfig:
    """MongoDB Document Store configuration"""
    host: str = "localhost"
    port: int = 27017
    database: str = "advanced_rag_db"
    collection: str = "documents"
    username: Optional[str] = None
    password: Optional[str] = None
    
    @property
    def uri(self) -> str:
        if self.username and self.password:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"mongodb://{self.host}:{self.port}/{self.database}"

@dataclass
class Neo4jConfig:
    """Neo4j Graph Store configuration"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"  # 実際のパスワードに変更
    database: str = "neo4j"

@dataclass
class RAGConfig:
    """Advanced RAG System configuration"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5
    response_mode: str = "compact"
    streaming: bool = True
    
    # Query engine settings
    use_reranking: bool = True
    enable_citation: bool = True
    max_tokens: int = 2000

# Main configuration class
@dataclass
class Config:
    """Main configuration container"""
    ollama: OllamaConfig = OllamaConfig()
    milvus: MilvusConfig = MilvusConfig()
    redis: RedisConfig = RedisConfig()
    mongodb: MongoDBConfig = MongoDBConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    rag: RAGConfig = RAGConfig()
    
    def __post_init__(self):
        """Override with environment variables if available"""
        # Ollama
        self.ollama.base_url = os.getenv("OLLAMA_BASE_URL", self.ollama.base_url)
        self.ollama.llm_model = os.getenv("OLLAMA_LLM_MODEL", self.ollama.llm_model)
        self.ollama.embed_model = os.getenv("OLLAMA_EMBED_MODEL", self.ollama.embed_model)
        
        # Milvus
        self.milvus.host = os.getenv("MILVUS_HOST", self.milvus.host)
        self.milvus.port = int(os.getenv("MILVUS_PORT", str(self.milvus.port)))
        
        # Redis
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", str(self.redis.port)))
        self.redis.password = os.getenv("REDIS_PASSWORD", self.redis.password)
        
        # MongoDB
        self.mongodb.host = os.getenv("MONGODB_HOST", self.mongodb.host)
        self.mongodb.port = int(os.getenv("MONGODB_PORT", str(self.mongodb.port)))
        self.mongodb.username = os.getenv("MONGODB_USERNAME", self.mongodb.username)
        self.mongodb.password = os.getenv("MONGODB_PASSWORD", self.mongodb.password)
        
        # Neo4j
        self.neo4j.uri = os.getenv("NEO4J_URI", self.neo4j.uri)
        self.neo4j.username = os.getenv("NEO4J_USERNAME", self.neo4j.username)
        self.neo4j.password = os.getenv("NEO4J_PASSWORD", self.neo4j.password)

# Global configuration instance
config = Config()