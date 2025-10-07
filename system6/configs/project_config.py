from pydantic import BaseModel

class ProjectConfig(BaseModel):
    """プロジェクト全体の設定"""
    project_name: str = "MyProject"
    default_language: str = "en"
    max_retries: int = 3
    retry_delay: int = 2  # seconds
    log_level: str = "INFO"
    temp_file_dir: str = "/tmp/myproject"
    allowed_file_types: list[str] = ["pdf", "docx", "txt", "md"]
    max_file_size_mb: int = 50  # Maximum file size in megabytes

    # Milvus Settings
    milvus_collection_name: str = "rag_system"
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "COSINE"
    mongodb_database: str = "rag_system"
    # Chunking Settings
    chunk_size: int = 800
    chunk_overlap: int = 50
    # Search Settings
    top_k: int = 10
    similarity_threshold: float = 0.7

