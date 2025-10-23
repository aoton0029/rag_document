"""
埋め込みモジュール
マルチベクター埋め込み、ファインチューニング、Advanced RAG埋め込み戦略
"""

from .base_embedder import (
    BaseEmbedder,
    EmbeddingResult,
    EmbeddingProvider,
    EmbeddingValidator
)

from .providers import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    CustomEmbedder
)

from .multi_vector import MultiVectorEmbedder

from .fine_tuning import (
    EmbeddingFineTuner,
    DomainAdapter,
    FineTunedEmbedder,
    ContrastiveLearningDataset
)

from .embedder_factory import (
    EmbedderFactory,
    create_embedder,
    get_available_providers,
    benchmark_embedding_providers,
    create_multi_vector_embedder
)

# バージョン情報
__version__ = "1.0.0"

# パブリックAPI
__all__ = [
    # Base classes
    "BaseEmbedder",
    "EmbeddingResult",
    "EmbeddingProvider", 
    "EmbeddingValidator",
    
    # Provider implementations
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "HuggingFaceEmbedder",
    "OpenAIEmbedder",
    "CustomEmbedder",
    
    # Advanced embedding strategies
    "MultiVectorEmbedder",
    
    # Fine-tuning
    "EmbeddingFineTuner",
    "DomainAdapter",
    "FineTunedEmbedder",
    "ContrastiveLearningDataset",
    
    # Factory and utilities
    "EmbedderFactory",
    "create_embedder",
    "get_available_providers",
    "benchmark_embedding_providers", 
    "create_multi_vector_embedder"
]

# デフォルト設定
DEFAULT_EMBEDDING_CONFIGS = {
    "ollama": {
        "model_name": "qwen3-embedding:8b",
        "embed_batch_size": 10,
        "dimensions": 1536
    },
    "sentence_transformers": {
        "model_name": "all-MiniLM-L6-v2",
        "embed_batch_size": 32,
        "normalize_embeddings": True
    },
    "multi_vector": {
        "embedders": {
            "st1": {
                "provider": "sentence_transformers",
                "model_name": "all-MiniLM-L6-v2",
                "weight": 0.6
            },
            "st2": {
                "provider": "sentence_transformers", 
                "model_name": "all-mpnet-base-v2",
                "weight": 0.4
            }
        },
        "fusion_strategy": "weighted_average"
    }
}


def quick_embed(texts: list, provider: str = "sentence_transformers", **kwargs) -> EmbeddingResult:
    """クイック埋め込み関数"""
    config = DEFAULT_EMBEDDING_CONFIGS.get(provider, {})
    config.update(kwargs)
    
    embedder = create_embedder(provider, config)
    return embedder.encode_texts(texts)


def validate_embeddings(embeddings: list, expected_dimensions: int) -> dict:
    """埋め込み品質検証"""
    validator = EmbeddingValidator()
    return validator.validate_embeddings(embeddings, expected_dimensions)