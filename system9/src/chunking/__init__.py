"""
チャンキングモジュール
テキストの高度な分割処理とAdvanced RAGチャンキング戦略
"""

from .base_chunker import (
    BaseChunker,
    Chunk,
    ChunkType,
    ChunkingStrategy,
    ChunkEvaluator
)

from .semantic_chunker import (
    SemanticChunker,
    HierarchicalSemanticChunker
)

from .adaptive_chunker import AdaptiveChunker

from .traditional_chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceBasedChunker,
    TokenBasedChunker
)

from .chunker_factory import (
    ChunkerFactory,
    create_chunker,
    get_chunking_strategies,
    benchmark_chunking_strategies
)

# バージョン情報
__version__ = "1.0.0"

# パブリックAPI
__all__ = [
    # Base classes
    "BaseChunker",
    "Chunk", 
    "ChunkType",
    "ChunkingStrategy",
    "ChunkEvaluator",
    
    # Chunker implementations
    "SemanticChunker",
    "HierarchicalSemanticChunker", 
    "AdaptiveChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceBasedChunker",
    "TokenBasedChunker",
    
    # Factory and utilities
    "ChunkerFactory",
    "create_chunker",
    "get_chunking_strategies",
    "benchmark_chunking_strategies"
]

# デフォルトチャンカー設定
DEFAULT_CHUNKING_CONFIGS = {
    "semantic": {
        "chunk_size": 512,
        "similarity_threshold": 0.7,
        "min_chunk_size": 100,
        "max_chunk_size": 2048
    },
    "adaptive": {
        "min_chunk_size": 200,
        "max_chunk_size": 1500,
        "target_chunk_size": 800
    },
    "fixed_size": {
        "chunk_size": 1024,
        "chunk_overlap": 100
    }
}


def quick_chunk(text: str, strategy: str = "adaptive", **kwargs) -> list:
    """クイックチャンキング関数"""
    config = DEFAULT_CHUNKING_CONFIGS.get(strategy, {})
    config.update(kwargs)
    
    chunker = create_chunker(strategy, config)
    return chunker.chunk_text(text)


def evaluate_chunking(chunks: list, original_text: str) -> dict:
    """チャンキング結果を評価"""
    evaluator = ChunkEvaluator()
    return evaluator.evaluate_chunks(chunks, original_text)