"""
Retrieval Module
Advanced RAG retrieval functionality
"""

from .base_retriever import (
    BaseRetriever,
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    MultiStageRetriever,
    SearchQuery,
    SearchResult
)

from .hybrid_retriever import (
    HybridRetriever as AdvancedHybridRetriever,
    FusionMethod
)

from .query_expansion import (
    BaseQueryExpander,
    SynonymExpander,
    SemanticExpander,
    EntityExpander,
    ContextExpander,
    CompositeQueryExpander,
    QueryAnalyzer,
    ExpansionStrategy
)

from .reranking import (
    BaseReranker,
    CrossEncoderReranker,
    LLMReranker,
    FeatureBasedReranker,
    DiversityReranker,
    EnsembleReranker,
    RerankingResult,
    RerankingStrategy,
    LlamaIndexRerankerPostprocessor
)

from .retriever_factory import (
    RetrieverFactory,
    RetrievalPipeline,
    RetrieverBenchmark,
    RetrieverType,
    RerankingType,
    QueryExpansionType,
    create_retriever,
    create_reranker,
    create_expander,
    create_retrieval_pipeline,
    get_retriever_factory,
    register_custom_retriever,
    register_custom_reranker,
    register_custom_expander
)

__all__ = [
    # Base classes
    "BaseRetriever",
    "BaseReranker", 
    "BaseQueryExpander",
    
    # Retrievers
    "VectorRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "AdvancedHybridRetriever",
    "MultiStageRetriever",
    
    # Query expansion
    "SynonymExpander",
    "SemanticExpander",
    "EntityExpander",
    "ContextExpander",
    "CompositeQueryExpander",
    "QueryAnalyzer",
    "ExpansionStrategy",
    
    # Reranking
    "CrossEncoderReranker",
    "LLMReranker",
    "FeatureBasedReranker",
    "DiversityReranker",
    "EnsembleReranker",
    "RerankingResult",
    "RerankingStrategy",
    "LlamaIndexRerankerPostprocessor",
    
    # Factory and pipeline
    "RetrieverFactory",
    "RetrievalPipeline",
    "RetrieverBenchmark",
    "RetrieverType",
    "RerankingType",
    "QueryExpansionType",
    
    # Data types
    "SearchQuery",
    "SearchResult",
    "FusionMethod",
    
    # Utility functions
    "create_retriever",
    "create_reranker", 
    "create_expander",
    "create_retrieval_pipeline",
    "get_retriever_factory",
    "register_custom_retriever",
    "register_custom_reranker",
    "register_custom_expander"
]