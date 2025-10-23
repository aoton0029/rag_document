"""
Retriever Factory
各種Retrieverの作成・管理機能
"""

from typing import Dict, Any, Optional, List, Type, Union
from enum import Enum
import importlib

from llama_index.core.retrievers import BaseRetriever as LlamaIndexRetriever
from llama_index.core.vector_stores import VectorStore
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms import LLM

from .base_retriever import (
    BaseRetriever, VectorRetriever, BM25Retriever, 
    HybridRetriever, MultiStageRetriever, SearchQuery
)
from .hybrid_retriever import HybridRetriever as AdvancedHybridRetriever
from .reranking import (
    BaseReranker, CrossEncoderReranker, LLMReranker,
    FeatureBasedReranker, DiversityReranker, EnsembleReranker
)
from .query_expansion import (
    BaseQueryExpander, SynonymExpander, SemanticExpander,
    EntityExpander, ContextExpander, CompositeQueryExpander
)
from ..utils import get_logger


class RetrieverType(Enum):
    """Retrieverタイプ"""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    ADVANCED_HYBRID = "advanced_hybrid"
    MULTI_STAGE = "multi_stage"
    CUSTOM = "custom"


class RerankingType(Enum):
    """再ランキングタイプ"""
    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based"
    FEATURE_BASED = "feature_based"
    DIVERSITY = "diversity"
    ENSEMBLE = "ensemble"


class QueryExpansionType(Enum):
    """クエリ拡張タイプ"""
    SYNONYM = "synonym"
    SEMANTIC = "semantic"
    ENTITY = "entity"
    CONTEXT = "context"
    COMPOSITE = "composite"


class RetrieverFactory:
    """Retriever作成ファクトリー"""
    
    def __init__(self):
        self.logger = get_logger("retriever_factory")
        
        # 登録済みRetriever
        self._registered_retrievers: Dict[str, Type[BaseRetriever]] = {
            RetrieverType.VECTOR.value: VectorRetriever,
            RetrieverType.BM25.value: BM25Retriever,
            RetrieverType.HYBRID.value: HybridRetriever,
            RetrieverType.ADVANCED_HYBRID.value: AdvancedHybridRetriever,
            RetrieverType.MULTI_STAGE.value: MultiStageRetriever,
        }
        
        # 登録済みReranker
        self._registered_rerankers: Dict[str, Type[BaseReranker]] = {
            RerankingType.CROSS_ENCODER.value: CrossEncoderReranker,
            RerankingType.LLM_BASED.value: LLMReranker,
            RerankingType.FEATURE_BASED.value: FeatureBasedReranker,
            RerankingType.DIVERSITY.value: DiversityReranker,
            RerankingType.ENSEMBLE.value: EnsembleReranker,
        }
        
        # 登録済みQueryExpander
        self._registered_expanders: Dict[str, Type[BaseQueryExpander]] = {
            QueryExpansionType.SYNONYM.value: SynonymExpander,
            QueryExpansionType.SEMANTIC.value: SemanticExpander,
            QueryExpansionType.ENTITY.value: EntityExpander,
            QueryExpansionType.CONTEXT.value: ContextExpander,
            QueryExpansionType.COMPOSITE.value: CompositeQueryExpander,
        }
    
    def register_retriever(
        self, 
        name: str, 
        retriever_class: Type[BaseRetriever]
    ):
        """カスタムRetrieverを登録"""
        self._registered_retrievers[name] = retriever_class
        self.logger.info(f"Registered custom retriever: {name}")
    
    def register_reranker(
        self, 
        name: str, 
        reranker_class: Type[BaseReranker]
    ):
        """カスタムRerankerを登録"""
        self._registered_rerankers[name] = reranker_class
        self.logger.info(f"Registered custom reranker: {name}")
    
    def register_expander(
        self, 
        name: str, 
        expander_class: Type[BaseQueryExpander]
    ):
        """カスタムQueryExpanderを登録"""
        self._registered_expanders[name] = expander_class
        self.logger.info(f"Registered custom expander: {name}")
    
    def create_retriever(
        self, 
        retriever_type: Union[str, RetrieverType],
        config: Dict[str, Any],
        **kwargs
    ) -> BaseRetriever:
        """Retrieverを作成"""
        
        if isinstance(retriever_type, RetrieverType):
            retriever_type = retriever_type.value
        
        if retriever_type not in self._registered_retrievers:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        retriever_class = self._registered_retrievers[retriever_type]
        
        try:
            retriever = retriever_class(config, **kwargs)
            self.logger.info(f"Created retriever: {retriever_type}")
            return retriever
        except Exception as e:
            self.logger.error(f"Failed to create retriever {retriever_type}: {e}")
            raise
    
    def create_reranker(
        self, 
        reranker_type: Union[str, RerankingType],
        config: Dict[str, Any],
        llm: Optional[LLM] = None
    ) -> BaseReranker:
        """Rerankerを作成"""
        
        if isinstance(reranker_type, RerankingType):
            reranker_type = reranker_type.value
        
        if reranker_type not in self._registered_rerankers:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
        
        reranker_class = self._registered_rerankers[reranker_type]
        
        try:
            # LLMが必要なRerankerの場合
            if reranker_type == RerankingType.LLM_BASED.value:
                reranker = reranker_class(config, llm)
            elif reranker_type == RerankingType.ENSEMBLE.value:
                reranker = reranker_class(config, llm)
            else:
                reranker = reranker_class(config)
            
            self.logger.info(f"Created reranker: {reranker_type}")
            return reranker
        except Exception as e:
            self.logger.error(f"Failed to create reranker {reranker_type}: {e}")
            raise
    
    def create_expander(
        self, 
        expander_type: Union[str, QueryExpansionType],
        config: Dict[str, Any]
    ) -> BaseQueryExpander:
        """QueryExpanderを作成"""
        
        if isinstance(expander_type, QueryExpansionType):
            expander_type = expander_type.value
        
        if expander_type not in self._registered_expanders:
            raise ValueError(f"Unknown expander type: {expander_type}")
        
        expander_class = self._registered_expanders[expander_type]
        
        try:
            expander = expander_class(config)
            self.logger.info(f"Created expander: {expander_type}")
            return expander
        except Exception as e:
            self.logger.error(f"Failed to create expander {expander_type}: {e}")
            raise
    
    def create_pipeline(self, config: Dict[str, Any], **kwargs) -> 'RetrievalPipeline':
        """Retrievalパイプラインを作成"""
        
        pipeline_config = config.get("pipeline", {})
        
        # Retriever作成
        retriever_config = config.get("retriever", {})
        retriever_type = retriever_config.get("type", RetrieverType.VECTOR.value)
        retriever = self.create_retriever(retriever_type, retriever_config, **kwargs)
        
        # QueryExpander作成（オプション）
        expander = None
        if config.get("use_query_expansion", False):
            expander_config = config.get("query_expansion", {})
            expander_type = expander_config.get("type", QueryExpansionType.SYNONYM.value)
            expander = self.create_expander(expander_type, expander_config)
        
        # Reranker作成（オプション）
        reranker = None
        if config.get("use_reranking", False):
            reranker_config = config.get("reranking", {})
            reranker_type = reranker_config.get("type", RerankingType.FEATURE_BASED.value)
            llm = kwargs.get("llm")
            reranker = self.create_reranker(reranker_type, reranker_config, llm)
        
        # パイプライン作成
        pipeline = RetrievalPipeline(
            retriever=retriever,
            expander=expander,
            reranker=reranker,
            config=pipeline_config
        )
        
        return pipeline
    
    def get_available_retrievers(self) -> List[str]:
        """利用可能なRetriever一覧を取得"""
        return list(self._registered_retrievers.keys())
    
    def get_available_rerankers(self) -> List[str]:
        """利用可能なReranker一覧を取得"""
        return list(self._registered_rerankers.keys())
    
    def get_available_expanders(self) -> List[str]:
        """利用可能なExpander一覧を取得"""
        return list(self._registered_expanders.keys())


class RetrievalPipeline:
    """Retrievalパイプライン"""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        expander: Optional[BaseQueryExpander] = None,
        reranker: Optional[BaseReranker] = None,
        config: Dict[str, Any] = None
    ):
        self.retriever = retriever
        self.expander = expander
        self.reranker = reranker
        self.config = config or {}
        
        self.logger = get_logger("retrieval_pipeline")
    
    async def retrieve(
        self, 
        query: Union[str, SearchQuery], 
        **kwargs
    ) -> Dict[str, Any]:
        """パイプライン実行"""
        
        # SearchQueryオブジェクト作成
        if isinstance(query, str):
            search_query = SearchQuery(text=query)
        else:
            search_query = query
        
        # クエリ拡張
        if self.expander:
            try:
                search_query = self.expander.expand_search_query(search_query)
                self.logger.debug(f"Query expanded: {search_query.metadata.get('expansion_terms', [])}")
            except Exception as e:
                self.logger.warning(f"Query expansion failed: {e}")
        
        # 検索実行
        try:
            search_results = await self.retriever.retrieve(search_query, **kwargs)
            self.logger.debug(f"Retrieved {len(search_results)} results")
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return {
                "results": [],
                "query": search_query,
                "error": str(e),
                "pipeline_metadata": {}
            }
        
        # 再ランキング
        reranking_result = None
        if self.reranker and search_results:
            try:
                reranking_result = self.reranker.rerank(search_query, search_results)
                final_results = reranking_result.reranked_results
                self.logger.debug(f"Reranked to {len(final_results)} results")
            except Exception as e:
                self.logger.warning(f"Reranking failed: {e}")
                final_results = search_results
        else:
            final_results = search_results
        
        # メタデータ収集
        pipeline_metadata = {
            "retriever_type": self.retriever.__class__.__name__,
            "expander_used": self.expander is not None,
            "reranker_used": self.reranker is not None,
            "original_query": query if isinstance(query, str) else query.text,
            "expanded_query": search_query.text if self.expander else None,
            "original_count": len(search_results) if search_results else 0,
            "final_count": len(final_results)
        }
        
        if reranking_result:
            pipeline_metadata["reranking_metadata"] = reranking_result.metadata
        
        return {
            "results": final_results,
            "query": search_query,
            "reranking_result": reranking_result,
            "pipeline_metadata": pipeline_metadata
        }


class RetrieverBenchmark:
    """Retrieverベンチマーク"""
    
    def __init__(self, factory: RetrieverFactory):
        self.factory = factory
        self.logger = get_logger("retriever_benchmark")
    
    async def benchmark_retrievers(
        self,
        retriever_configs: List[Dict[str, Any]],
        test_queries: List[Union[str, SearchQuery]],
        **kwargs
    ) -> Dict[str, Any]:
        """複数のRetrieverをベンチマーク"""
        
        results = {}
        
        for config in retriever_configs:
            retriever_name = config.get("name", config.get("type", "unknown"))
            
            try:
                retriever = self.factory.create_retriever(
                    config.get("type"),
                    config,
                    **kwargs
                )
                
                retriever_results = await self._benchmark_single_retriever(
                    retriever, test_queries, **kwargs
                )
                
                results[retriever_name] = retriever_results
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {retriever_name}: {e}")
                results[retriever_name] = {"error": str(e)}
        
        return results
    
    async def _benchmark_single_retriever(
        self,
        retriever: BaseRetriever,
        test_queries: List[Union[str, SearchQuery]],
        **kwargs
    ) -> Dict[str, Any]:
        """単一Retrieverのベンチマーク"""
        
        import time
        
        total_time = 0
        total_results = 0
        query_results = []
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                results = await retriever.retrieve(query, **kwargs)
                end_time = time.time()
                
                query_time = end_time - start_time
                result_count = len(results)
                
                total_time += query_time
                total_results += result_count
                
                query_results.append({
                    "query": query if isinstance(query, str) else query.text,
                    "result_count": result_count,
                    "time": query_time,
                    "avg_score": sum(r.score for r in results) / len(results) if results else 0.0
                })
                
            except Exception as e:
                self.logger.error(f"Query failed: {e}")
                query_results.append({
                    "query": query if isinstance(query, str) else query.text,
                    "error": str(e),
                    "time": 0,
                    "result_count": 0
                })
        
        return {
            "total_queries": len(test_queries),
            "total_time": total_time,
            "avg_time_per_query": total_time / len(test_queries) if test_queries else 0,
            "total_results": total_results,
            "avg_results_per_query": total_results / len(test_queries) if test_queries else 0,
            "query_results": query_results
        }


# グローバルファクトリーインスタンス
_global_factory = RetrieverFactory()


def create_retriever(
    retriever_type: Union[str, RetrieverType],
    config: Dict[str, Any],
    **kwargs
) -> BaseRetriever:
    """Retrieverを作成（便利関数）"""
    return _global_factory.create_retriever(retriever_type, config, **kwargs)


def create_reranker(
    reranker_type: Union[str, RerankingType],
    config: Dict[str, Any],
    llm: Optional[LLM] = None
) -> BaseReranker:
    """Rerankerを作成（便利関数）"""
    return _global_factory.create_reranker(reranker_type, config, llm)


def create_expander(
    expander_type: Union[str, QueryExpansionType],
    config: Dict[str, Any]
) -> BaseQueryExpander:
    """QueryExpanderを作成（便利関数）"""
    return _global_factory.create_expander(expander_type, config)


def create_retrieval_pipeline(
    config: Dict[str, Any], 
    **kwargs
) -> RetrievalPipeline:
    """Retrievalパイプラインを作成（便利関数）"""
    return _global_factory.create_pipeline(config, **kwargs)


def get_retriever_factory() -> RetrieverFactory:
    """グローバルファクトリーを取得"""
    return _global_factory


def register_custom_retriever(name: str, retriever_class: Type[BaseRetriever]):
    """カスタムRetrieverを登録（便利関数）"""
    _global_factory.register_retriever(name, retriever_class)


def register_custom_reranker(name: str, reranker_class: Type[BaseReranker]):
    """カスタムRerankerを登録（便利関数）"""
    _global_factory.register_reranker(name, reranker_class)


def register_custom_expander(name: str, expander_class: Type[BaseQueryExpander]):
    """カスタムExpanderを登録（便利関数）"""
    _global_factory.register_expander(name, expander_class)