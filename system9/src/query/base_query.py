"""
Query Module - クエリエンジンの実装
llama_indexのQueryEngineを活用した高度なクエリ処理機能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

from llama_index.core.query_engine import (
    BaseQueryEngine, SQLJoinQueryEngine, CustomQueryEngine, PandasQueryEngine, 
    RouterQueryEngine, RetryQueryEngine, MultiStepQueryEngine, 
    JSONalyzeQueryEngine, TransformQueryEngine, 
    SQLAutoVectorQueryEngine, RetrieverQueryEngine, SQLTableRetrieverQueryEngine, 
    ToolRetrieverRouterQueryEngine, SimpleMultiModalQueryEngine, 
    KnowledgeGraphQueryEngine, RetrieverRouterQueryEngine, RetrySourceQueryEngine,
)
from llama_index.core.tools import ToolMetadata, RetrieverTool, QueryEngineTool
from llama_index.core.selectors import (
    MultiSelection, SingleSelection, LLMSingleSelector, 
    LLMMultiSelector, EmbeddingSingleSelector, 
    PydanticMultiSelector, PydanticSingleSelector,
)
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.llms import LLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response import Response

from ..utils import get_logger


class QueryType(Enum):
    """クエリタイプ"""
    SIMPLE = "simple"
    ROUTER = "router"
    MULTI_STEP = "multi_step"
    RETRY = "retry"
    TRANSFORM = "transform"
    CUSTOM = "custom"


class QueryMode(Enum):
    """クエリモード"""
    RETRIEVAL = "retrieval"
    SQL = "sql"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TOOL = "tool"
    MULTIMODAL = "multimodal"


@dataclass
class QueryConfig:
    """クエリ設定"""
    query_type: QueryType
    query_mode: QueryMode
    similarity_top_k: int = 5
    streaming: bool = False
    metadata_filters: Optional[MetadataFilters] = None
    response_synthesizer_config: Optional[Dict[str, Any]] = None
    custom_config: Optional[Dict[str, Any]] = None


@dataclass
class QueryResult:
    """クエリ結果"""
    response: Response
    source_nodes: List[NodeWithScore]
    query: str
    query_config: QueryConfig
    metadata: Dict[str, Any]


class BaseCustomQueryEngine(ABC):
    """カスタムクエリエンジン基底クラス"""
    
    def __init__(self, config: QueryConfig):
        self.config = config
        self.logger = get_logger(f"query_engine_{self.__class__.__name__}")
    
    @abstractmethod
    async def aquery(self, query: Union[str, QueryBundle]) -> QueryResult:
        """非同期クエリ実行"""
        pass
    
    def query(self, query: Union[str, QueryBundle]) -> QueryResult:
        """同期クエリ実行"""
        return asyncio.run(self.aquery(query))


class SimpleQueryEngine(BaseCustomQueryEngine):
    """シンプルクエリエンジン"""
    
    def __init__(
        self, 
        config: QueryConfig,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None
    ):
        super().__init__(config)
        self.retriever = retriever
        self.llm = llm or Settings.llm
        
        # Response Synthesizerを設定
        response_config = config.response_synthesizer_config or {}
        self.response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            **response_config
        )
        
        # llama_indexのRetrieverQueryEngineを使用
        self.engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer
        )
    
    async def aquery(self, query: Union[str, QueryBundle]) -> QueryResult:
        """非同期クエリ実行"""
        
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        # メタデータフィルター適用
        if self.config.metadata_filters:
            query_bundle.metadata_filters = self.config.metadata_filters
        
        # クエリ実行
        response = await self.engine.aquery(query_bundle)
        
        return QueryResult(
            response=response,
            source_nodes=response.source_nodes,
            query=query_bundle.query_str,
            query_config=self.config,
            metadata={}
        )


class RouterBasedQueryEngine(BaseCustomQueryEngine):
    """ルーターベースクエリエンジン"""
    
    def __init__(
        self,
        config: QueryConfig,
        query_engines: Dict[str, BaseQueryEngine],
        selector: Optional[Any] = None,
        llm: Optional[LLM] = None
    ):
        super().__init__(config)
        self.query_engines = query_engines
        self.llm = llm or Settings.llm
        
        # セレクター設定
        if selector is None:
            self.selector = LLMSingleSelector.from_defaults(llm=self.llm)
        else:
            self.selector = selector
        
        # ツール作成
        tools = []
        for name, engine in query_engines.items():
            tool = QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name=name,
                    description=f"Query engine for {name}"
                )
            )
            tools.append(tool)
        
        # RouterQueryEngine作成
        self.engine = RouterQueryEngine(
            selector=self.selector,
            query_engine_tools=tools
        )
    
    async def aquery(self, query: Union[str, QueryBundle]) -> QueryResult:
        """非同期クエリ実行"""
        
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        # クエリ実行
        response = await self.engine.aquery(query_bundle)
        
        return QueryResult(
            response=response,
            source_nodes=response.source_nodes or [],
            query=query_bundle.query_str,
            query_config=self.config,
            metadata={
                "selected_engine": getattr(response, 'metadata', {}).get('selector_result')
            }
        )


class MultiStepQueryEngine(BaseCustomQueryEngine):
    """マルチステップクエリエンジン"""
    
    def __init__(
        self,
        config: QueryConfig,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        num_steps: int = 3
    ):
        super().__init__(config)
        self.retriever = retriever
        self.llm = llm or Settings.llm
        self.num_steps = num_steps
        
        # Response Synthesizer設定
        response_config = config.response_synthesizer_config or {}
        self.response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            **response_config
        )
        
        # MultiStepQueryEngineを作成
        self.engine = MultiStepQueryEngine(
            query_engine=RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=self.response_synthesizer
            ),
            llm=self.llm,
            num_steps=self.num_steps
        )
    
    async def aquery(self, query: Union[str, QueryBundle]) -> QueryResult:
        """非同期クエリ実行"""
        
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        # クエリ実行
        response = await self.engine.aquery(query_bundle)
        
        return QueryResult(
            response=response,
            source_nodes=response.source_nodes or [],
            query=query_bundle.query_str,
            query_config=self.config,
            metadata={
                "num_steps": self.num_steps,
                "intermediate_steps": getattr(response, 'metadata', {}).get('intermediate_steps', [])
            }
        )


class RetryQueryEngine(BaseCustomQueryEngine):
    """リトライクエリエンジン"""
    
    def __init__(
        self,
        config: QueryConfig,
        base_query_engine: BaseQueryEngine,
        evaluator: Optional[Callable] = None,
        max_retries: int = 3,
        llm: Optional[LLM] = None
    ):
        super().__init__(config)
        self.base_query_engine = base_query_engine
        self.evaluator = evaluator
        self.max_retries = max_retries
        self.llm = llm or Settings.llm
        
        # RetryQueryEngineを作成
        self.engine = RetryQueryEngine(
            query_engine=base_query_engine,
            evaluator=evaluator,
            max_retries=max_retries
        )
    
    async def aquery(self, query: Union[str, QueryBundle]) -> QueryResult:
        """非同期クエリ実行"""
        
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        # クエリ実行
        response = await self.engine.aquery(query_bundle)
        
        return QueryResult(
            response=response,
            source_nodes=response.source_nodes or [],
            query=query_bundle.query_str,
            query_config=self.config,
            metadata={
                "max_retries": self.max_retries,
                "actual_retries": getattr(response, 'metadata', {}).get('retries', 0)
            }
        )


class TransformQueryEngine(BaseCustomQueryEngine):
    """変換クエリエンジン"""
    
    def __init__(
        self,
        config: QueryConfig,
        base_query_engine: BaseQueryEngine,
        query_transform: Callable[[str], str],
        response_transform: Optional[Callable[[Response], Response]] = None
    ):
        super().__init__(config)
        self.base_query_engine = base_query_engine
        self.query_transform = query_transform
        self.response_transform = response_transform
        
        # TransformQueryEngineを作成
        self.engine = TransformQueryEngine(
            query_engine=base_query_engine,
            query_transform=query_transform,
            response_transform=response_transform
        )
    
    async def aquery(self, query: Union[str, QueryBundle]) -> QueryResult:
        """非同期クエリ実行"""
        
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        original_query = query_bundle.query_str
        
        # クエリ実行
        response = await self.engine.aquery(query_bundle)
        
        return QueryResult(
            response=response,
            source_nodes=response.source_nodes or [],
            query=original_query,
            query_config=self.config,
            metadata={
                "original_query": original_query,
                "transformed_query": self.query_transform(original_query) if self.query_transform else original_query
            }
        )


class HybridQueryEngine(BaseCustomQueryEngine):
    """ハイブリッドクエリエンジン"""
    
    def __init__(
        self,
        config: QueryConfig,
        query_engines: List[BaseQueryEngine],
        weights: Optional[List[float]] = None,
        combination_method: str = "weighted_sum"
    ):
        super().__init__(config)
        self.query_engines = query_engines
        self.weights = weights or [1.0] * len(query_engines)
        self.combination_method = combination_method
        
        if len(self.weights) != len(self.query_engines):
            raise ValueError("Number of weights must match number of query engines")
    
    async def aquery(self, query: Union[str, QueryBundle]) -> QueryResult:
        """非同期クエリ実行"""
        
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
        
        # 全エンジンで並列実行
        tasks = [engine.aquery(query_bundle) for engine in self.query_engines]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果の結合
        combined_response = self._combine_responses(responses, query_bundle.query_str)
        
        return QueryResult(
            response=combined_response,
            source_nodes=combined_response.source_nodes or [],
            query=query_bundle.query_str,
            query_config=self.config,
            metadata={
                "num_engines": len(self.query_engines),
                "weights": self.weights,
                "combination_method": self.combination_method
            }
        )
    
    def _combine_responses(self, responses: List[Union[Response, Exception]], query: str) -> Response:
        """レスポンスを結合"""
        
        valid_responses = [r for r in responses if isinstance(r, Response)]
        
        if not valid_responses:
            # 全て失敗した場合
            return Response(
                response="エラー: 全てのクエリエンジンが失敗しました",
                source_nodes=[],
                metadata={"error": "All query engines failed"}
            )
        
        if len(valid_responses) == 1:
            return valid_responses[0]
        
        # 重み付き結合
        if self.combination_method == "weighted_sum":
            return self._weighted_sum_responses(valid_responses)
        elif self.combination_method == "best_score":
            return max(valid_responses, key=lambda r: getattr(r, 'score', 0.0))
        else:
            # デフォルトは最初の有効なレスポンス
            return valid_responses[0]
    
    def _weighted_sum_responses(self, responses: List[Response]) -> Response:
        """重み付きレスポンス結合"""
        
        # テキスト結合
        combined_text = ""
        all_source_nodes = []
        
        for i, response in enumerate(responses):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            
            # 重みに基づいてテキストを結合
            if response.response:
                combined_text += f"[エンジン{i+1} (重み:{weight})]: {response.response}\n\n"
            
            # ソースノード収集
            if response.source_nodes:
                all_source_nodes.extend(response.source_nodes)
        
        # 重複ノード除去
        unique_nodes = {}
        for node in all_source_nodes:
            node_id = node.node.node_id
            if node_id not in unique_nodes or node.score > unique_nodes[node_id].score:
                unique_nodes[node_id] = node
        
        return Response(
            response=combined_text.strip(),
            source_nodes=list(unique_nodes.values()),
            metadata={"combination_method": "weighted_sum"}
        )


class QueryEngineManager:
    """クエリエンジン管理"""
    
    def __init__(self):
        self.logger = get_logger("query_engine_manager")
        self._engines: Dict[str, BaseCustomQueryEngine] = {}
    
    def register_engine(self, name: str, engine: BaseCustomQueryEngine):
        """エンジンを登録"""
        self._engines[name] = engine
        self.logger.info(f"Registered query engine: {name}")
    
    def get_engine(self, name: str) -> Optional[BaseCustomQueryEngine]:
        """エンジンを取得"""
        return self._engines.get(name)
    
    def list_engines(self) -> List[str]:
        """登録済みエンジン一覧"""
        return list(self._engines.keys())
    
    async def query_all(self, query: Union[str, QueryBundle]) -> Dict[str, QueryResult]:
        """全エンジンでクエリ実行"""
        
        tasks = []
        for name, engine in self._engines.items():
            tasks.append(self._safe_query(name, engine, query))
        
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def _safe_query(
        self, 
        name: str, 
        engine: BaseCustomQueryEngine, 
        query: Union[str, QueryBundle]
    ) -> Tuple[str, QueryResult]:
        """安全なクエリ実行"""
        
        try:
            result = await engine.aquery(query)
            return name, result
        except Exception as e:
            self.logger.error(f"Engine {name} failed: {e}")
            
            # エラー用のQueryResult作成
            error_result = QueryResult(
                response=Response(
                    response=f"エラー: {str(e)}",
                    source_nodes=[],
                    metadata={"error": str(e)}
                ),
                source_nodes=[],
                query=query if isinstance(query, str) else query.query_str,
                query_config=engine.config,
                metadata={"error": str(e)}
            )
            
            return name, error_result


# Utility functions
def create_simple_query_engine(
    retriever: BaseRetriever,
    llm: Optional[LLM] = None,
    similarity_top_k: int = 5,
    **kwargs
) -> SimpleQueryEngine:
    """シンプルクエリエンジン作成"""
    
    config = QueryConfig(
        query_type=QueryType.SIMPLE,
        query_mode=QueryMode.RETRIEVAL,
        similarity_top_k=similarity_top_k,
        **kwargs
    )
    
    return SimpleQueryEngine(config, retriever, llm)


def create_router_query_engine(
    query_engines: Dict[str, BaseQueryEngine],
    selector: Optional[Any] = None,
    llm: Optional[LLM] = None,
    **kwargs
) -> RouterBasedQueryEngine:
    """ルータークエリエンジン作成"""
    
    config = QueryConfig(
        query_type=QueryType.ROUTER,
        query_mode=QueryMode.RETRIEVAL,
        **kwargs
    )
    
    return RouterBasedQueryEngine(config, query_engines, selector, llm)


def create_multi_step_query_engine(
    retriever: BaseRetriever,
    llm: Optional[LLM] = None,
    num_steps: int = 3,
    **kwargs
) -> MultiStepQueryEngine:
    """マルチステップクエリエンジン作成"""
    
    config = QueryConfig(
        query_type=QueryType.MULTI_STEP,
        query_mode=QueryMode.RETRIEVAL,
        **kwargs
    )
    
    return MultiStepQueryEngine(config, retriever, llm, num_steps)
