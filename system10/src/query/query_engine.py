"""
Query Engine Module
各種QueryEngineの実装
"""

import logging
from typing import List, Optional, Dict, Any, Sequence

from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
    RouterQueryEngine,
    RetryQueryEngine,
    MultiStepQueryEngine,
    TransformQueryEngine,
    RetrySourceQueryEngine
)
from llama_index.core.retrievers.base import BaseRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
    PydanticSingleSelector
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.indices.base import BaseIndex

logger = logging.getLogger(__name__)


class QueryEngineFactory:
    """
    QueryEngineファクトリークラス
    設定に基づいてQueryEngineを生成
    """
    
    @staticmethod
    def create_query_engine(
        index: BaseIndex,
        query_engine_type: str = "retriever",
        **kwargs
    ) -> BaseQueryEngine:
        """
        QueryEngineを作成
        
        Args:
            index: インデックス
            query_engine_type: QueryEngineタイプ
            **kwargs: QueryEngineパラメータ
            
        Returns:
            BaseQueryEngine
        """
        if query_engine_type == "retriever":
            builder = RetrieverQueryEngineBuilder()
            return builder.build(index=index, **kwargs)
        elif query_engine_type == "router":
            builder = RouterQueryEngineBuilder()
            return builder.build(**kwargs)
        else:
            logger.warning(f"未対応のquery_engine_type: {query_engine_type}、デフォルトのRetrieverQueryEngineを使用")
            builder = RetrieverQueryEngineBuilder()
            return builder.build(index=index, **kwargs)


class RetrieverQueryEngineBuilder:
    """
    RetrieverQueryEngineビルダー
    """
    
    def build(
        self,
        index: Optional[BaseIndex] = None,
        retriever: Optional[BaseRetriever] = None,
        response_mode: str = "compact",
        similarity_top_k: int = 10,
        use_async: bool = False,
        streaming: bool = False,
        node_postprocessors: Optional[List[Any]] = None,
        **kwargs
    ) -> RetrieverQueryEngine:
        """
        RetrieverQueryEngineを構築
        
        Args:
            index: インデックス
            retriever: Retriever（指定しない場合はindexから作成）
            response_mode: レスポンスモード
            similarity_top_k: 取得する上位k件
            use_async: 非同期実行するか
            streaming: ストリーミングするか
            node_postprocessors: ノード後処理
            **kwargs: 追加パラメータ
            
        Returns:
            RetrieverQueryEngine
        """
        try:
            # Retrieverを取得または作成
            if retriever is None:
                if index is None:
                    raise ValueError("indexまたはretrieverのいずれかを指定してください")
                retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            
            # ResponseSynthesizerを作成
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode(response_mode),
                use_async=use_async,
                streaming=streaming
            )
            
            # QueryEngineを構築
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=node_postprocessors,
                **kwargs
            )
            
            logger.info(f"RetrieverQueryEngineを構築: response_mode={response_mode}")
            return query_engine
        except Exception as e:
            logger.error(f"RetrieverQueryEngine構築エラー: {e}")
            raise


class RouterQueryEngineBuilder:
    """
    RouterQueryEngineビルダー
    """
    
    def build(
        self,
        query_engine_tools: List[QueryEngineTool],
        selector_type: str = "llm_single",
        llm: Optional[Any] = None,
        **kwargs
    ) -> RouterQueryEngine:
        """
        RouterQueryEngineを構築
        
        Args:
            query_engine_tools: QueryEngineToolのリスト
            selector_type: セレクタータイプ
            llm: LLMインスタンス
            **kwargs: 追加パラメータ
            
        Returns:
            RouterQueryEngine
        """
        try:
            # セレクターを作成
            if selector_type == "llm_single":
                if llm is None:
                    raise ValueError("llm_singleセレクターにはLLMが必要です")
                selector = LLMSingleSelector.from_defaults(llm=llm)
            elif selector_type == "llm_multi":
                if llm is None:
                    raise ValueError("llm_multiセレクターにはLLMが必要です")
                selector = LLMMultiSelector.from_defaults(llm=llm)
            elif selector_type == "pydantic":
                selector = PydanticSingleSelector.from_defaults()
            else:
                logger.warning(f"未対応のselector_type: {selector_type}、デフォルトのPydanticSingleSelectorを使用")
                selector = PydanticSingleSelector.from_defaults()
            
            # RouterQueryEngineを構築
            query_engine = RouterQueryEngine(
                selector=selector,
                query_engine_tools=query_engine_tools,
                **kwargs
            )
            
            logger.info(f"RouterQueryEngineを構築: selector_type={selector_type}")
            return query_engine
        except Exception as e:
            logger.error(f"RouterQueryEngine構築エラー: {e}")
            raise


class RetryQueryEngineBuilder:
    """
    RetryQueryEngineビルダー
    """
    
    def build(
        self,
        query_engine: BaseQueryEngine,
        evaluator: Any,
        max_retries: int = 3,
        **kwargs
    ) -> RetryQueryEngine:
        """
        RetryQueryEngineを構築
        
        Args:
            query_engine: ベースとなるQueryEngine
            evaluator: 評価器
            max_retries: 最大リトライ回数
            **kwargs: 追加パラメータ
            
        Returns:
            RetryQueryEngine
        """
        try:
            retry_query_engine = RetryQueryEngine(
                query_engine=query_engine,
                evaluator=evaluator,
                max_retries=max_retries,
                **kwargs
            )
            
            logger.info(f"RetryQueryEngineを構築: max_retries={max_retries}")
            return retry_query_engine
        except Exception as e:
            logger.error(f"RetryQueryEngine構築エラー: {e}")
            raise


class MultiStepQueryEngineBuilder:
    """
    MultiStepQueryEngineビルダー
    """
    
    def build(
        self,
        query_engine: BaseQueryEngine,
        query_transform: Any,
        index_summary: str = "",
        num_steps: int = 3,
        **kwargs
    ) -> MultiStepQueryEngine:
        """
        MultiStepQueryEngineを構築
        
        Args:
            query_engine: ベースとなるQueryEngine
            query_transform: クエリ変換
            index_summary: インデックスサマリー
            num_steps: ステップ数
            **kwargs: 追加パラメータ
            
        Returns:
            MultiStepQueryEngine
        """
        try:
            multi_step_query_engine = MultiStepQueryEngine(
                query_engine=query_engine,
                query_transform=query_transform,
                index_summary=index_summary,
                num_steps=num_steps,
                **kwargs
            )
            
            logger.info(f"MultiStepQueryEngineを構築: num_steps={num_steps}")
            return multi_step_query_engine
        except Exception as e:
            logger.error(f"MultiStepQueryEngine構築エラー: {e}")
            raise


class ToolFactory:
    """
    Toolファクトリークラス
    from llama_index.core.tools import ToolMetadata, RetrieverTool, QueryEngineTool
    """
    
    @staticmethod
    def create_query_engine_tool(
        query_engine: BaseQueryEngine,
        name: str,
        description: str,
        **kwargs
    ) -> QueryEngineTool:
        """
        QueryEngineToolを作成
        
        Args:
            query_engine: QueryEngine
            name: ツール名
            description: ツール説明
            **kwargs: 追加パラメータ
            
        Returns:
            QueryEngineTool
        """
        try:
            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=name,
                description=description,
                **kwargs
            )
            logger.info(f"QueryEngineToolを作成: {name}")
            return tool
        except Exception as e:
            logger.error(f"QueryEngineTool作成エラー: {e}")
            raise
    
    @staticmethod
    def create_retriever_tool(
        retriever: BaseRetriever,
        name: str,
        description: str,
        **kwargs
    ):
        """
        RetrieverToolを作成
        
        Args:
            retriever: Retriever
            name: ツール名
            description: ツール説明
            **kwargs: 追加パラメータ
            
        Returns:
            RetrieverTool
        """
        try:
            from llama_index.core.tools import RetrieverTool
            
            tool = RetrieverTool.from_defaults(
                retriever=retriever,
                name=name,
                description=description,
                **kwargs
            )
            logger.info(f"RetrieverToolを作成: {name}")
            return tool
        except Exception as e:
            logger.error(f"RetrieverTool作成エラー: {e}")
            raise


class SelectorFactory:
    """
    Selectorファクトリークラス
    from llama_index.core.selectors import (
        MultiSelection, SingleSelection, LLMSingleSelector,
        LLMMultiSelector, EmbeddingSingleSelector,
        PydanticMultiSelector, PydanticSingleSelector
    )
    """
    
    @staticmethod
    def create_llm_single_selector(llm: Any, **kwargs):
        """
        LLMSingleSelectorを作成
        
        Args:
            llm: LLMインスタンス
            **kwargs: 追加パラメータ
            
        Returns:
            LLMSingleSelector
        """
        try:
            selector = LLMSingleSelector.from_defaults(llm=llm, **kwargs)
            logger.info("LLMSingleSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"LLMSingleSelector作成エラー: {e}")
            raise
    
    @staticmethod
    def create_llm_multi_selector(llm: Any, **kwargs):
        """
        LLMMultiSelectorを作成
        
        Args:
            llm: LLMインスタンス
            **kwargs: 追加パラメータ
            
        Returns:
            LLMMultiSelector
        """
        try:
            selector = LLMMultiSelector.from_defaults(llm=llm, **kwargs)
            logger.info("LLMMultiSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"LLMMultiSelector作成エラー: {e}")
            raise
    
    @staticmethod
    def create_pydantic_single_selector(**kwargs):
        """
        PydanticSingleSelectorを作成
        
        Args:
            **kwargs: 追加パラメータ
            
        Returns:
            PydanticSingleSelector
        """
        try:
            selector = PydanticSingleSelector.from_defaults(**kwargs)
            logger.info("PydanticSingleSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"PydanticSingleSelector作成エラー: {e}")
            raise
    
    @staticmethod
    def create_pydantic_multi_selector(**kwargs):
        """
        PydanticMultiSelectorを作成
        
        Args:
            **kwargs: 追加パラメータ
            
        Returns:
            PydanticMultiSelector
        """
        try:
            from llama_index.core.selectors import PydanticMultiSelector
            
            selector = PydanticMultiSelector.from_defaults(**kwargs)
            logger.info("PydanticMultiSelectorを作成")
            return selector
        except Exception as e:
            logger.error(f"PydanticMultiSelector作成エラー: {e}")
            raise


class MetadataFilterFactory:
    """
    MetadataFilterファクトリークラス
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
    """
    
    @staticmethod
    def create_metadata_filter(
        key: str,
        value: Any,
        operator: str = "==",
        **kwargs
    ):
        """
        MetadataFilterを作成
        
        Args:
            key: フィルターキー
            value: フィルター値
            operator: 演算子
            **kwargs: 追加パラメータ
            
        Returns:
            MetadataFilter
        """
        try:
            from llama_index.core.vector_stores import MetadataFilter, FilterOperator
            
            # 演算子のマッピング
            operator_map = {
                "==": FilterOperator.EQ,
                "!=": FilterOperator.NE,
                ">": FilterOperator.GT,
                ">=": FilterOperator.GTE,
                "<": FilterOperator.LT,
                "<=": FilterOperator.LTE,
                "in": FilterOperator.IN,
                "nin": FilterOperator.NIN,
            }
            
            filter_operator = operator_map.get(operator, FilterOperator.EQ)
            
            metadata_filter = MetadataFilter(
                key=key,
                value=value,
                operator=filter_operator,
                **kwargs
            )
            logger.info(f"MetadataFilterを作成: {key} {operator} {value}")
            return metadata_filter
        except Exception as e:
            logger.error(f"MetadataFilter作成エラー: {e}")
            raise
    
    @staticmethod
    def create_metadata_filters(
        filters: List[Any],
        condition: str = "and",
        **kwargs
    ):
        """
        MetadataFiltersを作成
        
        Args:
            filters: フィルターのリスト
            condition: 条件 ("and" or "or")
            **kwargs: 追加パラメータ
            
        Returns:
            MetadataFilters
        """
        try:
            from llama_index.core.vector_stores import MetadataFilters, FilterCondition
            
            filter_condition = FilterCondition.AND if condition == "and" else FilterCondition.OR
            
            metadata_filters = MetadataFilters(
                filters=filters,
                condition=filter_condition,
                **kwargs
            )
            logger.info(f"MetadataFiltersを作成: {len(filters)}個のフィルター")
            return metadata_filters
        except Exception as e:
            logger.error(f"MetadataFilters作成エラー: {e}")
            raise

