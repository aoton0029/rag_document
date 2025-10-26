"""
Retriever Module
各種Retrieverの実装
"""

import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
    SummaryIndexRetriever,
    RouterRetriever,
    TreeRootRetriever,
    TransformRetriever,
    QueryFusionRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    TreeSelectLeafRetriever,
    SummaryIndexEmbeddingRetriever,
    VectorIndexAutoRetriever,
    KnowledgeGraphRAGRetriever
)
from llama_index.core.retrievers.base import BaseRetriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor
)
from llama_index.core.indices.base import BaseIndex

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """
    Retrieverファクトリークラス
    設定に基づいてRetrieverを生成
    """
    
    @staticmethod
    def create_retriever(
        index: BaseIndex,
        retriever_type: str = "vector",
        **kwargs
    ) -> BaseRetriever:
        """
        Retrieverを作成
        
        Args:
            index: インデックス
            retriever_type: Retrieverタイプ
            **kwargs: Retrieverパラメータ
            
        Returns:
            BaseRetriever
        """
        if retriever_type == "vector":
            return VectorRetriever.create(index, **kwargs)
        elif retriever_type == "keyword":
            return KeywordRetriever.create(index, **kwargs)
        elif retriever_type == "summary":
            return SummaryRetriever.create(index, **kwargs)
        elif retriever_type == "hybrid":
            return HybridRetriever.create(index, **kwargs)
        else:
            logger.warning(f"未対応のretriever_type: {retriever_type}、デフォルトのベクトルRetrieverを使用")
            return VectorRetriever.create(index, **kwargs)


class VectorRetriever:
    """
    ベクトルRetriever
    """
    
    @staticmethod
    def create(
        index: BaseIndex,
        similarity_top_k: int = 10,
        filters: Optional[Any] = None,
        alpha: Optional[float] = None,
        **kwargs
    ) -> VectorIndexRetriever:
        """
        VectorIndexRetrieverを作成
        
        Args:
            index: インデックス
            similarity_top_k: 取得する上位k件
            filters: メタデータフィルター
            alpha: ハイブリッド検索のアルファ値
            **kwargs: 追加パラメータ
            
        Returns:
            VectorIndexRetriever
        """
        try:
            retriever_kwargs = {
                "similarity_top_k": similarity_top_k,
                **kwargs
            }
            
            if filters is not None:
                retriever_kwargs["filters"] = filters
            
            if alpha is not None:
                retriever_kwargs["alpha"] = alpha
            
            retriever = index.as_retriever(**retriever_kwargs)
            logger.info(f"VectorIndexRetrieverを作成: top_k={similarity_top_k}")
            return retriever
        except Exception as e:
            logger.error(f"VectorIndexRetriever作成エラー: {e}")
            raise


class KeywordRetriever:
    """
    キーワードRetriever
    """
    
    @staticmethod
    def create(
        index: BaseIndex,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        **kwargs
    ) -> BaseRetriever:
        """
        KeywordTableSimpleRetrieverを作成
        
        Args:
            index: インデックス
            max_keywords_per_query: クエリあたりの最大キーワード数
            num_chunks_per_query: クエリあたりの取得チャンク数
            **kwargs: 追加パラメータ
            
        Returns:
            BaseRetriever
        """
        try:
            from llama_index.core import KeywordTableIndex
            
            if not isinstance(index, KeywordTableIndex):
                logger.warning("KeywordTableIndexでない場合、キーワードRetrieverは使用できません")
                # フォールバック: 通常のRetrieverを返す
                return index.as_retriever(**kwargs)
            
            retriever = KeywordTableSimpleRetriever(
                index=index,
                max_keywords_per_query=max_keywords_per_query,
                num_chunks_per_query=num_chunks_per_query,
                **kwargs
            )
            logger.info(f"KeywordTableSimpleRetrieverを作成")
            return retriever
        except Exception as e:
            logger.error(f"KeywordTableSimpleRetriever作成エラー: {e}")
            raise


class SummaryRetriever:
    """
    サマリーRetriever
    """
    
    @staticmethod
    def create(
        index: BaseIndex,
        **kwargs
    ) -> BaseRetriever:
        """
        SummaryIndexRetrieverを作成
        
        Args:
            index: インデックス
            **kwargs: 追加パラメータ
            
        Returns:
            BaseRetriever
        """
        try:
            from llama_index.core import SummaryIndex
            
            if not isinstance(index, SummaryIndex):
                logger.warning("SummaryIndexでない場合、サマリーRetrieverは使用できません")
                return index.as_retriever(**kwargs)
            
            retriever = SummaryIndexRetriever(
                index=index,
                **kwargs
            )
            logger.info(f"SummaryIndexRetrieverを作成")
            return retriever
        except Exception as e:
            logger.error(f"SummaryIndexRetriever作成エラー: {e}")
            raise


class HybridRetriever:
    """
    ハイブリッドRetriever
    複数のRetrieverを組み合わせる
    """
    
    @staticmethod
    def create(
        index: BaseIndex,
        vector_top_k: int = 10,
        use_query_fusion: bool = False,
        num_queries: int = 4,
        **kwargs
    ) -> BaseRetriever:
        """
        ハイブリッドRetrieverを作成
        
        Args:
            index: インデックス
            vector_top_k: ベクトル検索の上位k件
            use_query_fusion: QueryFusionを使用するか
            num_queries: 生成するクエリ数（QueryFusion使用時）
            **kwargs: 追加パラメータ
            
        Returns:
            BaseRetriever
        """
        try:
            # ベースとなるベクトルRetrieverを作成
            base_retriever = index.as_retriever(
                similarity_top_k=vector_top_k,
                **kwargs
            )
            
            if use_query_fusion:
                # QueryFusionRetrieverを使用
                retriever = QueryFusionRetriever(
                    retrievers=[base_retriever],
                    num_queries=num_queries,
                    mode="reciprocal_rerank",
                    use_async=True
                )
                logger.info(f"QueryFusionRetrieverを作成")
            else:
                retriever = base_retriever
                logger.info(f"ベクトルRetrieverを作成（ハイブリッドモード）")
            
            return retriever
        except Exception as e:
            logger.error(f"ハイブリッドRetriever作成エラー: {e}")
            raise


class TreeRootRetrieverBuilder:
    """
    TreeRootRetrieverビルダー
    """
    
    @staticmethod
    def create(index: BaseIndex, **kwargs) -> TreeRootRetriever:
        """TreeRootRetrieverを作成"""
        try:
            from llama_index.core import TreeIndex
            if not isinstance(index, TreeIndex):
                logger.warning("TreeIndexではありません")
            retriever = TreeRootRetriever(index=index, **kwargs)
            logger.info("TreeRootRetrieverを作成")
            return retriever
        except Exception as e:
            logger.error(f"TreeRootRetriever作成エラー: {e}")
            raise


class AutoMergingRetrieverBuilder:
    """
    AutoMergingRetrieverビルダー
    """
    
    @staticmethod
    def create(
        index: BaseIndex,
        storage_context: Any,
        simple_ratio_thresh: float = 0.5,
        **kwargs
    ) -> AutoMergingRetriever:
        """AutoMergingRetrieverを作成"""
        try:
            vector_retriever = index.as_retriever(**kwargs)
            retriever = AutoMergingRetriever(
                vector_retriever=vector_retriever,
                storage_context=storage_context,
                simple_ratio_thresh=simple_ratio_thresh
            )
            logger.info("AutoMergingRetrieverを作成")
            return retriever
        except Exception as e:
            logger.error(f"AutoMergingRetriever作成エラー: {e}")
            raise


class RecursiveRetrieverBuilder:
    """
    RecursiveRetrieverビルダー
    """
    
    @staticmethod
    def create(
        root_id: str,
        retriever_dict: Dict[str, BaseRetriever],
        **kwargs
    ) -> RecursiveRetriever:
        """RecursiveRetrieverを作成"""
        try:
            retriever = RecursiveRetriever(
                root_id=root_id,
                retriever_dict=retriever_dict,
                **kwargs
            )
            logger.info("RecursiveRetrieverを作成")
            return retriever
        except Exception as e:
            logger.error(f"RecursiveRetriever作成エラー: {e}")
            raise


class VectorAutoRetrieverBuilder:
    """
    VectorIndexAutoRetrieverビルダー
    """
    
    @staticmethod
    def create(
        index: BaseIndex,
        vector_store_info: Any,
        llm: Optional[Any] = None,
        **kwargs
    ) -> VectorIndexAutoRetriever:
        """VectorIndexAutoRetrieverを作成"""
        try:
            retriever_kwargs = {
                "index": index,
                "vector_store_info": vector_store_info,
                **kwargs
            }
            if llm is not None:
                retriever_kwargs["llm"] = llm
            retriever = VectorIndexAutoRetriever(**retriever_kwargs)
            logger.info("VectorIndexAutoRetrieverを作成")
            return retriever
        except Exception as e:
            logger.error(f"VectorIndexAutoRetriever作成エラー: {e}")
            raise


class KnowledgeGraphRetrieverBuilder:
    """
    KnowledgeGraphRAGRetrieverビルダー
    """
    
    @staticmethod
    def create(index: BaseIndex, **kwargs) -> KnowledgeGraphRAGRetriever:
        """KnowledgeGraphRAGRetrieverを作成"""
        try:
            from llama_index.core import KnowledgeGraphIndex
            if not isinstance(index, KnowledgeGraphIndex):
                logger.warning("KnowledgeGraphIndexではありません")
            retriever = KnowledgeGraphRAGRetriever(
                storage_context=index.storage_context,
                **kwargs
            )
            logger.info("KnowledgeGraphRAGRetrieverを作成")
            return retriever
        except Exception as e:
            logger.error(f"KnowledgeGraphRAGRetriever作成エラー: {e}")
            raise


class PostprocessorFactory:
    """
    Postprocessorファクトリークラス
    """
    
    @staticmethod
    def create_similarity_postprocessor(
        similarity_cutoff: float = 0.7
    ) -> SimilarityPostprocessor:
        """
        SimilarityPostprocessorを作成
        
        Args:
            similarity_cutoff: 類似度カットオフ
            
        Returns:
            SimilarityPostprocessor
        """
        return SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    
    @staticmethod
    def create_keyword_postprocessor(
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None
    ) -> KeywordNodePostprocessor:
        """
        KeywordNodePostprocessorを作成
        
        Args:
            required_keywords: 必須キーワード
            exclude_keywords: 除外キーワード
            
        Returns:
            KeywordNodePostprocessor
        """
        return KeywordNodePostprocessor(
            required_keywords=required_keywords,
            exclude_keywords=exclude_keywords
        )
