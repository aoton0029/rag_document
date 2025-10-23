"""
検索基底クラス
Advanced RAG検索戦略とハイブリッド検索の実装
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    VectorContextRetriever,
    KnowledgeGraphRAGRetriever
)
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core import Settings
from llama_index.core.schema import BaseNode, QueryBundle, NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever as LlamaBaseRetriever

from ..utils import get_logger, performance_monitor


class SearchMode(Enum):
    """検索モード"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    FUSION = "fusion"
    ADAPTIVE = "adaptive"


class RetrievalStrategy(Enum):
    """検索戦略"""
    SIMPLE = "simple"
    RECURSIVE = "recursive"
    AUTO_MERGING = "auto_merging"
    FUSION = "fusion"
    ROUTER = "router"
    TRANSFORM = "transform"
    MULTI_STAGE = "multi_stage"


@dataclass
class SearchQuery:
    """検索クエリ"""
    text: str
    filters: Optional[Dict[str, Any]] = None
    similarity_top_k: int = 5
    mode: SearchMode = SearchMode.HYBRID
    metadata: Optional[Dict[str, Any]] = None
    
    def to_query_bundle(self) -> QueryBundle:
        """LlamaIndexのQueryBundleに変換"""
        return QueryBundle(
            query_str=self.text,
            custom_embedding_strs=[self.text] if self.text else None
        )


@dataclass
class RetrievalResult:
    """検索結果"""
    nodes: List[NodeWithScore]
    query: SearchQuery
    retrieval_time: float
    total_nodes_considered: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # 基本統計を追加
        if self.nodes:
            scores = [node.score for node in self.nodes if node.score is not None]
            self.metadata.update({
                "retrieved_count": len(self.nodes),
                "avg_score": np.mean(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0
            })
    
    def get_texts(self) -> List[str]:
        """取得したノードのテキスト一覧を取得"""
        return [node.node.get_content() for node in self.nodes]
    
    def get_top_k_nodes(self, k: int) -> List[NodeWithScore]:
        """上位k件のノードを取得"""
        return self.nodes[:k]
    
    def filter_by_score(self, min_score: float) -> List[NodeWithScore]:
        """スコア閾値でフィルタリング"""
        return [node for node in self.nodes if node.score and node.score >= min_score]


class BaseRetriever(ABC):
    """検索基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"retriever_{self.__class__.__name__}")
        
        # 基本パラメータ
        self.similarity_top_k = config.get("similarity_top_k", 5)
        self.similarity_cutoff = config.get("similarity_cutoff", 0.0)
        
        # 後処理設定
        self.enable_postprocessing = config.get("enable_postprocessing", True)
        self.postprocessors = self._setup_postprocessors() if self.enable_postprocessing else []
        
        # 並列処理設定
        self.max_workers = config.get("max_workers", 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def _setup_postprocessors(self) -> List[Any]:
        """後処理器を設定"""
        postprocessors = []
        
        postprocessor_config = self.config.get("postprocessors", {})
        
        # 類似度フィルター
        if postprocessor_config.get("similarity_filter", True):
            cutoff = postprocessor_config.get("similarity_cutoff", self.similarity_cutoff)
            if cutoff > 0:
                postprocessors.append(SimilarityPostprocessor(similarity_cutoff=cutoff))
        
        # キーワードフィルター
        if postprocessor_config.get("keyword_filter", False):
            keywords = postprocessor_config.get("required_keywords", [])
            if keywords:
                postprocessors.append(KeywordNodePostprocessor(keywords=keywords))
        
        return postprocessors
    
    @abstractmethod
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """内部的な検索処理（実装必須）"""
        pass
    
    def retrieve(self, query: Union[str, SearchQuery]) -> RetrievalResult:
        """検索実行"""
        
        # クエリ正規化
        if isinstance(query, str):
            search_query = SearchQuery(text=query)
        else:
            search_query = query
        
        self.logger.info("Starting retrieval", 
                        query_text=search_query.text[:100],
                        mode=search_query.mode.value,
                        top_k=search_query.similarity_top_k)
        
        import time
        start_time = time.time()
        
        with performance_monitor(f"retrieval_{search_query.mode.value}"):
            # 検索実行
            nodes = self._retrieve_internal(search_query)
            
            # 後処理適用
            if self.postprocessors:
                nodes = self._apply_postprocessing(nodes, search_query)
            
            # 結果数制限
            if len(nodes) > search_query.similarity_top_k:
                nodes = nodes[:search_query.similarity_top_k]
        
        retrieval_time = time.time() - start_time
        
        result = RetrievalResult(
            nodes=nodes,
            query=search_query,
            retrieval_time=retrieval_time,
            total_nodes_considered=len(nodes),
            metadata={
                "postprocessor_count": len(self.postprocessors)
            }
        )
        
        self.logger.info("Retrieval completed",
                        retrieved_count=len(nodes),
                        retrieval_time=retrieval_time)
        
        return result
    
    def _apply_postprocessing(self, nodes: List[NodeWithScore], 
                            query: SearchQuery) -> List[NodeWithScore]:
        """後処理を適用"""
        processed_nodes = nodes
        
        for postprocessor in self.postprocessors:
            try:
                # QueryBundleが必要な場合
                query_bundle = query.to_query_bundle()
                processed_nodes = postprocessor.postprocess_nodes(
                    processed_nodes, query_bundle
                )
            except Exception as e:
                self.logger.warning(f"Postprocessor {postprocessor} failed: {e}")
        
        return processed_nodes
    
    def batch_retrieve(self, queries: List[Union[str, SearchQuery]]) -> List[RetrievalResult]:
        """バッチ検索"""
        results = []
        
        for query in queries:
            try:
                result = self.retrieve(query)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to retrieve for query {query}: {e}")
                # エラー時は空の結果を追加
                search_query = SearchQuery(text=str(query)) if isinstance(query, str) else query
                results.append(RetrievalResult(
                    nodes=[],
                    query=search_query,
                    retrieval_time=0,
                    total_nodes_considered=0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    async def retrieve_async(self, query: Union[str, SearchQuery]) -> RetrievalResult:
        """非同期検索"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.retrieve, query)
    
    async def batch_retrieve_async(self, queries: List[Union[str, SearchQuery]]) -> List[RetrievalResult]:
        """非同期バッチ検索"""
        tasks = [self.retrieve_async(query) for query in queries]
        return await asyncio.gather(*tasks)
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """検索器統計を取得"""
        return {
            "retriever_type": self.__class__.__name__,
            "similarity_top_k": self.similarity_top_k,
            "similarity_cutoff": self.similarity_cutoff,
            "postprocessor_count": len(self.postprocessors),
            "max_workers": self.max_workers
        }
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class MultiStageRetriever(BaseRetriever):
    """マルチステージ検索器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.stages = config.get("stages", [])
        self.stage_retrievers = []
        
        # 各ステージの検索器を初期化
        for stage_config in self.stages:
            retriever = self._create_stage_retriever(stage_config)
            if retriever:
                self.stage_retrievers.append((stage_config, retriever))
    
    def _create_stage_retriever(self, stage_config: Dict[str, Any]) -> Optional[Any]:
        """ステージ検索器を作成"""
        # 実装は具体的な検索器に依存
        # ここでは基本的な構造のみ定義
        return None
    
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """マルチステージ検索"""
        all_nodes = []
        
        for stage_config, retriever in self.stage_retrievers:
            try:
                # ステージ固有の設定を適用
                stage_query = SearchQuery(
                    text=query.text,
                    filters=query.filters,
                    similarity_top_k=stage_config.get("top_k", query.similarity_top_k),
                    mode=SearchMode(stage_config.get("mode", query.mode.value)),
                    metadata={**query.metadata, "stage": stage_config.get("name", "unknown")}
                )
                
                # LlamaIndex retrieverの場合
                if hasattr(retriever, 'retrieve'):
                    query_bundle = stage_query.to_query_bundle()
                    stage_nodes = retriever.retrieve(query_bundle)
                else:
                    # カスタム検索器の場合
                    stage_result = retriever.retrieve(stage_query)
                    stage_nodes = stage_result.nodes if hasattr(stage_result, 'nodes') else stage_result
                
                # ステージメタデータを追加
                for node in stage_nodes:
                    if hasattr(node, 'metadata'):
                        node.metadata = node.metadata or {}
                        node.metadata["retrieval_stage"] = stage_config.get("name", "unknown")
                
                all_nodes.extend(stage_nodes)
                
            except Exception as e:
                self.logger.error(f"Stage {stage_config.get('name')} failed: {e}")
        
        # 重複除去とスコアベースソート
        unique_nodes = self._deduplicate_nodes(all_nodes)
        sorted_nodes = sorted(unique_nodes, 
                            key=lambda x: x.score if x.score else 0, 
                            reverse=True)
        
        return sorted_nodes
    
    def _deduplicate_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """ノードの重複除去"""
        seen_ids = set()
        unique_nodes = []
        
        for node in nodes:
            node_id = getattr(node.node, 'node_id', None) or getattr(node.node, 'id_', None)
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_nodes.append(node)
            elif not node_id:
                # IDがない場合はテキストで判定
                text_hash = hash(node.node.get_content()[:100])
                if text_hash not in seen_ids:
                    seen_ids.add(text_hash)
                    unique_nodes.append(node)
        
        return unique_nodes


class VectorRetriever(BaseRetriever):
    """ベクトル検索器"""
    
    def __init__(self, config: Dict[str, Any], vector_index: Any):
        super().__init__(config)
        self.vector_index = vector_index
        
        # VectorIndexRetrieverを作成
        self.llama_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self.similarity_top_k
        )
    
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """ベクトル検索実行"""
        query_bundle = query.to_query_bundle()
        
        # LlamaIndexの検索器を使用
        nodes = self.llama_retriever.retrieve(query_bundle)
        
        return nodes


class KeywordRetriever(BaseRetriever):
    """キーワード検索器"""
    
    def __init__(self, config: Dict[str, Any], keyword_index: Any):
        super().__init__(config)
        self.keyword_index = keyword_index
        
        # KeywordTableSimpleRetrieverを作成
        self.llama_retriever = KeywordTableSimpleRetriever(
            index=keyword_index
        )
    
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """キーワード検索実行"""
        query_bundle = query.to_query_bundle()
        
        nodes = self.llama_retriever.retrieve(query_bundle)
        
        return nodes


class FusionRetriever(BaseRetriever):
    """フュージョン検索器"""
    
    def __init__(self, config: Dict[str, Any], retrievers: List[Any]):
        super().__init__(config)
        
        # QueryFusionRetrieverを作成
        self.llama_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=self.similarity_top_k,
            num_queries=config.get("num_queries", 4)
        )
    
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """フュージョン検索実行"""
        query_bundle = query.to_query_bundle()
        
        nodes = self.llama_retriever.retrieve(query_bundle)
        
        return nodes


class AutoMergingRetriever(BaseRetriever):
    """自動マージ検索器"""
    
    def __init__(self, config: Dict[str, Any], vector_retriever: Any, storage_context: Any):
        super().__init__(config)
        
        # AutoMergingRetrieverを作成
        self.llama_retriever = AutoMergingRetriever(
            vector_retriever=vector_retriever,
            storage_context=storage_context,
            similarity_top_k=self.similarity_top_k
        )
    
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """自動マージ検索実行"""
        query_bundle = query.to_query_bundle()
        
        nodes = self.llama_retriever.retrieve(query_bundle)
        
        return nodes


class RecursiveRetriever(BaseRetriever):
    """再帰検索器"""
    
    def __init__(self, config: Dict[str, Any], retriever_dict: Dict[str, Any]):
        super().__init__(config)
        
        # RecursiveRetrieverを作成
        self.llama_retriever = RecursiveRetriever(
            root_id=config.get("root_id", "root"),
            retriever_dict=retriever_dict
        )
    
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """再帰検索実行"""
        query_bundle = query.to_query_bundle()
        
        nodes = self.llama_retriever.retrieve(query_bundle)
        
        return nodes


class RouterRetriever(BaseRetriever):
    """ルーター検索器"""
    
    def __init__(self, config: Dict[str, Any], selector: Any, retriever_tools: List[Any]):
        super().__init__(config)
        
        # RouterRetrieverを作成
        self.llama_retriever = RouterRetriever(
            selector=selector,
            retriever_tools=retriever_tools
        )
    
    def _retrieve_internal(self, query: SearchQuery) -> List[NodeWithScore]:
        """ルーター検索実行"""
        query_bundle = query.to_query_bundle()
        
        nodes = self.llama_retriever.retrieve(query_bundle)
        
        return nodes
