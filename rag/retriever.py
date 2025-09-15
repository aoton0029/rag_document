import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from ..database.document_db.mongo_client import MongoClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.vector_db.milvus_client import MilvusClient
from ..database.graph_db.neo4j_client import Neo4jClient
from ..embedding.embedding_service import EmbeddingService, EmbeddingConfig


@dataclass
class SearchConfig:
    """検索設定"""
    vector_top_k: int = 10
    keyword_top_k: int = 5
    hybrid_alpha: float = 0.7  # ベクトル検索の重み (0-1)
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    enable_graph_expansion: bool = True
    max_graph_depth: int = 2
    cache_results: bool = True
    cache_expire_seconds: int = 1800


@dataclass
class SearchResult:
    """検索結果"""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_type: str  # "vector", "keyword", "graph"
    rank: int


class RAGRetriever(BaseRetriever):
    """
    統合検索・取得エンジン
    ベクトル検索、キーワード検索、グラフ検索を統合
    """
    
    def __init__(
        self,
        config: SearchConfig = None,
        embedding_service: EmbeddingService = None,
        mongo_client: MongoClient = None,
        redis_client: RedisClient = None,
        milvus_client: MilvusClient = None,
        neo4j_client: Neo4jClient = None
    ):
        super().__init__()
        self.config = config or SearchConfig()
        self.logger = logging.getLogger(__name__)
        
        # サービス初期化
        self.embedding_service = embedding_service or self._create_embedding_service()
        self.mongo_client = mongo_client or MongoClient()
        self.redis_client = redis_client or RedisClient()
        self.milvus_client = milvus_client or MilvusClient()
        self.neo4j_client = neo4j_client or Neo4jClient()
        
        # 統計情報
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "vector_searches": 0,
            "keyword_searches": 0,
            "graph_searches": 0,
            "average_response_time": 0.0
        }
        
        self.logger.info("RAGRetriever初期化完了")
    
    def _create_embedding_service(self) -> EmbeddingService:
        """埋め込みサービス作成"""
        config = EmbeddingConfig()
        return EmbeddingService(config)
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """非同期検索実行（LlamaIndex BaseRetriever実装）"""
        query = query_bundle.query_str
        
        # 検索実行
        search_results = await self.hybrid_search(
            query=query,
            top_k=self.config.vector_top_k,
            filters=getattr(query_bundle, 'custom_embedding_strs', None)
        )
        
        # NodeWithScoreに変換
        nodes_with_scores = []
        for result in search_results:
            text_node = TextNode(
                text=result.content,
                metadata=result.metadata
            )
            text_node.node_id = result.chunk_id
            
            node_with_score = NodeWithScore(
                node=text_node,
                score=result.score
            )
            nodes_with_scores.append(node_with_score)
        
        return nodes_with_scores
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """同期検索実行（LlamaIndex BaseRetriever実装）"""
        return asyncio.run(self._aretrieve(query_bundle))
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        search_types: List[str] = None
    ) -> List[SearchResult]:
        """
        ハイブリッド検索（ベクトル + キーワード + グラフ）
        
        Args:
            query: 検索クエリ
            top_k: 取得件数
            filters: 検索フィルター
            search_types: 検索タイプリスト ["vector", "keyword", "graph"]
        
        Returns:
            検索結果リスト
        """
        start_time = datetime.utcnow()
        top_k = top_k or self.config.vector_top_k
        search_types = search_types or ["vector", "keyword", "graph"]
        
        # クエリハッシュ生成（キャッシュキー用）
        query_hash = self._generate_query_hash(query, top_k, filters, search_types)
        
        # キャッシュチェック
        if self.config.cache_results:
            cached_results = await self._get_cached_results(query_hash)
            if cached_results:
                self.search_stats["cache_hits"] += 1
                return cached_results
        
        try:
            # 並行検索実行
            search_tasks = []
            
            if "vector" in search_types:
                search_tasks.append(self._vector_search(query, top_k, filters))
            
            if "keyword" in search_types:
                search_tasks.append(self._keyword_search(query, top_k, filters))
            
            if "graph" in search_types and self.config.enable_graph_expansion:
                search_tasks.append(self._graph_search(query, top_k, filters))
            
            # 検索結果取得
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # 結果統合
            all_results = []
            for i, result in enumerate(search_results):
                if not isinstance(result, Exception) and result:
                    all_results.extend(result)
            
            # スコア正規化と統合
            final_results = await self._merge_and_rerank_results(
                all_results, query, top_k
            )
            
            # キャッシュ保存
            if self.config.cache_results and final_results:
                await self._cache_results(query_hash, final_results)
            
            # 統計更新
            self._update_search_stats(start_time, search_types)
            
            self.logger.info(f"ハイブリッド検索完了: クエリ='{query}', 結果数={len(final_results)}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"ハイブリッド検索エラー: {e}")
            return []
    
    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """ベクトル類似検索"""
        try:
            # クエリ埋め込み生成
            query_embeddings = await self.embedding_service.create_embeddings(
                [query], model_name="ollama"
            )
            
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Milvus検索
            milvus_results = self.milvus_client.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                document_filter=filters.get("document_id") if filters else None
            )
            
            # 結果変換
            results = []
            for i, result in enumerate(milvus_results):
                if result["score"] >= self.config.similarity_threshold:
                    search_result = SearchResult(
                        document_id=result["document_id"],
                        chunk_id=result["chunk_id"],
                        content=result["text"],
                        score=result["score"],
                        metadata={"source": "milvus"},
                        source_type="vector",
                        rank=i + 1
                    )
                    results.append(search_result)
            
            self.search_stats["vector_searches"] += 1
            self.logger.info(f"ベクトル検索完了: {len(results)} 件")
            return results
            
        except Exception as e:
            self.logger.error(f"ベクトル検索エラー: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """キーワード検索（MongoDB全文検索）"""
        try:
            # MongoDB全文検索
            search_query = {"$text": {"$search": query}}
            
            # フィルター適用
            if filters:
                search_query.update(filters)
            
            # MongoDB検索実行
            cursor = self.mongo_client.documents.find(
                search_query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
            
            results = []
            for i, doc in enumerate(cursor):
                search_result = SearchResult(
                    document_id=doc.get("document_id", ""),
                    chunk_id=doc.get("chunk_id", str(doc["_id"])),
                    content=doc.get("content", ""),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                    source_type="keyword",
                    rank=i + 1
                )
                results.append(search_result)
            
            self.search_stats["keyword_searches"] += 1
            self.logger.info(f"キーワード検索完了: {len(results)} 件")
            return results
            
        except Exception as e:
            self.logger.error(f"キーワード検索エラー: {e}")
            return []
    
    async def _graph_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """グラフ検索（関連エンティティ経由）"""
        try:
            # クエリからエンティティ抽出（簡易実装）
            entities = await self._extract_query_entities(query)
            
            if not entities:
                return []
            
            results = []
            for entity in entities[:3]:  # 最大3エンティティ
                # 関連ドキュメント検索
                related_docs = self.neo4j_client.find_related_documents(
                    entity_id=entity,
                    max_depth=self.config.max_graph_depth
                )
                
                for doc_info in related_docs[:top_k // len(entities)]:
                    # MongoDBから実際のコンテンツ取得
                    doc_content = self.mongo_client.get_document(doc_info["document_id"])
                    
                    if doc_content:
                        search_result = SearchResult(
                            document_id=doc_info["document_id"],
                            chunk_id=f"{doc_info['document_id']}_graph",
                            content=doc_content.get("content", "")[:1000],  # 最初の1000文字
                            score=1.0 / (doc_info.get("distance", 1) + 1),  # 距離逆数
                            metadata={
                                **doc_info.get("metadata", {}),
                                "graph_entity": entity,
                                "graph_distance": doc_info.get("distance", 0)
                            },
                            source_type="graph",
                            rank=len(results) + 1
                        )
                        results.append(search_result)
            
            self.search_stats["graph_searches"] += 1
            self.logger.info(f"グラフ検索完了: {len(results)} 件")
            return results
            
        except Exception as e:
            self.logger.error(f"グラフ検索エラー: {e}")
            return []
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """クエリからエンティティを抽出（簡易実装）"""
        # 実際の実装では、NERモデルや固有表現抽出を使用
        import re
        
        # 大文字で始まる単語を固有名詞として抽出
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # 重複除去して返す
        return list(set(entities))
    
    async def _merge_and_rerank_results(
        self,
        all_results: List[SearchResult],
        query: str,
        top_k: int
    ) -> List[SearchResult]:
        """検索結果をマージして再ランキング"""
        if not all_results:
            return []
        
        # 重複除去（chunk_idベース）
        unique_results = {}
        for result in all_results:
            key = f"{result.document_id}_{result.chunk_id}"
            if key not in unique_results or result.score > unique_results[key].score:
                unique_results[key] = result
        
        deduplicated_results = list(unique_results.values())
        
        # 再ランキング
        if self.config.enable_reranking and len(deduplicated_results) > 1:
            reranked_results = await self._rerank_results(deduplicated_results, query)
        else:
            # スコアでソート
            reranked_results = sorted(
                deduplicated_results,
                key=lambda x: x.score,
                reverse=True
            )
        
        # 上位top_k件を返す
        final_results = reranked_results[:top_k]
        
        # ランク更新
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        return final_results
    
    async def _rerank_results(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """結果の再ランキング（クロスエンコーダーや複合スコアリング）"""
        try:
            # 複合スコアリング実装
            for result in results:
                # 基本スコア
                base_score = result.score
                
                # ソースタイプ重み
                source_weights = {
                    "vector": 1.0,
                    "keyword": 0.8,
                    "graph": 0.6
                }
                source_weight = source_weights.get(result.source_type, 0.5)
                
                # コンテンツ長さボーナス（適度な長さを優遇）
                content_length = len(result.content)
                length_bonus = 1.0
                if 100 <= content_length <= 1000:
                    length_bonus = 1.1
                elif content_length > 1000:
                    length_bonus = 0.9
                
                # クエリとの語彙重複度
                query_words = set(query.lower().split())
                content_words = set(result.content.lower().split())
                overlap_ratio = len(query_words & content_words) / len(query_words) if query_words else 0
                overlap_bonus = 1 + (overlap_ratio * 0.2)
                
                # 複合スコア計算
                final_score = base_score * source_weight * length_bonus * overlap_bonus
                result.score = final_score
            
            # スコアでソート
            return sorted(results, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"再ランキングエラー: {e}")
            return results
    
    def _generate_query_hash(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        search_types: List[str]
    ) -> str:
        """クエリハッシュ生成"""
        hash_input = f"{query}_{top_k}_{str(filters)}_{str(sorted(search_types))}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def _get_cached_results(self, query_hash: str) -> Optional[List[SearchResult]]:
        """キャッシュから検索結果取得"""
        try:
            cached_data = self.redis_client.get_search_results(query_hash)
            if cached_data:
                # 辞書からSearchResultオブジェクトに復元
                results = []
                for item in cached_data:
                    result = SearchResult(**item)
                    results.append(result)
                return results
        except Exception as e:
            self.logger.warning(f"キャッシュ取得エラー: {e}")
        return None
    
    async def _cache_results(self, query_hash: str, results: List[SearchResult]):
        """検索結果をキャッシュ"""
        try:
            # SearchResultオブジェクトを辞書に変換
            serializable_results = []
            for result in results:
                result_dict = {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                    "source_type": result.source_type,
                    "rank": result.rank
                }
                serializable_results.append(result_dict)
            
            self.redis_client.set_search_results(
                query_hash,
                serializable_results,
                self.config.cache_expire_seconds
            )
        except Exception as e:
            self.logger.warning(f"キャッシュ保存エラー: {e}")
    
    def _update_search_stats(self, start_time: datetime, search_types: List[str]):
        """検索統計更新"""
        elapsed_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.search_stats["total_searches"] += 1
        
        # 平均応答時間更新
        total_time = (self.search_stats["average_response_time"] * 
                     (self.search_stats["total_searches"] - 1) + elapsed_time)
        self.search_stats["average_response_time"] = total_time / self.search_stats["total_searches"]
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """検索統計取得"""
        return {
            **self.search_stats,
            "cache_hit_rate": (self.search_stats["cache_hits"] / 
                              max(self.search_stats["total_searches"], 1)) * 100,
            "config": {
                "vector_top_k": self.config.vector_top_k,
                "similarity_threshold": self.config.similarity_threshold,
                "enable_reranking": self.config.enable_reranking,
                "enable_graph_expansion": self.config.enable_graph_expansion
            }
        }
    
    async def semantic_search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """純粋なベクトル類似検索"""
        filters = {"document_id": {"$in": document_ids}} if document_ids else None
        return await self._vector_search(query, top_k, filters)
    
    async def keyword_only_search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """純粋なキーワード検索"""
        filters = {"document_id": {"$in": document_ids}} if document_ids else None
        return await self._keyword_search(query, top_k, filters)
    
    async def clear_search_cache(self) -> bool:
        """検索キャッシュクリア"""
        try:
            cache_keys = self.redis_client.get_keys_by_pattern("search:*")
            for key in cache_keys:
                self.redis_client.delete_cache(key)
            self.logger.info(f"検索キャッシュクリア完了: {len(cache_keys)} 件")
            return True
        except Exception as e:
            self.logger.error(f"検索キャッシュクリアエラー: {e}")
            return False


# 使用例とテスト用のメイン関数
async def main():
    """テスト用メイン関数"""
    logging.basicConfig(level=logging.INFO)
    
    # 設定
    config = SearchConfig(
        vector_top_k=5,
        keyword_top_k=3,
        enable_reranking=True,
        enable_graph_expansion=True
    )
    
    # Retriever初期化
    retriever = RAGRetriever(config)
    
    # テスト検索
    test_queries = [
        "機械学習のアルゴリズムについて教えて",
        "自然言語処理の最新技術",
        "RAGシステムの仕組み"
    ]
    
    for query in test_queries:
        print(f"\n=== 検索テスト: {query} ===")
        
        try:
            # ハイブリッド検索
            results = await retriever.hybrid_search(query, top_k=3)
            
            print(f"検索結果数: {len(results)}")
            for i, result in enumerate(results):
                print(f"{i+1}. [スコア: {result.score:.3f}] {result.content[:100]}...")
                print(f"   ソース: {result.source_type}, ドキュメント: {result.document_id}")
        
        except Exception as e:
            print(f"検索エラー: {e}")
    
    # 統計情報表示
    stats = await retriever.get_search_stats()
    print(f"\n=== 検索統計 ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
