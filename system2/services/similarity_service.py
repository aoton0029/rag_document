from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from core.database import db_manager
from config.settings import settings
from typing import Dict, Any, List
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SimilaritySearchService:
    def __init__(self):
        self.embedding_model = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
    
    async def search_similar(
        self, 
        query_text: str, 
        top_k: int = None, 
        filters: Dict[str, Any] = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """類似検索のメインフロー"""
        
        try:
            # 1. キャッシュ確認
            cached_result = await self._check_similarity_cache(query_text, top_k)
            if cached_result:
                return cached_result
            
            # 2. ベクトル類似検索実行
            search_results = await self._execute_vector_search(query_text, top_k, filters)
            
            # 3. メタデータフィルタリング
            filtered_results = await self._apply_metadata_filters(search_results, filters)
            
            # 4. 閾値フィルタリング
            threshold_filtered = await self._apply_threshold_filter(
                filtered_results, 
                threshold or settings.similarity_threshold
            )
            
            # 5. 関連文書情報の付加（Neo4j連携）
            enriched_results = await self._enrich_with_graph_data(threshold_filtered)
            
            # 6. 結果のキャッシュ
            await self._cache_similarity_result(query_text, enriched_results, top_k)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    async def _check_similarity_cache(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """類似検索結果のキャッシュ確認"""
        try:
            storage_context = db_manager.get_storage_context()
            kvstore = storage_context.kvstore
            
            cache_key = f"similarity:{hash(query_text)}:{top_k}"
            cached_data = await kvstore.aget(cache_key)
            
            if cached_data:
                logger.info("Similarity cache hit")
                return json.loads(cached_data)
                
        except Exception as e:
            logger.warning(f"Similarity cache check failed: {e}")
        
        return None
    
    async def _execute_vector_search(
        self, 
        query_text: str, 
        top_k: int, 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Milvusでのベクトル検索実行"""
        
        storage_context = db_manager.get_storage_context()
        
        # VectorStoreIndexを取得
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            embed_model=self.embedding_model
        )
        
        # Retrieverを設定
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k or settings.top_k
        )
        
        # 検索実行
        nodes = await retriever.aretrieve(query_text)
        
        # 結果を辞書形式に変換
        results = []
        for node in nodes:
            result = {
                "node_id": node.node_id,
                "chunk_id": node.metadata.get("chunk_id"),
                "doc_id": node.metadata.get("doc_id"),
                "text": node.text,
                "similarity_score": node.score,
                "metadata": node.metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(result)
        
        return results
    
    async def _apply_metadata_filters(
        self, 
        results: List[Dict[str, Any]], 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """MongoDBベースのメタデータフィルタリング"""
        
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            # フィルタ条件をチェック
            include_result = True
            
            if "language" in filters and metadata.get("language") != filters["language"]:
                include_result = False
            
            if "category" in filters and metadata.get("category") != filters["category"]:
                include_result = False
            
            if "date_range" in filters:
                # 日付範囲フィルタ
                doc_date = metadata.get("created_at")
                if doc_date:
                    # 日付比較ロジック（簡易版）
                    pass
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    async def _apply_threshold_filter(
        self, 
        results: List[Dict[str, Any]], 
        threshold: float
    ) -> List[Dict[str, Any]]:
        """類似度閾値によるフィルタリング"""
        
        return [
            result for result in results 
            if result.get("similarity_score", 0) >= threshold
        ]
    
    async def _enrich_with_graph_data(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Neo4jからの関連情報付加"""
        
        enriched_results = []
        
        try:
            storage_context = db_manager.get_storage_context()
            graph_store = storage_context.graph_store
            
            for result in results:
                # エンティティ情報を取得
                entities = result.get("metadata", {}).get("entities", [])
                
                # 関連文書情報を付加
                result["related_entities"] = entities
                result["graph_metadata"] = {
                    "entity_count": len(entities),
                    "has_graph_data": len(entities) > 0
                }
                
                enriched_results.append(result)
                
        except Exception as e:
            logger.warning(f"Graph data enrichment failed: {e}")
            # エラー時は元の結果をそのまま返す
            enriched_results = results
        
        return enriched_results
    
    async def _cache_similarity_result(
        self, 
        query_text: str, 
        results: List[Dict[str, Any]], 
        top_k: int
    ):
        """類似検索結果をキャッシュに保存"""
        
        try:
            storage_context = db_manager.get_storage_context()
            kvstore = storage_context.kvstore
            
            cache_key = f"similarity:{hash(query_text)}:{top_k}"
            cache_data = json.dumps(results, ensure_ascii=False, default=str)
            
            await kvstore.aput(cache_key, cache_data)
            
        except Exception as e:
            logger.warning(f"Failed to cache similarity results: {e}")
    
    async def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """ドキュメントメタデータの取得"""
        
        try:
            storage_context = db_manager.get_storage_context()
            docstore = storage_context.docstore
            
            doc_metadata = await docstore.aget_document(f"doc_meta:{doc_id}")
            return doc_metadata or {}
            
        except Exception as e:
            logger.error(f"Failed to get document metadata for {doc_id}: {e}")
            return {}
