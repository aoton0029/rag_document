from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from core.database import db_manager
from config.settings import settings
from typing import Dict, Any, List
import uuid
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.llm = Ollama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url
        )
        self.embedding_model = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
    
    async def query(self, query_text: str, user_id: str = None, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """RAG検索のメインフロー"""
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # 1. 入力クエリの正規化
            normalized_query = await self._normalize_query(query_text)
            
            # 2. キャッシュ確認
            cached_result = await self._check_cache(normalized_query)
            if cached_result:
                return cached_result
            
            # 3. ベクトル検索実行
            retrieval_results = await self._vector_search(normalized_query, filters)
            
            # 4. 再ランキング（オプション）
            reranked_results = await self._rerank_results(normalized_query, retrieval_results)
            
            # 5. コンテキスト構築
            context = await self._build_context(reranked_results)
            
            # 6. LLM回答生成
            response = await self._generate_response(normalized_query, context)
            
            # 7. 結果の構造化と保存
            result = await self._format_and_save_result(
                query_id, query_text, response, reranked_results, user_id, start_time
            )
            
            # 8. キャッシュ保存
            await self._cache_result(normalized_query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"RAG query failed for query_id {query_id}: {e}")
            raise
    
    async def _normalize_query(self, query_text: str) -> str:
        """クエリの正規化"""
        # 基本的なクリーニング
        normalized = query_text.strip()
        
        # 言語判定等の前処理
        return normalized
    
    async def _check_cache(self, query_text: str) -> Dict[str, Any]:
        """Redis キャッシュから結果確認"""
        try:
            storage_context = db_manager.get_storage_context()
            kvstore = storage_context.kvstore
            
            cache_key = f"rag_query:{hash(query_text)}"
            cached_data = await kvstore.aget(cache_key)
            
            if cached_data:
                logger.info("Cache hit for query")
                return json.loads(cached_data)
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None
    
    async def _vector_search(self, query_text: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """ベクトル類似検索の実行"""
        storage_context = db_manager.get_storage_context()
        
        # VectorStoreIndexから検索
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            embed_model=self.embedding_model
        )
        
        # Retrieverを設定
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=settings.top_k
        )
        
        # 検索実行
        nodes = await retriever.aretrieve(query_text)
        
        # 結果を辞書形式に変換
        results = []
        for node in nodes:
            results.append({
                "node_id": node.node_id,
                "text": node.text,
                "score": node.score,
                "metadata": node.metadata
            })
        
        return results
    
    async def _rerank_results(self, query_text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """結果の再ランキング（LLMベース）"""
        # 簡易版：スコアによるソート
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    
    async def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """LLM用コンテキストの構築"""
        context_parts = []
        
        for i, result in enumerate(results[:5]):  # 上位5件
            context_parts.append(f"[文書{i+1}] {result['text']}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """LLMによる回答生成"""
        prompt = f"""
以下の文書を参考にして、質問に答えてください。
回答には必ず出典を明記してください。

質問: {query}

参考文書:
{context}

回答:
"""
        
        response = await self.llm.acomplete(prompt)
        return response.text
    
    async def _format_and_save_result(
        self, 
        query_id: str, 
        original_query: str, 
        response: str, 
        retrieved_docs: List[Dict[str, Any]], 
        user_id: str, 
        start_time: datetime
    ) -> Dict[str, Any]:
        """結果の整形と保存"""
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        result = {
            "query_id": query_id,
            "query_text": original_query,
            "response": response,
            "sources": [
                {
                    "chunk_id": doc["metadata"].get("chunk_id"),
                    "doc_id": doc["metadata"].get("doc_id"),
                    "score": doc["score"],
                    "text_snippet": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                } 
                for doc in retrieved_docs[:3]
            ],
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # MongoDB にクエリログを保存
        try:
            storage_context = db_manager.get_storage_context()
            docstore = storage_context.docstore
            
            log_doc = {
                "query_id": query_id,
                "user_id": user_id,
                "query_text": original_query,
                "retrieved_chunk_ids": [doc["metadata"].get("chunk_id") for doc in retrieved_docs],
                "retrieval_scores": [doc["score"] for doc in retrieved_docs],
                "response_text": response,
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow()
            }
            
            await docstore.aset_document(f"query_log:{query_id}", log_doc)
            
        except Exception as e:
            logger.error(f"Failed to save query log: {e}")
        
        return result
    
    async def _cache_result(self, query_text: str, result: Dict[str, Any]):
        """結果をキャッシュに保存"""
        try:
            storage_context = db_manager.get_storage_context()
            kvstore = storage_context.kvstore
            
            cache_key = f"rag_query:{hash(query_text)}"
            await kvstore.aput(cache_key, json.dumps(result, ensure_ascii=False))
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
