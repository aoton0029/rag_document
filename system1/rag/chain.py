import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import uuid
import re
from llama_index.core import QueryBundle
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response import Response
from llama_index.core.callbacks import CallbackManager
from .retriever import RAGRetriever, SearchConfig, SearchResult
from .generator import RAGGenerator, GenerationConfig, GenerationResult
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.document_db.mongo_client import MongoClient
from ..database.relational_db.models import QueryLog


@dataclass
class RAGChainConfig:
    """RAGチェーン設定"""
    retrieval_config: SearchConfig = None
    generation_config: GenerationConfig = None
    enable_conversation_memory: bool = True
    max_conversation_turns: int = 10
    enable_query_preprocessing: bool = True
    enable_response_postprocessing: bool = True
    enable_evaluation: bool = False
    log_conversations: bool = True
    conversation_expire_seconds: int = 7200  # 2時間


@dataclass
class ConversationTurn:
    """会話ターン"""
    turn_id: str
    user_query: str
    assistant_response: str
    search_results: List[SearchResult]
    generation_info: Dict[str, Any]
    timestamp: datetime
    evaluation_score: Optional[float] = None


@dataclass
class RAGChainResult:
    """RAGチェーン実行結果"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    turn_id: str
    retrieval_time: float
    generation_time: float
    total_time: float
    search_results_count: int
    cached_retrieval: bool
    cached_generation: bool
    metadata: Dict[str, Any]


class RAGChain(BaseQueryEngine):
    """
    RAGチェーン - 統合検索・生成エンジン
    Retriever と Generator を組み合わせて完全なRAGパイプラインを提供
    """
    
    def __init__(
        self,
        config: RAGChainConfig = None,
        retriever: RAGRetriever = None,
        generator: RAGGenerator = None,
        redis_client: RedisClient = None,
        mongo_client: MongoClient = None,
        callback_manager: CallbackManager = None
    ):
        super().__init__(callback_manager=callback_manager)
        
        self.config = config or RAGChainConfig()
        self.logger = logging.getLogger(__name__)
        
        # サブコンポーネント初期化
        self.retriever = retriever or RAGRetriever(
            self.config.retrieval_config or SearchConfig()
        )
        self.generator = generator or RAGGenerator(
            self.config.generation_config or GenerationConfig()
        )
        
        # Database clients
        self.redis_client = redis_client or RedisClient()
        self.mongo_client = mongo_client or MongoClient()
        
        # クエリ前処理パイプライン
        self.query_preprocessors = self._initialize_query_preprocessors()
        
        # レスポンス後処理パイプライン
        self.response_postprocessors = self._initialize_response_postprocessors()
        
        # 統計情報
        self.chain_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_total_time": 0.0,
            "average_retrieval_time": 0.0,
            "average_generation_time": 0.0,
            "conversations_created": 0,
            "total_turns": 0
        }
        
        self.logger.info("RAGChain初期化完了")
    
    def _initialize_query_preprocessors(self) -> List[callable]:
        """クエリ前処理パイプライン初期化"""
        preprocessors = []
        
        if self.config.enable_query_preprocessing:
            preprocessors.extend([
                self._normalize_query,
                self._expand_acronyms,
                self._detect_intent,
                self._extract_filters
            ])
        
        return preprocessors
    
    def _initialize_response_postprocessors(self) -> List[callable]:
        """レスポンス後処理パイプライン初期化"""
        postprocessors = []
        
        if self.config.enable_response_postprocessing:
            postprocessors.extend([
                self._format_response,
                self._add_related_questions,
                self._validate_response
            ])
        
        return postprocessors
    
    async def query(
        self,
        query: Union[str, QueryBundle],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        custom_filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RAGChainResult:
        """
        RAGクエリ実行
        
        Args:
            query: ユーザークエリ
            conversation_id: 会話ID（継続会話の場合）
            user_id: ユーザーID
            custom_filters: カスタム検索フィルター
        
        Returns:
            RAG実行結果
        """
        start_time = time.time()
        
        # クエリ文字列取得
        if isinstance(query, QueryBundle):
            query_str = query.query_str
        else:
            query_str = str(query)
        
        # 会話ID生成
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            self.chain_stats["conversations_created"] += 1
        
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"RAGクエリ開始: {query_str[:100]}... (会話: {conversation_id})")
        
        try:
            # 1. クエリ前処理
            processed_query = query_str
            query_metadata = {}
            
            for preprocessor in self.query_preprocessors:
                processed_query, metadata = await preprocessor(processed_query, query_metadata)
                query_metadata.update(metadata)
            
            # 2. 会話履歴取得
            conversation_history = []
            if self.config.enable_conversation_memory and conversation_id:
                conversation_history = await self._get_conversation_history(conversation_id)
            
            # 3. 検索実行
            retrieval_start = time.time()
            search_results = await self.retriever.hybrid_search(
                query=processed_query,
                top_k=self.config.retrieval_config.vector_top_k if self.config.retrieval_config else 10,
                filters=custom_filters
            )
            retrieval_time = time.time() - retrieval_start
            
            # 4. 応答生成
            generation_start = time.time()
            generation_result = await self.generator.generate_response(
                query=processed_query,
                search_results=search_results,
                conversation_history=conversation_history
            )
            generation_time = time.time() - generation_start
            
            # 5. レスポンス後処理
            processed_response = generation_result.response
            for postprocessor in self.response_postprocessors:
                processed_response = await postprocessor(
                    processed_response, query_str, search_results
                )
            
            # 6. 会話履歴更新
            if self.config.enable_conversation_memory:
                await self._update_conversation_history(
                    conversation_id,
                    turn_id,
                    query_str,
                    processed_response,
                    search_results,
                    generation_result.model_info
                )
            
            # 7. 結果作成
            total_time = time.time() - start_time
            result = RAGChainResult(
                query=query_str,
                response=processed_response,
                sources=generation_result.sources,
                conversation_id=conversation_id,
                turn_id=turn_id,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                search_results_count=len(search_results),
                cached_retrieval=False,  # TODO: キャッシュ検出実装
                cached_generation=generation_result.cached,
                metadata={
                    **query_metadata,
                    "model_info": generation_result.model_info,
                    "user_id": user_id
                }
            )
            
            # 8. 統計更新
            self._update_chain_stats(result, success=True)
            
            # 9. ログ記録
            if self.config.log_conversations:
                await self._log_conversation_turn(result)
            
            self.logger.info(f"RAGクエリ完了: {total_time:.2f}s (検索: {retrieval_time:.2f}s, 生成: {generation_time:.2f}s)")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.chain_stats["failed_queries"] += 1
            self.logger.error(f"RAGクエリエラー: {e}")
            
            # エラー時のフォールバック応答
            return RAGChainResult(
                query=query_str,
                response=f"申し訳ありませんが、処理中にエラーが発生しました: {str(e)}",
                sources=[],
                conversation_id=conversation_id,
                turn_id=turn_id,
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=total_time,
                search_results_count=0,
                cached_retrieval=False,
                cached_generation=False,
                metadata={"error": str(e), "user_id": user_id}
            )
    
    async def _query(self, query_bundle: QueryBundle) -> Response:
        """LlamaIndex BaseQueryEngine実装"""
        result = await self.query(query_bundle)
        
        # LlamaIndex Response形式に変換
        return Response(
            response=result.response,
            source_nodes=[],  # TODO: NodeWithScore変換実装
            metadata={
                "conversation_id": result.conversation_id,
                "turn_id": result.turn_id,
                "total_time": result.total_time,
                "sources": result.sources
            }
        )
    
    async def stream_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        ストリーミングRAGクエリ
        
        Args:
            query: ユーザークエリ
            conversation_id: 会話ID
            user_id: ユーザーID
        
        Yields:
            ストリーミング応答チャンク
        """
        start_time = time.time()
        
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        
        try:
            # メタデータ送信
            yield {
                "type": "metadata",
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "query": query
            }
            
            # 検索実行
            yield {"type": "status", "message": "検索中..."}
            
            search_results = await self.retriever.hybrid_search(query)
            
            yield {
                "type": "search_complete",
                "results_count": len(search_results),
                "sources": [{"title": r.metadata.get("filename", "Unknown")} for r in search_results[:3]]
            }
            
            # ストリーミング生成
            yield {"type": "status", "message": "応答生成中..."}
            
            response_parts = []
            async for chunk in self.generator.generate_streaming_response(
                query, search_results
            ):
                response_parts.append(chunk)
                yield {
                    "type": "response_chunk",
                    "content": chunk
                }
            
            # 完了情報
            total_time = time.time() - start_time
            full_response = "".join(response_parts)
            
            yield {
                "type": "complete",
                "full_response": full_response,
                "total_time": total_time,
                "sources": self.generator._prepare_sources(search_results)
            }
            
            # 会話履歴更新
            if self.config.enable_conversation_memory:
                await self._update_conversation_history(
                    conversation_id, turn_id, query, full_response, search_results, {}
                )
            
        except Exception as e:
            yield {
                "type": "error",
                "message": f"エラーが発生しました: {str(e)}"
            }
    
    # クエリ前処理メソッド群
    async def _normalize_query(self, query: str, metadata: Dict[str, Any]) -> tuple:
        """クエリ正規化"""
        normalized = query.strip()
        
        # 全角文字を半角に変換
        normalized = normalized.translate(str.maketrans(
            "０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "0123456789abcdefghijklmnopqrstuvwxyz"
        ))
        
        # 重複スペース除去
        normalized = re.sub(r'\s+', ' ', normalized)
        
        metadata["normalization_applied"] = normalized != query
        return normalized, metadata
    
    async def _expand_acronyms(self, query: str, metadata: Dict[str, Any]) -> tuple:
        """略語展開"""
        acronym_dict = {
            "AI": "人工知能",
            "ML": "機械学習",
            "DL": "深層学習",
            "NLP": "自然言語処理",
            "RAG": "検索拡張生成",
            "LLM": "大規模言語モデル"
        }
        
        expanded_query = query
        expanded_terms = []
        
        for acronym, expansion in acronym_dict.items():
            if acronym in query and expansion not in query:
                expanded_query += f" {expansion}"
                expanded_terms.append(f"{acronym}->{expansion}")
        
        if expanded_terms:
            metadata["acronym_expansions"] = expanded_terms
        
        return expanded_query, metadata
    
    async def _detect_intent(self, query: str, metadata: Dict[str, Any]) -> tuple:
        """クエリ意図検出"""
        intent_patterns = {
            "question": r"[？?]|何|どう|なぜ|いつ|どこ|誰",
            "request": r"教えて|説明|について|に関して",
            "comparison": r"比較|違い|差|vs",
            "definition": r"とは|定義|意味",
            "procedure": r"方法|手順|やり方|どうやって"
        }
        
        detected_intents = []
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, query):
                detected_intents.append(intent)
        
        if detected_intents:
            metadata["detected_intents"] = detected_intents
        
        return query, metadata
    
    async def _extract_filters(self, query: str, metadata: Dict[str, Any]) -> tuple:
        """フィルター条件抽出"""
        # 日付パターン
        date_pattern = r"(\d{4})年|(\d{1,2})月|(\d{1,2})日"
        date_matches = re.findall(date_pattern, query)
        
        if date_matches:
            metadata["date_filters"] = date_matches
        
        # 著者パターン
        author_pattern = r"著者[:：]?\s*([^\s、。]+)"
        author_matches = re.findall(author_pattern, query)
        
        if author_matches:
            metadata["author_filters"] = author_matches
        
        return query, metadata
    
    # レスポンス後処理メソッド群
    async def _format_response(self, response: str, query: str, search_results: List[SearchResult]) -> str:
        """レスポンス整形"""
        # 改行の正規化
        formatted = re.sub(r'\n{3,}', '\n\n', response)
        
        # 不要な空白除去
        formatted = formatted.strip()
        
        return formatted
    
    async def _add_related_questions(self, response: str, query: str, search_results: List[SearchResult]) -> str:
        """関連質問追加"""
        # 検索結果から関連質問を生成（簡易実装）
        if len(search_results) >= 2:
            related_topics = []
            for result in search_results[:3]:
                # メタデータから関連トピック抽出
                if "keywords" in result.metadata:
                    related_topics.extend(result.metadata["keywords"].split(","))
            
            if related_topics:
                unique_topics = list(set(related_topics))[:3]
                related_questions = [f"- {topic}について詳しく教えて" for topic in unique_topics]
                
                response += "\n\n**関連する質問:**\n" + "\n".join(related_questions)
        
        return response
    
    async def _validate_response(self, response: str, query: str, search_results: List[SearchResult]) -> str:
        """レスポンス妥当性検証"""
        # 最小文字数チェック
        if len(response.strip()) < 10:
            return "申し訳ありませんが、十分な情報を見つけることができませんでした。より具体的な質問をお試しください。"
        
        # 検索結果との関連性チェック（簡易実装）
        if not search_results:
            response = "検索結果が見つかりませんでしたが、一般的な情報として:\n\n" + response
        
        return response
    
    # 会話履歴管理
    async def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """会話履歴取得"""
        try:
            history_key = f"conversation:{conversation_id}"
            cached_history = self.redis_client.get_cache(history_key)
            
            if cached_history and "turns" in cached_history:
                # 最新のN件のみ返す
                recent_turns = cached_history["turns"][-self.config.max_conversation_turns:]
                return [
                    {
                        "user": turn["user_query"],
                        "assistant": turn["assistant_response"]
                    }
                    for turn in recent_turns
                ]
            
            return []
            
        except Exception as e:
            self.logger.error(f"会話履歴取得エラー: {e}")
            return []
    
    async def _update_conversation_history(
        self,
        conversation_id: str,
        turn_id: str,
        user_query: str,
        assistant_response: str,
        search_results: List[SearchResult],
        generation_info: Dict[str, Any]
    ):
        """会話履歴更新"""
        try:
            history_key = f"conversation:{conversation_id}"
            
            # 既存履歴取得
            conversation_data = self.redis_client.get_cache(history_key) or {
                "conversation_id": conversation_id,
                "created_at": datetime.utcnow().isoformat(),
                "turns": []
            }
            
            # 新しいターン追加
            new_turn = {
                "turn_id": turn_id,
                "user_query": user_query,
                "assistant_response": assistant_response,
                "timestamp": datetime.utcnow().isoformat(),
                "search_results_count": len(search_results),
                "generation_info": generation_info
            }
            
            conversation_data["turns"].append(new_turn)
            conversation_data["updated_at"] = datetime.utcnow().isoformat()
            
            # 履歴長制限
            if len(conversation_data["turns"]) > self.config.max_conversation_turns:
                conversation_data["turns"] = conversation_data["turns"][-self.config.max_conversation_turns:]
            
            # Redis保存
            self.redis_client.set_cache(
                history_key,
                conversation_data,
                self.config.conversation_expire_seconds
            )
            
            self.chain_stats["total_turns"] += 1
            
        except Exception as e:
            self.logger.error(f"会話履歴更新エラー: {e}")
    
    async def _log_conversation_turn(self, result: RAGChainResult):
        """会話ターンログ記録"""
        try:
            log_data = {
                "conversation_id": result.conversation_id,
                "turn_id": result.turn_id,
                "query": result.query,
                "response_length": len(result.response),
                "sources_count": len(result.sources),
                "timing": {
                    "retrieval_time": result.retrieval_time,
                    "generation_time": result.generation_time,
                    "total_time": result.total_time
                },
                "cached": {
                    "retrieval": result.cached_retrieval,
                    "generation": result.cached_generation
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            log_key = f"conversation_log:{result.conversation_id}:{result.turn_id}"
            self.redis_client.set_cache(log_key, log_data, 86400)  # 24時間保持
            
            # SQL database logging (new)
            query_log = QueryLog(
                user_id=result.metadata.get("user_id"),
                query_text=result.query,
                response_text=result.response[:1000],  # Limit response length
                conversation_id=result.conversation_id,
                turn_id=result.turn_id,
                search_results_count=result.search_results_count,
                retrieval_time=result.retrieval_time,
                generation_time=result.generation_time,
                total_time=result.total_time,
                success=True,
                metadata_json={
                    "sources": result.sources[:5],  # Limit sources
                    "cached_retrieval": result.cached_retrieval,
                    "cached_generation": result.cached_generation,
                    "user_id": result.metadata.get("user_id")
                },
                created_at=datetime.utcnow()
            )
            
            # Save to SQL database
            # with get_db() as session:
            #     session.add(query_log)
            #     session.commit()
            
        except Exception as e:
            self.logger.warning(f"会話ログ記録エラー: {e}")
    
    def _update_chain_stats(self, result: RAGChainResult, success: bool):
        """チェーン統計更新"""
        self.chain_stats["total_queries"] += 1
        
        if success:
            self.chain_stats["successful_queries"] += 1
            
            # 平均時間更新
            total_successful = self.chain_stats["successful_queries"]
            
            # 平均総時間
            total_avg = self.chain_stats["average_total_time"]
            self.chain_stats["average_total_time"] = (
                (total_avg * (total_successful - 1) + result.total_time) / total_successful
            )
            
            # 平均検索時間
            retrieval_avg = self.chain_stats["average_retrieval_time"]
            self.chain_stats["average_retrieval_time"] = (
                (retrieval_avg * (total_successful - 1) + result.retrieval_time) / total_successful
            )
            
            # 平均生成時間
            generation_avg = self.chain_stats["average_generation_time"]
            self.chain_stats["average_generation_time"] = (
                (generation_avg * (total_successful - 1) + result.generation_time) / total_successful
            )
    
    # ユーティリティメソッド
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """会話データ取得"""
        try:
            history_key = f"conversation:{conversation_id}"
            return self.redis_client.get_cache(history_key)
        except Exception as e:
            self.logger.error(f"会話データ取得エラー: {e}")
            return None
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """会話削除"""
        try:
            history_key = f"conversation:{conversation_id}"
            return self.redis_client.delete_cache(history_key)
        except Exception as e:
            self.logger.error(f"会話削除エラー: {e}")
            return False
    
    async def get_chain_stats(self) -> Dict[str, Any]:
        """チェーン統計取得"""
        return {
            **self.chain_stats,
            "success_rate": (
                self.chain_stats["successful_queries"] / 
                max(self.chain_stats["total_queries"], 1)
            ) * 100,
            "config": {
                "enable_conversation_memory": self.config.enable_conversation_memory,
                "max_conversation_turns": self.config.max_conversation_turns,
                "enable_query_preprocessing": self.config.enable_query_preprocessing,
                "enable_response_postprocessing": self.config.enable_response_postprocessing
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        health = {}
        
        try:
            # Retriever ヘルスチェック
            retriever_stats = await self.retriever.get_search_stats()
            health["retriever"] = "healthy" if retriever_stats else "unhealthy"
        except Exception as e:
            health["retriever"] = f"error: {str(e)}"
        
        try:
            # Generator ヘルスチェック
            generator_health = await self.generator.health_check()
            health["generator"] = generator_health["status"]
        except Exception as e:
            health["generator"] = f"error: {str(e)}"
        
        try:
            # Redis接続確認
            test_key = "health_check_test"
            self.redis_client.set_cache(test_key, "test", 10)
            test_value = self.redis_client.get_cache(test_key)
            health["redis"] = "healthy" if test_value == "test" else "unhealthy"
            self.redis_client.delete_cache(test_key)
        except Exception as e:
            health["redis"] = f"error: {str(e)}"
        
        # 全体ステータス
        all_healthy = all(
            status == "healthy" for status in health.values() 
            if not status.startswith("error:")
        )
        health["overall"] = "healthy" if all_healthy else "unhealthy"
        
        return health


# 使用例とテスト用のメイン関数
async def main():
    """テスト用メイン関数"""
    logging.basicConfig(level=logging.INFO)
    
    # 設定
    config = RAGChainConfig(
        retrieval_config=SearchConfig(vector_top_k=5),
        generation_config=GenerationConfig(
            model_name="llama3.2:3b",
            temperature=0.1,
            include_sources=True
        ),
        enable_conversation_memory=True,
        enable_query_preprocessing=True,
        enable_response_postprocessing=True
    )
    
    # RAGチェーン初期化
    rag_chain = RAGChain(config)
    
    # ヘルスチェック
    health = await rag_chain.health_check()
    print("ヘルスチェック:", json.dumps(health, indent=2, ensure_ascii=False))
    
    # テストクエリ
    test_queries = [
        "機械学習について教えてください",
        "RAGシステムの仕組みを説明してください",
        "前の質問に関連して、具体的な実装方法は？"
    ]
    
    conversation_id = None
    
    for i, query in enumerate(test_queries):
        print(f"\n=== クエリ {i+1}: {query} ===")
        
        try:
            # RAGクエリ実行
            result = await rag_chain.query(
                query=query,
                conversation_id=conversation_id,
                user_id="test_user"
            )
            
            conversation_id = result.conversation_id
            
            print(f"応答: {result.response[:200]}...")
            print(f"時間: {result.total_time:.2f}s (検索: {result.retrieval_time:.2f}s, 生成: {result.generation_time:.2f}s)")
            print(f"ソース数: {result.search_results_count}")
            print(f"会話ID: {result.conversation_id}")
            
        except Exception as e:
            print(f"クエリ実行エラー: {e}")
    
    # ストリーミングテスト
    print(f"\n=== ストリーミングテスト ===")
    try:
        async for chunk in rag_chain.stream_query(
            "ストリーミングテストです",
            conversation_id=conversation_id
        ):
            print(f"チャンク: {chunk}")
    except Exception as e:
        print(f"ストリーミングエラー: {e}")
    
    # 統計情報表示
    stats = await rag_chain.get_chain_stats()
    print(f"\n=== チェーン統計 ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
