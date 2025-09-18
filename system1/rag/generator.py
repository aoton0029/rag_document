import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import json
import re
from llama_index.core.llms import LLM
from llama_index.core.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.ollama import Ollama
from ..llms.ollama_connector import OllamaConnector
from ..database.keyvalue_db.redis_client import RedisClient
from .retriever import SearchResult


@dataclass
class GenerationConfig:
    """生成設定"""
    model_name: str = "llama3.2:3b"
    temperature: float = 0.1
    max_tokens: int = 2048
    context_window: int = 4096
    system_prompt: str = ""
    response_mode: str = "compact"  # "compact", "tree_summarize", "accumulate"
    streaming: bool = False
    include_sources: bool = True
    max_source_length: int = 500
    citation_style: str = "numbered"  # "numbered", "markdown", "academic"


@dataclass
class GenerationResult:
    """生成結果"""
    response: str
    sources: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    generation_time: float
    token_count: Optional[int] = None
    cached: bool = False


class RAGGenerator:
    """
    RAG応答生成エンジン
    検索結果を基に文脈を考慮した応答を生成
    """
    
    def __init__(
        self,
        config: GenerationConfig = None,
        ollama: OllamaConnector = None,
        redis_client: RedisClient = None
    ):
        self.config = config or GenerationConfig()
        self.logger = logging.getLogger(__name__)
        
        self.redis_client = redis_client or RedisClient()
        
        # Response Synthesizer
        self.response_synthesizer = self._initialize_synthesizer()
        
        # プロンプトテンプレート
        self.prompt_templates = self._load_prompt_templates()
        
        # 統計情報
        self.generation_stats = {
            "total_generations": 0,
            "cache_hits": 0,
            "average_generation_time": 0.0,
            "total_tokens_generated": 0,
            "error_count": 0
        }
        
        self.logger.info("RAGGenerator初期化完了")
        
    def _initialize_synthesizer(self) -> BaseSynthesizer:
        """Response Synthesizer初期化"""
        return get_response_synthesizer(
            llm=self.llm,
            response_mode=self.config.response_mode,
            streaming=self.config.streaming
        )
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """プロンプトテンプレート読み込み"""
        return {
            "system": self.config.system_prompt or """
あなたは知識豊富なAIアシスタントです。提供された文脈情報を基に、正確で有用な回答を提供してください。

以下の指針に従ってください：
1. 提供された文脈情報のみを使用して回答する
2. 文脈にない情報については「提供された情報では確認できません」と答える
3. 回答は簡潔で分かりやすくする
4. 適切な場合は引用元を明示する
5. 日本語で回答する
""",
            
            "context_prompt": """
文脈情報:
{context}

質問: {query}

上記の文脈情報を基に、質問に対する適切な回答を提供してください。
""",
            
            "streaming_prompt": """
文脈情報を参考に、以下の質問に段階的に回答してください：

文脈: {context}

質問: {query}

回答：
""",
            
            "summary_prompt": """
以下の情報を要約してください：

{content}

要約（3-5文程度）：
""",
            
            "citation_prompt": """
以下の文脈情報を使用して質問に回答し、適切な引用を含めてください：

{context_with_sources}

質問: {query}

回答（引用付き）：
"""
        }
    
    async def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        custom_prompt: Optional[str] = None
    ) -> GenerationResult:
        """
        検索結果を基にRAG応答を生成
        
        Args:
            query: ユーザークエリ
            search_results: 検索結果リスト
            conversation_history: 会話履歴
            custom_prompt: カスタムプロンプト
        
        Returns:
            生成結果
        """
        start_time = datetime.utcnow()
        
        try:
            # キャッシュチェック
            cache_key = self._generate_cache_key(query, search_results)
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                self.generation_stats["cache_hits"] += 1
                return cached_result
            
            # 文脈準備
            context = await self._prepare_context(search_results, query)
            
            # プロンプト構築
            if custom_prompt:
                final_prompt = custom_prompt.format(context=context, query=query)
            else:
                final_prompt = await self._build_prompt(
                    query, context, conversation_history
                )
            
            # 応答生成
            if self.config.streaming:
                response = await self._generate_streaming_response(final_prompt)
            else:
                response = await self._generate_standard_response(final_prompt)
            
            # ソース情報準備
            sources = self._prepare_sources(search_results)
            
            # 引用追加
            if self.config.include_sources:
                response = self._add_citations(response, sources)
            
            # 結果作成
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            result = GenerationResult(
                response=response,
                sources=sources,
                model_info={
                    "model_name": self.config.model_name,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                },
                generation_time=generation_time,
                cached=False
            )
            
            # キャッシュ保存
            await self._cache_response(cache_key, result)
            
            # 統計更新
            self._update_generation_stats(generation_time)
            
            self.logger.info(f"応答生成完了: クエリ='{query[:50]}...', 時間={generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.generation_stats["error_count"] += 1
            self.logger.error(f"応答生成エラー: {e}")
            
            # エラー時のフォールバック応答
            return GenerationResult(
                response="申し訳ありませんが、応答の生成中にエラーが発生しました。",
                sources=[],
                model_info={"error": str(e)},
                generation_time=0.0,
                cached=False
            )
    
    async def _prepare_context(
        self,
        search_results: List[SearchResult],
        query: str
    ) -> str:
        """検索結果から文脈を準備"""
        if not search_results:
            return "関連する情報が見つかりませんでした。"
        
        context_parts = []
        for i, result in enumerate(search_results):
            # コンテンツ長さ制限
            content = result.content
            if len(content) > self.config.max_source_length:
                content = content[:self.config.max_source_length] + "..."
            
            # ソース情報付きでコンテンツ追加
            source_info = f"[ソース {i+1}]"
            if result.metadata.get("filename"):
                source_info += f" {result.metadata['filename']}"
            if result.metadata.get("page_number"):
                source_info += f" (ページ {result.metadata['page_number']})"
            
            context_part = f"{source_info}\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """プロンプト構築"""
        # システムプロンプト
        prompt_parts = [self.prompt_templates["system"]]
        
        # 会話履歴
        if conversation_history:
            history_text = "\n".join([
                f"ユーザー: {turn.get('user', '')}\nアシスタント: {turn.get('assistant', '')}"
                for turn in conversation_history[-3:]  # 直近3ターンのみ
            ])
            prompt_parts.append(f"\n会話履歴:\n{history_text}\n")
        
        # メインプロンプト
        if self.config.include_sources:
            main_prompt = self.prompt_templates["citation_prompt"].format(
                context_with_sources=context,
                query=query
            )
        else:
            main_prompt = self.prompt_templates["context_prompt"].format(
                context=context,
                query=query
            )
        
        prompt_parts.append(main_prompt)
        
        return "\n".join(prompt_parts)
    
    async def _generate_standard_response(self, prompt: str) -> str:
        """標準応答生成"""
        try:
            response = await self.llm.acomplete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"LLM応答生成エラー: {e}")
            return "応答の生成中にエラーが発生しました。"
    
    async def _generate_streaming_response(self, prompt: str) -> str:
        """ストリーミング応答生成"""
        try:
            response_parts = []
            async for chunk in self.llm.astream_complete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            ):
                if chunk.delta:
                    response_parts.append(chunk.delta)
            
            return "".join(response_parts).strip()
            
        except Exception as e:
            self.logger.error(f"ストリーミング応答生成エラー: {e}")
            return "応答の生成中にエラーが発生しました。"
    
    def _prepare_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """ソース情報準備"""
        sources = []
        for i, result in enumerate(search_results):
            source = {
                "id": i + 1,
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "source_type": result.source_type,
                "metadata": result.metadata,
                "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
            }
            sources.append(source)
        
        return sources
    
    def _add_citations(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """引用を応答に追加"""
        if not sources:
            return response
        
        if self.config.citation_style == "numbered":
            # 番号付き引用
            citations = "\n\n参考文献:\n"
            for source in sources:
                filename = source["metadata"].get("filename", "Unknown")
                citations += f"[{source['id']}] {filename}\n"
            
            return response + citations
            
        elif self.config.citation_style == "markdown":
            # Markdown形式引用
            citations = "\n\n## 参考文献\n"
            for source in sources:
                filename = source["metadata"].get("filename", "Unknown")
                citations += f"- **{filename}** (スコア: {source['score']:.3f})\n"
            
            return response + citations
            
        elif self.config.citation_style == "academic":
            # 学術論文形式引用
            citations = "\n\n## References\n"
            for source in sources:
                metadata = source["metadata"]
                title = metadata.get("title", "Unknown Title")
                author = metadata.get("author", "Unknown Author")
                year = metadata.get("published_date", "Unknown Year")
                citations += f"{author} ({year}). {title}.\n"
            
            return response + citations
        
        return response
    
    async def summarize_content(
        self,
        content: str,
        max_length: int = 200
    ) -> str:
        """コンテンツ要約"""
        try:
            if len(content) <= max_length:
                return content
            
            prompt = self.prompt_templates["summary_prompt"].format(content=content)
            
            response = await self.llm.acomplete(
                prompt,
                temperature=0.1,
                max_tokens=max_length
            )
            
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"要約生成エラー: {e}")
            return content[:max_length] + "..."
    
    async def generate_streaming_response(
        self,
        query: str,
        search_results: List[SearchResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """ストリーミング応答生成（非同期ジェネレーター）"""
        try:
            # 文脈準備
            context = await self._prepare_context(search_results, query)
            
            # プロンプト構築
            prompt = await self._build_prompt(query, context, conversation_history)
            
            # ストリーミング生成
            async for chunk in self.llm.astream_complete(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            ):
                if chunk.delta:
                    yield chunk.delta
            
            # ソース情報を最後に追加
            if self.config.include_sources:
                sources = self._prepare_sources(search_results)
                citations = self._add_citations("", sources)
                if citations.strip():
                    yield citations
                    
        except Exception as e:
            self.logger.error(f"ストリーミング応答エラー: {e}")
            yield "応答の生成中にエラーが発生しました。"
    
    def _generate_cache_key(
        self,
        query: str,
        search_results: List[SearchResult]
    ) -> str:
        """キャッシュキー生成"""
        # 検索結果のハッシュを含める
        result_ids = [f"{r.document_id}_{r.chunk_id}_{r.score:.3f}" for r in search_results]
        hash_input = f"{query}_{self.config.model_name}_{str(sorted(result_ids))}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[GenerationResult]:
        """キャッシュから応答取得"""
        try:
            cached_data = self.redis_client.get_llm_response(f"rag_{cache_key}")
            if cached_data:
                # 辞書からGenerationResultに復元
                return GenerationResult(
                    response=cached_data["response"],
                    sources=cached_data["sources"],
                    model_info=cached_data["model_info"],
                    generation_time=cached_data["generation_time"],
                    cached=True
                )
        except Exception as e:
            self.logger.warning(f"キャッシュ取得エラー: {e}")
        return None
    
    async def _cache_response(self, cache_key: str, result: GenerationResult):
        """応答をキャッシュ"""
        try:
            cache_data = {
                "response": result.response,
                "sources": result.sources,
                "model_info": result.model_info,
                "generation_time": result.generation_time
            }
            
            self.redis_client.cache_llm_response(
                f"rag_{cache_key}",
                json.dumps(cache_data, ensure_ascii=False),
                expire_seconds=3600  # 1時間
            )
        except Exception as e:
            self.logger.warning(f"キャッシュ保存エラー: {e}")
    
    def _update_generation_stats(self, generation_time: float):
        """生成統計更新"""
        self.generation_stats["total_generations"] += 1
        
        # 平均生成時間更新
        total_time = (self.generation_stats["average_generation_time"] * 
                     (self.generation_stats["total_generations"] - 1) + generation_time)
        self.generation_stats["average_generation_time"] = total_time / self.generation_stats["total_generations"]
    
    async def get_generation_stats(self) -> Dict[str, Any]:
        """生成統計取得"""
        return {
            **self.generation_stats,
            "cache_hit_rate": (self.generation_stats["cache_hits"] / 
                              max(self.generation_stats["total_generations"], 1)) * 100,
            "config": {
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_mode": self.config.response_mode,
                "include_sources": self.config.include_sources
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        try:
            # Ollama接続確認
            ollama_status = self.ollama_connector.check_connection()
            
            # 簡単なテスト生成
            test_prompt = "Hello"
            test_response = await self.llm.acomplete(test_prompt, max_tokens=10)
            llm_status = bool(test_response.text.strip())
            
            return {
                "ollama_connection": ollama_status,
                "llm_generation": llm_status,
                "model_name": self.config.model_name,
                "status": "healthy" if (ollama_status and llm_status) else "unhealthy"
            }
            
        except Exception as e:
            return {
                "ollama_connection": False,
                "llm_generation": False,
                "error": str(e),
                "status": "unhealthy"
            }


# 使用例とテスト用のメイン関数
async def main():
    """テスト用メイン関数"""
    logging.basicConfig(level=logging.INFO)
    
    # 設定
    config = GenerationConfig(
        model_name="llama3.2:3b",
        temperature=0.1,
        include_sources=True,
        citation_style="numbered"
    )
    
    # Generator初期化
    generator = RAGGenerator(config)
    
    # ヘルスチェック
    health = await generator.health_check()
    print("ヘルスチェック:", health)
    
    # テスト用検索結果（模擬）
    mock_search_results = [
        SearchResult(
            document_id="doc_001",
            chunk_id="chunk_001",
            content="機械学習は人工知能の一分野で、コンピューターがデータから学習する技術です。",
            score=0.95,
            metadata={"filename": "ml_basics.pdf", "page_number": 1},
            source_type="vector",
            rank=1
        ),
        SearchResult(
            document_id="doc_002",
            chunk_id="chunk_002",
            content="深層学習はニューラルネットワークを使った機械学習の手法です。",
            score=0.88,
            metadata={"filename": "deep_learning.pdf", "page_number": 5},
            source_type="vector",
            rank=2
        )
    ]
    
    # テスト質問
    test_query = "機械学習について教えてください"
    
    try:
        print(f"\n=== 応答生成テスト ===")
        print(f"質問: {test_query}")
        
        # 応答生成
        result = await generator.generate_response(test_query, mock_search_results)
        
        print(f"\n応答: {result.response}")
        print(f"生成時間: {result.generation_time:.2f}秒")
        print(f"ソース数: {len(result.sources)}")
        print(f"キャッシュ済み: {result.cached}")
        
        # 統計情報
        stats = await generator.get_generation_stats()
        print(f"\n=== 生成統計 ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"テスト実行エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())
