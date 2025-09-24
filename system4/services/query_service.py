from llama_index.core.query_engine import (
    BaseQueryEngine, SQLJoinQueryEngine, CustomQueryEngine, PandasQueryEngine, 
    RouterQueryEngine, RetryQueryEngine, MultiStepQueryEngine, 
    JSONalyzeQueryEngine, TransformQueryEngine, 
    SQLAutoVectorQueryEngine, RetrieverQueryEngine, SQLTableRetrieverQueryEngine, 
    ToolRetrieverRouterQueryEngine, SimpleMultiModalQueryEngine, 
    KnowledgeGraphQueryEngine, RetrieverRouterQueryEngine, RetrySourceQueryEngine
)
import logging
from typing import List, Dict, Any, Optional
from llama_index.core.tools import ToolMetadata, RetrieverTool, QueryEngineTool
from llama_index.core.selectors import (
    MultiSelection, SingleSelection, LLMSingleSelector, 
    LLMMultiSelector, EmbeddingSingleSelector, 
    PydanticMultiSelector, PydanticSingleSelector
)
from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llm.ollama_connector import OllamaConnector
from db.database_manager import db_manager
from services.index_service import IndexingService
from services.retriever_service import RetrieverService

class QueryService:
    def __init__(self, 
                 ollama: OllamaConnector):
        self.ollama = ollama
        self.logger = logging.getLogger(__name__)
        self.indexing_service = IndexingService(ollama)
        self.retriever_service = RetrieverService(ollama)
        
        # Settings configuration
        Settings.llm = ollama.llm
        Settings.embed_model = ollama.embedding_model
    
    def create_retriever_query_engine(self, 
                                    retriever_type: str = 'vector',
                                    response_mode: str = "compact",
                                    **retriever_kwargs) -> RetrieverQueryEngine:
        """リトリーバーベースのクエリエンジンを作成"""
        try:
            self.logger.info(f"Creating Retriever Query Engine with {retriever_type} retriever...")
            
            # リトリーバーを作成
            retriever = self.retriever_service.get_retriever_by_type(
                retriever_type, **retriever_kwargs
            )
            
            # レスポンスシンセサイザーを作成
            response_synthesizer = get_response_synthesizer(
                response_mode=response_mode,
                llm=self.ollama.llm
            )
            
            # クエリエンジンを作成
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            self.logger.info("Retriever Query Engine created successfully")
            return query_engine
            
        except Exception as e:
            self.logger.error(f"Failed to create Retriever Query Engine: {e}")
            raise
    
    def create_knowledge_graph_query_engine(self, 
                                          retriever_mode: str = "keyword",
                                          response_mode: str = "compact",
                                          include_text: bool = True) -> KnowledgeGraphQueryEngine:
        """ナレッジグラフクエリエンジンを作成"""
        try:
            self.logger.info("Creating Knowledge Graph Query Engine...")
            
            # KnowledgeGraphIndexを取得
            kg_index = self.indexing_service.load_index('knowledge_graph')
            
            # クエリエンジンを作成
            query_engine = KnowledgeGraphQueryEngine(
                index=kg_index,
                retriever_mode=retriever_mode,
                response_mode=response_mode,
                include_text=include_text
            )
            
            self.logger.info("Knowledge Graph Query Engine created successfully")
            return query_engine
            
        except Exception as e:
            self.logger.error(f"Failed to create Knowledge Graph Query Engine: {e}")
            raise
    
    def create_router_query_engine(self, 
                                 query_engines: List[BaseQueryEngine],
                                 descriptions: List[str],
                                 selector_type: str = "llm") -> RouterQueryEngine:
        """ルータークエリエンジンを作成"""
        try:
            self.logger.info("Creating Router Query Engine...")
            
            # ツールメタデータを作成
            tools = []
            for i, (engine, description) in enumerate(zip(query_engines, descriptions)):
                tool = QueryEngineTool(
                    query_engine=engine,
                    metadata=ToolMetadata(
                        name=f"query_engine_{i}",
                        description=description
                    )
                )
                tools.append(tool)
            
            # セレクターを選択
            if selector_type == "llm":
                selector = LLMSingleSelector.from_defaults(llm=self.ollama.llm)
            elif selector_type == "embedding":
                selector = EmbeddingSingleSelector.from_defaults(
                    embed_model=self.ollama.embedding_model
                )
            else:
                raise ValueError(f"Unknown selector type: {selector_type}")
            
            # ルータークエリエンジンを作成
            router_query_engine = RouterQueryEngine(
                selector=selector,
                query_engine_tools=tools
            )
            
            self.logger.info("Router Query Engine created successfully")
            return router_query_engine
            
        except Exception as e:
            self.logger.error(f"Failed to create Router Query Engine: {e}")
            raise
    
    def create_multi_step_query_engine(self, 
                                     base_query_engine: BaseQueryEngine,
                                     num_steps: int = 3) -> MultiStepQueryEngine:
        """マルチステップクエリエンジンを作成"""
        try:
            self.logger.info("Creating Multi-Step Query Engine...")
            
            query_engine = MultiStepQueryEngine(
                query_engine=base_query_engine,
                query_transform=None,  # デフォルト変換を使用
                num_steps=num_steps
            )
            
            self.logger.info("Multi-Step Query Engine created successfully")
            return query_engine
            
        except Exception as e:
            self.logger.error(f"Failed to create Multi-Step Query Engine: {e}")
            raise
    
    def create_retry_query_engine(self, 
                                base_query_engine: BaseQueryEngine,
                                max_retries: int = 3) -> RetryQueryEngine:
        """リトライクエリエンジンを作成"""
        try:
            self.logger.info("Creating Retry Query Engine...")
            
            query_engine = RetryQueryEngine(
                query_engine=base_query_engine,
                max_retries=max_retries
            )
            
            self.logger.info("Retry Query Engine created successfully")
            return query_engine
            
        except Exception as e:
            self.logger.error(f"Failed to create Retry Query Engine: {e}")
            raise
    
    def create_comprehensive_query_engine(self) -> RouterQueryEngine:
        """包括的なクエリエンジンを作成（複数のエンジンを組み合わせ）"""
        try:
            self.logger.info("Creating Comprehensive Query Engine...")
            
            # 各タイプのクエリエンジンを作成
            vector_engine = self.create_retriever_query_engine(
                retriever_type='vector',
                similarity_top_k=5
            )
            
            keyword_engine = self.create_retriever_query_engine(
                retriever_type='keyword',
                keyword_top_k=5
            )
            
            hybrid_engine = self.create_retriever_query_engine(
                retriever_type='hybrid',
                similarity_top_k=5,
                keyword_top_k=3
            )
            
            kg_engine = self.create_knowledge_graph_query_engine()
            
            # 説明文を定義
            descriptions = [
                "ベクトル類似度検索に基づく質問応答。セマンティックな類似性を重視。",
                "キーワードマッチングに基づく質問応答。正確なキーワードマッチを重視。",
                "ベクトル検索とキーワード検索を組み合わせたハイブリッド検索。",
                "知識グラフベースの質問応答。エンティティ間の関係性を重視。"
            ]
            
            # ルータークエリエンジンを作成
            comprehensive_engine = self.create_router_query_engine(
                query_engines=[vector_engine, keyword_engine, hybrid_engine, kg_engine],
                descriptions=descriptions,
                selector_type="llm"
            )
            
            # リトライ機能を追加
            final_engine = self.create_retry_query_engine(
                base_query_engine=comprehensive_engine,
                max_retries=2
            )
            
            self.logger.info("Comprehensive Query Engine created successfully")
            return final_engine
            
        except Exception as e:
            self.logger.error(f"Failed to create Comprehensive Query Engine: {e}")
            raise
    
    def query(self, 
            query_text: str, 
            engine_type: str = "comprehensive") -> str:
        """クエリを実行"""
        try:
            self.logger.info(f"Executing query: {query_text[:100]}...")
            
            # エンジンタイプに応じてクエリエンジンを作成
            if engine_type == "comprehensive":
                query_engine = self.create_comprehensive_query_engine()
            elif engine_type == "vector":
                query_engine = self.create_retriever_query_engine(retriever_type='vector')
            elif engine_type == "keyword":
                query_engine = self.create_retriever_query_engine(retriever_type='keyword')
            elif engine_type == "hybrid":
                query_engine = self.create_retriever_query_engine(retriever_type='hybrid')
            elif engine_type == "knowledge_graph":
                query_engine = self.create_knowledge_graph_query_engine()
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
            
            # クエリを実行
            response = query_engine.query(query_text)
            
            self.logger.info("Query executed successfully")
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise
    
    def batch_query(self, 
                   queries: List[str], 
                   engine_type: str = "comprehensive") -> List[str]:
        """複数のクエリを一括実行"""
        try:
            self.logger.info(f"Executing batch queries: {len(queries)} queries")
            
            results = []
            for query in queries:
                try:
                    result = self.query(query, engine_type)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to execute query '{query[:50]}...': {e}")
                    results.append(f"Error: {str(e)}")
            
            self.logger.info("Batch queries executed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to execute batch queries: {e}")
            raise