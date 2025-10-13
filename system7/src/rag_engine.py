"""
Advanced RAG Engine implementation
高度なRAGシステムのコア機能を実装
"""
from typing import List, Optional, Any, Dict
from llama_index.core import VectorStoreIndex, PropertyGraphIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from src.storage_setup import storage_manager
from src.model_setup import model_manager
from config.config import config
from loguru import logger

class AdvancedRAGEngine:
    """
    Advanced RAG System の中核エンジン
    複数のRetrieverとQueryEngineを統合して高度な質問応答を実現
    """
    
    def __init__(self):
        self.vector_index: Optional[VectorStoreIndex] = None
        self.graph_index: Optional[PropertyGraphIndex] = None
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.query_engines: Dict[str, BaseQueryEngine] = {}
        self.callback_manager: Optional[CallbackManager] = None
        
        # デバッグハンドラーの設定
        self._setup_debug_handler()
    
    def _setup_debug_handler(self):
        """デバッグハンドラーの設定"""
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        self.callback_manager = CallbackManager([llama_debug])
    
    def initialize(self, documents: Optional[List[Any]] = None):
        """RAGエンジンの初期化"""
        try:
            logger.info("Initializing Advanced RAG Engine...")
            
            # モデルとストレージの初期化
            model_manager.setup_global_settings()
            storage_context = storage_manager.get_storage_context()
            
            if documents:
                # 新しいドキュメントでインデックス作成
                self._create_indexes_from_documents(documents, storage_context)
            else:
                # 既存のインデックスから復元
                self._load_existing_indexes(storage_context)
            
            # Retrieverの設定
            self._setup_retrievers()
            
            # QueryEngineの設定
            self._setup_query_engines()
            
            logger.success("Advanced RAG Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced RAG Engine: {str(e)}")
            raise
    
    def _create_indexes_from_documents(self, documents: List[Any], storage_context):
        """ドキュメントからインデックスを作成"""
        try:
            logger.info("Creating indexes from documents...")
            
            # Vector Indexの作成
            self.vector_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                callback_manager=self.callback_manager,
                show_progress=True
            )
            
            # Property Graph Indexの作成（知識グラフ用）
            self.graph_index = PropertyGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                callback_manager=self.callback_manager,
                show_progress=True,
                # グラフ抽出の設定
                kg_extractors=[
                    # エンティティとリレーション抽出の設定
                ],
                include_embeddings=True
            )
            
            logger.success("Indexes created successfully from documents")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise
    
    def _load_existing_indexes(self, storage_context):
        """既存のインデックスをロード"""
        try:
            logger.info("Loading existing indexes...")
            
            # Vector Indexのロード
            try:
                self.vector_index = VectorStoreIndex.from_vector_store(
                    storage_context.vector_store,
                    storage_context=storage_context,
                    callback_manager=self.callback_manager
                )
                logger.info("Vector index loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load existing vector index: {str(e)}")
            
            # Property Graph Indexのロード
            try:
                self.graph_index = PropertyGraphIndex.from_existing(
                    storage_context=storage_context,
                    callback_manager=self.callback_manager
                )
                logger.info("Graph index loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load existing graph index: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to load existing indexes: {str(e)}")
            raise
    
    def _setup_retrievers(self):
        """各種Retrieverの設定"""
        try:
            logger.info("Setting up retrievers...")
            
            # Vector Retriever
            if self.vector_index:
                self.retrievers["vector"] = VectorIndexRetriever(
                    index=self.vector_index,
                    similarity_top_k=config.rag.similarity_top_k,
                    callback_manager=self.callback_manager
                )
                logger.info("Vector retriever configured")
            
            # Graph Retriever
            if self.graph_index:
                self.retrievers["graph"] = self.graph_index.as_retriever(
                    similarity_top_k=config.rag.similarity_top_k,
                    callback_manager=self.callback_manager
                )
                logger.info("Graph retriever configured")
            
            logger.success("All retrievers configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup retrievers: {str(e)}")
            raise
    
    def _setup_query_engines(self):
        """各種QueryEngineの設定"""
        try:
            logger.info("Setting up query engines...")
            
            # Response Synthesizerの設定
            response_synthesizer = get_response_synthesizer(
                response_mode=config.rag.response_mode,
                streaming=config.rag.streaming,
                callback_manager=self.callback_manager
            )
            
            # ポストプロセッサーの設定
            postprocessors = []
            
            if config.rag.use_reranking:
                # 類似性再ランキング
                similarity_postprocessor = SimilarityPostprocessor(
                    similarity_cutoff=0.7
                )
                postprocessors.append(similarity_postprocessor)
            
            # キーワードフィルタリング
            keyword_postprocessor = KeywordNodePostprocessor(
                required_keywords=None,  # 必要に応じて設定
                exclude_keywords=["advertisement", "spam"]
            )
            postprocessors.append(keyword_postprocessor)
            
            # Vector Query Engine
            if "vector" in self.retrievers:
                self.query_engines["vector"] = RetrieverQueryEngine.from_args(
                    retriever=self.retrievers["vector"],
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=postprocessors,
                    callback_manager=self.callback_manager
                )
                logger.info("Vector query engine configured")
            
            # Graph Query Engine
            if "graph" in self.retrievers:
                self.query_engines["graph"] = RetrieverQueryEngine.from_args(
                    retriever=self.retrievers["graph"],
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=postprocessors,
                    callback_manager=self.callback_manager
                )
                logger.info("Graph query engine configured")
            
            # Hybrid Query Engine (Vector + Graph)
            if "vector" in self.retrievers and "graph" in self.retrievers:
                self.query_engines["hybrid"] = self._create_hybrid_query_engine(
                    response_synthesizer, postprocessors
                )
                logger.info("Hybrid query engine configured")
            
            logger.success("All query engines configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup query engines: {str(e)}")
            raise
    
    def _create_hybrid_query_engine(self, response_synthesizer, postprocessors) -> BaseQueryEngine:
        """ハイブリッドクエリエンジンの作成（ベクトル検索とグラフ検索の組み合わせ）"""
        
        class HybridRetriever(BaseRetriever):
            def __init__(self, vector_retriever, graph_retriever):
                self.vector_retriever = vector_retriever
                self.graph_retriever = graph_retriever
            
            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                # ベクトル検索の結果を取得
                vector_nodes = self.vector_retriever.retrieve(query_bundle)
                
                # グラフ検索の結果を取得
                graph_nodes = self.graph_retriever.retrieve(query_bundle)
                
                # 結果をマージし、重複を除去
                all_nodes = vector_nodes + graph_nodes
                unique_nodes = {}
                
                for node in all_nodes:
                    node_id = node.node.node_id
                    if node_id not in unique_nodes or node.score > unique_nodes[node_id].score:
                        unique_nodes[node_id] = node
                
                # スコア順にソート
                merged_nodes = list(unique_nodes.values())
                merged_nodes.sort(key=lambda x: x.score, reverse=True)
                
                # 上位k個を返す
                return merged_nodes[:config.rag.similarity_top_k]
        
        hybrid_retriever = HybridRetriever(
            self.retrievers["vector"],
            self.retrievers["graph"]
        )
        
        return RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
            callback_manager=self.callback_manager
        )
    
    def query(self, 
              question: str, 
              engine_type: str = "hybrid", 
              include_sources: bool = True) -> Dict[str, Any]:
        """質問応答の実行"""
        try:
            logger.info(f"Processing query with {engine_type} engine: {question[:50]}...")
            
            if engine_type not in self.query_engines:
                available_engines = list(self.query_engines.keys())
                logger.warning(f"Engine {engine_type} not available. Using: {available_engines[0]}")
                engine_type = available_engines[0]
            
            query_engine = self.query_engines[engine_type]
            
            # クエリ実行
            response = query_engine.query(question)
            
            # レスポンスの構築
            result = {
                "question": question,
                "answer": str(response),
                "engine_type": engine_type,
                "sources": []
            }
            
            # ソース情報の追加
            if include_sources and hasattr(response, 'source_nodes'):
                for i, source_node in enumerate(response.source_nodes):
                    source_info = {
                        "rank": i + 1,
                        "score": getattr(source_node, 'score', 0.0),
                        "text": source_node.node.text[:200] + "..." if len(source_node.node.text) > 200 else source_node.node.text,
                        "metadata": source_node.node.metadata
                    }
                    result["sources"].append(source_info)
            
            logger.success(f"Query processed successfully with {len(result.get('sources', []))} sources")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], engine_type: str = "hybrid") -> str:
        """チャット形式での対話"""
        try:
            # 最新のメッセージを質問として使用
            if not messages:
                raise ValueError("No messages provided")
            
            latest_message = messages[-1]["content"]
            
            # コンテキストを考慮したクエリ実行
            result = self.query(latest_message, engine_type, include_sources=False)
            
            return result["answer"]
            
        except Exception as e:
            logger.error(f"Failed to process chat: {str(e)}")
            raise
    
    def get_engine_status(self) -> Dict[str, Any]:
        """エンジンの状態を取得"""
        return {
            "available_engines": list(self.query_engines.keys()),
            "available_retrievers": list(self.retrievers.keys()),
            "vector_index_loaded": self.vector_index is not None,
            "graph_index_loaded": self.graph_index is not None,
            "storage_connections": storage_manager.test_connections(),
            "model_status": {
                "llm_initialized": model_manager.llm is not None,
                "embed_model_initialized": model_manager.embed_model is not None
            }
        }

# グローバルRAGエンジンインスタンス
rag_engine = AdvancedRAGEngine()