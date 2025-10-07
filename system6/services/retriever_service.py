import logging
from typing import List, Dict, Any, Optional
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
from db.database_manager import db_manager
from services.index_service import IndexingService

class RetrieverService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indexing_service = IndexingService()
        self.logger.info("RetrieverService initialized")

            
    def create_vector_retriever(self, 
                              similarity_top_k: int = 5,
                              similarity_cutoff: float = 0.7) -> VectorIndexRetriever:
        """ベクトルインデックスからのリトリーバーを作成"""
        try:
            self.logger.info("Creating Vector Retriever...")
            
            # VectorStoreIndexを取得または作成
            try:
                vector_index = self.indexing_service.load_index('vector_store')
            except:
                self.logger.warning("Vector index not found, creating new one...")
                # インデックスが存在しない場合は空のインデックスを作成
                from llama_index.core import VectorStoreIndex
                storage_context = db_manager.get_storage_context()
                vector_index = VectorStoreIndex([], storage_context=storage_context)
            
            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=similarity_top_k
            )
            
            # 後処理でスコアフィルタリング
            retriever = self._add_postprocessors(retriever, similarity_cutoff)
            
            self.logger.info("Vector Retriever created successfully")
            return retriever
            
        except Exception as e:
            self.logger.error(f"Failed to create Vector Retriever: {e}")
            raise
    
    def create_keyword_retriever(self, 
                               keyword_top_k: int = 5) -> KeywordTableSimpleRetriever:
        """キーワードテーブルからのリトリーバーを作成"""
        try:
            self.logger.info("Creating Keyword Retriever...")
            
            # KeywordTableIndexを取得または作成
            try:
                keyword_index = self.indexing_service.load_index('keyword_table')
            except:
                self.logger.warning("Keyword index not found, creating new one...")
                from llama_index.core import KeywordTableIndex
                storage_context = db_manager.get_storage_context()
                keyword_index = KeywordTableIndex([], storage_context=storage_context)
            
            retriever = KeywordTableSimpleRetriever(
                index=keyword_index,
                top_k=keyword_top_k
            )
            
            self.logger.info("Keyword Retriever created successfully")
            return retriever
            
        except Exception as e:
            self.logger.error(f"Failed to create Keyword Retriever: {e}")
            raise
    
    def create_summary_retriever(self) -> SummaryIndexRetriever:
        """サマリーインデックスからのリトリーバーを作成"""
        try:
            self.logger.info("Creating Summary Retriever...")
            
            try:
                summary_index = self.indexing_service.load_index('summary')
            except:
                self.logger.warning("Summary index not found, creating new one...")
                from llama_index.core import SummaryIndex
                storage_context = db_manager.get_storage_context()
                summary_index = SummaryIndex([], storage_context=storage_context)
            
            retriever = SummaryIndexRetriever(index=summary_index)
            
            self.logger.info("Summary Retriever created successfully")
            return retriever
            
        except Exception as e:
            self.logger.error(f"Failed to create Summary Retriever: {e}")
            raise
    
    def create_knowledge_graph_retriever(self, 
                                       retriever_mode: str = "keyword",
                                       include_text: bool = True) -> KnowledgeGraphRAGRetriever:
        """ナレッジグラフからのリトリーバーを作成"""
        try:
            self.logger.info("Creating Knowledge Graph Retriever...")
            
            try:
                kg_index = self.indexing_service.load_index('knowledge_graph')
            except:
                self.logger.warning("Knowledge Graph index not found, creating new one...")
                from llama_index.core import KnowledgeGraphIndex
                storage_context = db_manager.get_storage_context()
                kg_index = KnowledgeGraphIndex([], storage_context=storage_context)
            
            retriever = KnowledgeGraphRAGRetriever(
                index=kg_index,
                retriever_mode=retriever_mode,
                include_text=include_text
            )
            
            self.logger.info("Knowledge Graph Retriever created successfully")
            return retriever
            
        except Exception as e:
            self.logger.error(f"Failed to create Knowledge Graph Retriever: {e}")
            raise
    
    def create_fusion_retriever(self, 
                              retrievers: List[Any],
                              similarity_top_k: int = 5,
                              num_queries: int = 4) -> QueryFusionRetriever:
        """複数のリトリーバーを融合したリトリーバーを作成"""
        try:
            self.logger.info("Creating Fusion Retriever...")
            
            fusion_retriever = QueryFusionRetriever(
                retrievers=retrievers,
                similarity_top_k=similarity_top_k,
                num_queries=num_queries,
                mode="reciprocal_rerank",
                use_async=True
            )
            
            self.logger.info("Fusion Retriever created successfully")
            return fusion_retriever
            
        except Exception as e:
            self.logger.error(f"Failed to create Fusion Retriever: {e}")
            raise
    
    def create_hybrid_retriever(self, 
                              similarity_top_k: int = 5,
                              keyword_top_k: int = 3,
                              similarity_cutoff: float = 0.7) -> QueryFusionRetriever:
        """ベクトル検索とキーワード検索を組み合わせたハイブリッドリトリーバー"""
        try:
            self.logger.info("Creating Hybrid Retriever...")
            
            # ベクトルリトリーバー
            vector_retriever = self.create_vector_retriever(
                similarity_top_k=similarity_top_k,
                similarity_cutoff=similarity_cutoff
            )
            
            # キーワードリトリーバー
            keyword_retriever = self.create_keyword_retriever(
                keyword_top_k=keyword_top_k
            )
            
            # 融合リトリーバー
            hybrid_retriever = self.create_fusion_retriever(
                retrievers=[vector_retriever, keyword_retriever],
                similarity_top_k=max(similarity_top_k, keyword_top_k)
            )
            
            self.logger.info("Hybrid Retriever created successfully")
            return hybrid_retriever
            
        except Exception as e:
            self.logger.error(f"Failed to create Hybrid Retriever: {e}")
            raise
    
    def _add_postprocessors(self, retriever, similarity_cutoff: float = 0.7):
        """リトリーバーに後処理を追加"""
        try:
            # 類似度フィルタリング
            similarity_postprocessor = SimilarityPostprocessor(
                similarity_cutoff=similarity_cutoff
            )
            
            # 後処理を適用
            retriever.node_postprocessors = [similarity_postprocessor]
            
            return retriever
            
        except Exception as e:
            self.logger.error(f"Failed to add postprocessors: {e}")
            return retriever
    
    def get_retriever_by_type(self, 
                            retriever_type: str,
                            **kwargs) -> Any:
        """タイプに応じたリトリーバーを取得"""
        retrievers = {
            'vector': self.create_vector_retriever,
            'keyword': self.create_keyword_retriever,
            'summary': self.create_summary_retriever,
            'knowledge_graph': self.create_knowledge_graph_retriever,
            'hybrid': self.create_hybrid_retriever,
            'fusion': self.create_fusion_retriever
        }
        
        if retriever_type not in retrievers:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        return retrievers[retriever_type](**kwargs)