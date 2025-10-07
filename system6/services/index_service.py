import logging
from typing import List
from llama_index.core import Settings, VectorStoreIndex, SummaryIndex, TreeIndex, KeywordTableIndex, KnowledgeGraphIndex, DocumentSummaryIndex
from llama_index.core.schema import BaseNode
from db.database_manager import db_manager
from llama_index.core import Document, load_index_from_storage
from configs import ProcessingConfig

logger = logging.getLogger(__name__)

class IndexingService:
    def __init__(self):
        logger.info("IndexingService initialized")

    
    def create_vector_store_index(self, documents: List[BaseNode]) -> VectorStoreIndex:
        """VectorStoreIndexを作成"""
        try:
            logger.info("Creating VectorStoreIndex...")
            storage_context = db_manager.get_storage_context()
            
            for node in documents:
                node.metadata['embedding_model'] = Settings.embed_model.model_name
            
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info("VectorStoreIndex created and persisted successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create VectorStoreIndex: {e}")
            raise
    
    def create_summary_index(self, documents: List[Document]) -> SummaryIndex:
        """SummaryIndexを作成"""
        try:
            logger.info("Creating SummaryIndex...")
            storage_context = db_manager.get_storage_context()
            
            for node in documents:
                node.metadata['embedding_model'] = Settings.embed_model.model_name

            index = SummaryIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info("SummaryIndex created and persisted successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create SummaryIndex: {e}")
            raise
    
    def create_tree_index(self, documents: List[Document]) -> TreeIndex:
        """TreeIndexを作成"""
        try:
            logger.info("Creating TreeIndex...")
            storage_context = db_manager.get_storage_context()
            
            for node in documents:
                node.metadata['embedding_model'] = Settings.embed_model.model_name

            index = TreeIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info("TreeIndex created and persisted successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create TreeIndex: {e}")
            raise
    
    def create_keyword_table_index(self, documents: List[Document]) -> KeywordTableIndex:
        """KeywordTableIndexを作成"""
        try:
            logger.info("Creating KeywordTableIndex...")
            storage_context = db_manager.get_storage_context()
            
            index = KeywordTableIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            
            logger.info("KeywordTableIndex created and persisted successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create KeywordTableIndex: {e}")
            raise
    
    def create_knowledge_graph_index(self, documents: List[Document]) -> KnowledgeGraphIndex:
        """KnowledgeGraphIndexを作成"""
        try:
            logger.info("Creating KnowledgeGraphIndex...")
            storage_context = db_manager.get_storage_context()
            
            index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                max_triplets_per_chunk=10,
                include_embeddings=True
            )
            
            logger.info("KnowledgeGraphIndex created and persisted successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create KnowledgeGraphIndex: {e}")
            raise
    
    def create_document_summary_index(self, documents: List[Document]) -> DocumentSummaryIndex:
        """DocumentSummaryIndexを作成"""
        try:
            logger.info("Creating DocumentSummaryIndex...")
            storage_context = db_manager.get_storage_context()
            
            index = DocumentSummaryIndex.from_documents(
                documents,
                storage_context=storage_context,
                response_synthesizer=None  # デフォルトを使用
            )
            
            logger.info("DocumentSummaryIndex created and persisted successfully")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create DocumentSummaryIndex: {e}")
            raise
    
    def create_all_indexes(self, documents: List[Document]) -> dict:
        """すべてのインデックスを作成"""
        indexes = {}
        
        try:
            indexes['vector_store'] = self.create_vector_store_index(documents)
            indexes['summary'] = self.create_summary_index(documents)
            indexes['tree'] = self.create_tree_index(documents)
            indexes['keyword_table'] = self.create_keyword_table_index(documents)
            indexes['knowledge_graph'] = self.create_knowledge_graph_index(documents)
            indexes['document_summary'] = self.create_document_summary_index(documents)
            
            logger.info("All indexes created successfully")
            return indexes
            
        except Exception as e:
            logger.error(f"Failed to create all indexes: {e}")
            raise
    
    def load_index(self, index_type: str, storage_context=None):
        """保存されたインデックスを読み込み"""
        try:
            if storage_context is None:
                storage_context = db_manager.get_storage_context()
            
            if index_type == 'vector_store':
                return load_index_from_storage(storage_context, index_id="vector_store")
            elif index_type == 'summary':
                return load_index_from_storage(storage_context, index_id="summary")
            elif index_type == 'tree':
                return load_index_from_storage(storage_context, index_id="tree")
            elif index_type == 'keyword_table':
                return load_index_from_storage(storage_context, index_id="keyword_table")
            elif index_type == 'knowledge_graph':
                return load_index_from_storage(storage_context, index_id="knowledge_graph")
            elif index_type == 'document_summary':
                return load_index_from_storage(storage_context, index_id="document_summary")
            else:
                raise ValueError(f"Unknown index type: {index_type}")
                
        except Exception as e:
            logger.error(f"Failed to load {index_type} index: {e}")
            raise