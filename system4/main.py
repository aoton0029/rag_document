import asyncio
import uvicorn
from api.main import app
from core.database import db_manager
import logging

logger = logging.getLogger(__name__)

async def initialize_system():
    """システム初期化"""
    try:
        # データベース接続の初期化
        db_manager.initialize_connections()
        logger.info("System initialized successfully")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

def main():
    """メイン実行関数"""
    # システム初期化
    asyncio.run(initialize_system())
    
    # FastAPIアプリケーション起動
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()



import logging
from typing import List, Dict, Any
from llama_index.core import (
    Settings, VectorStoreIndex, SummaryIndex, TreeIndex, 
    KeywordTableIndex, KnowledgeGraphIndex, DocumentSummaryIndex,
    Document, load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llm.ollama_connector import OllamaConnector
from db.database_manager import db_manager

class EnhancedIndexingService:
    def __init__(self, ollama: OllamaConnector):
        self.logger = logging.getLogger(__name__)
        self.ollama = ollama
        Settings.llm = ollama.get_llm()
        Settings.embed_model = ollama.get_embedding_model()

    def create_tree_index(self, documents: List[Document]) -> TreeIndex:
        """TreeIndexを修正版で作成（階層構造を適切に構築）"""
        try:
            self.logger.info("Creating TreeIndex with proper tree structure...")
            storage_context = db_manager.get_storage_context()
            
            # TreeIndex用の専用ノードパーサー
            node_parser = SentenceSplitter(
                chunk_size=512,  # TreeIndexには小さめのチャンクが適している
                chunk_overlap=50,
                separator=" "
            )
            
            # TreeIndexを階層構造付きで作成
            index = TreeIndex.from_documents(
                documents,
                storage_context=storage_context,
                node_parser=node_parser,
                build_tree=True,  # 重要：階層構造を構築
                num_children=10,  # 各ノードの子ノード数を制限
                show_progress=True
            )
            
            # インデックスIDを設定して保存
            index.set_index_id("tree_index")
            index.storage_context.persist()
            
            # データが正しく保存されたか確認
            self._verify_tree_index_data(index)
            
            self.logger.info("TreeIndex created and verified successfully")
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to create TreeIndex: {e}")
            raise

    def create_summary_index(self, documents: List[Document]) -> SummaryIndex:
        """SummaryIndexを修正版で作成（要約機能を適切に実装）"""
        try:
            self.logger.info("Creating SummaryIndex with proper summarization...")
            storage_context = db_manager.get_storage_context()
            
            # SummaryIndex用の専用ノードパーサー（大きめのチャンク）
            node_parser = SentenceSplitter(
                chunk_size=2048,  # 要約には大きめのチャンクが適している
                chunk_overlap=100,
                separator="\n\n"
            )
            
            # Response Synthesizerを設定
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",  # 階層的要約モード
                use_async=False
            )
            
            # SummaryIndexを適切な設定で作成
            index = SummaryIndex.from_documents(
                documents,
                storage_context=storage_context,
                node_parser=node_parser,
                response_synthesizer=response_synthesizer,
                show_progress=True
            )
            
            # インデックスIDを設定して保存
            index.set_index_id("summary_index")
            index.storage_context.persist()
            
            # データが正しく保存されたか確認
            self._verify_summary_index_data(index)
            
            self.logger.info("SummaryIndex created and verified successfully")
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to create SummaryIndex: {e}")
            raise

    def create_vector_store_index(self, documents: List[Document]) -> VectorStoreIndex:
        """VectorStoreIndexを作成"""
        try:
            self.logger.info("Creating VectorStoreIndex...")
            storage_context = db_manager.get_storage_context()
            
            node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                node_parser=node_parser,
                show_progress=True
            )
            
            index.set_index_id("vector_store_index")
            index.storage_context.persist()
            
            self.logger.info("VectorStoreIndex created successfully")
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to create VectorStoreIndex: {e}")
            raise

    def create_document_summary_index(self, documents: List[Document]) -> DocumentSummaryIndex:
        """DocumentSummaryIndexを作成（SummaryIndexとは別物）"""
        try:
            self.logger.info("Creating DocumentSummaryIndex...")
            storage_context = db_manager.get_storage_context()
            
            # DocumentSummaryIndex用の設定
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                use_async=False
            )
            
            index = DocumentSummaryIndex.from_documents(
                documents,
                storage_context=storage_context,
                response_synthesizer=response_synthesizer,
                show_progress=True
            )
            
            index.set_index_id("document_summary_index")
            index.storage_context.persist()
            
            self.logger.info("DocumentSummaryIndex created successfully")
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to create DocumentSummaryIndex: {e}")
            raise

    def create_all_indexes(self, documents: List[Document]) -> Dict[str, Any]:
        """すべてのインデックスを作成（修正版）"""
        indexes = {}
        
        try:
            # 各インデックスを順次作成
            self.logger.info("Creating all indexes with enhanced configuration...")
            
            indexes['vector_store'] = self.create_vector_store_index(documents)
            indexes['summary'] = self.create_summary_index(documents)
            indexes['tree'] = self.create_tree_index(documents)
            indexes['document_summary'] = self.create_document_summary_index(documents)
            
            # 全インデックスの検証
            self._verify_all_indexes(indexes)
            
            self.logger.info("All indexes created and verified successfully")
            return indexes
            
        except Exception as e:
            self.logger.error(f"Failed to create all indexes: {e}")
            raise

    def _verify_tree_index_data(self, index: TreeIndex):
        """TreeIndexのデータ検証"""
        try:
            # ルートノードの確認
            if hasattr(index, '_root_nodes') and index._root_nodes:
                self.logger.info(f"TreeIndex root nodes count: {len(index._root_nodes)}")
                
                # 各ルートノードの子ノード確認
                for i, root_node in enumerate(index._root_nodes):
                    if hasattr(root_node, 'child_nodes'):
                        child_count = len(root_node.child_nodes) if root_node.child_nodes else 0
                        self.logger.info(f"Root node {i} has {child_count} children")
            
            # ドキュメントストアの確認
            docstore = index.storage_context.docstore
            doc_count = len(docstore.docs)
            self.logger.info(f"TreeIndex docstore contains {doc_count} documents")
            
            if doc_count == 0:
                raise ValueError("TreeIndex docstore is empty!")
                
        except Exception as e:
            self.logger.error(f"TreeIndex data verification failed: {e}")
            raise

    def _verify_summary_index_data(self, index: SummaryIndex):
        """SummaryIndexのデータ検証"""
        try:
            # ノード数の確認
            if hasattr(index, '_index_struct') and hasattr(index._index_struct, 'nodes'):
                node_count = len(index._index_struct.nodes)
                self.logger.info(f"SummaryIndex contains {node_count} nodes")
                
                if node_count == 0:
                    raise ValueError("SummaryIndex has no nodes!")
            
            # ドキュメントストアの確認
            docstore = index.storage_context.docstore
            doc_count = len(docstore.docs)
            self.logger.info(f"SummaryIndex docstore contains {doc_count} documents")
            
            if doc_count == 0:
                raise ValueError("SummaryIndex docstore is empty!")
                
            # テストクエリで動作確認
            query_engine = index.as_query_engine()
            test_response = query_engine.query("この文書の要約を教えてください")
            
            if not test_response or len(str(test_response)) < 10:
                raise ValueError("SummaryIndex query returned empty response!")
                
            self.logger.info("SummaryIndex query test passed")
            
        except Exception as e:
            self.logger.error(f"SummaryIndex data verification failed: {e}")
            raise

    def _verify_all_indexes(self, indexes: Dict[str, Any]):
        """全インデックスの統合検証"""
        self.logger.info("Performing comprehensive index verification...")
        
        for index_name, index in indexes.items():
            try:
                # 基本的な検索テスト
                query_engine = index.as_query_engine()
                test_query = "テストクエリ"
                response = query_engine.query(test_query)
                
                self.logger.info(f"{index_name} query test: {'PASSED' if response else 'FAILED'}")
                
            except Exception as e:
                self.logger.error(f"{index_name} verification failed: {e}")

    def test_all_indexes(self, test_query: str = "この文書について教えてください") -> Dict[str, str]:
        """全インデックスのクエリテスト"""
        results = {}
        
        try:
            storage_context = db_manager.get_storage_context()
            
            # 各インデックスタイプをロードしてテスト
            index_types = ['vector_store_index', 'summary_index', 'tree_index', 'document_summary_index']
            
            for index_type in index_types:
                try:
                    index = load_index_from_storage(storage_context, index_id=index_type)
                    query_engine = index.as_query_engine()
                    response = query_engine.query(test_query)
                    
                    results[index_type] = str(response)
                    self.logger.info(f"{index_type} test completed successfully")
                    
                except Exception as e:
                    results[index_type] = f"Error: {str(e)}"
                    self.logger.error(f"{index_type} test failed: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Index testing failed: {e}")
            return {"error": str(e)}

    def load_index(self, index_type: str, storage_context=None):
        """保存されたインデックスを読み込み（修正版）"""
        try:
            if storage_context is None:
                storage_context = db_manager.get_storage_context()
            
            # インデックスIDのマッピング
            index_id_mapping = {
                'vector_store': 'vector_store_index',
                'summary': 'summary_index',
                'tree': 'tree_index',
                'document_summary': 'document_summary_index'
            }
            
            index_id = index_id_mapping.get(index_type, index_type)
            index = load_index_from_storage(storage_context, index_id=index_id)
            
            self.logger.info(f"Successfully loaded {index_type} index")
            return index
                
        except Exception as e:
            self.logger.error(f"Failed to load {index_type} index: {e}")
            raise
