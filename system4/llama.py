import logging
from typing import List
from llama_index.core import VectorStoreIndex, Settings
from llm.ollama_connector import OllamaConnector
from db.database_manager import db_manager
from llama_index.core import Document, load_index_from_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_system():
    """システムの初期化"""
    logger.info("システム初期化開始")
    
    # データベース接続初期化
    db_manager.initialize_connections()
    
    # Ollama接続初期化
    ollama = OllamaConnector.generate()
    
    # グローバル設定
    Settings.llm = ollama.llm
    Settings.embed_model = ollama.embedding_model
    
    logger.info("システム初期化完了")
    return ollama


def create_sample_documents() -> List[Document]:
    """サンプルドキュメントを作成"""
    documents = [
        Document(
            text="人工知能（AI）は、機械学習と深層学習の技術を使用して、人間のような知的な動作を模倣するコンピューターシステムです。",
            metadata={"source": "ai_intro", "category": "technology", "id": "doc_1"}
        ),
        Document(
            text="機械学習は、データからパターンを学習し、新しいデータに対して予測や分類を行う技術です。",
            metadata={"source": "ml_basics", "category": "technology", "id": "doc_2"}
        ),
        Document(
            text="自然言語処理（NLP）は、コンピューターが人間の言語を理解し、処理する技術分野です。",
            metadata={"source": "nlp_overview", "category": "technology", "id": "doc_3"}
        ),
        Document(
            text="ベクトルデータベースは、高次元ベクトルデータを効率的に保存・検索するためのデータベースシステムです。",
            metadata={"source": "vector_db", "category": "database", "id": "doc_4"}
        )
    ]
    return documents


def build_and_store_index():
    """インデックスを構築してストレージに保存"""
    logger.info("=== インデックス構築・保存開始 ===")
    
    # サンプルドキュメント作成
    documents = create_sample_documents()
    logger.info(f"作成されたドキュメント数: {len(documents)}")
    
    # ストレージコンテキスト取得
    storage_context = db_manager.get_storage_context()
    
    # VectorStoreIndexを構築
    logger.info("VectorStoreIndex構築中...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    
    logger.info("インデックス構築完了")
    return index


def verify_index_and_storage():
    """インデックス情報と各DBの保存データを確認"""
    logger.info("=== インデックス・ストレージ確認開始 ===")
    
    # 各データベースの保存データを確認
    db_manager.verify_stored_data()
    
    # インデックスからの検索テスト
    logger.info("--- インデックス検索テスト ---")
    try:
        storage_context = db_manager.get_storage_context()
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        
        # クエリエンジン作成
        query_engine = index.as_query_engine()
        
        # テストクエリ実行
        test_query = "機械学習とは何ですか？"
        response = query_engine.query(test_query)
        
        logger.info(f"テストクエリ: {test_query}")
        logger.info(f"応答: {response}")
        
    except Exception as e:
        logger.error(f"インデックス検索テストエラー: {e}")


if __name__ == '__main__':
    try:
        # システム初期化
        ollama = initialize_system()
        
        # インデックス構築・保存
        index = build_and_store_index()
        
        # データ確認
        verify_index_and_storage()
        
        logger.info("=== 処理完了 ===")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise