"""
Database module usage example
データベースモジュールの使用例
"""

import logging
from typing import Dict, Any
from src.database import DatabaseManager, DatabaseConfig


# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_database_usage():
    """データベースモジュールの基本的な使用例"""
    
    # 1. 設定の作成
    config = DatabaseConfig(
        # MongoDB設定
        mongodb_host="localhost",
        mongodb_port=27017,
        mongodb_database="rag_evaluation",
        mongodb_username=None,  # 認証が必要な場合は設定
        mongodb_password=None,
        
        # Redis設定
        redis_host="localhost", 
        redis_port=6379,
        redis_password=None,
        redis_db=0,
        
        # Milvus設定
        milvus_host="localhost",
        milvus_port=19530,
        milvus_user=None,
        milvus_password=None,
        milvus_collection="document_vectors",
        milvus_dim=1536,
        
        # Neo4j設定
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j"
    )
    
    # 2. DatabaseManagerの作成
    db_manager = DatabaseManager(config)
    
    try:
        # 3. 個別クライアントの使用例
        logger.info("=== 個別クライアントの使用例 ===")
        
        # MongoDBクライアント
        mongodb_client = db_manager.get_mongodb_client()
        logger.info(f"MongoDB健康状態: {mongodb_client.health_check()}")
        
        # Redisクライアント
        redis_client = db_manager.get_redis_client()
        logger.info(f"Redis健康状態: {redis_client.health_check()}")
        
        # Milvusクライアント
        milvus_client = db_manager.get_milvus_client()
        logger.info(f"Milvus健康状態: {milvus_client.health_check()}")
        
        # Neo4jクライアント
        neo4j_client = db_manager.get_neo4j_client()
        logger.info(f"Neo4j健康状態: {neo4j_client.health_check()}")
        
        # 4. llama_indexストアの取得例
        logger.info("=== llama_indexストアの取得例 ===")
        
        # ドキュメントストア（MongoDB）
        docstore = db_manager.get_docstore(namespace="experiment_1")
        logger.info(f"ドキュメントストア取得: {type(docstore)}")
        
        # インデックスストア（Redis）
        index_store = db_manager.get_index_store(namespace="experiment_1")
        logger.info(f"インデックスストア取得: {type(index_store)}")
        
        # ベクトルストア（Milvus）
        vector_store = db_manager.get_vector_store(collection_name="exp1_vectors")
        logger.info(f"ベクトルストア取得: {type(vector_store)}")
        
        # グラフストア（Neo4j）
        graph_store = db_manager.get_graph_store(
            node_label="Document", 
            rel_type="REFERENCES"
        )
        logger.info(f"グラフストア取得: {type(graph_store)}")
        
        # 5. StorageContextの構築
        logger.info("=== StorageContextの構築例 ===")
        storage_context = db_manager.get_storage_context(
            docstore_namespace="experiment_1",
            index_namespace="experiment_1", 
            vector_collection="exp1_vectors",
            graph_node_label="Document",
            graph_rel_type="REFERENCES"
        )
        logger.info(f"StorageContext構築完了: {type(storage_context)}")
        
        # 6. 全データベースのヘルスチェック
        logger.info("=== 全データベース健康状態チェック ===")
        health_status = db_manager.health_check_all()
        for db_name, status in health_status.items():
            logger.info(f"{db_name}: {'OK' if status else 'ERROR'}")
    
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
    
    finally:
        # 7. 接続の切断
        db_manager.disconnect_all()
        logger.info("全データベース接続を切断しました")


def example_config_from_dict():
    """設定辞書からDatabaseManagerを作成する例"""
    
    config_dict = {
        "mongodb_host": "localhost",
        "mongodb_port": 27017,
        "mongodb_database": "rag_test",
        "redis_host": "localhost",
        "redis_port": 6379,
        "milvus_host": "localhost",
        "milvus_port": 19530,
        "milvus_collection": "test_vectors",
        "milvus_dim": 768,
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "test_password"
    }
    
    # 設定辞書からDatabaseManagerを作成
    db_manager = DatabaseManager.from_config_dict(config_dict)
    
    try:
        # ヘルスチェック
        health_status = db_manager.health_check_all()
        logger.info(f"設定辞書から作成したDatabaseManager: {health_status}")
    
    except Exception as e:
        logger.error(f"設定辞書からの作成でエラー: {e}")
    
    finally:
        db_manager.disconnect_all()


def example_context_manager():
    """コンテキストマネージャーを使用した例"""
    
    config = DatabaseConfig(
        mongodb_database="context_test",
        milvus_collection="context_vectors"
    )
    
    # コンテキストマネージャーとして使用
    with DatabaseManager(config) as db_manager:
        logger.info("コンテキストマネージャーで自動接続されました")
        
        # StorageContextを取得
        storage_context = db_manager.get_storage_context()
        logger.info("StorageContextを取得しました")
        
        # 何らかの処理...
        
    # ここで自動的に全接続が切断される
    logger.info("コンテキストマネージャーで自動切断されました")


if __name__ == "__main__":
    logger.info("Database module usage examples")
    
    print("\n1. 基本的な使用例")
    example_database_usage()
    
    print("\n2. 設定辞書からの作成例")
    example_config_from_dict()
    
    print("\n3. コンテキストマネージャーの使用例")
    example_context_manager()
    
    logger.info("全ての例が完了しました")