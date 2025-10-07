import logging
from db.database_manager import db_manager

logger = logging.getLogger(__name__)

def verify_stored_data():
    """各データベースに保存されたデータを確認"""
    logger.info("=== データベース確認開始 ===")
    
    # Milvus (Vector Store) の確認
    _verify_milvus_data()
    
    # MongoDB (Document Store) の確認
    _verify_mongodb_data()
    
    # Redis (KV Store & Index Store) の確認
    _verify_redis_data()
    
    # Neo4j (Graph Store) の確認
    _verify_neo4j_data()

def _verify_milvus_data():
    """Milvusのデータ確認"""
    try:
        logger.info("--- Milvus Vector Store 確認 ---")
        if db_manager.milvus:
            collection_info = db_manager.milvus.get_collection_info()
            logger.info(f"Collection: {collection_info}")
            
            # ベクトル数を確認
            vector_count = db_manager.milvus.count_vectors()
            logger.info(f"保存されたベクトル数: {vector_count}")
    except Exception as e:
        logger.error(f"Milvus確認エラー: {e}")

def _verify_mongodb_data():
    """MongoDBのデータ確認"""
    try:
        logger.info("--- MongoDB Document Store 確認 ---")
        if db_manager.mongo:
            collections = db_manager.mongo.list_collections()
            logger.info(f"Collections: {collections}")
            
            for collection_name in collections:
                count = db_manager.mongo.count_documents(collection_name)
                logger.info(f"{collection_name}: {count} documents")
                
                # サンプルドキュメントを表示
                if count > 0:
                    sample = db_manager.mongo.find_documents(collection_name, limit=1)
                    if sample:
                        logger.info(f"Sample document keys: {list(sample[0].keys())}")
    except Exception as e:
        logger.error(f"MongoDB確認エラー: {e}")

def _verify_redis_data():
    """Redisのデータ確認"""
    try:
        logger.info("--- Redis KV/Index Store 確認 ---")
        if db_manager.redis:
            all_keys = db_manager.redis.get_all_keys()
            logger.info(f"Redis keys 総数: {len(all_keys)}")
            
            # キーの種類別に分類
            index_keys = [k for k in all_keys if 'index' in k.lower()]
            other_keys = [k for k in all_keys if 'index' not in k.lower()]
            
            logger.info(f"Index関連キー: {len(index_keys)}")
            logger.info(f"その他のキー: {len(other_keys)}")
            
            # サンプルキーを表示
            if all_keys:
                sample_keys = all_keys[:5]
                logger.info(f"Sample keys: {sample_keys}")
    except Exception as e:
        logger.error(f"Redis確認エラー: {e}")

def _verify_neo4j_data():
    """Neo4jのデータ確認"""
    try:
        logger.info("--- Neo4j Graph Store 確認 ---")
        if db_manager.neo4j:
            node_count = db_manager.neo4j.count_nodes()
            relationship_count = db_manager.neo4j.count_relationships()
            
            logger.info(f"ノード数: {node_count}")
            logger.info(f"リレーション数: {relationship_count}")
            
            # ノードタイプを確認
            node_types = db_manager.neo4j.get_node_labels()
            logger.info(f"ノードタイプ: {node_types}")
    except Exception as e:
        logger.error(f"Neo4j確認エラー: {e}")

