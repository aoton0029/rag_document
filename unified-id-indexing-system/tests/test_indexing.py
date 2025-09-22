import unittest
from src.indexing.mongo_indexer import MongoIndexer
from src.indexing.milvus_indexer import MilvusIndexer
from src.indexing.neo4j_indexer import Neo4jIndexer
from src.indexing.redis_indexer import RedisIndexer
from src.db.database_manager import db_manager

class TestIndexing(unittest.TestCase):

    def setUp(self):
        self.mongo_indexer = MongoIndexer(db_manager.mongo)
        self.milvus_indexer = MilvusIndexer(db_manager.milvus)
        self.neo4j_indexer = Neo4jIndexer(db_manager.neo4j)
        self.redis_indexer = RedisIndexer(db_manager.redis)

    def test_mongo_indexing(self):
        # Test MongoDB indexing functionality
        result = self.mongo_indexer.index_document({"test": "data"})
        self.assertTrue(result.success)

    def test_milvus_indexing(self):
        # Test Milvus indexing functionality
        result = self.milvus_indexer.index_vector([0.1, 0.2, 0.3], {"meta": "data"})
        self.assertTrue(result.success)

    def test_neo4j_indexing(self):
        # Test Neo4j indexing functionality
        result = self.neo4j_indexer.index_node({"name": "Test Node"})
        self.assertTrue(result.success)

    def test_redis_indexing(self):
        # Test Redis indexing functionality
        result = self.redis_indexer.index_data("test_key", "test_value")
        self.assertTrue(result.success)

if __name__ == '__main__':
    unittest.main()