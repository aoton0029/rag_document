import unittest
from src.indexing.index_registry import IndexRegistry

class TestIndexRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = IndexRegistry()

    def test_add_index(self):
        self.registry.add_index("test_index", "mongodb", {"status": "creating"})
        index_info = self.registry.get_index("test_index")
        self.assertEqual(index_info["database"], "mongodb")
        self.assertEqual(index_info["status"], "creating")

    def test_update_index_status(self):
        self.registry.add_index("test_index", "mongodb", {"status": "creating"})
        self.registry.update_index_status("test_index", "completed")
        index_info = self.registry.get_index("test_index")
        self.assertEqual(index_info["status"], "completed")

    def test_get_non_existent_index(self):
        index_info = self.registry.get_index("non_existent_index")
        self.assertIsNone(index_info)

    def test_remove_index(self):
        self.registry.add_index("test_index", "mongodb", {"status": "creating"})
        self.registry.remove_index("test_index")
        index_info = self.registry.get_index("test_index")
        self.assertIsNone(index_info)

if __name__ == '__main__':
    unittest.main()