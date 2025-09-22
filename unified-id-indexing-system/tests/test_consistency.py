import unittest
from src.utils.consistency_checker import ConsistencyChecker
from src.db.database_manager import db_manager

class TestConsistency(unittest.TestCase):

    def setUp(self):
        self.checker = ConsistencyChecker(db_manager)

    def test_check_unified_id_consistency(self):
        result = self.checker.check_unified_id_consistency()
        self.assertTrue(result['consistent'], "Unified IDs are inconsistent across databases.")

    def test_check_data_integrity(self):
        result = self.checker.check_data_integrity()
        self.assertTrue(result['integrity'], "Data integrity check failed.")

    def test_check_index_status(self):
        result = self.checker.check_index_status()
        self.assertTrue(result['all_indexes_completed'], "Not all indexes are completed.")

if __name__ == '__main__':
    unittest.main()