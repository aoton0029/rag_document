import unittest
from src.core.unified_id import UnifiedID

class TestUnifiedID(unittest.TestCase):

    def setUp(self):
        self.unified_id_generator = UnifiedID()

    def test_generate_unified_id(self):
        unified_id = self.unified_id_generator.generate()
        self.assertIsNotNone(unified_id)
        self.assertIsInstance(unified_id, str)

    def test_unified_id_format(self):
        unified_id = self.unified_id_generator.generate()
        self.assertTrue(len(unified_id) > 0)
        self.assertTrue(unified_id.count('-') == 4)  # UUID format has 4 hyphens

    def test_generate_multiple_unified_ids(self):
        ids = {self.unified_id_generator.generate() for _ in range(100)}
        self.assertEqual(len(ids), 100)  # Ensure all generated IDs are unique

if __name__ == '__main__':
    unittest.main()