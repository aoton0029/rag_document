"""
RAG評価フレームワークのテストスイート
"""
import unittest
import os
import tempfile
from src.chunking.chunker_factory import ChunkerFactory
from src.embedding.embedding_factory import EmbeddingFactory
from src.utils import load_yaml_config

class TestChunking(unittest.TestCase):
    """チャンキング機能のテスト"""
    
    def setUp(self):
        self.test_text = """
        これは論文のタイトルです。
        
        Abstract
        この論文では新しい手法を提案します。
        
        1. Introduction
        背景として以下の問題があります。
        
        2. Methodology
        提案手法は次の通りです。
        """
        
    def test_fixed_size_chunker(self):
        """固定サイズチャンカーのテスト"""
        config = {
            'chunk_size': 100,
            'chunk_overlap': 20,
            'separator': '\n\n'
        }
        
        chunker = ChunkerFactory.create_chunker('fixed_size', config)
        chunks = chunker.chunk(self.test_text)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) <= config['chunk_size'] * 1.5 for chunk in chunks))

class TestEmbedding(unittest.TestCase):
    """埋め込み機能のテスト"""
    
    def test_embedding_factory(self):
        """埋め込みファクトリーのテスト"""
        config = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimensions': 384
        }
        
        try:
            embedding = EmbeddingFactory.create_embedding('huggingface', 'sentence-transformers/all-MiniLM-L6-v2', config)
            self.assertIsNotNone(embedding)
        except Exception as e:
            self.skipTest(f"Embedding model not available: {e}")

class TestConfig(unittest.TestCase):
    """設定ファイルのテスト"""
    
    def test_load_yaml_config(self):
        """YAML設定読み込みのテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
test_config:
  chunk_size: 1024
  model: test_model
""")
            f.flush()
            
            config = load_yaml_config(f.name)
            self.assertIn('test_config', config)
            self.assertEqual(config['test_config']['chunk_size'], 1024)
            
            os.unlink(f.name)

if __name__ == '__main__':
    unittest.main()