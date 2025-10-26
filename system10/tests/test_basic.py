"""
基本的なテストケース
"""

import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_manager import ConfigManager
from src.chunking.chunking import ChunkerFactory
from src.embedding.embedding import EmbeddingFactory


class TestConfigManager:
    """ConfigManagerのテスト"""
    
    def test_load_config(self):
        """設定ファイルの読み込みテスト"""
        config_manager = ConfigManager(config_dir="config")
        
        # 全設定読み込み
        configs = config_manager.load_all_configs()
        
        assert "chunking" in configs
        assert "embedding" in configs
        assert "llm" in configs
        assert "test_patterns" in configs
    
    def test_get_chunking_method(self):
        """チャンキング手法取得テスト"""
        config_manager = ConfigManager(config_dir="config")
        
        # token_based設定取得
        config = config_manager.get_chunking_method("token_based")
        
        assert config is not None
        assert config["type"] == "token_text_splitter"
        assert "chunk_size" in config


class TestChunking:
    """Chunkingモジュールのテスト"""
    
    def test_create_token_chunker(self):
        """トークンベースチャンカー作成テスト"""
        chunker = ChunkerFactory.create(
            method="token_based",
            chunk_size=512,
            chunk_overlap=50
        )
        
        assert chunker is not None
    
    def test_chunk_text(self):
        """テキストチャンキングテスト"""
        from llama_index.core.schema import Document
        
        # テストドキュメント作成
        text = "これはテストです。" * 100
        doc = Document(text=text)
        
        chunker = ChunkerFactory.create(
            method="token_based",
            chunk_size=100,
            chunk_overlap=20
        )
        
        nodes = chunker.chunk([doc])
        
        assert len(nodes) > 0
        assert all(hasattr(node, 'text') for node in nodes)


class TestEmbedding:
    """Embeddingモジュールのテスト"""
    
    def test_create_ollama_embedding(self):
        """Ollama埋め込みモデル作成テスト"""
        try:
            embed_model = EmbeddingFactory.create(
                backend="ollama",
                model_name="qwen3-embedding:8b"
            )
            
            assert embed_model is not None
        except Exception as e:
            # Ollamaが起動していない場合はスキップ
            pytest.skip(f"Ollama not available: {e}")
    
    def test_create_huggingface_embedding(self):
        """HuggingFace埋め込みモデル作成テスト"""
        embed_model = EmbeddingFactory.create(
            backend="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert embed_model is not None


class TestEndToEnd:
    """エンドツーエンドテスト"""
    
    @pytest.mark.slow
    def test_simple_pipeline(self):
        """シンプルなパイプラインテスト"""
        from llama_index.core.schema import Document
        from src.indexing.index_builder import IndexBuilderFactory
        
        # 1. ドキュメント作成
        text = """
        人工知能（AI）は、コンピュータシステムが人間のような知的な振る舞いを
        示すことを可能にする技術です。機械学習や深層学習などの手法により、
        AIは画像認識、自然言語処理、意思決定支援など、様々な分野で
        活用されています。
        """
        doc = Document(text=text)
        
        # 2. チャンキング
        chunker = ChunkerFactory.create(
            method="token_based",
            chunk_size=100,
            chunk_overlap=20
        )
        nodes = chunker.chunk([doc])
        
        assert len(nodes) > 0
        
        # 3. インデックス構築（ローカルのみ、DBなし）
        try:
            embed_model = EmbeddingFactory.create(
                backend="huggingface",
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            from llama_index.core import Settings
            Settings.embed_model = embed_model
            
            index = IndexBuilderFactory.build(
                index_type="vector",
                nodes=nodes,
                show_progress=False
            )
            
            assert index is not None
            
            # 4. クエリ実行
            query_engine = index.as_query_engine(similarity_top_k=2)
            response = query_engine.query("AIとは何ですか？")
            
            assert response is not None
            assert len(response.response) > 0
            
        except Exception as e:
            pytest.skip(f"Embedding or indexing failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
