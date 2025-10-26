"""
Embedding Module
Embeddingsのアダプター実装
要件定義に基づいた正確なインポート
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """埋め込み設定"""
    model_name: str = "qwen3-embedding:8b"
    backend: str = "ollama"
    dimension: int = 1024
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True


class BaseEmbeddingAdapter(ABC):
    """
    埋め込みアダプター基底クラス
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        BaseEmbeddingAdapterの初期化
        
        Args:
            config: 埋め込み設定
        """
        self.config = config or EmbeddingConfig()
        self._embed_model = None
    
    @abstractmethod
    def get_embed_model(self) -> BaseEmbedding:
        """埋め込みモデルを取得"""
        pass


class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Ollama埋め込みアダプター
    from llama_index.embeddings.ollama import OllamaEmbedding
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-embedding:8b",
        base_url: str = "http://localhost:11434",
        embed_batch_size: int = 10,
        **kwargs
    ):
        """
        OllamaEmbeddingAdapterの初期化
        
        Args:
            model_name: モデル名
            base_url: OllamaのベースURL
            embed_batch_size: バッチサイズ
            **kwargs: その他のパラメータ
        """
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.embed_batch_size = embed_batch_size
        self.kwargs = kwargs
    
    def get_embed_model(self) -> OllamaEmbedding:
        """
        Ollama埋め込みモデルを取得
        
        Returns:
            OllamaEmbedding
        """
        if self._embed_model is None:
            try:
                self._embed_model = OllamaEmbedding(
                    model_name=self.model_name,
                    base_url=self.base_url,
                    embed_batch_size=self.embed_batch_size,
                    **self.kwargs
                )
                logger.info(f"Ollama埋め込みモデルを初期化: {self.model_name}")
            except Exception as e:
                logger.error(f"OllamaEmbedding初期化エラー: {e}")
                raise
        
        return self._embed_model


class LangchainEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Langchain埋め込みアダプター
    from llama_index.embeddings.langchain import LangchainEmbedding
    """
    
    def __init__(
        self,
        langchain_embeddings: Any,
        **kwargs
    ):
        """
        LangchainEmbeddingAdapterの初期化
        
        Args:
            langchain_embeddings: LangchainのEmbeddingsインスタンス
            **kwargs: その他のパラメータ
        """
        super().__init__()
        self.langchain_embeddings = langchain_embeddings
        self.kwargs = kwargs
    
    def get_embed_model(self) -> LangchainEmbedding:
        """
        Langchain埋め込みモデルを取得
        
        Returns:
            LangchainEmbedding
        """
        if self._embed_model is None:
            try:
                self._embed_model = LangchainEmbedding(
                    langchain_embeddings=self.langchain_embeddings,
                    **self.kwargs
                )
                logger.info("Langchain埋め込みモデルを初期化")
            except Exception as e:
                logger.error(f"LangchainEmbedding初期化エラー: {e}")
                raise
        
        return self._embed_model


class HuggingFaceEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    HuggingFace埋め込みアダプター
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: Optional[int] = None,
        normalize: bool = True,
        **kwargs
    ):
        """
        HuggingFaceEmbeddingAdapterの初期化
        
        Args:
            model_name: HuggingFaceモデル名
            max_length: 最大長
            normalize: 正規化するか
            **kwargs: その他のパラメータ
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.kwargs = kwargs
    
    def get_embed_model(self) -> HuggingFaceEmbedding:
        """
        HuggingFace埋め込みモデルを取得
        
        Returns:
            HuggingFaceEmbedding
        """
        if self._embed_model is None:
            try:
                embed_kwargs = {
                    "model_name": self.model_name,
                    "normalize": self.normalize,
                    **self.kwargs
                }
                
                if self.max_length is not None:
                    embed_kwargs["max_length"] = self.max_length
                
                self._embed_model = HuggingFaceEmbedding(**embed_kwargs)
                logger.info(f"HuggingFace埋め込みモデルを初期化: {self.model_name}")
            except Exception as e:
                logger.error(f"HuggingFaceEmbedding初期化エラー: {e}")
                raise
        
        return self._embed_model


class EmbeddingFactory:
    """
    埋め込みモデルファクトリー
    """
    
    @staticmethod
    def create_embedding(
        backend: str = "ollama",
        **kwargs
    ) -> BaseEmbedding:
        """
        埋め込みモデルを作成
        
        Args:
            backend: バックエンド ("ollama", "langchain", "huggingface")
            **kwargs: バックエンド固有のパラメータ
            
        Returns:
            BaseEmbedding
        """
        if backend == "ollama":
            adapter = OllamaEmbeddingAdapter(**kwargs)
            return adapter.get_embed_model()
        elif backend == "langchain":
            adapter = LangchainEmbeddingAdapter(**kwargs)
            return adapter.get_embed_model()
        elif backend == "huggingface":
            adapter = HuggingFaceEmbeddingAdapter(**kwargs)
            return adapter.get_embed_model()
        else:
            raise ValueError(f"未対応の埋め込みバックエンド: {backend}")
