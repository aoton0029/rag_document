"""
埋め込みファクトリー
設定に基づいて適切な埋め込み生成器を作成
"""

from typing import Dict, Any, Optional, Type, List, Union
from .base_embedder import BaseEmbedder, EmbeddingProvider
from .providers import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    CustomEmbedder
)
from .multi_vector import MultiVectorEmbedder
from .fine_tuning import FineTunedEmbedder, EmbeddingFineTuner
from ..utils import get_logger, ConfigManager


class EmbedderFactory:
    """埋め込み生成器ファクトリークラス"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager
        self.logger = get_logger("embedder_factory")
        
        # 利用可能な埋め込み生成器を登録
        self._embedders: Dict[str, Type[BaseEmbedder]] = {
            "ollama": OllamaEmbedder,
            "sentence_transformers": SentenceTransformerEmbedder,
            "huggingface": HuggingFaceEmbedder,
            "openai": OpenAIEmbedder,
            "custom": CustomEmbedder,
            "fine_tuned": FineTunedEmbedder
        }
        
        self.logger.info("EmbedderFactory initialized",
                        available_embedders=list(self._embedders.keys()))
    
    def create_embedder(self, provider: str, 
                       config: Optional[Dict[str, Any]] = None) -> BaseEmbedder:
        """指定されたプロバイダーで埋め込み生成器を作成"""
        
        if provider not in self._embedders:
            raise ValueError(f"Unknown embedding provider: {provider}. "
                           f"Available providers: {list(self._embedders.keys())}")
        
        # 設定を取得
        embedder_config = self._get_embedder_config(provider, config)
        
        # 埋め込み生成器を作成
        embedder_class = self._embedders[provider]
        
        try:
            if provider == "fine_tuned":
                # ファインチューニング済みモデルの場合は特別処理
                model_path = embedder_config.get("model_path")
                if not model_path:
                    raise ValueError("model_path is required for fine_tuned embedder")
                embedder = embedder_class(model_path, embedder_config)
            else:
                embedder = embedder_class(embedder_config)
            
            self.logger.info("Embedder created", 
                           provider=provider,
                           model_name=embedder_config.get("model_name", "unknown"))
            
            return embedder
            
        except Exception as e:
            self.logger.error(f"Failed to create embedder for {provider}: {e}")
            raise
    
    def create_multi_vector_embedder(self, 
                                   config: Optional[Dict[str, Any]] = None) -> MultiVectorEmbedder:
        """マルチベクター埋め込み生成器を作成"""
        
        # デフォルト設定
        default_config = {
            "embedders": {
                "sentence_transformers": {
                    "provider": "sentence_transformers",
                    "model_name": "all-MiniLM-L6-v2",
                    "weight": 1.0
                }
            },
            "fusion_strategy": "weighted_average",
            "parallel_encoding": True
        }
        
        # 設定をマージ
        multi_config = default_config.copy()
        if config:
            multi_config.update(config)
        
        # 設定マネージャーから追加設定を取得
        if self.config_manager:
            try:
                file_config = self.config_manager.get_embedding_config("multi_vector")
                multi_config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Failed to load multi-vector config: {e}")
        
        return MultiVectorEmbedder(multi_config)
    
    def create_fine_tuner(self, config: Optional[Dict[str, Any]] = None) -> EmbeddingFineTuner:
        """埋め込みファインチューナーを作成"""
        
        default_config = {
            "base_model": {
                "type": "sentence_transformers",
                "model_name": "all-MiniLM-L6-v2"
            },
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 3,
            "output_dir": "fine_tuned_models"
        }
        
        fine_tune_config = default_config.copy()
        if config:
            fine_tune_config.update(config)
        
        return EmbeddingFineTuner(fine_tune_config)
    
    def _get_embedder_config(self, provider: str, 
                           override_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """埋め込み生成器の設定を取得"""
        
        # デフォルト設定から開始
        base_config = self._get_default_config(provider)
        
        # 設定マネージャーから設定を取得
        if self.config_manager:
            try:
                file_config = self.config_manager.get_embedding_config(provider)
                base_config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config for {provider}: {e}")
        
        # オーバーライド設定を適用
        if override_config:
            base_config.update(override_config)
        
        return base_config
    
    def _get_default_config(self, provider: str) -> Dict[str, Any]:
        """プロバイダー別のデフォルト設定を取得"""
        
        defaults = {
            "ollama": {
                "model_name": "qwen3-embedding:8b",
                "base_url": "http://localhost:11434",
                "embed_batch_size": 10,
                "dimensions": 1536
            },
            
            "sentence_transformers": {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "embed_batch_size": 32,
                "normalize_embeddings": True
            },
            
            "huggingface": {
                "model_name": "bert-base-multilingual-cased",
                "device": "cpu",
                "embed_batch_size": 16,
                "max_length": 512,
                "pooling_strategy": "mean"
            },
            
            "openai": {
                "model_name": "text-embedding-ada-002",
                "embed_batch_size": 100,
                "dimensions": 1536,
                "max_retries": 3
            },
            
            "custom": {
                "embed_batch_size": 32,
                "dimensions": 768,
                "request_timeout": 30
            },
            
            "fine_tuned": {
                "embed_batch_size": 32
            }
        }
        
        return defaults.get(provider, {})
    
    def get_available_providers(self) -> List[str]:
        """利用可能なプロバイダー一覧を取得"""
        return list(self._embedders.keys())
    
    def register_embedder(self, provider: str, embedder_class: Type[BaseEmbedder]) -> None:
        """新しい埋め込み生成器を登録"""
        if not issubclass(embedder_class, BaseEmbedder):
            raise ValueError("Embedder class must inherit from BaseEmbedder")
        
        self._embedders[provider] = embedder_class
        self.logger.info("New embedder registered", 
                        provider=provider,
                        class_name=embedder_class.__name__)
    
    def create_embedders_from_config(self, 
                                   embedders_config: Dict[str, Dict[str, Any]]) -> Dict[str, BaseEmbedder]:
        """設定から複数の埋め込み生成器を作成"""
        embedders = {}
        
        for name, config in embedders_config.items():
            provider = config.get("provider")
            if provider:
                embedders[name] = self.create_embedder(provider, config)
            else:
                self.logger.warning(f"No provider specified for embedder {name}")
        
        return embedders
    
    def benchmark_embedders(self, 
                          texts: List[str],
                          providers: List[str],
                          configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """複数の埋め込み生成器でベンチマークを実行"""
        from ..utils import performance_monitor
        
        results = {}
        
        for provider in providers:
            config = configs.get(provider, {}) if configs else {}
            
            try:
                embedder = self.create_embedder(provider, config)
                
                # パフォーマンス測定付きで埋め込み生成
                @performance_monitor(f"embedding_{provider}")
                def embed_with_monitoring():
                    return embedder.encode_texts(texts)
                
                embedding_result = embed_with_monitoring()
                
                # 結果をまとめる
                results[provider] = {
                    "success": True,
                    "embedding_count": len(embedding_result.embeddings),
                    "dimensions": embedding_result.dimensions,
                    "model_name": embedding_result.model_name,
                    "metadata": embedding_result.metadata
                }
                
            except Exception as e:
                self.logger.error(f"Benchmarking failed for {provider}: {e}")
                results[provider] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def validate_embedder_availability(self, provider: str) -> Dict[str, Any]:
        """埋め込み生成器の利用可能性を検証"""
        validation_result = {
            "provider": provider,
            "available": False,
            "issues": []
        }
        
        if provider not in self._embedders:
            validation_result["issues"].append(f"Unknown provider: {provider}")
            return validation_result
        
        try:
            # 簡単なテスト埋め込みを試行
            test_config = self._get_default_config(provider)
            
            if provider == "fine_tuned":
                validation_result["issues"].append("Fine-tuned embedder requires model_path")
                return validation_result
            
            embedder = self._embedders[provider](test_config)
            test_result = embedder.encode_texts(["test"])
            
            if test_result.embeddings and len(test_result.embeddings[0]) > 0:
                validation_result["available"] = True
                validation_result["dimensions"] = test_result.dimensions
                validation_result["model_name"] = test_result.model_name
            else:
                validation_result["issues"].append("Empty or invalid embeddings returned")
                
        except Exception as e:
            validation_result["issues"].append(f"Initialization error: {str(e)}")
        
        return validation_result


# モジュールレベルの便利関数
def create_embedder(provider: str, 
                   config: Optional[Dict[str, Any]] = None,
                   config_manager: Optional[ConfigManager] = None) -> BaseEmbedder:
    """便利関数：埋め込み生成器を作成"""
    factory = EmbedderFactory(config_manager)
    return factory.create_embedder(provider, config)


def get_available_providers() -> List[str]:
    """便利関数：利用可能なプロバイダー一覧を取得"""
    factory = EmbedderFactory()
    return factory.get_available_providers()


def benchmark_embedding_providers(texts: List[str], 
                                providers: Optional[List[str]] = None,
                                configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """便利関数：埋め込みプロバイダーのベンチマーク"""
    factory = EmbedderFactory()
    
    if providers is None:
        # 利用可能なプロバイダーでベンチマーク（fine_tunedは除く）
        providers = [p for p in factory.get_available_providers() if p != "fine_tuned"]
    
    return factory.benchmark_embedders(texts, providers, configs)


def create_multi_vector_embedder(config: Optional[Dict[str, Any]] = None) -> MultiVectorEmbedder:
    """便利関数：マルチベクター埋め込み生成器を作成"""
    factory = EmbedderFactory()
    return factory.create_multi_vector_embedder(config)