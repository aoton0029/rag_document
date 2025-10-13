"""
LLM and Embedding models setup using Ollama
OllamaベースのLLMと埋め込みモデルの初期化
"""
from typing import Optional
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from config.config import config
from loguru import logger
import ollama

class ModelManager:
    """
    OllamaベースのLLMと埋め込みモデルの管理クラス
    """
    
    def __init__(self):
        self.llm: Optional[Ollama] = None
        self.embed_model: Optional[OllamaEmbedding] = None
        
    def check_ollama_availability(self) -> bool:
        """Ollamaサービスの可用性をチェック"""
        try:
            client = ollama.Client(host=config.ollama.base_url)
            # サービス状態を確認
            models = client.list()
            logger.info(f"Ollama service is available at {config.ollama.base_url}")
            return True
        except Exception as e:
            logger.error(f"Ollama service is not available: {str(e)}")
            return False
    
    def list_available_models(self) -> dict:
        """利用可能なOllamaモデルをリスト"""
        try:
            client = ollama.Client(host=config.ollama.base_url)
            models_response = client.list()
            
            available_models = {
                "llm_models": [],
                "embedding_models": []
            }
            
            for model in models_response.get('models', []):
                model_name = model.get('name', '')
                if 'embed' in model_name.lower() or 'embedding' in model_name.lower():
                    available_models["embedding_models"].append(model_name)
                else:
                    available_models["llm_models"].append(model_name)
            
            logger.info(f"Available LLM models: {available_models['llm_models']}")
            logger.info(f"Available embedding models: {available_models['embedding_models']}")
            
            return available_models
            
        except Exception as e:
            logger.error(f"Failed to list available models: {str(e)}")
            return {"llm_models": [], "embedding_models": []}
    
    def setup_llm(self) -> Ollama:
        """Ollama LLMの初期化"""
        try:
            logger.info(f"Initializing Ollama LLM: {config.ollama.llm_model}")
            
            self.llm = Ollama(
                model=config.ollama.llm_model,
                base_url=config.ollama.base_url,
                temperature=config.ollama.temperature,
                request_timeout=config.ollama.request_timeout,
                additional_kwargs={
                    "num_ctx": 4096,  # コンテキスト長
                    "num_predict": config.rag.max_tokens,  # 最大生成トークン数
                }
            )
            
            # テストクエリで動作確認
            test_response = self.llm.complete("Hello, this is a test.")
            logger.info(f"LLM test successful: {test_response.text[:50]}...")
            
            logger.success(f"Ollama LLM initialized successfully: {config.ollama.llm_model}")
            return self.llm
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            raise
    
    def setup_embedding_model(self) -> OllamaEmbedding:
        """Ollama埋め込みモデルの初期化"""
        try:
            logger.info(f"Initializing Ollama Embedding: {config.ollama.embed_model}")
            
            self.embed_model = OllamaEmbedding(
                model_name=config.ollama.embed_model,
                base_url=config.ollama.base_url,
                ollama_additional_kwargs={
                    "mirostat": 0,
                    "num_ctx": 2048
                }
            )
            
            # テストクエリで動作確認
            test_embedding = self.embed_model.get_text_embedding("This is a test sentence.")
            logger.info(f"Embedding test successful, dimension: {len(test_embedding)}")
            
            # 設定で指定された次元数と一致するか確認
            if len(test_embedding) != config.milvus.dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {config.milvus.dimension}, "
                    f"got {len(test_embedding)}"
                )
                # Milvus設定を更新
                config.milvus.dimension = len(test_embedding)
                logger.info(f"Updated Milvus dimension to {len(test_embedding)}")
            
            logger.success(f"Ollama Embedding initialized successfully: {config.ollama.embed_model}")
            return self.embed_model
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama Embedding: {str(e)}")
            raise
    
    def setup_global_settings(self):
        """LlamaIndexのグローバル設定を更新"""
        try:
            if self.llm is None:
                self.setup_llm()
            if self.embed_model is None:
                self.setup_embedding_model()
            
            # LlamaIndexのグローバル設定
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = config.rag.chunk_size
            Settings.chunk_overlap = config.rag.chunk_overlap
            
            logger.success("Global LlamaIndex settings configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup global settings: {str(e)}")
            raise
    
    def pull_model_if_needed(self, model_name: str) -> bool:
        """必要に応じてモデルをダウンロード"""
        try:
            client = ollama.Client(host=config.ollama.base_url)
            
            # 利用可能なモデルを確認
            available_models = [m['name'] for m in client.list().get('models', [])]
            
            if model_name not in available_models:
                logger.info(f"Model {model_name} not found locally. Downloading...")
                client.pull(model_name)
                logger.success(f"Successfully downloaded model: {model_name}")
                return True
            else:
                logger.info(f"Model {model_name} is already available locally")
                return True
                
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            return False
    
    def get_llm(self) -> Ollama:
        """LLMインスタンスの取得（初期化されていない場合は初期化）"""
        if self.llm is None:
            self.setup_llm()
        return self.llm
    
    def get_embed_model(self) -> OllamaEmbedding:
        """埋め込みモデルインスタンスの取得（初期化されていない場合は初期化）"""
        if self.embed_model is None:
            self.setup_embedding_model()
        return self.embed_model

# グローバルモデルマネージャインスタンス
model_manager = ModelManager()