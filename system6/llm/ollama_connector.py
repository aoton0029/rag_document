import requests
import json
from enum import Enum 
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class LlmModel(BaseModel):
    model_name: str = None
    vector_dim: int = None
    
class LlmModelType(Enum):
    qwen3embedding: str = "qwen3-embedding:8b"
    elyza8b: str = "hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF:Q4_K_M"


class OllamaConnector:
    """Ollamaサーバーとの接続とモデル管理を行うクラス"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.llm_model: LlmModel = None
        self.llm: Ollama = None
        self.embedding_model: LlmModel = None
        self.embedding: OllamaEmbedding = None
        self.available_modellist = {
            LlmModelType.qwen3embedding: LlmModel(model_name=LlmModelType.qwen3embedding, vector_dim=1536),
            LlmModelType.elyza8b: LlmModel(model_name=LlmModelType.elyza8b, vector_dim=4096),
        } 
    
    def initialize_llm(self, model_name: str) -> Ollama:
        """LLMモデルを初期化"""
        self.llm_model = self.available_modellist.get(model_name)
        self.llm = Ollama(
            model=model_name,
            base_url=self.base_url,
            temperature=0.1
        )
        return self.llm

    def initialize_embedding(self, model_name: str) -> OllamaEmbedding:
        """埋め込みモデルを初期化"""
        self.embedding_model = self.available_modellist.get(model_name)
        self.embedding = OllamaEmbedding(
            model_name=model_name,
            base_url=self.base_url
        )
        return self.embedding

    def get_available_models(self) -> List[str]:
        """利用可能なモデル一覧を取得"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json()
            return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            print(f"モデル一覧取得エラー: {e}")
            return []
    
    def check_connection(self) -> bool:
        """Ollamaサーバーへの接続確認"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @staticmethod
    def generate(base_url: str, llm_model_name: str, embed_model_name: str) -> 'OllamaConnector':
        connector = OllamaConnector(base_url=base_url)
        connector.initialize_llm(model_name=llm_model_name)
        connector.initialize_embedding(model_name=embed_model_name)  # Use embedding model
        return connector