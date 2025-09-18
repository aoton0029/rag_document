import requests
import json
from typing import List, Dict, Any, Optional
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class OllamaConnector:
    """Ollamaサーバーとの接続とモデル管理を行うクラス"""
    
    def __init__(self, base_url: str = "http://ollama:11434"):
        self.base_url = base_url
        self.llms = {}
        self.embedding_models = {}
    
    def initialize_llm(self, model_name: str = "llama3.2:3b") -> Ollama:
        """LLMモデルを初期化"""
        self.llms[model_name] = Ollama(
            model=model_name,
            base_url=self.base_url,
            temperature=0.1
        )
        return self.llms[model_name]
    
    def initialize_embedding(self, model_name: str = "nomic-embed-text") -> OllamaEmbedding:
        """埋め込みモデルを初期化"""
        self.embedding_models[model_name] = OllamaEmbedding(
            model_name=model_name,
            base_url=self.base_url
        )
        return self.embedding_models[model_name]

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