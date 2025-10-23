"""
各種プロバイダー埋め込み実装
Ollama、HuggingFace、OpenAI等の埋め込みモデル
"""

from typing import List, Dict, Any, Optional
import numpy as np
import requests
import json

from .base_embedder import BaseEmbedder, EmbeddingProvider
from ..utils import get_logger

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OllamaEmbedder(BaseEmbedder):
    """Ollama埋め込み生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not OLLAMA_AVAILABLE:
            raise ImportError("llama-index-embeddings-ollama is required for OllamaEmbedder")
        
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "qwen3-embedding:8b")
        self.request_timeout = config.get("request_timeout", 60)
        
        # Ollama embedding インスタンスを作成
        try:
            self.embedder = OllamaEmbedding(
                model_name=self.model_name,
                base_url=self.base_url,
                request_timeout=self.request_timeout
            )
            
            # 次元数を取得するため、テストエンコーディング
            test_embedding = self.embedder.get_text_embedding("test")
            self.dimensions = len(test_embedding)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama embedder: {e}")
            raise
        
        self.logger.info("OllamaEmbedder initialized", 
                        model_name=self.model_name,
                        dimensions=self.dimensions)
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Ollamaでテキストを埋め込み"""
        embeddings = []
        
        for text in texts:
            try:
                embedding = self.embedder.get_text_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Failed to encode text with Ollama: {e}")
                # エラー時はゼロベクトルを返す
                embeddings.append([0.0] * self.dimensions)
        
        return embeddings
    
    def get_embedding_dimensions(self) -> int:
        return self.dimensions


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformers埋め込み生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for SentenceTransformerEmbedder")
        
        self.model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self.device = config.get("device", "cpu")
        self.normalize_embeddings = config.get("normalize_embeddings", True)
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise
        
        self.logger.info("SentenceTransformerEmbedder initialized",
                        model_name=self.model_name,
                        dimensions=self.dimensions,
                        device=self.device)
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """SentenceTransformersでテキストを埋め込み"""
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # NumPy配列をリストに変換
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to encode texts with SentenceTransformer: {e}")
            # エラー時はゼロベクトルを返す
            return [[0.0] * self.dimensions for _ in texts]
    
    def get_embedding_dimensions(self) -> int:
        return self.dimensions


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace Transformers埋め込み生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for HuggingFaceEmbedder")
        
        self.model_name = config.get("model_name", "bert-base-multilingual-cased")
        self.device = config.get("device", "cpu")
        self.max_length = config.get("max_length", 512)
        self.pooling_strategy = config.get("pooling_strategy", "mean")  # mean, cls, max
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 次元数を取得
            test_inputs = self.tokenizer("test", return_tensors="pt", 
                                       max_length=self.max_length, 
                                       truncation=True, padding=True)
            test_inputs = {k: v.to(self.device) for k, v in test_inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**test_inputs)
                hidden_states = outputs.last_hidden_state
                pooled = self._pool_embeddings(hidden_states, test_inputs["attention_mask"])
                self.dimensions = pooled.shape[-1]
            
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            raise
        
        self.logger.info("HuggingFaceEmbedder initialized",
                        model_name=self.model_name,
                        dimensions=self.dimensions,
                        device=self.device)
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """HuggingFace Transformersでテキストを埋め込み"""
        try:
            # バッチトークン化
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # デバイスに移動
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # プーリング戦略を適用
                pooled_embeddings = self._pool_embeddings(hidden_states, inputs["attention_mask"])
                
                # CPU NumPy配列に変換してリスト化
                embeddings = pooled_embeddings.cpu().numpy().tolist()
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode texts with HuggingFace: {e}")
            # エラー時はゼロベクトルを返す
            return [[0.0] * self.dimensions for _ in texts]
    
    def _pool_embeddings(self, hidden_states: torch.Tensor, 
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """埋め込みプーリング戦略を適用"""
        if self.pooling_strategy == "cls":
            # [CLS] トークンの埋め込みを使用
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # マスクを考慮した平均プーリング
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == "max":
            # マックスプーリング
            masked_embeddings = hidden_states.clone()
            masked_embeddings[attention_mask == 0] = -1e9
            return torch.max(masked_embeddings, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def get_embedding_dimensions(self) -> int:
        return self.dimensions


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API埋め込み生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required for OpenAIEmbedder")
        
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "text-embedding-ada-002")
        self.api_base = config.get("api_base")
        self.max_retries = config.get("max_retries", 3)
        
        # OpenAIクライアント設定
        client_kwargs = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        
        self.client = openai.OpenAI(**client_kwargs)
        
        # 次元数設定（モデル別）
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        self.dimensions = config.get("dimensions", model_dimensions.get(self.model_name, 1536))
        
        self.logger.info("OpenAIEmbedder initialized",
                        model_name=self.model_name,
                        dimensions=self.dimensions)
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """OpenAI APIでテキストを埋め込み"""
        embeddings = []
        
        for text in texts:
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=text
                    )
                    
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for OpenAI embedding: {e}")
                    if attempt == self.max_retries - 1:
                        # 最後の試行でも失敗した場合はゼロベクトル
                        embeddings.append([0.0] * self.dimensions)
        
        return embeddings
    
    def get_embedding_dimensions(self) -> int:
        return self.dimensions


class CustomEmbedder(BaseEmbedder):
    """カスタム埋め込み生成器（ローカルAPIやカスタムモデル用）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.api_endpoint = config.get("api_endpoint")
        self.api_headers = config.get("api_headers", {})
        self.request_timeout = config.get("request_timeout", 30)
        self.dimensions = config.get("dimensions", 768)
        
        if not self.api_endpoint:
            raise ValueError("api_endpoint is required for CustomEmbedder")
        
        self.logger.info("CustomEmbedder initialized",
                        api_endpoint=self.api_endpoint,
                        dimensions=self.dimensions)
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """カスタムAPIでテキストを埋め込み"""
        embeddings = []
        
        for text in texts:
            try:
                payload = {
                    "text": text,
                    "model": self.model_name
                }
                
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    headers=self.api_headers,
                    timeout=self.request_timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                # APIレスポンス形式に応じて調整
                if "embedding" in result:
                    embedding = result["embedding"]
                elif "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0]["embedding"]
                else:
                    raise ValueError("Unknown API response format")
                
                embeddings.append(embedding)
                
            except Exception as e:
                self.logger.error(f"Failed to encode text with custom API: {e}")
                embeddings.append([0.0] * self.dimensions)
        
        return embeddings
    
    def get_embedding_dimensions(self) -> int:
        return self.dimensions