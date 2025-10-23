"""
Ollama埋め込みモデル実装
"""
from typing import List, Dict, Any, Union
import time
import numpy as np
import requests
import json
from . import BaseEmbedding, EmbeddingResult

class OllamaEmbedding(BaseEmbedding):
    """Ollamaを使用した埋め込み"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.timeout = config.get('timeout', 30)
        self._check_connection()
        
    def _check_connection(self):
        """Ollamaサーバーへの接続をチェック"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"Connected to Ollama server at {self.base_url}")
                # モデルが利用可能かチェック
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name not in model_names:
                    print(f"Warning: Model {self.model_name} not found in available models: {model_names}")
            else:
                print(f"Warning: Cannot connect to Ollama server at {self.base_url}")
        except Exception as e:
            print(f"Warning: Failed to connect to Ollama server: {e}")
    
    def embed_text(self, text: Union[str, List[str]]) -> EmbeddingResult:
        """テキストを埋め込みベクトルに変換"""
        start_time = time.time()
        
        # 前処理
        if isinstance(text, str):
            processed_text = [self._preprocess_text(text)]
        else:
            processed_text = [self._preprocess_text(t) for t in text]
        
        embeddings = []
        
        for text_item in processed_text:
            try:
                # Ollama API呼び出し
                payload = {
                    "model": self.model_name,
                    "prompt": text_item
                }
                
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding', [])
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        raise ValueError("Empty embedding returned")
                else:
                    raise RuntimeError(f"Ollama API returned status {response.status_code}: {response.text}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to get embedding for text: {e}")
        
        # numpy配列に変換
        embeddings = np.array(embeddings)
        
        # 後処理
        embeddings = self._postprocess_embeddings(embeddings)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimensions=embeddings.shape[1],
            metadata={
                'input_texts_count': len(processed_text),
                'base_url': self.base_url,
                'processing_time': processing_time
            },
            processing_time=processing_time
        )
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[EmbeddingResult]:
        """バッチでテキストを埋め込み（Ollamaは並列処理に制限があるため小さめのバッチサイズ）"""
        return super().embed_batch(texts, batch_size)

class LlamaIndexOllamaEmbedding(BaseEmbedding):
    """LlamaIndex経由でOllamaを使用した埋め込み"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self._load_model()
        
    def _load_model(self):
        """LlamaIndex OllamaEmbeddingをロード"""
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            
            self.model = OllamaEmbedding(
                model_name=self.model_name,
                base_url=self.base_url
            )
            print(f"Loaded LlamaIndex Ollama embedding: {self.model_name}")
        except Exception as e:
            print(f"Failed to load LlamaIndex Ollama embedding: {e}")
            self.model = None
    
    def embed_text(self, text: Union[str, List[str]]) -> EmbeddingResult:
        """テキストを埋め込みベクトルに変換"""
        if self.model is None:
            raise RuntimeError(f"Model {self.model_name} is not loaded")
            
        start_time = time.time()
        
        # 前処理
        if isinstance(text, str):
            processed_text = [self._preprocess_text(text)]
        else:
            processed_text = [self._preprocess_text(t) for t in text]
        
        try:
            # LlamaIndex経由で埋め込み計算
            if len(processed_text) == 1:
                embeddings = [self.model.get_text_embedding(processed_text[0])]
            else:
                embeddings = self.model.get_text_embedding_batch(processed_text)
            
            embeddings = np.array(embeddings)
            
            # 後処理
            embeddings = self._postprocess_embeddings(embeddings)
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                dimensions=embeddings.shape[1],
                metadata={
                    'input_texts_count': len(processed_text),
                    'base_url': self.base_url,
                    'processing_time': processing_time
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            raise RuntimeError(f"LlamaIndex Ollama embedding failed: {e}")