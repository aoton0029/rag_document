"""
OpenAI埋め込みモデル実装
"""
from typing import List, Dict, Any, Union
import time
import numpy as np
from . import BaseEmbedding, EmbeddingResult

class OpenAIEmbedding(BaseEmbedding):
    """OpenAI APIを使用した埋め込み"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = self._get_api_key()
        self.client = None
        self._setup_client()
        
    def _get_api_key(self) -> str:
        """API キーを取得"""
        import os
        api_key_env = self.config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"Warning: {api_key_env} environment variable not found")
        return api_key
    
    def _setup_client(self):
        """OpenAI クライアントをセットアップ"""
        if not self.api_key:
            return
            
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            print(f"OpenAI client initialized for model: {self.model_name}")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
    
    def embed_text(self, text: Union[str, List[str]]) -> EmbeddingResult:
        """テキストを埋め込みベクトルに変換"""
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized")
            
        start_time = time.time()
        
        # 前処理
        if isinstance(text, str):
            processed_text = [self._preprocess_text(text)]
        else:
            processed_text = [self._preprocess_text(t) for t in text]
        
        try:
            # OpenAI API呼び出し
            response = self.client.embeddings.create(
                model=self.model_name,
                input=processed_text
            )
            
            # 埋め込みを抽出
            embeddings = np.array([item.embedding for item in response.data])
            
            # 後処理
            embeddings = self._postprocess_embeddings(embeddings)
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                dimensions=embeddings.shape[1],
                metadata={
                    'input_texts_count': len(processed_text),
                    'usage': response.usage.dict() if hasattr(response, 'usage') else None,
                    'processing_time': processing_time
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

class AzureOpenAIEmbedding(OpenAIEmbedding):
    """Azure OpenAI Service を使用した埋め込み"""
    
    def __init__(self, config: Dict[str, Any]):
        self.azure_endpoint = config.get('azure_endpoint')
        self.api_version = config.get('api_version', '2023-05-15')
        super().__init__(config)
        
    def _setup_client(self):
        """Azure OpenAI クライアントをセットアップ"""
        if not self.api_key or not self.azure_endpoint:
            return
            
        try:
            import openai
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            print(f"Azure OpenAI client initialized for model: {self.model_name}")
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI client: {e}")