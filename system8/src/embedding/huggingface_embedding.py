"""
Hugging Face埋め込みモデル実装
"""
from typing import List, Dict, Any, Union
import time
import numpy as np
from . import BaseEmbedding, EmbeddingResult

class HuggingFaceEmbedding(BaseEmbedding):
    """Hugging Faceモデルを使用した埋め込み"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cpu')
        self._load_model()
        
    def _load_model(self):
        """モデルをロード"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Loaded HuggingFace model: {self.model_name}")
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
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
        
        # 埋め込み計算
        embeddings = self.model.encode(
            processed_text,
            convert_to_numpy=True,
            show_progress_bar=len(processed_text) > 10
        )
        
        # 後処理
        embeddings = self._postprocess_embeddings(embeddings)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimensions=embeddings.shape[1],
            metadata={
                'input_texts_count': len(processed_text),
                'device': self.device,
                'processing_time': processing_time
            },
            processing_time=processing_time
        )

class TransformersEmbedding(BaseEmbedding):
    """Transformersライブラリを直接使用した埋め込み"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cpu')
        self._load_model()
        
    def _load_model(self):
        """モデルとトークナイザーをロード"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded Transformers model: {self.model_name}")
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
    
    def embed_text(self, text: Union[str, List[str]]) -> EmbeddingResult:
        """テキストを埋め込みベクトルに変換"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model {self.model_name} is not loaded")
            
        import torch
        
        start_time = time.time()
        
        # 前処理
        if isinstance(text, str):
            processed_text = [self._preprocess_text(text)]
        else:
            processed_text = [self._preprocess_text(t) for t in text]
        
        # トークン化
        inputs = self.tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 埋め込み計算
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # プーリング戦略（平均プーリング）
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = embeddings.cpu().numpy()
        
        # 後処理
        embeddings = self._postprocess_embeddings(embeddings)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimensions=embeddings.shape[1],
            metadata={
                'input_texts_count': len(processed_text),
                'device': self.device,
                'processing_time': processing_time
            },
            processing_time=processing_time
        )
    
    def _mean_pooling(self, model_output, attention_mask):
        """平均プーリング"""
        import torch
        
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class JapaneseEmbedding(HuggingFaceEmbedding):
    """日本語特化埋め込みモデル"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def _preprocess_text(self, text: str) -> str:
        """日本語テキストの前処理"""
        # 基本的な前処理
        text = super()._preprocess_text(text)
        
        # 日本語特有の処理
        if self.config.get('japanese_preprocessing', {}).get('normalize_japanese', True):
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        
        return text