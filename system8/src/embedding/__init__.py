"""
埋め込みモジュールの基底クラスと共通インターフェース
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class EmbeddingResult:
    """埋め込み結果のデータクラス"""
    embeddings: np.ndarray
    model_name: str
    dimensions: int
    metadata: Dict[str, Any]
    processing_time: float = 0.0

class BaseEmbedding(ABC):
    """埋め込みモデルの基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'unknown')
        self.dimensions = config.get('dimensions', 768)
        self.max_sequence_length = config.get('max_sequence_length', 512)
        
    @abstractmethod
    def embed_text(self, text: Union[str, List[str]]) -> EmbeddingResult:
        """テキストを埋め込みベクトルに変換"""
        pass
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[EmbeddingResult]:
        """バッチでテキストを埋め込み"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.embed_text(batch)
            results.append(result)
            
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """テキストの前処理"""
        preprocessing = self.config.get('preprocessing', {})
        
        if preprocessing.get('remove_special_chars', False):
            import re
            text = re.sub(r'[^\w\s]', '', text)
            
        if preprocessing.get('lowercase', False):
            text = text.lower()
            
        if preprocessing.get('remove_stopwords', False):
            text = self._remove_stopwords(text)
            
        # 長さ制限
        if len(text) > self.max_sequence_length:
            text = text[:self.max_sequence_length]
            
        return text.strip()
    
    def _remove_stopwords(self, text: str) -> str:
        """ストップワードを除去"""
        # 簡単な日本語・英語ストップワード
        stop_words = {
            'は', 'が', 'を', 'に', 'の', 'で', 'と', 'から', 'まで',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    def _postprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """埋め込みの後処理"""
        postprocessing = self.config.get('postprocessing', {})
        
        if postprocessing.get('normalize_l2', True):
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, norm='l2')
            
        if postprocessing.get('dimension_reduction', False):
            components = postprocessing.get('pca_components')
            if components and components < embeddings.shape[1]:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=components)
                embeddings = pca.fit_transform(embeddings)
                
        return embeddings