import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class SentenceTransformerConnector:
    """SentenceTransformersを使用したモデル管理を行うクラス"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.embedding_dimensions = {}  # モデルごとの次元数を保存
        self.logger = logging.getLogger(__name__)
    
    def initialize_embedding(self, model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        """埋め込みモデルを初期化"""
        try:
            if model_name not in self.embedding_models:
                self.embedding_models[model_name] = SentenceTransformer(
                    model_name,
                    device=self.device
                )
                # 次元数を取得して保存
                self.embedding_dimensions[model_name] = self.embedding_models[model_name].get_sentence_embedding_dimension()
                self.logger.info(f"モデル '{model_name}' を {self.device} で初期化しました (次元数: {self.embedding_dimensions[model_name]})")
            return self.embedding_models[model_name]
        except Exception as e:
            self.logger.error(f"モデル初期化エラー: {e}")
            raise
    
    def get_embedding_model(self, model_name: str) -> Optional[SentenceTransformer]:
        """初期化済みの埋め込みモデルを取得"""
        return self.embedding_models.get(model_name)
    
    def get_embedding_dimension(self, model_name: str) -> Optional[int]:
        """指定モデルの埋め込み次元数を取得"""
        if model_name in self.embedding_dimensions:
            return self.embedding_dimensions[model_name]
        elif model_name in self.embedding_models:
            # 既存モデルの次元数を取得
            dim = self.embedding_models[model_name].get_sentence_embedding_dimension()
            self.embedding_dimensions[model_name] = dim
            return dim
        return None
    
    def encode_text(self, text: str, model_name: str = "all-MiniLM-L6-v2", **kwargs) -> np.ndarray:
        """単一テキストを埋め込みベクトルに変換"""
        model = self.get_embedding_model(model_name)
        if model is None:
            model = self.initialize_embedding(model_name)
        
        embedding = model.encode([text], convert_to_numpy=True, **kwargs)[0]
        return embedding
    
    def encode_texts(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32, **kwargs) -> List[np.ndarray]:
        """複数テキストを埋め込みベクトルに変換"""
        model = self.get_embedding_model(model_name)
        if model is None:
            model = self.initialize_embedding(model_name)
        
        embeddings = model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True, 
            **kwargs
        )
        return embeddings
    
    def add_model(self, model_name: str, force_reload: bool = False) -> SentenceTransformer:
        """新しいモデルを追加（既存の場合は再読み込みオプション）"""
        if model_name in self.embedding_models and not force_reload:
            self.logger.info(f"モデル '{model_name}' は既に初期化済みです")
            return self.embedding_models[model_name]
        
        if force_reload and model_name in self.embedding_models:
            self.unload_model(model_name)
        
        return self.initialize_embedding(model_name)
    
    def get_available_models(self) -> List[str]:
        """よく使用されるSentenceTransformersモデル一覧を取得"""
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "all-distilroberta-v1",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            "sentence-transformers/msmarco-distilbert-base-v4",
            "sentence-transformers/all-roberta-large-v1",
            "sentence-transformers/stsb-roberta-large"
        ]
    
    def get_loaded_models(self) -> List[str]:
        """現在ロード済みのモデル一覧を取得"""
        return list(self.embedding_models.keys())
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """モデル情報を取得"""
        if model_name:
            if model_name not in self.embedding_models:
                return {"error": f"モデル '{model_name}' は初期化されていません"}
            
            return {
                "model_name": model_name,
                "embedding_dimension": self.get_embedding_dimension(model_name),
                "device": self.device,
                "is_loaded": True
            }
        else:
            # 全モデル情報
            info = {
                "device": self.device,
                "loaded_models": {},
                "available_models": self.get_available_models()
            }
            
            for name in self.get_loaded_models():
                info["loaded_models"][name] = {
                    "embedding_dimension": self.get_embedding_dimension(name)
                }
            
            return info
    
    def unload_model(self, model_name: str) -> bool:
        """指定したモデルをメモリから解放"""
        if model_name in self.embedding_models:
            del self.embedding_models[model_name]
            if model_name in self.embedding_dimensions:
                del self.embedding_dimensions[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"モデル '{model_name}' を解放しました")
            return True
        return False
    
    def unload_all_models(self) -> None:
        """すべてのモデルをメモリから解放"""
        self.embedding_models.clear()
        self.embedding_dimensions.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("すべてのモデルを解放しました")
    
    def check_device(self) -> Dict[str, Any]:
        """デバイス情報を確認"""
        info = {
            "current_device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
            info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
        
        return info
    
    def similarity(self, text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
        """2つのテキスト間の類似度を計算"""
        embeddings = self.encode_texts([text1, text2], model_name=model_name)
        # コサイン類似度を計算
        embedding1, embedding2 = embeddings[0], embeddings[1]
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    
    def similarities(self, query: str, texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[float]:
        """クエリと複数テキスト間の類似度を計算"""
        all_texts = [query] + texts
        embeddings = self.encode_texts(all_texts, model_name=model_name)
        
        query_embedding = embeddings[0]
        text_embeddings = embeddings[1:]
        
        similarities = []
        for text_embedding in text_embeddings:
            similarity = np.dot(query_embedding, text_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
            )
            similarities.append(float(similarity))
        
        return similarities