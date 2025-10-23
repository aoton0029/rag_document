"""
埋め込み基底クラス
各種埋め込みプロバイダーの共通インターフェース
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

from ..utils import get_logger, performance_monitor


class EmbeddingProvider(Enum):
    """埋め込みプロバイダー列挙"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CUSTOM = "custom"


@dataclass
class EmbeddingResult:
    """埋め込み結果データクラス"""
    embeddings: List[List[float]]
    texts: List[str]
    model_name: str
    dimensions: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初期化後処理"""
        if len(self.embeddings) != len(self.texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        if self.metadata is None:
            self.metadata = {}
        
        # 基本統計を追加
        if self.embeddings:
            self.metadata.update({
                "embedding_count": len(self.embeddings),
                "avg_embedding_norm": float(np.mean([np.linalg.norm(emb) for emb in self.embeddings])),
                "dimensions": len(self.embeddings[0]) if self.embeddings else 0
            })
    
    def get_embedding_matrix(self) -> np.ndarray:
        """埋め込みをNumPy配列として取得"""
        return np.array(self.embeddings)
    
    def get_text_embedding_pairs(self) -> List[Tuple[str, List[float]]]:
        """テキストと埋め込みのペアを取得"""
        return list(zip(self.texts, self.embeddings))
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "embeddings": self.embeddings,
            "texts": self.texts,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingResult":
        """辞書からインスタンス作成"""
        return cls(
            embeddings=data["embeddings"],
            texts=data["texts"],
            model_name=data["model_name"],
            dimensions=data["dimensions"],
            metadata=data.get("metadata", {})
        )


class BaseEmbedder(ABC):
    """埋め込み生成器基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "unknown")
        self.embed_batch_size = config.get("embed_batch_size", 32)
        self.dimensions = config.get("dimensions", 768)
        self.logger = get_logger(f"embedder_{self.__class__.__name__}")
        
        # キャッシュ設定
        self.use_cache = config.get("use_cache", True)
        self.cache = {} if self.use_cache else None
        
        # 並列処理設定
        self.max_workers = config.get("max_workers", 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    @abstractmethod
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """テキストを埋め込みベクトルに変換（実装必須）"""
        pass
    
    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """埋め込み次元数を取得（実装必須）"""
        pass
    
    def encode_texts(self, texts: List[str], 
                    batch_size: Optional[int] = None,
                    show_progress: bool = True) -> EmbeddingResult:
        """テキストのリストを埋め込みベクトルに変換"""
        if not texts:
            return EmbeddingResult([], [], self.model_name, self.dimensions)
        
        batch_size = batch_size or self.embed_batch_size
        
        self.logger.info("Starting text encoding", 
                        text_count=len(texts),
                        batch_size=batch_size)
        
        with performance_monitor(f"encode_texts_{len(texts)}"):
            # キャッシュをチェック
            cached_embeddings = []
            remaining_texts = []
            
            for text in texts:
                if self.use_cache:
                    cached_embedding = self._get_cached_embedding(text)
                    if cached_embedding is not None:
                        cached_embeddings.append((text, cached_embedding))
                        continue
                
                remaining_texts.append(text)
            
            # 未キャッシュのテキストを処理
            new_embeddings = []
            if remaining_texts:
                new_embeddings = self._encode_batch_texts(remaining_texts, batch_size)
                
                # 新しい埋め込みをキャッシュ
                if self.use_cache:
                    for text, embedding in zip(remaining_texts, new_embeddings):
                        self._cache_embedding(text, embedding)
            
            # 結果をマージ
            all_embeddings = self._merge_embeddings(texts, cached_embeddings, 
                                                  remaining_texts, new_embeddings)
        
        self.logger.info("Text encoding completed", 
                        total_texts=len(texts),
                        cached_count=len(cached_embeddings),
                        new_count=len(remaining_texts))
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            texts=texts,
            model_name=self.model_name,
            dimensions=self.get_embedding_dimensions(),
            metadata={
                "batch_size": batch_size,
                "cached_count": len(cached_embeddings),
                "new_count": len(remaining_texts)
            }
        )
    
    def _encode_batch_texts(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """バッチ処理でテキストを埋め込み"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_embeddings = self._encode_texts(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                self.logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                self.logger.error(f"Failed to encode batch starting at index {i}: {e}")
                # エラー時はゼロベクトルで埋める
                zero_embedding = [0.0] * self.get_embedding_dimensions()
                all_embeddings.extend([zero_embedding] * len(batch_texts))
        
        return all_embeddings
    
    def _merge_embeddings(self, original_texts: List[str], 
                         cached_embeddings: List[Tuple[str, List[float]]],
                         new_texts: List[str], 
                         new_embeddings: List[List[float]]) -> List[List[float]]:
        """キャッシュされた埋め込みと新しい埋め込みをマージ"""
        
        # テキスト -> 埋め込みのマッピングを作成
        embedding_map = dict(cached_embeddings)
        
        if len(new_texts) == len(new_embeddings):
            embedding_map.update(zip(new_texts, new_embeddings))
        
        # 元の順序で埋め込みを配列
        result_embeddings = []
        for text in original_texts:
            if text in embedding_map:
                result_embeddings.append(embedding_map[text])
            else:
                # 見つからない場合はゼロベクトル
                zero_embedding = [0.0] * self.get_embedding_dimensions()
                result_embeddings.append(zero_embedding)
                self.logger.warning(f"No embedding found for text: {text[:50]}...")
        
        return result_embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """テキストのキャッシュキーを生成"""
        # テキストをハッシュ化してキーとする
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{self.model_name}:{text_hash}"
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """キャッシュから埋め込みを取得"""
        if not self.use_cache or self.cache is None:
            return None
        
        cache_key = self._get_cache_key(text)
        return self.cache.get(cache_key)
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """埋め込みをキャッシュに保存"""
        if not self.use_cache or self.cache is None:
            return
        
        cache_key = self._get_cache_key(text)
        self.cache[cache_key] = embedding
    
    def save_cache(self, filepath: str) -> None:
        """キャッシュをファイルに保存"""
        if self.cache:
            with open(filepath, 'wb') as f:
                pickle.dump(self.cache, f)
            self.logger.info(f"Cache saved to {filepath}", cache_size=len(self.cache))
    
    def load_cache(self, filepath: str) -> None:
        """ファイルからキャッシュを読み込み"""
        try:
            with open(filepath, 'rb') as f:
                self.cache = pickle.load(f)
            self.logger.info(f"Cache loaded from {filepath}", cache_size=len(self.cache))
        except Exception as e:
            self.logger.warning(f"Failed to load cache from {filepath}: {e}")
            self.cache = {}
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        if not self.cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "memory_usage": sum(len(str(k)) + len(str(v)) for k, v in self.cache.items())
        }
    
    async def encode_texts_async(self, texts: List[str], 
                               batch_size: Optional[int] = None) -> EmbeddingResult:
        """非同期でテキストを埋め込み"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.encode_texts, 
            texts, 
            batch_size
        )
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class EmbeddingValidator:
    """埋め込み品質検証クラス"""
    
    @staticmethod
    def validate_embeddings(embeddings: List[List[float]], 
                          expected_dimensions: int) -> Dict[str, Any]:
        """埋め込みの妥当性を検証"""
        if not embeddings:
            return {"valid": False, "error": "No embeddings provided"}
        
        results = {
            "valid": True,
            "embedding_count": len(embeddings),
            "expected_dimensions": expected_dimensions,
            "issues": []
        }
        
        # 次元数チェック
        dimension_issues = []
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dimensions:
                dimension_issues.append({
                    "index": i,
                    "expected": expected_dimensions,
                    "actual": len(embedding)
                })
        
        if dimension_issues:
            results["issues"].append({
                "type": "dimension_mismatch",
                "count": len(dimension_issues),
                "examples": dimension_issues[:5]  # 最初の5つの例のみ
            })
        
        # NaN/Inf値チェック
        invalid_values = []
        for i, embedding in enumerate(embeddings):
            if any(not np.isfinite(val) for val in embedding):
                invalid_values.append(i)
        
        if invalid_values:
            results["issues"].append({
                "type": "invalid_values",
                "count": len(invalid_values),
                "indices": invalid_values[:10]  # 最初の10個のインデックス
            })
        
        # ゼロベクトルチェック
        zero_vectors = []
        for i, embedding in enumerate(embeddings):
            if all(val == 0.0 for val in embedding):
                zero_vectors.append(i)
        
        if zero_vectors:
            results["issues"].append({
                "type": "zero_vectors",
                "count": len(zero_vectors),
                "indices": zero_vectors[:10]
            })
        
        # 品質統計
        if embeddings and all(len(emb) == expected_dimensions for emb in embeddings):
            embedding_array = np.array(embeddings)
            results["statistics"] = {
                "mean_norm": float(np.mean(np.linalg.norm(embedding_array, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embedding_array, axis=1))),
                "min_value": float(np.min(embedding_array)),
                "max_value": float(np.max(embedding_array)),
                "mean_value": float(np.mean(embedding_array))
            }
        
        results["valid"] = len(results["issues"]) == 0
        return results