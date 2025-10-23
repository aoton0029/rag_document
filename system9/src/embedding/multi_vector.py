"""
マルチベクター埋め込み実装
複数の埋め込み手法を組み合わせたAdvanced RAG手法
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_embedder import BaseEmbedder, EmbeddingResult
from .providers import (
    OllamaEmbedder, SentenceTransformerEmbedder, 
    HuggingFaceEmbedder, OpenAIEmbedder, CustomEmbedder
)
from ..utils import get_logger, performance_monitor


class MultiVectorEmbedder:
    """マルチベクター埋め込み生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("multi_vector_embedder")
        
        # 複数の埋め込み生成器を設定
        self.embedders = {}
        self.embedder_weights = {}
        
        embedder_configs = config.get("embedders", {})
        for name, embedder_config in embedder_configs.items():
            self.embedders[name] = self._create_embedder(embedder_config)
            self.embedder_weights[name] = embedder_config.get("weight", 1.0)
        
        if not self.embedders:
            raise ValueError("At least one embedder must be configured")
        
        # 融合戦略設定
        self.fusion_strategy = config.get("fusion_strategy", "weighted_average")
        self.dimensionality_reduction = config.get("dimensionality_reduction", {})
        
        # 非同期処理設定
        self.parallel_encoding = config.get("parallel_encoding", True)
        self.max_workers = config.get("max_workers", len(self.embedders))
        
        self.logger.info("MultiVectorEmbedder initialized",
                        embedders=list(self.embedders.keys()),
                        fusion_strategy=self.fusion_strategy)
    
    def _create_embedder(self, config: Dict[str, Any]) -> BaseEmbedder:
        """設定に基づいて埋め込み生成器を作成"""
        provider = config.get("provider", "sentence_transformers")
        
        embedder_map = {
            "ollama": OllamaEmbedder,
            "sentence_transformers": SentenceTransformerEmbedder,
            "huggingface": HuggingFaceEmbedder,
            "openai": OpenAIEmbedder,
            "custom": CustomEmbedder
        }
        
        if provider not in embedder_map:
            raise ValueError(f"Unknown embedder provider: {provider}")
        
        return embedder_map[provider](config)
    
    def encode_texts(self, texts: List[str]) -> EmbeddingResult:
        """マルチベクター埋め込みを生成"""
        self.logger.info("Starting multi-vector encoding", 
                        text_count=len(texts),
                        embedder_count=len(self.embedders))
        
        with performance_monitor(f"multi_vector_encode_{len(texts)}"):
            # 各埋め込み生成器で埋め込みを生成
            if self.parallel_encoding:
                embedder_results = self._encode_parallel(texts)
            else:
                embedder_results = self._encode_sequential(texts)
            
            # 埋め込みを融合
            fused_embeddings = self._fuse_embeddings(embedder_results, texts)
            
            # 次元削減（オプション）
            if self.dimensionality_reduction.get("enabled", False):
                fused_embeddings = self._apply_dimensionality_reduction(fused_embeddings)
        
        # 結果のメタデータ
        metadata = {
            "embedders_used": list(self.embedders.keys()),
            "fusion_strategy": self.fusion_strategy,
            "embedder_weights": self.embedder_weights,
            "parallel_encoding": self.parallel_encoding
        }
        
        # 個別の埋め込み結果も保存
        for name, result in embedder_results.items():
            metadata[f"{name}_dimensions"] = result.dimensions
            metadata[f"{name}_success"] = len(result.embeddings) == len(texts)
        
        # 最終次元数を計算
        final_dimensions = len(fused_embeddings[0]) if fused_embeddings else 0
        
        self.logger.info("Multi-vector encoding completed",
                        final_dimensions=final_dimensions,
                        fusion_strategy=self.fusion_strategy)
        
        return EmbeddingResult(
            embeddings=fused_embeddings,
            texts=texts,
            model_name="multi_vector_ensemble",
            dimensions=final_dimensions,
            metadata=metadata
        )
    
    def _encode_parallel(self, texts: List[str]) -> Dict[str, EmbeddingResult]:
        """並列で埋め込みを生成"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # タスクを投入
            future_to_name = {
                executor.submit(embedder.encode_texts, texts): name
                for name, embedder in self.embedders.items()
            }
            
            # 結果を収集
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                    self.logger.debug(f"Embedder {name} completed successfully")
                except Exception as e:
                    self.logger.error(f"Embedder {name} failed: {e}")
                    # 失敗した場合はゼロベクトルで埋める
                    dimensions = self.embedders[name].get_embedding_dimensions()
                    zero_embeddings = [[0.0] * dimensions for _ in texts]
                    results[name] = EmbeddingResult(
                        embeddings=zero_embeddings,
                        texts=texts,
                        model_name=f"{name}_failed",
                        dimensions=dimensions,
                        metadata={"error": str(e)}
                    )
        
        return results
    
    def _encode_sequential(self, texts: List[str]) -> Dict[str, EmbeddingResult]:
        """順次で埋め込みを生成"""
        results = {}
        
        for name, embedder in self.embedders.items():
            try:
                result = embedder.encode_texts(texts)
                results[name] = result
                self.logger.debug(f"Embedder {name} completed successfully")
            except Exception as e:
                self.logger.error(f"Embedder {name} failed: {e}")
                # 失敗した場合はゼロベクトルで埋める
                dimensions = embedder.get_embedding_dimensions()
                zero_embeddings = [[0.0] * dimensions for _ in texts]
                results[name] = EmbeddingResult(
                    embeddings=zero_embeddings,
                    texts=texts,
                    model_name=f"{name}_failed",
                    dimensions=dimensions,
                    metadata={"error": str(e)}
                )
        
        return results
    
    def _fuse_embeddings(self, embedder_results: Dict[str, EmbeddingResult], 
                        texts: List[str]) -> List[List[float]]:
        """複数の埋め込み結果を融合"""
        
        if self.fusion_strategy == "concatenation":
            return self._fuse_by_concatenation(embedder_results)
        
        elif self.fusion_strategy == "weighted_average":
            return self._fuse_by_weighted_average(embedder_results)
        
        elif self.fusion_strategy == "learned_fusion":
            return self._fuse_by_learned_fusion(embedder_results)
        
        elif self.fusion_strategy == "attention_fusion":
            return self._fuse_by_attention(embedder_results)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _fuse_by_concatenation(self, embedder_results: Dict[str, EmbeddingResult]) -> List[List[float]]:
        """連結による融合"""
        if not embedder_results:
            return []
        
        # 全ての埋め込みを連結
        fused_embeddings = []
        text_count = len(next(iter(embedder_results.values())).texts)
        
        for i in range(text_count):
            concatenated = []
            for name in sorted(embedder_results.keys()):  # 一貫した順序のため
                result = embedder_results[name]
                if i < len(result.embeddings):
                    concatenated.extend(result.embeddings[i])
                else:
                    # 不足分はゼロで埋める
                    concatenated.extend([0.0] * result.dimensions)
            
            fused_embeddings.append(concatenated)
        
        return fused_embeddings
    
    def _fuse_by_weighted_average(self, embedder_results: Dict[str, EmbeddingResult]) -> List[List[float]]:
        """重み付き平均による融合"""
        if not embedder_results:
            return []
        
        # 全ての埋め込みを同じ次元にリサイズ
        normalized_results = self._normalize_embedding_dimensions(embedder_results)
        
        fused_embeddings = []
        text_count = len(next(iter(normalized_results.values())).embeddings)
        
        for i in range(text_count):
            weighted_sum = None
            total_weight = 0
            
            for name, result in normalized_results.items():
                if i < len(result.embeddings):
                    embedding = np.array(result.embeddings[i])
                    weight = self.embedder_weights.get(name, 1.0)
                    
                    if weighted_sum is None:
                        weighted_sum = weight * embedding
                    else:
                        weighted_sum += weight * embedding
                    
                    total_weight += weight
            
            if total_weight > 0:
                fused_embedding = (weighted_sum / total_weight).tolist()
            else:
                # 全てのウェイトが0の場合はゼロベクトル
                fused_embedding = [0.0] * len(weighted_sum)
            
            fused_embeddings.append(fused_embedding)
        
        return fused_embeddings
    
    def _fuse_by_learned_fusion(self, embedder_results: Dict[str, EmbeddingResult]) -> List[List[float]]:
        """学習ベースの融合（簡易実装）"""
        # より高度な実装では、学習可能な重みや変換行列を使用
        # ここでは簡易的にPCAを使用した次元削減と再結合を行う
        
        concatenated = self._fuse_by_concatenation(embedder_results)
        if not concatenated:
            return []
        
        # PCAを使用して重要な特徴を抽出
        target_dim = min(512, len(concatenated[0]))  # 最大512次元に削減
        
        if len(concatenated) < 2 or len(concatenated[0]) <= target_dim:
            return concatenated
        
        try:
            pca = PCA(n_components=target_dim, random_state=42)
            reduced_embeddings = pca.fit_transform(concatenated)
            return reduced_embeddings.tolist()
        except Exception as e:
            self.logger.warning(f"PCA fusion failed: {e}, falling back to concatenation")
            return concatenated
    
    def _fuse_by_attention(self, embedder_results: Dict[str, EmbeddingResult]) -> List[List[float]]:
        """アテンション機構による融合（簡易実装）"""
        normalized_results = self._normalize_embedding_dimensions(embedder_results)
        
        if not normalized_results:
            return []
        
        fused_embeddings = []
        text_count = len(next(iter(normalized_results.values())).embeddings)
        
        for i in range(text_count):
            embeddings_matrix = []
            embedder_names = []
            
            for name, result in normalized_results.items():
                if i < len(result.embeddings):
                    embeddings_matrix.append(result.embeddings[i])
                    embedder_names.append(name)
            
            if not embeddings_matrix:
                continue
            
            # 簡易アテンション計算
            embeddings_array = np.array(embeddings_matrix)
            
            # 各埋め込みの重要度を計算（簡易版：ノルムベース）
            norms = np.linalg.norm(embeddings_array, axis=1)
            if np.sum(norms) > 0:
                attention_weights = norms / np.sum(norms)
            else:
                attention_weights = np.ones(len(embeddings_array)) / len(embeddings_array)
            
            # アテンション重み付き結合
            fused_embedding = np.average(embeddings_array, axis=0, weights=attention_weights)
            fused_embeddings.append(fused_embedding.tolist())
        
        return fused_embeddings
    
    def _normalize_embedding_dimensions(self, embedder_results: Dict[str, EmbeddingResult]) -> Dict[str, EmbeddingResult]:
        """埋め込みの次元を統一"""
        if not embedder_results:
            return {}
        
        # 最大次元数を取得
        max_dim = max(result.dimensions for result in embedder_results.values())
        
        normalized_results = {}
        
        for name, result in embedder_results.items():
            if result.dimensions == max_dim:
                normalized_results[name] = result
            else:
                # 次元を拡張（ゼロパディング）
                padded_embeddings = []
                for embedding in result.embeddings:
                    padded = embedding + [0.0] * (max_dim - len(embedding))
                    padded_embeddings.append(padded)
                
                normalized_results[name] = EmbeddingResult(
                    embeddings=padded_embeddings,
                    texts=result.texts,
                    model_name=f"{result.model_name}_padded",
                    dimensions=max_dim,
                    metadata={**result.metadata, "original_dimensions": result.dimensions}
                )
        
        return normalized_results
    
    def _apply_dimensionality_reduction(self, embeddings: List[List[float]]) -> List[List[float]]:
        """次元削減を適用"""
        if not embeddings:
            return embeddings
        
        reduction_config = self.dimensionality_reduction
        method = reduction_config.get("method", "pca")
        target_dimensions = reduction_config.get("target_dimensions", 256)
        
        if len(embeddings[0]) <= target_dimensions:
            return embeddings
        
        try:
            if method == "pca":
                pca = PCA(n_components=target_dimensions, random_state=42)
                reduced = pca.fit_transform(embeddings)
                
                # 分散説明率をログ出力
                explained_variance = np.sum(pca.explained_variance_ratio_)
                self.logger.info(f"PCA dimensionality reduction completed",
                               original_dim=len(embeddings[0]),
                               target_dim=target_dimensions,
                               explained_variance_ratio=explained_variance)
                
                return reduced.tolist()
            
            else:
                self.logger.warning(f"Unknown dimensionality reduction method: {method}")
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Dimensionality reduction failed: {e}")
            return embeddings
    
    def get_embedding_similarities(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """埋め込み間の類似度を計算"""
        embedder_results = {}
        
        for name, embedder in self.embedders.items():
            try:
                result = embedder.encode_texts(texts)
                embeddings_matrix = np.array(result.embeddings)
                
                # コサイン類似度計算
                similarities = cosine_similarity(embeddings_matrix)
                embedder_results[name] = similarities
                
            except Exception as e:
                self.logger.error(f"Failed to compute similarities for {name}: {e}")
        
        return embedder_results
    
    async def encode_texts_async(self, texts: List[str]) -> EmbeddingResult:
        """非同期でマルチベクター埋め込みを生成"""
        loop = asyncio.get_event_loop()
        
        # 各埋め込み生成器を非同期で実行
        tasks = []
        for name, embedder in self.embedders.items():
            task = loop.run_in_executor(None, embedder.encode_texts, texts)
            tasks.append((name, task))
        
        # 結果を収集
        embedder_results = {}
        for name, task in tasks:
            try:
                result = await task
                embedder_results[name] = result
            except Exception as e:
                self.logger.error(f"Async embedder {name} failed: {e}")
                # エラー時の処理は同期版と同様
        
        # 融合処理は同期的に実行
        fused_embeddings = self._fuse_embeddings(embedder_results, texts)
        
        if self.dimensionality_reduction.get("enabled", False):
            fused_embeddings = self._apply_dimensionality_reduction(fused_embeddings)
        
        final_dimensions = len(fused_embeddings[0]) if fused_embeddings else 0
        
        return EmbeddingResult(
            embeddings=fused_embeddings,
            texts=texts,
            model_name="multi_vector_ensemble_async",
            dimensions=final_dimensions,
            metadata={
                "embedders_used": list(self.embedders.keys()),
                "fusion_strategy": self.fusion_strategy,
                "async_processing": True
            }
        )