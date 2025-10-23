"""
ハイブリッド検索器
複数の検索手法を組み合わせたAdvanced RAG検索
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from enum import Enum
from sklearn.preprocessing import MinMaxScaler

from .base_retriever import BaseRetriever, SearchQuery, RetrievalResult, SearchMode
from ..utils import get_logger, performance_monitor


class FusionMethod(Enum):
    """融合手法"""
    WEIGHTED_SUM = "weighted_sum"
    RANK_FUSION = "rank_fusion"
    RECIPROCAL_RANK = "reciprocal_rank"
    MAX_SCORE = "max_score"
    LEARNED_FUSION = "learned_fusion"


class HybridRetriever(BaseRetriever):
    """ハイブリッド検索器"""
    
    def __init__(self, config: Dict[str, Any], retrievers: Dict[str, BaseRetriever]):
        super().__init__(config)
        
        self.retrievers = retrievers
        self.fusion_method = FusionMethod(config.get("fusion_method", "weighted_sum"))
        
        # 各検索器の重み
        self.weights = config.get("weights", {})
        self._normalize_weights()
        
        # 正規化設定
        self.normalize_scores = config.get("normalize_scores", True)
        self.scaler = MinMaxScaler() if self.normalize_scores else None
        
        # 閾値設定
        self.min_score_threshold = config.get("min_score_threshold", 0.0)
        self.diversity_threshold = config.get("diversity_threshold", 0.7)
        
        # 多様性促進
        self.enable_diversity = config.get("enable_diversity", False)
        
        self.logger.info("HybridRetriever initialized",
                        retrievers=list(self.retrievers.keys()),
                        fusion_method=self.fusion_method.value,
                        weights=self.weights)
    
    def _normalize_weights(self):
        """重みを正規化"""
        if not self.weights:
            # 均等重み
            num_retrievers = len(self.retrievers)
            self.weights = {name: 1.0/num_retrievers for name in self.retrievers.keys()}
        else:
            # 重みの合計を1.0に正規化
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def _retrieve_internal(self, query: SearchQuery) -> List[Any]:
        """ハイブリッド検索実行"""
        
        # 各検索器で検索実行
        retriever_results = {}
        
        for name, retriever in self.retrievers.items():
            try:
                # 各検索器に適した設定を適用
                adapted_query = self._adapt_query_for_retriever(query, name)
                result = retriever.retrieve(adapted_query)
                retriever_results[name] = result.nodes
                
                self.logger.debug(f"Retrieved {len(result.nodes)} nodes from {name}")
                
            except Exception as e:
                self.logger.error(f"Retrieval failed for {name}: {e}")
                retriever_results[name] = []
        
        # 結果を融合
        fused_nodes = self._fuse_results(retriever_results, query)
        
        # 多様性フィルタリング（オプション）
        if self.enable_diversity:
            fused_nodes = self._apply_diversity_filtering(fused_nodes)
        
        # スコア閾値フィルタリング
        if self.min_score_threshold > 0:
            fused_nodes = [node for node in fused_nodes 
                          if node.score and node.score >= self.min_score_threshold]
        
        return fused_nodes
    
    def _adapt_query_for_retriever(self, query: SearchQuery, retriever_name: str) -> SearchQuery:
        """検索器に応じてクエリを適応"""
        
        # 検索器固有の設定
        retriever_config = self.config.get("retriever_configs", {}).get(retriever_name, {})
        
        adapted_query = SearchQuery(
            text=query.text,
            filters=query.filters,
            similarity_top_k=retriever_config.get("top_k", query.similarity_top_k * 2),  # 多めに取得
            mode=SearchMode(retriever_config.get("mode", query.mode.value)),
            metadata={**query.metadata, "retriever": retriever_name}
        )
        
        return adapted_query
    
    def _fuse_results(self, retriever_results: Dict[str, List[Any]], 
                     query: SearchQuery) -> List[Any]:
        """検索結果を融合"""
        
        if self.fusion_method == FusionMethod.WEIGHTED_SUM:
            return self._weighted_sum_fusion(retriever_results)
        elif self.fusion_method == FusionMethod.RANK_FUSION:
            return self._rank_fusion(retriever_results)
        elif self.fusion_method == FusionMethod.RECIPROCAL_RANK:
            return self._reciprocal_rank_fusion(retriever_results)
        elif self.fusion_method == FusionMethod.MAX_SCORE:
            return self._max_score_fusion(retriever_results)
        elif self.fusion_method == FusionMethod.LEARNED_FUSION:
            return self._learned_fusion(retriever_results, query)
        else:
            return self._weighted_sum_fusion(retriever_results)
    
    def _weighted_sum_fusion(self, retriever_results: Dict[str, List[Any]]) -> List[Any]:
        """重み付き合計融合"""
        
        # 全ノードを収集
        all_nodes = {}  # node_id -> (node, retriever_scores)
        
        for retriever_name, nodes in retriever_results.items():
            weight = self.weights.get(retriever_name, 0)
            
            for i, node in enumerate(nodes):
                node_id = self._get_node_id(node)
                
                if node_id not in all_nodes:
                    all_nodes[node_id] = {"node": node, "scores": {}, "ranks": {}}
                
                # スコア正規化
                score = node.score if node.score else 0
                all_nodes[node_id]["scores"][retriever_name] = score
                all_nodes[node_id]["ranks"][retriever_name] = i + 1
        
        # スコア正規化（検索器ごと）
        if self.normalize_scores:
            all_nodes = self._normalize_scores_across_retrievers(all_nodes)
        
        # 重み付き合計計算
        fused_nodes = []
        for node_data in all_nodes.values():
            weighted_score = 0
            total_weight = 0
            
            for retriever_name, score in node_data["scores"].items():
                weight = self.weights.get(retriever_name, 0)
                weighted_score += weight * score
                total_weight += weight
            
            # 正規化
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0
            
            # NodeWithScoreオブジェクトを作成
            node_with_score = node_data["node"]
            node_with_score.score = final_score
            
            # メタデータに融合情報を追加
            if hasattr(node_with_score.node, 'metadata'):
                node_with_score.node.metadata = node_with_score.node.metadata or {}
                node_with_score.node.metadata.update({
                    "fusion_method": "weighted_sum",
                    "retriever_scores": node_data["scores"],
                    "final_score": final_score
                })
            
            fused_nodes.append(node_with_score)
        
        # スコアでソート
        fused_nodes.sort(key=lambda x: x.score if x.score else 0, reverse=True)
        
        return fused_nodes
    
    def _rank_fusion(self, retriever_results: Dict[str, List[Any]]) -> List[Any]:
        """ランクベース融合"""
        
        all_nodes = {}
        
        for retriever_name, nodes in retriever_results.items():
            weight = self.weights.get(retriever_name, 0)
            
            for rank, node in enumerate(nodes):
                node_id = self._get_node_id(node)
                
                if node_id not in all_nodes:
                    all_nodes[node_id] = {"node": node, "rank_scores": {}}
                
                # ランクスコア（順位の逆数）
                rank_score = 1.0 / (rank + 1)
                all_nodes[node_id]["rank_scores"][retriever_name] = weight * rank_score
        
        # ランクスコア合計
        fused_nodes = []
        for node_data in all_nodes.values():
            total_rank_score = sum(node_data["rank_scores"].values())
            
            node_with_score = node_data["node"]
            node_with_score.score = total_rank_score
            
            # メタデータ更新
            if hasattr(node_with_score.node, 'metadata'):
                node_with_score.node.metadata = node_with_score.node.metadata or {}
                node_with_score.node.metadata.update({
                    "fusion_method": "rank_fusion",
                    "rank_scores": node_data["rank_scores"],
                    "total_rank_score": total_rank_score
                })
            
            fused_nodes.append(node_with_score)
        
        fused_nodes.sort(key=lambda x: x.score if x.score else 0, reverse=True)
        
        return fused_nodes
    
    def _reciprocal_rank_fusion(self, retriever_results: Dict[str, List[Any]]) -> List[Any]:
        """相互ランク融合（RRF）"""
        
        k = self.config.get("rrf_k", 60)  # RRFパラメータ
        all_nodes = {}
        
        for retriever_name, nodes in retriever_results.items():
            weight = self.weights.get(retriever_name, 0)
            
            for rank, node in enumerate(nodes):
                node_id = self._get_node_id(node)
                
                if node_id not in all_nodes:
                    all_nodes[node_id] = {"node": node, "rrf_scores": {}}
                
                # RRFスコア
                rrf_score = weight / (k + rank + 1)
                all_nodes[node_id]["rrf_scores"][retriever_name] = rrf_score
        
        # RRFスコア合計
        fused_nodes = []
        for node_data in all_nodes.values():
            total_rrf_score = sum(node_data["rrf_scores"].values())
            
            node_with_score = node_data["node"]
            node_with_score.score = total_rrf_score
            
            if hasattr(node_with_score.node, 'metadata'):
                node_with_score.node.metadata = node_with_score.node.metadata or {}
                node_with_score.node.metadata.update({
                    "fusion_method": "reciprocal_rank",
                    "rrf_scores": node_data["rrf_scores"],
                    "total_rrf_score": total_rrf_score
                })
            
            fused_nodes.append(node_with_score)
        
        fused_nodes.sort(key=lambda x: x.score if x.score else 0, reverse=True)
        
        return fused_nodes
    
    def _max_score_fusion(self, retriever_results: Dict[str, List[Any]]) -> List[Any]:
        """最大スコア融合"""
        
        all_nodes = {}
        
        for retriever_name, nodes in retriever_results.items():
            for node in nodes:
                node_id = self._get_node_id(node)
                
                current_score = node.score if node.score else 0
                
                if node_id not in all_nodes:
                    all_nodes[node_id] = {"node": node, "max_score": current_score, "source": retriever_name}
                else:
                    if current_score > all_nodes[node_id]["max_score"]:
                        all_nodes[node_id]["max_score"] = current_score
                        all_nodes[node_id]["source"] = retriever_name
        
        fused_nodes = []
        for node_data in all_nodes.values():
            node_with_score = node_data["node"]
            node_with_score.score = node_data["max_score"]
            
            if hasattr(node_with_score.node, 'metadata'):
                node_with_score.node.metadata = node_with_score.node.metadata or {}
                node_with_score.node.metadata.update({
                    "fusion_method": "max_score",
                    "best_source": node_data["source"],
                    "max_score": node_data["max_score"]
                })
            
            fused_nodes.append(node_with_score)
        
        fused_nodes.sort(key=lambda x: x.score if x.score else 0, reverse=True)
        
        return fused_nodes
    
    def _learned_fusion(self, retriever_results: Dict[str, List[Any]], 
                       query: SearchQuery) -> List[Any]:
        """学習ベース融合（簡易実装）"""
        
        # 簡易的な特徴量ベース融合
        # 実際の実装では機械学習モデルを使用
        
        all_nodes = {}
        
        for retriever_name, nodes in retriever_results.items():
            for rank, node in enumerate(nodes):
                node_id = self._get_node_id(node)
                
                if node_id not in all_nodes:
                    all_nodes[node_id] = {"node": node, "features": {}}
                
                # 特徴量計算
                score = node.score if node.score else 0
                rank_feature = 1.0 / (rank + 1)
                
                all_nodes[node_id]["features"][f"{retriever_name}_score"] = score
                all_nodes[node_id]["features"][f"{retriever_name}_rank"] = rank_feature
        
        # 簡易的な線形結合
        fused_nodes = []
        for node_data in all_nodes.values():
            features = node_data["features"]
            
            # 特徴量の重み付き合計（簡易実装）
            learned_score = 0
            for feature_name, feature_value in features.items():
                if "score" in feature_name:
                    learned_score += 0.7 * feature_value
                elif "rank" in feature_name:
                    learned_score += 0.3 * feature_value
            
            node_with_score = node_data["node"]
            node_with_score.score = learned_score
            
            if hasattr(node_with_score.node, 'metadata'):
                node_with_score.node.metadata = node_with_score.node.metadata or {}
                node_with_score.node.metadata.update({
                    "fusion_method": "learned_fusion",
                    "features": features,
                    "learned_score": learned_score
                })
            
            fused_nodes.append(node_with_score)
        
        fused_nodes.sort(key=lambda x: x.score if x.score else 0, reverse=True)
        
        return fused_nodes
    
    def _normalize_scores_across_retrievers(self, all_nodes: Dict[str, Dict]) -> Dict[str, Dict]:
        """検索器間でスコアを正規化"""
        
        # 検索器ごとのスコア収集
        retriever_scores = {}
        for node_data in all_nodes.values():
            for retriever_name, score in node_data["scores"].items():
                if retriever_name not in retriever_scores:
                    retriever_scores[retriever_name] = []
                retriever_scores[retriever_name].append(score)
        
        # 検索器ごとの正規化
        for retriever_name, scores in retriever_scores.items():
            if scores and len(scores) > 1:
                scores_array = np.array(scores).reshape(-1, 1)
                scaler = MinMaxScaler()
                normalized_scores = scaler.fit_transform(scores_array).flatten()
                
                # 正規化されたスコアを戻す
                score_idx = 0
                for node_data in all_nodes.values():
                    if retriever_name in node_data["scores"]:
                        node_data["scores"][retriever_name] = normalized_scores[score_idx]
                        score_idx += 1
        
        return all_nodes
    
    def _apply_diversity_filtering(self, nodes: List[Any]) -> List[Any]:
        """多様性フィルタリング"""
        
        if not nodes:
            return nodes
        
        diverse_nodes = [nodes[0]]  # 最初のノードは常に含める
        
        for node in nodes[1:]:
            # 既存ノードとの類似度をチェック
            is_diverse = True
            
            for existing_node in diverse_nodes:
                similarity = self._calculate_text_similarity(
                    node.node.get_content(),
                    existing_node.node.get_content()
                )
                
                if similarity > self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_nodes.append(node)
        
        return diverse_nodes
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキスト類似度計算（簡易版）"""
        
        # Jaccard類似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _get_node_id(self, node: Any) -> str:
        """ノードIDを取得"""
        
        if hasattr(node, 'node') and hasattr(node.node, 'node_id'):
            return node.node.node_id
        elif hasattr(node, 'node') and hasattr(node.node, 'id_'):
            return node.node.id_
        else:
            # IDがない場合はテキストハッシュを使用
            text = node.node.get_content() if hasattr(node, 'node') else str(node)
            return str(hash(text[:100]))
    
    def update_weights(self, new_weights: Dict[str, float]):
        """検索器の重みを更新"""
        self.weights.update(new_weights)
        self._normalize_weights()
        
        self.logger.info("Weights updated", weights=self.weights)
    
    def get_retriever_performance(self, queries: List[SearchQuery]) -> Dict[str, Dict[str, float]]:
        """各検索器の性能を評価"""
        
        performance = {}
        
        for query in queries:
            for name, retriever in self.retrievers.items():
                try:
                    result = retriever.retrieve(query)
                    
                    if name not in performance:
                        performance[name] = {"total_time": 0, "total_results": 0, "query_count": 0}
                    
                    performance[name]["total_time"] += result.retrieval_time
                    performance[name]["total_results"] += len(result.nodes)
                    performance[name]["query_count"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Performance evaluation failed for {name}: {e}")
        
        # 平均値計算
        for name, stats in performance.items():
            if stats["query_count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["query_count"]
                stats["avg_results"] = stats["total_results"] / stats["query_count"]
        
        return performance