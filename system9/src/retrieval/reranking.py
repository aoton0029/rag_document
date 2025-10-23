"""
再ランキングモジュール
検索結果の再順序付け機能
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math
import re

from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import BaseNodePostprocessor
from llama_index.core.llms import LLM

from .base_retriever import SearchResult, SearchQuery
from ..utils import get_logger


class RerankingStrategy(Enum):
    """再ランキング戦略"""
    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based"
    FEATURE_BASED = "feature_based"
    DIVERSITY = "diversity"
    FUSION = "fusion"


@dataclass
class RerankingResult:
    """再ランキング結果"""
    original_results: List[SearchResult]
    reranked_results: List[SearchResult]
    reranking_scores: List[float]
    metadata: Dict[str, Any]


class BaseReranker(ABC):
    """再ランキング基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"reranker_{self.__class__.__name__}")
        
        self.top_k = config.get("top_k", 10)
        self.score_threshold = config.get("score_threshold", 0.0)
    
    @abstractmethod
    def rerank(
        self, 
        query: Union[str, SearchQuery], 
        search_results: List[SearchResult]
    ) -> RerankingResult:
        """検索結果を再ランキング（実装必須）"""
        pass
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """スコアを正規化"""
        
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized
    
    def _apply_threshold(
        self, 
        results: List[SearchResult], 
        scores: List[float]
    ) -> tuple[List[SearchResult], List[float]]:
        """スコア閾値を適用"""
        
        filtered_results = []
        filtered_scores = []
        
        for result, score in zip(results, scores):
            if score >= self.score_threshold:
                filtered_results.append(result)
                filtered_scores.append(score)
        
        return filtered_results, filtered_scores


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder再ランキング"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.batch_size = config.get("batch_size", 32)
        
        # Cross-Encoderモデル
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            self.logger.error(f"Failed to load CrossEncoder model: {e}")
    
    def rerank(
        self, 
        query: Union[str, SearchQuery], 
        search_results: List[SearchResult]
    ) -> RerankingResult:
        """Cross-Encoderによる再ランキング"""
        
        if isinstance(query, SearchQuery):
            query_text = query.text
        else:
            query_text = query
        
        if not self.model:
            # モデルが利用できない場合は元の順序を維持
            return RerankingResult(
                original_results=search_results,
                reranked_results=search_results,
                reranking_scores=[result.score for result in search_results],
                metadata={"error": "CrossEncoder model not available"}
            )
        
        # クエリとドキュメントのペア作成
        pairs = []
        for result in search_results:
            document_text = result.node.get_content()[:512]  # テキスト長制限
            pairs.append([query_text, document_text])
        
        # 再ランキングスコア計算
        try:
            scores = self.model.predict(pairs).tolist()
        except Exception as e:
            self.logger.error(f"CrossEncoder prediction failed: {e}")
            scores = [result.score for result in search_results]
        
        # 正規化
        normalized_scores = self._normalize_scores(scores)
        
        # スコアで並び替え
        scored_results = list(zip(search_results, normalized_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 上位k件とスコア閾値適用
        top_results = scored_results[:self.top_k]
        reranked_results = [result for result, score in top_results]
        reranking_scores = [score for result, score in top_results]
        
        # 閾値適用
        filtered_results, filtered_scores = self._apply_threshold(reranked_results, reranking_scores)
        
        return RerankingResult(
            original_results=search_results,
            reranked_results=filtered_results,
            reranking_scores=filtered_scores,
            metadata={
                "model": self.model_name,
                "original_count": len(search_results),
                "reranked_count": len(filtered_results)
            }
        )


class LLMReranker(BaseReranker):
    """LLMベース再ランキング"""
    
    def __init__(self, config: Dict[str, Any], llm: Optional[LLM] = None):
        super().__init__(config)
        
        self.llm = llm
        self.max_candidates = config.get("max_candidates", 20)  # LLM処理する最大候補数
        self.prompt_template = config.get("prompt_template", self._get_default_prompt())
    
    def _get_default_prompt(self) -> str:
        """デフォルトプロンプトテンプレート"""
        return """
以下のクエリに対して、候補文書を関連性の高い順に並び替えてください。

クエリ: {query}

候補文書:
{candidates}

指示:
1. 各文書をクエリとの関連性で評価してください
2. 最も関連性の高いものから順番に番号を付けてください
3. 出力形式: 1, 2, 3, ... （番号のみ）

ランキング:
"""
    
    def rerank(
        self, 
        query: Union[str, SearchQuery], 
        search_results: List[SearchResult]
    ) -> RerankingResult:
        """LLMによる再ランキング"""
        
        if isinstance(query, SearchQuery):
            query_text = query.text
        else:
            query_text = query
        
        if not self.llm:
            # LLMが利用できない場合は元の順序を維持
            return RerankingResult(
                original_results=search_results,
                reranked_results=search_results,
                reranking_scores=[result.score for result in search_results],
                metadata={"error": "LLM not available"}
            )
        
        # 候補数を制限
        candidates = search_results[:self.max_candidates]
        
        # プロンプト作成
        candidates_text = ""
        for i, result in enumerate(candidates, 1):
            content = result.node.get_content()[:200]  # 文字数制限
            candidates_text += f"{i}. {content}\n\n"
        
        prompt = self.prompt_template.format(
            query=query_text,
            candidates=candidates_text
        )
        
        try:
            # LLM推論
            response = self.llm.complete(prompt)
            ranking = self._parse_ranking(response.text, len(candidates))
            
            # 再順序付け
            reranked_results = [candidates[i-1] for i in ranking if 1 <= i <= len(candidates)]
            
            # スコア計算（順位の逆数）
            reranking_scores = [1.0 / (i + 1) for i in range(len(reranked_results))]
            
            # 元の候補にない結果も追加
            remaining_results = [r for r in search_results if r not in reranked_results]
            reranked_results.extend(remaining_results)
            reranking_scores.extend([0.1 / (i + 1) for i in range(len(remaining_results))])
            
        except Exception as e:
            self.logger.error(f"LLM reranking failed: {e}")
            reranked_results = search_results
            reranking_scores = [result.score for result in search_results]
        
        # 上位k件
        top_k_results = reranked_results[:self.top_k]
        top_k_scores = reranking_scores[:self.top_k]
        
        # 閾値適用
        filtered_results, filtered_scores = self._apply_threshold(top_k_results, top_k_scores)
        
        return RerankingResult(
            original_results=search_results,
            reranked_results=filtered_results,
            reranking_scores=filtered_scores,
            metadata={
                "llm_model": str(self.llm),
                "original_count": len(search_results),
                "reranked_count": len(filtered_results)
            }
        )
    
    def _parse_ranking(self, response: str, max_candidates: int) -> List[int]:
        """LLM応答からランキングを解析"""
        
        # 数字のパターンを抽出
        numbers = re.findall(r'\d+', response)
        
        ranking = []
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= max_candidates and num not in ranking:
                    ranking.append(num)
            except ValueError:
                continue
        
        # 不足分を補完
        for i in range(1, max_candidates + 1):
            if i not in ranking:
                ranking.append(i)
        
        return ranking[:max_candidates]


class FeatureBasedReranker(BaseReranker):
    """特徴量ベース再ランキング"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 特徴量の重み
        self.feature_weights = config.get("feature_weights", {
            "semantic_similarity": 0.4,
            "lexical_overlap": 0.2,
            "length_score": 0.1,
            "position_score": 0.1,
            "freshness_score": 0.1,
            "quality_score": 0.1
        })
    
    def rerank(
        self, 
        query: Union[str, SearchQuery], 
        search_results: List[SearchResult]
    ) -> RerankingResult:
        """特徴量ベース再ランキング"""
        
        if isinstance(query, SearchQuery):
            query_text = query.text
        else:
            query_text = query
        
        # 各結果の特徴量スコアを計算
        scored_results = []
        
        for i, result in enumerate(search_results):
            features = self._extract_features(query_text, result, i, len(search_results))
            
            # 加重スコア計算
            total_score = sum(
                features.get(feature, 0.0) * weight 
                for feature, weight in self.feature_weights.items()
            )
            
            scored_results.append((result, total_score, features))
        
        # スコアで並び替え
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 上位k件
        top_results = scored_results[:self.top_k]
        reranked_results = [result for result, score, features in top_results]
        reranking_scores = [score for result, score, features in top_results]
        
        # 閾値適用
        filtered_results, filtered_scores = self._apply_threshold(reranked_results, reranking_scores)
        
        return RerankingResult(
            original_results=search_results,
            reranked_results=filtered_results,
            reranking_scores=filtered_scores,
            metadata={
                "feature_weights": self.feature_weights,
                "original_count": len(search_results),
                "reranked_count": len(filtered_results)
            }
        )
    
    def _extract_features(
        self, 
        query: str, 
        result: SearchResult, 
        position: int, 
        total_count: int
    ) -> Dict[str, float]:
        """特徴量を抽出"""
        
        content = result.node.get_content()
        
        features = {
            "semantic_similarity": result.score,  # 元のスコア
            "lexical_overlap": self._calculate_lexical_overlap(query, content),
            "length_score": self._calculate_length_score(content),
            "position_score": self._calculate_position_score(position, total_count),
            "freshness_score": self._calculate_freshness_score(result),
            "quality_score": self._calculate_quality_score(content)
        }
        
        return features
    
    def _calculate_lexical_overlap(self, query: str, content: str) -> float:
        """語彙重複度を計算"""
        
        query_words = set(re.findall(r'\w+', query.lower()))
        content_words = set(re.findall(r'\w+', content.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    def _calculate_length_score(self, content: str) -> float:
        """長さスコアを計算"""
        
        length = len(content)
        
        # 適度な長さ（200-2000文字）を最高スコアに
        optimal_min, optimal_max = 200, 2000
        
        if optimal_min <= length <= optimal_max:
            return 1.0
        elif length < optimal_min:
            return length / optimal_min
        else:
            return max(0.1, optimal_max / length)
    
    def _calculate_position_score(self, position: int, total_count: int) -> float:
        """位置スコアを計算（元の順位も考慮）"""
        
        # 元の順位が高いほど高スコア（指数減衰）
        return math.exp(-0.1 * position)
    
    def _calculate_freshness_score(self, result: SearchResult) -> float:
        """新しさスコアを計算"""
        
        # メタデータから日付情報を取得
        metadata = result.node.metadata or {}
        
        # 簡易的な実装：メタデータに基づく
        if "created_at" in metadata or "updated_at" in metadata:
            # 実際の実装では日付解析が必要
            return 0.8
        
        return 0.5  # デフォルト
    
    def _calculate_quality_score(self, content: str) -> float:
        """品質スコアを計算"""
        
        # 簡易的な品質指標
        score = 0.0
        
        # 文の数
        sentences = len(re.findall(r'[。！？]', content))
        if sentences > 0:
            score += 0.3
        
        # 数値・統計の存在
        if re.search(r'\d+%|\d+年|\d+人|\d+件', content):
            score += 0.3
        
        # 構造化された情報（箇条書きなど）
        if re.search(r'[・\-\*]\s', content):
            score += 0.2
        
        # 引用・参考文献
        if re.search(r'参考|引用|出典|文献', content):
            score += 0.2
        
        return min(1.0, score)


class DiversityReranker(BaseReranker):
    """多様性再ランキング"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.diversity_threshold = config.get("diversity_threshold", 0.8)
        self.diversity_weight = config.get("diversity_weight", 0.5)
    
    def rerank(
        self, 
        query: Union[str, SearchQuery], 
        search_results: List[SearchResult]
    ) -> RerankingResult:
        """多様性を考慮した再ランキング"""
        
        if len(search_results) <= 1:
            return RerankingResult(
                original_results=search_results,
                reranked_results=search_results,
                reranking_scores=[result.score for result in search_results],
                metadata={}
            )
        
        # MMR (Maximal Marginal Relevance) アルゴリズム
        selected = []
        remaining = search_results.copy()
        
        # 最初は最高スコアを選択
        best_idx = 0
        selected.append(remaining.pop(best_idx))
        
        # 残りを多様性を考慮して選択
        while remaining and len(selected) < self.top_k:
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # 関連性スコア
                relevance_score = candidate.score
                
                # 既選択との最大類似度
                max_similarity = max(
                    self._calculate_similarity(candidate, selected_item)
                    for selected_item in selected
                )
                
                # MMRスコア
                mmr_score = (
                    self.diversity_weight * relevance_score - 
                    (1 - self.diversity_weight) * max_similarity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        # スコア計算
        reranking_scores = [1.0 / (i + 1) for i in range(len(selected))]
        
        return RerankingResult(
            original_results=search_results,
            reranked_results=selected,
            reranking_scores=reranking_scores,
            metadata={
                "diversity_threshold": self.diversity_threshold,
                "diversity_weight": self.diversity_weight,
                "original_count": len(search_results),
                "reranked_count": len(selected)
            }
        )
    
    def _calculate_similarity(self, result1: SearchResult, result2: SearchResult) -> float:
        """2つの結果の類似度を計算"""
        
        content1 = result1.node.get_content()
        content2 = result2.node.get_content()
        
        # 簡易的なJaccard係数
        words1 = set(re.findall(r'\w+', content1.lower()))
        words2 = set(re.findall(r'\w+', content2.lower()))
        
        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class EnsembleReranker(BaseReranker):
    """アンサンブル再ランキング"""
    
    def __init__(self, config: Dict[str, Any], llm: Optional[LLM] = None):
        super().__init__(config)
        
        # 各再ランカーの重み
        self.reranker_weights = config.get("reranker_weights", {
            "cross_encoder": 0.4,
            "feature_based": 0.3,
            "diversity": 0.2,
            "llm_based": 0.1
        })
        
        # 各再ランカーを初期化
        self.rerankers = {}
        
        if config.get("use_cross_encoder", True):
            self.rerankers["cross_encoder"] = CrossEncoderReranker(
                config.get("cross_encoder_config", {})
            )
        
        if config.get("use_feature_based", True):
            self.rerankers["feature_based"] = FeatureBasedReranker(
                config.get("feature_based_config", {})
            )
        
        if config.get("use_diversity", True):
            self.rerankers["diversity"] = DiversityReranker(
                config.get("diversity_config", {})
            )
        
        if config.get("use_llm_based", False) and llm:
            self.rerankers["llm_based"] = LLMReranker(
                config.get("llm_based_config", {}), llm
            )
    
    def rerank(
        self, 
        query: Union[str, SearchQuery], 
        search_results: List[SearchResult]
    ) -> RerankingResult:
        """アンサンブル再ランキング"""
        
        # 各再ランカーの結果を取得
        reranker_results = {}
        
        for name, reranker in self.rerankers.items():
            try:
                result = reranker.rerank(query, search_results)
                reranker_results[name] = result
            except Exception as e:
                self.logger.error(f"Reranker {name} failed: {e}")
        
        # スコアをアンサンブル
        final_scores = {}  # result_id -> score
        
        for result in search_results:
            result_id = id(result.node)  # ノードのIDを使用
            final_scores[result_id] = 0.0
        
        for name, reranker_result in reranker_results.items():
            weight = self.reranker_weights.get(name, 0.0)
            
            for result, score in zip(reranker_result.reranked_results, reranker_result.reranking_scores):
                result_id = id(result.node)
                if result_id in final_scores:
                    final_scores[result_id] += weight * score
        
        # スコアで並び替え
        scored_results = [(result, final_scores[id(result.node)]) for result in search_results]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 上位k件
        top_results = scored_results[:self.top_k]
        reranked_results = [result for result, score in top_results]
        reranking_scores = [score for result, score in top_results]
        
        # 閾値適用
        filtered_results, filtered_scores = self._apply_threshold(reranked_results, reranking_scores)
        
        return RerankingResult(
            original_results=search_results,
            reranked_results=filtered_results,
            reranking_scores=filtered_scores,
            metadata={
                "reranker_weights": self.reranker_weights,
                "reranker_results": {name: len(result.reranked_results) for name, result in reranker_results.items()},
                "original_count": len(search_results),
                "final_count": len(filtered_results)
            }
        )


# llama_index統合用のPostprocessor
class LlamaIndexRerankerPostprocessor(BaseNodePostprocessor):
    """llama_index用再ランキングポストプロセッサー"""
    
    def __init__(self, reranker: BaseReranker, query: Optional[str] = None):
        super().__init__()
        self.reranker = reranker
        self.query = query
    
    def _postprocess_nodes(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: Optional[Any] = None
    ) -> List[NodeWithScore]:
        """ノードをポストプロセス"""
        
        # SearchResultに変換
        search_results = [
            SearchResult(node=node.node, score=node.score or 0.0, metadata={})
            for node in nodes
        ]
        
        # クエリテキスト取得
        query_text = self.query
        if query_bundle and hasattr(query_bundle, 'query_str'):
            query_text = query_bundle.query_str
        
        if not query_text:
            return nodes
        
        # 再ランキング実行
        reranking_result = self.reranker.rerank(query_text, search_results)
        
        # NodeWithScoreに変換して返す
        reranked_nodes = [
            NodeWithScore(node=result.node, score=score)
            for result, score in zip(
                reranking_result.reranked_results,
                reranking_result.reranking_scores
            )
        ]
        
        return reranked_nodes