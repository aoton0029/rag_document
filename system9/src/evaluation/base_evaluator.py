"""
Evaluation Module
llama_indexのevaluationとRAGASを活用した高度な評価機能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner
)
from llama_index.core import ServiceContext
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.response import Response

# RAGAS import (optional)
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from ..utils import get_logger


class EvaluationMetric(Enum):
    """評価指標"""
    # llama_index metrics
    FAITHFULNESS = "faithfulness"
    RELEVANCY = "relevancy"
    CORRECTNESS = "correctness"
    
    # RAGAS metrics  
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RELEVANCY = "context_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ANSWER_CORRECTNESS = "answer_correctness"
    ANSWER_SIMILARITY = "answer_similarity"
    
    # Custom metrics
    RESPONSE_TIME = "response_time"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ROUGE = "rouge"
    BLEU = "bleu"


class EvaluationLevel(Enum):
    """評価レベル"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    RESEARCH = "research"


@dataclass
class EvaluationConfig:
    """評価設定"""
    metrics: List[EvaluationMetric]
    level: EvaluationLevel = EvaluationLevel.STANDARD
    use_ragas: bool = True
    use_llm_based: bool = True
    batch_size: int = 10
    async_evaluation: bool = True
    custom_config: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """評価結果"""
    query: str
    response: str
    reference_answer: Optional[str]
    retrieved_contexts: List[str]
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    evaluation_config: EvaluationConfig


@dataclass
class BatchEvaluationResult:
    """バッチ評価結果"""
    individual_results: List[EvaluationResult]
    aggregate_scores: Dict[str, float]
    statistics: Dict[str, Dict[str, float]]  # mean, std, min, max for each metric
    evaluation_config: EvaluationConfig
    metadata: Dict[str, Any]


class BaseCustomEvaluator(ABC):
    """カスタム評価器基底クラス"""
    
    def __init__(self, config: EvaluationConfig, llm: Optional[LLM] = None):
        self.config = config
        self.llm = llm
        self.logger = get_logger(f"evaluator_{self.__class__.__name__}")
    
    @abstractmethod
    async def evaluate_single(
        self,
        query: str,
        response: str,
        reference_answer: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """単一評価実行"""
        pass
    
    @abstractmethod
    async def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> BatchEvaluationResult:
        """バッチ評価実行"""
        pass


class LlamaIndexEvaluator(BaseCustomEvaluator):
    """llama_index評価器"""
    
    def __init__(self, config: EvaluationConfig, llm: Optional[LLM] = None):
        super().__init__(config, llm)
        
        # llama_index evaluatorsを初期化
        self.evaluators = {}
        
        if EvaluationMetric.FAITHFULNESS in config.metrics:
            self.evaluators[EvaluationMetric.FAITHFULNESS] = FaithfulnessEvaluator(
                llm=self.llm
            )
        
        if EvaluationMetric.RELEVANCY in config.metrics:
            self.evaluators[EvaluationMetric.RELEVANCY] = RelevancyEvaluator(
                llm=self.llm
            )
        
        if EvaluationMetric.CORRECTNESS in config.metrics:
            self.evaluators[EvaluationMetric.CORRECTNESS] = CorrectnessEvaluator(
                llm=self.llm
            )
    
    async def evaluate_single(
        self,
        query: str,
        response: str,
        reference_answer: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """単一評価実行"""
        
        scores = {}
        
        # QueryBundleとResponseオブジェクト作成
        query_bundle = QueryBundle(query_str=query)
        response_obj = Response(
            response=response,
            source_nodes=[
                NodeWithScore(node=self._create_text_node(context))
                for context in (retrieved_contexts or [])
            ]
        )
        
        # 各評価器で評価
        for metric, evaluator in self.evaluators.items():
            try:
                if metric == EvaluationMetric.FAITHFULNESS:
                    result = await evaluator.aevaluate_response(
                        query=query_bundle,
                        response=response_obj
                    )
                    scores[metric.value] = float(result.score)
                
                elif metric == EvaluationMetric.RELEVANCY:
                    result = await evaluator.aevaluate_response(
                        query=query_bundle,
                        response=response_obj
                    )
                    scores[metric.value] = float(result.score)
                
                elif metric == EvaluationMetric.CORRECTNESS:
                    if reference_answer:
                        result = await evaluator.aevaluate_response(
                            query=query_bundle,
                            response=response_obj,
                            reference=reference_answer
                        )
                        scores[metric.value] = float(result.score)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for {metric}: {e}")
                scores[metric.value] = 0.0
        
        return EvaluationResult(
            query=query,
            response=response,
            reference_answer=reference_answer,
            retrieved_contexts=retrieved_contexts or [],
            scores=scores,
            metadata={"evaluator": "llama_index"},
            evaluation_config=self.config
        )
    
    async def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> BatchEvaluationResult:
        """バッチ評価実行"""
        
        # 個別評価
        individual_tasks = []
        for item in evaluation_data:
            task = self.evaluate_single(
                query=item.get("query", ""),
                response=item.get("response", ""),
                reference_answer=item.get("reference_answer"),
                retrieved_contexts=item.get("retrieved_contexts", [])
            )
            individual_tasks.append(task)
        
        individual_results = await asyncio.gather(*individual_tasks)
        
        # 統計計算
        aggregate_scores, statistics = self._calculate_statistics(individual_results)
        
        return BatchEvaluationResult(
            individual_results=individual_results,
            aggregate_scores=aggregate_scores,
            statistics=statistics,
            evaluation_config=self.config,
            metadata={"evaluator": "llama_index"}
        )
    
    def _create_text_node(self, text: str):
        """TextNodeを作成"""
        from llama_index.core.schema import TextNode
        return TextNode(text=text)
    
    def _calculate_statistics(
        self, 
        results: List[EvaluationResult]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """統計計算"""
        
        # 全スコアを収集
        all_scores = {}
        for result in results:
            for metric, score in result.scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        # 統計計算
        aggregate_scores = {}
        statistics = {}
        
        for metric, scores in all_scores.items():
            if scores:
                aggregate_scores[metric] = np.mean(scores)
                statistics[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "count": len(scores)
                }
        
        return aggregate_scores, statistics


class RAGASEvaluator(BaseCustomEvaluator):
    """RAGAS評価器"""
    
    def __init__(self, config: EvaluationConfig, llm: Optional[LLM] = None):
        super().__init__(config, llm)
        
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is not available. Install with: pip install ragas")
        
        # RAGASメトリクスマッピング
        self.ragas_metrics = []
        
        metric_mapping = {
            EvaluationMetric.ANSWER_RELEVANCY: answer_relevancy,
            EvaluationMetric.FAITHFULNESS: faithfulness,
            EvaluationMetric.CONTEXT_RELEVANCY: context_relevancy,
            EvaluationMetric.CONTEXT_PRECISION: context_precision,
            EvaluationMetric.CONTEXT_RECALL: context_recall,
            EvaluationMetric.ANSWER_CORRECTNESS: answer_correctness,
            EvaluationMetric.ANSWER_SIMILARITY: answer_similarity,
        }
        
        for metric in config.metrics:
            if metric in metric_mapping:
                self.ragas_metrics.append(metric_mapping[metric])
    
    async def evaluate_single(
        self,
        query: str,
        response: str,
        reference_answer: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """単一評価実行"""
        
        # RAGASデータセット作成
        dataset_dict = {
            "question": [query],
            "answer": [response],
            "contexts": [retrieved_contexts or []],
        }
        
        if reference_answer:
            dataset_dict["ground_truth"] = [reference_answer]
        
        try:
            # RAGAS評価実行
            from datasets import Dataset
            dataset = Dataset.from_dict(dataset_dict)
            
            result = evaluate(
                dataset=dataset,
                metrics=self.ragas_metrics
            )
            
            # スコア抽出
            scores = {}
            for metric in self.config.metrics:
                if hasattr(result, metric.value):
                    scores[metric.value] = float(getattr(result, metric.value))
            
        except Exception as e:
            self.logger.error(f"RAGAS evaluation failed: {e}")
            scores = {metric.value: 0.0 for metric in self.config.metrics}
        
        return EvaluationResult(
            query=query,
            response=response,
            reference_answer=reference_answer,
            retrieved_contexts=retrieved_contexts or [],
            scores=scores,
            metadata={"evaluator": "ragas"},
            evaluation_config=self.config
        )
    
    async def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> BatchEvaluationResult:
        """バッチ評価実行"""
        
        # RAGASバッチ評価用データ準備
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for item in evaluation_data:
            questions.append(item.get("query", ""))
            answers.append(item.get("response", ""))
            contexts.append(item.get("retrieved_contexts", []))
            if item.get("reference_answer"):
                ground_truths.append(item.get("reference_answer"))
        
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths and len(ground_truths) == len(questions):
            dataset_dict["ground_truth"] = ground_truths
        
        try:
            # RAGAS バッチ評価
            from datasets import Dataset
            dataset = Dataset.from_dict(dataset_dict)
            
            result = evaluate(
                dataset=dataset,
                metrics=self.ragas_metrics
            )
            
            # 個別結果作成
            individual_results = []
            for i, item in enumerate(evaluation_data):
                scores = {}
                for metric in self.config.metrics:
                    if hasattr(result, metric.value):
                        metric_scores = getattr(result, metric.value)
                        if isinstance(metric_scores, list) and i < len(metric_scores):
                            scores[metric.value] = float(metric_scores[i])
                
                individual_results.append(EvaluationResult(
                    query=item.get("query", ""),
                    response=item.get("response", ""),
                    reference_answer=item.get("reference_answer"),
                    retrieved_contexts=item.get("retrieved_contexts", []),
                    scores=scores,
                    metadata={"evaluator": "ragas"},
                    evaluation_config=self.config
                ))
            
            # 統計計算
            aggregate_scores, statistics = self._calculate_statistics(individual_results)
            
        except Exception as e:
            self.logger.error(f"RAGAS batch evaluation failed: {e}")
            
            # エラー時のフォールバック
            individual_results = []
            for item in evaluation_data:
                scores = {metric.value: 0.0 for metric in self.config.metrics}
                individual_results.append(EvaluationResult(
                    query=item.get("query", ""),
                    response=item.get("response", ""),
                    reference_answer=item.get("reference_answer"),
                    retrieved_contexts=item.get("retrieved_contexts", []),
                    scores=scores,
                    metadata={"evaluator": "ragas", "error": str(e)},
                    evaluation_config=self.config
                ))
            
            aggregate_scores, statistics = self._calculate_statistics(individual_results)
        
        return BatchEvaluationResult(
            individual_results=individual_results,
            aggregate_scores=aggregate_scores,
            statistics=statistics,
            evaluation_config=self.config,
            metadata={"evaluator": "ragas"}
        )
    
    def _calculate_statistics(
        self, 
        results: List[EvaluationResult]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """統計計算（LlamaIndexEvaluatorと同じ実装）"""
        
        all_scores = {}
        for result in results:
            for metric, score in result.scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
        
        aggregate_scores = {}
        statistics = {}
        
        for metric, scores in all_scores.items():
            if scores:
                aggregate_scores[metric] = np.mean(scores)
                statistics[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "count": len(scores)
                }
        
        return aggregate_scores, statistics


class HybridEvaluator(BaseCustomEvaluator):
    """ハイブリッド評価器（llama_index + RAGAS）"""
    
    def __init__(self, config: EvaluationConfig, llm: Optional[LLM] = None):
        super().__init__(config, llm)
        
        # llama_index metrics
        llama_metrics = [
            metric for metric in config.metrics
            if metric in [EvaluationMetric.FAITHFULNESS, EvaluationMetric.RELEVANCY, EvaluationMetric.CORRECTNESS]
        ]
        
        # RAGAS metrics
        ragas_metrics = [
            metric for metric in config.metrics
            if metric in [
                EvaluationMetric.ANSWER_RELEVANCY, EvaluationMetric.CONTEXT_RELEVANCY,
                EvaluationMetric.CONTEXT_PRECISION, EvaluationMetric.CONTEXT_RECALL,
                EvaluationMetric.ANSWER_CORRECTNESS, EvaluationMetric.ANSWER_SIMILARITY
            ]
        ]
        
        # 個別評価器を作成
        self.evaluators = {}
        
        if llama_metrics:
            llama_config = EvaluationConfig(
                metrics=llama_metrics,
                level=config.level,
                use_ragas=False
            )
            self.evaluators["llama_index"] = LlamaIndexEvaluator(llama_config, llm)
        
        if ragas_metrics and RAGAS_AVAILABLE:
            ragas_config = EvaluationConfig(
                metrics=ragas_metrics,
                level=config.level,
                use_llm_based=False
            )
            self.evaluators["ragas"] = RAGASEvaluator(ragas_config, llm)
    
    async def evaluate_single(
        self,
        query: str,
        response: str,
        reference_answer: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """単一評価実行"""
        
        # 各評価器で並列実行
        tasks = []
        for name, evaluator in self.evaluators.items():
            task = evaluator.evaluate_single(
                query=query,
                response=response,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts,
                **kwargs
            )
            tasks.append((name, task))
        
        # 結果を収集
        combined_scores = {}
        combined_metadata = {}
        
        for name, task in tasks:
            try:
                result = await task
                combined_scores.update(result.scores)
                combined_metadata[f"{name}_metadata"] = result.metadata
            except Exception as e:
                self.logger.error(f"Evaluator {name} failed: {e}")
                combined_metadata[f"{name}_error"] = str(e)
        
        return EvaluationResult(
            query=query,
            response=response,
            reference_answer=reference_answer,
            retrieved_contexts=retrieved_contexts or [],
            scores=combined_scores,
            metadata=combined_metadata,
            evaluation_config=self.config
        )
    
    async def evaluate_batch(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> BatchEvaluationResult:
        """バッチ評価実行"""
        
        # 各評価器で並列実行
        tasks = []
        for name, evaluator in self.evaluators.items():
            task = evaluator.evaluate_batch(evaluation_data)
            tasks.append((name, task))
        
        # 結果を結合
        all_individual_results = []
        combined_aggregate_scores = {}
        combined_statistics = {}
        combined_metadata = {}
        
        for name, task in tasks:
            try:
                batch_result = await task
                
                # 個別結果をマージ
                if not all_individual_results:
                    # 最初の評価器の結果をベースに
                    all_individual_results = batch_result.individual_results
                else:
                    # スコアをマージ
                    for i, result in enumerate(batch_result.individual_results):
                        if i < len(all_individual_results):
                            all_individual_results[i].scores.update(result.scores)
                
                # 集計結果をマージ
                combined_aggregate_scores.update(batch_result.aggregate_scores)
                combined_statistics.update(batch_result.statistics)
                combined_metadata[f"{name}_metadata"] = batch_result.metadata
                
            except Exception as e:
                self.logger.error(f"Batch evaluator {name} failed: {e}")
                combined_metadata[f"{name}_error"] = str(e)
        
        return BatchEvaluationResult(
            individual_results=all_individual_results,
            aggregate_scores=combined_aggregate_scores,
            statistics=combined_statistics,
            evaluation_config=self.config,
            metadata=combined_metadata
        )


class EvaluationManager:
    """評価管理"""
    
    def __init__(self):
        self.logger = get_logger("evaluation_manager")
        self._evaluators: Dict[str, BaseCustomEvaluator] = {}
    
    def register_evaluator(self, name: str, evaluator: BaseCustomEvaluator):
        """評価器を登録"""
        self._evaluators[name] = evaluator
        self.logger.info(f"Registered evaluator: {name}")
    
    def get_evaluator(self, name: str) -> Optional[BaseCustomEvaluator]:
        """評価器を取得"""
        return self._evaluators.get(name)
    
    def list_evaluators(self) -> List[str]:
        """登録済み評価器一覧"""
        return list(self._evaluators.keys())
    
    async def evaluate_all(
        self,
        query: str,
        response: str,
        reference_answer: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None
    ) -> Dict[str, EvaluationResult]:
        """全評価器で評価実行"""
        
        tasks = []
        for name, evaluator in self._evaluators.items():
            task = self._safe_evaluate(name, evaluator, query, response, reference_answer, retrieved_contexts)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def _safe_evaluate(
        self,
        name: str,
        evaluator: BaseCustomEvaluator,
        query: str,
        response: str,
        reference_answer: Optional[str],
        retrieved_contexts: Optional[List[str]]
    ) -> Tuple[str, EvaluationResult]:
        """安全な評価実行"""
        
        try:
            result = await evaluator.evaluate_single(
                query=query,
                response=response,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts
            )
            return name, result
        except Exception as e:
            self.logger.error(f"Evaluator {name} failed: {e}")
            
            # エラー用の結果作成
            error_result = EvaluationResult(
                query=query,
                response=response,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts or [],
                scores={},
                metadata={"error": str(e)},
                evaluation_config=evaluator.config
            )
            
            return name, error_result
    
    def export_results_to_dataframe(
        self, 
        results: List[EvaluationResult]
    ) -> pd.DataFrame:
        """結果をDataFrameにエクスポート"""
        
        data = []
        for result in results:
            row = {
                "query": result.query,
                "response": result.response,
                "reference_answer": result.reference_answer,
                "num_contexts": len(result.retrieved_contexts),
            }
            
            # スコアを追加
            row.update(result.scores)
            
            # メタデータを追加
            for key, value in result.metadata.items():
                row[f"meta_{key}"] = value
            
            data.append(row)
        
        return pd.DataFrame(data)


# Utility functions
def create_llama_index_evaluator(
    metrics: List[EvaluationMetric] = None,
    level: EvaluationLevel = EvaluationLevel.STANDARD,
    llm: Optional[LLM] = None,
    **kwargs
) -> LlamaIndexEvaluator:
    """llama_index評価器作成"""
    
    if metrics is None:
        metrics = [
            EvaluationMetric.FAITHFULNESS,
            EvaluationMetric.RELEVANCY,
            EvaluationMetric.CORRECTNESS
        ]
    
    config = EvaluationConfig(
        metrics=metrics,
        level=level,
        use_ragas=False,
        **kwargs
    )
    
    return LlamaIndexEvaluator(config, llm)


def create_ragas_evaluator(
    metrics: List[EvaluationMetric] = None,
    level: EvaluationLevel = EvaluationLevel.STANDARD,
    llm: Optional[LLM] = None,
    **kwargs
) -> RAGASEvaluator:
    """RAGAS評価器作成"""
    
    if metrics is None:
        metrics = [
            EvaluationMetric.ANSWER_RELEVANCY,
            EvaluationMetric.FAITHFULNESS,
            EvaluationMetric.CONTEXT_RELEVANCY
        ]
    
    config = EvaluationConfig(
        metrics=metrics,
        level=level,
        use_llm_based=False,
        **kwargs
    )
    
    return RAGASEvaluator(config, llm)


def create_hybrid_evaluator(
    llama_metrics: List[EvaluationMetric] = None,
    ragas_metrics: List[EvaluationMetric] = None,
    level: EvaluationLevel = EvaluationLevel.COMPREHENSIVE,
    llm: Optional[LLM] = None,
    **kwargs
) -> HybridEvaluator:
    """ハイブリッド評価器作成"""
    
    all_metrics = []
    
    if llama_metrics:
        all_metrics.extend(llama_metrics)
    else:
        all_metrics.extend([
            EvaluationMetric.FAITHFULNESS,
            EvaluationMetric.RELEVANCY,
            EvaluationMetric.CORRECTNESS
        ])
    
    if ragas_metrics:
        all_metrics.extend(ragas_metrics)
    else:
        all_metrics.extend([
            EvaluationMetric.ANSWER_RELEVANCY,
            EvaluationMetric.CONTEXT_RELEVANCY
        ])
    
    config = EvaluationConfig(
        metrics=all_metrics,
        level=level,
        **kwargs
    )
    
    return HybridEvaluator(config, llm)
