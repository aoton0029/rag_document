"""
Evaluator Module
RAG評価の実装
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner
)
from llama_index.core.base.response.schema import Response

logger = logging.getLogger(__name__)


class EvaluatorFactory:
    """
    Evaluatorファクトリークラス
    """
    
    @staticmethod
    def create_faithfulness_evaluator(
        llm: Optional[Any] = None,
        raise_error: bool = False
    ) -> FaithfulnessEvaluator:
        """
        FaithfulnessEvaluatorを作成
        
        Args:
            llm: LLMインスタンス
            raise_error: エラーを発生させるか
            
        Returns:
            FaithfulnessEvaluator
        """
        try:
            if llm:
                evaluator = FaithfulnessEvaluator(llm=llm, raise_error=raise_error)
            else:
                evaluator = FaithfulnessEvaluator(raise_error=raise_error)
            
            logger.info("FaithfulnessEvaluatorを作成")
            return evaluator
        except Exception as e:
            logger.error(f"FaithfulnessEvaluator作成エラー: {e}")
            raise
    
    @staticmethod
    def create_relevancy_evaluator(
        llm: Optional[Any] = None,
        raise_error: bool = False
    ) -> RelevancyEvaluator:
        """
        RelevancyEvaluatorを作成
        
        Args:
            llm: LLMインスタンス
            raise_error: エラーを発生させるか
            
        Returns:
            RelevancyEvaluator
        """
        try:
            if llm:
                evaluator = RelevancyEvaluator(llm=llm, raise_error=raise_error)
            else:
                evaluator = RelevancyEvaluator(raise_error=raise_error)
            
            logger.info("RelevancyEvaluatorを作成")
            return evaluator
        except Exception as e:
            logger.error(f"RelevancyEvaluator作成エラー: {e}")
            raise
    
    @staticmethod
    def create_correctness_evaluator(
        llm: Optional[Any] = None,
        raise_error: bool = False
    ) -> CorrectnessEvaluator:
        """
        CorrectnessEvaluatorを作成
        
        Args:
            llm: LLMインスタンス
            raise_error: エラーを発生させるか
            
        Returns:
            CorrectnessEvaluator
        """
        try:
            if llm:
                evaluator = CorrectnessEvaluator(llm=llm, raise_error=raise_error)
            else:
                evaluator = CorrectnessEvaluator(raise_error=raise_error)
            
            logger.info("CorrectnessEvaluatorを作成")
            return evaluator
        except Exception as e:
            logger.error(f"CorrectnessEvaluator作成エラー: {e}")
            raise


class FaithfulnessEvaluatorWrapper:
    """
    FaithfulnessEvaluatorラッパー
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        FaithfulnessEvaluatorWrapperの初期化
        
        Args:
            llm: LLMインスタンス
        """
        self.evaluator = EvaluatorFactory.create_faithfulness_evaluator(llm=llm)
    
    def evaluate(
        self,
        query: str,
        response: str,
        contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        忠実性を評価
        
        Args:
            query: クエリ
            response: レスポンス
            contexts: コンテキストリスト
            
        Returns:
            評価結果
        """
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts
            )
            
            return {
                "passing": result.passing,
                "score": result.score,
                "feedback": result.feedback
            }
        except Exception as e:
            logger.error(f"忠実性評価エラー: {e}")
            return {
                "passing": False,
                "score": 0.0,
                "feedback": f"エラー: {str(e)}"
            }


class RelevancyEvaluatorWrapper:
    """
    RelevancyEvaluatorラッパー
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        RelevancyEvaluatorWrapperの初期化
        
        Args:
            llm: LLMインスタンス
        """
        self.evaluator = EvaluatorFactory.create_relevancy_evaluator(llm=llm)
    
    def evaluate(
        self,
        query: str,
        response: str,
        contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        関連性を評価
        
        Args:
            query: クエリ
            response: レスポンス
            contexts: コンテキストリスト
            
        Returns:
            評価結果
        """
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts
            )
            
            return {
                "passing": result.passing,
                "score": result.score,
                "feedback": result.feedback
            }
        except Exception as e:
            logger.error(f"関連性評価エラー: {e}")
            return {
                "passing": False,
                "score": 0.0,
                "feedback": f"エラー: {str(e)}"
            }


class CorrectnessEvaluatorWrapper:
    """
    CorrectnessEvaluatorラッパー
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        CorrectnessEvaluatorWrapperの初期化
        
        Args:
            llm: LLMインスタンス
        """
        self.evaluator = EvaluatorFactory.create_correctness_evaluator(llm=llm)
    
    def evaluate(
        self,
        query: str,
        response: str,
        reference: str
    ) -> Dict[str, Any]:
        """
        正確性を評価
        
        Args:
            query: クエリ
            response: レスポンス
            reference: 参照回答
            
        Returns:
            評価結果
        """
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                reference=reference
            )
            
            return {
                "passing": result.passing,
                "score": result.score,
                "feedback": result.feedback
            }
        except Exception as e:
            logger.error(f"正確性評価エラー: {e}")
            return {
                "passing": False,
                "score": 0.0,
                "feedback": f"エラー: {str(e)}"
            }


class BatchEvaluationRunner:
    """
    バッチ評価実行クラス
    """
    
    def __init__(
        self,
        evaluators: Dict[str, Any],
        workers: int = 4,
        show_progress: bool = True
    ):
        """
        BatchEvaluationRunnerの初期化
        
        Args:
            evaluators: 評価器の辞書
            workers: ワーカー数
            show_progress: 進捗を表示するか
        """
        self.evaluators = evaluators
        self.workers = workers
        self.show_progress = show_progress
        self._runner = None
    
    def _get_runner(self) -> BatchEvalRunner:
        """BatchEvalRunnerを取得"""
        if self._runner is None:
            self._runner = BatchEvalRunner(
                evaluators=self.evaluators,
                workers=self.workers,
                show_progress=self.show_progress
            )
        return self._runner
    
    def evaluate_queries(
        self,
        queries: List[str],
        responses: List[Response]
    ) -> Dict[str, List[Any]]:
        """
        クエリとレスポンスを評価
        
        Args:
            queries: クエリのリスト
            responses: レスポンスのリスト
            
        Returns:
            評価結果
        """
        try:
            runner = self._get_runner()
            results = runner.evaluate_queries(
                queries=queries,
                responses=responses
            )
            
            logger.info(f"{len(queries)}クエリの評価完了")
            return results
        except Exception as e:
            logger.error(f"バッチ評価エラー: {e}")
            raise
    
    def save_results(
        self,
        results: Dict[str, List[Any]],
        output_path: str
    ):
        """
        評価結果を保存
        
        Args:
            results: 評価結果
            output_path: 出力パス
        """
        try:
            # 結果をJSON形式に変換
            serializable_results = {}
            for eval_name, eval_results in results.items():
                serializable_results[eval_name] = []
                for result in eval_results:
                    serializable_results[eval_name].append({
                        "passing": result.passing if hasattr(result, 'passing') else None,
                        "score": result.score if hasattr(result, 'score') else None,
                        "feedback": result.feedback if hasattr(result, 'feedback') else None
                    })
            
            # ファイルに保存
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"評価結果を保存: {output_path}")
        except Exception as e:
            logger.error(f"評価結果保存エラー: {e}")
            raise


class RAGEvaluator:
    """
    RAG評価統合クラス
    複数の評価指標を統合して評価を実行
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        embed_model: Optional[Any] = None,
        enable_faithfulness: bool = True,
        enable_relevancy: bool = True,
        enable_correctness: bool = False
    ):
        """
        RAGEvaluatorの初期化
        
        Args:
            llm: LLMインスタンス
            embed_model: 埋め込みモデル
            enable_faithfulness: 忠実性評価を有効化
            enable_relevancy: 関連性評価を有効化
            enable_correctness: 正確性評価を有効化
        """
        self.llm = llm
        self.embed_model = embed_model
        
        # 評価器を初期化
        self.evaluators = {}
        
        if enable_faithfulness:
            self.evaluators["faithfulness"] = FaithfulnessEvaluatorWrapper(llm=llm)
            
        if enable_relevancy:
            self.evaluators["relevancy"] = RelevancyEvaluatorWrapper(llm=llm)
            
        if enable_correctness:
            self.evaluators["correctness"] = CorrectnessEvaluatorWrapper(llm=llm)
        
        logger.info(f"RAGEvaluator初期化: {len(self.evaluators)}個の評価器")
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        contexts: Optional[List[str]] = None,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        単一レスポンスを評価
        
        Args:
            query: クエリ
            response: レスポンス
            contexts: コンテキストリスト
            reference: 参照回答（正確性評価用）
            
        Returns:
            評価結果の辞書
        """
        results = {}
        
        # 忠実性評価
        if "faithfulness" in self.evaluators:
            try:
                results["faithfulness"] = self.evaluators["faithfulness"].evaluate(
                    query=query,
                    response=response,
                    contexts=contexts
                )
            except Exception as e:
                logger.error(f"忠実性評価エラー: {e}")
                results["faithfulness"] = {
                    "passing": False,
                    "score": 0.0,
                    "feedback": f"エラー: {str(e)}"
                }
        
        # 関連性評価
        if "relevancy" in self.evaluators:
            try:
                results["relevancy"] = self.evaluators["relevancy"].evaluate(
                    query=query,
                    response=response,
                    contexts=contexts
                )
            except Exception as e:
                logger.error(f"関連性評価エラー: {e}")
                results["relevancy"] = {
                    "passing": False,
                    "score": 0.0,
                    "feedback": f"エラー: {str(e)}"
                }
        
        # 正確性評価
        if "correctness" in self.evaluators and reference:
            try:
                results["correctness"] = self.evaluators["correctness"].evaluate(
                    query=query,
                    response=response,
                    reference=reference
                )
            except Exception as e:
                logger.error(f"正確性評価エラー: {e}")
                results["correctness"] = {
                    "passing": False,
                    "score": 0.0,
                    "feedback": f"エラー: {str(e)}"
                }
        
        return results
    
    def evaluate_query_engine(
        self,
        query_engine: Any,
        queries: List[str],
        references: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        QueryEngineを評価
        
        Args:
            query_engine: QueryEngineインスタンス
            queries: クエリリスト
            references: 参照回答リスト
            **kwargs: 追加パラメータ
            
        Returns:
            評価結果
        """
        logger.info(f"QueryEngine評価開始: {len(queries)}クエリ")
        
        all_results = []
        responses = []
        
        # 各クエリを実行
        for i, query in enumerate(queries):
            try:
                # クエリ実行
                response = query_engine.query(query)
                responses.append(response)
                
                # コンテキスト抽出
                contexts = None
                if hasattr(response, 'source_nodes'):
                    contexts = [
                        node.node.get_content()
                        for node in response.source_nodes
                    ]
                
                # 参照回答取得
                reference = None
                if references and i < len(references):
                    reference = references[i]
                
                # 評価実行
                result = self.evaluate_response(
                    query=query,
                    response=str(response),
                    contexts=contexts,
                    reference=reference
                )
                
                # クエリと結果を保存
                result["query"] = query
                result["response"] = str(response)
                all_results.append(result)
                
                logger.info(f"クエリ {i+1}/{len(queries)} 評価完了")
                
            except Exception as e:
                logger.error(f"クエリ {i+1} 評価エラー: {e}")
                all_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        # 集計統計を計算
        aggregate_stats = self._compute_aggregate_stats(all_results)
        
        return {
            "individual_results": all_results,
            "aggregate_stats": aggregate_stats,
            "num_queries": len(queries),
            "num_successful": len([r for r in all_results if "error" not in r])
        }
    
    def _compute_aggregate_stats(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        集計統計を計算
        
        Args:
            results: 個別評価結果リスト
            
        Returns:
            集計統計
        """
        stats = {}
        
        # エラーを除外
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return stats
        
        # 各評価指標の統計
        for metric in ["faithfulness", "relevancy", "correctness"]:
            metric_scores = []
            metric_passing = []
            
            for result in valid_results:
                if metric in result and isinstance(result[metric], dict):
                    if "score" in result[metric]:
                        metric_scores.append(result[metric]["score"])
                    if "passing" in result[metric]:
                        metric_passing.append(result[metric]["passing"])
            
            if metric_scores:
                stats[metric] = {
                    "mean_score": sum(metric_scores) / len(metric_scores),
                    "min_score": min(metric_scores),
                    "max_score": max(metric_scores),
                    "num_samples": len(metric_scores)
                }
                
                if metric_passing:
                    stats[metric]["passing_rate"] = (
                        sum(metric_passing) / len(metric_passing)
                    )
        
        return stats
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        評価結果を保存
        
        Args:
            results: 評価結果
            output_path: 出力パス
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"評価結果を保存: {output_path}")
            
        except Exception as e:
            logger.error(f"評価結果保存エラー: {e}")
            raise
