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
