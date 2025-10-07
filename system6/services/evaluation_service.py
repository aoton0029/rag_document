from typing import List, Dict, Any, Optional
import pandas as pd
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner
)
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI


class EvaluationService:
    """RAGシステムの評価を行うサービスクラス"""
    
    def __init__(self, service_context: Optional[ServiceContext] = None):
        """
        評価サービスを初期化
        
        Args:
            service_context: llama_indexのサービスコンテキスト
        """
        self.service_context = service_context or ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo")
        )
        
        # 評価器を初期化
        self.faithfulness_evaluator = FaithfulnessEvaluator(
            service_context=self.service_context
        )
        self.relevancy_evaluator = RelevancyEvaluator(
            service_context=self.service_context
        )
        self.correctness_evaluator = CorrectnessEvaluator(
            service_context=self.service_context
        )
        
    def evaluate_response(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        単一の回答を評価
        
        Args:
            query: ユーザーの質問
            response: システムの回答
            contexts: 回答生成に使用されたコンテキスト
            reference_answer: 参考回答（正解）
            
        Returns:
            評価結果の辞書
        """
        results = {}
        
        # Faithfulness評価（回答がコンテキストに忠実かどうか）
        faithfulness_result = self.faithfulness_evaluator.evaluate_response(
            query=query,
            response=response,
            contexts=contexts
        )
        results['faithfulness'] = faithfulness_result.score
        
        # Relevancy評価（回答が質問に関連しているかどうか）
        relevancy_result = self.relevancy_evaluator.evaluate_response(
            query=query,
            response=response,
            contexts=contexts
        )
        results['relevancy'] = relevancy_result.score
        
        # Correctness評価（参考回答がある場合）
        if reference_answer:
            correctness_result = self.correctness_evaluator.evaluate_response(
                query=query,
                response=response,
                contexts=contexts,
                reference=reference_answer
            )
            results['correctness'] = correctness_result.score
        
        return results
    
    def batch_evaluate(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        バッチ評価を実行
        
        Args:
            evaluation_data: 評価データのリスト
                各要素は {query, response, contexts, reference_answer} の辞書
                
        Returns:
            評価結果のDataFrame
        """
        results = []
        
        for i, data in enumerate(evaluation_data):
            try:
                result = self.evaluate_response(
                    query=data['query'],
                    response=data['response'],
                    contexts=data['contexts'],
                    reference_answer=data.get('reference_answer')
                )
                result['query_id'] = i
                result['query'] = data['query']
                results.append(result)
            except Exception as e:
                print(f"評価エラー (query_id: {i}): {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        評価結果を分析
        
        Args:
            results_df: 評価結果のDataFrame
            
        Returns:
            分析結果の辞書
        """
        analysis = {
            'total_queries': len(results_df),
            'metrics': {}
        }
        
        # 各メトリクスの統計情報を計算
        for metric in ['faithfulness', 'relevancy', 'correctness']:
            if metric in results_df.columns:
                metric_data = results_df[metric].dropna()
                analysis['metrics'][metric] = {
                    'mean': metric_data.mean(),
                    'std': metric_data.std(),
                    'min': metric_data.min(),
                    'max': metric_data.max(),
                    'count': len(metric_data)
                }
        
        return analysis
    
    def export_results(
        self,
        results_df: pd.DataFrame,
        file_path: str,
        format: str = 'csv'
    ):
        """
        評価結果をファイルに出力
        
        Args:
            results_df: 評価結果のDataFrame
            file_path: 出力ファイルパス
            format: 出力形式 ('csv', 'json', 'excel')
        """
        if format == 'csv':
            results_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif format == 'json':
            results_df.to_json(file_path, orient='records', force_ascii=False, indent=2)
        elif format == 'excel':
            results_df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"サポートされていない形式: {format}")
    
    def create_evaluation_report(
        self,
        results_df: pd.DataFrame,
        report_path: str
    ):
        """
        評価レポートを生成
        
        Args:
            results_df: 評価結果のDataFrame
            report_path: レポート出力パス
        """
        analysis = self.analyze_results(results_df)
        
        report = f"""# RAGシステム評価レポート

## 概要
- 評価対象クエリ数: {analysis['total_queries']}
- 評価日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 評価結果
"""
        
        for metric, stats in analysis['metrics'].items():
            report += f"""
### {metric.capitalize()}
- 平均スコア: {stats['mean']:.3f}
- 標準偏差: {stats['std']:.3f}
- 最小値: {stats['min']:.3f}
- 最大値: {stats['max']:.3f}
- サンプル数: {stats['count']}
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
