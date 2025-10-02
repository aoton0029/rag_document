llama_index.core.evaluationを使ってEvaluationServiceクラスを作成します。以下のような実装を提示します：

```python
from typing import List, Dict, Any, Optional
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
    EvaluationResult
)
from llama_index.core.evaluation.dataset_generation import (
    DatasetGenerator,
    RagDatasetGenerator
)
from llama_index.core import QueryEngine, VectorStoreIndex, Document
from llama_index.core.schema import BaseNode
import asyncio

class EvaluationService:
    """
    llama_index.core.evaluationを使用した評価サービスクラス
    各Evaluatorの生成と各Datasetの生成を行う
    """
    
    def __init__(self, 
                 index: VectorStoreIndex,
                 documents: List[Document],
                 query_engine: QueryEngine):
        """
        初期化
        
        Args:
            index: 外部で作成済みのVectorStoreIndex
            documents: 外部で作成済みのDocumentリスト
            query_engine: 外部で作成済みのQueryEngine
        """
        self.index = index
        self.documents = documents
        self.query_engine = query_engine
        self.evaluators = {}
        self.datasets = {}
    
    def create_evaluators(self, 
                         include_faithfulness: bool = True,
                         include_relevancy: bool = True,
                         include_correctness: bool = True,
                         include_semantic_similarity: bool = True) -> Dict[str, Any]:
        """
        各種Evaluatorを生成
        
        Args:
            include_faithfulness: FaithfulnessEvaluatorを含めるか
            include_relevancy: RelevancyEvaluatorを含めるか
            include_correctness: CorrectnessEvaluatorを含めるか
            include_semantic_similarity: SemanticSimilarityEvaluatorを含めるか
            
        Returns:
            生成されたEvaluatorの辞書
        """
        evaluators = {}
        
        if include_faithfulness:
            evaluators['faithfulness'] = FaithfulnessEvaluator()
            
        if include_relevancy:
            evaluators['relevancy'] = RelevancyEvaluator()
            
        if include_correctness:
            evaluators['correctness'] = CorrectnessEvaluator()
            
        if include_semantic_similarity:
            evaluators['semantic_similarity'] = SemanticSimilarityEvaluator()
        
        self.evaluators = evaluators
        return evaluators
    
    def create_rag_dataset(self, 
                          num_questions_per_chunk: int = 2,
                          question_gen_query: Optional[str] = None) -> Dict[str, Any]:
        """
        RAG用のデータセットを生成
        
        Args:
            num_questions_per_chunk: チャンクごとの質問数
            question_gen_query: 質問生成用のクエリテンプレート
            
        Returns:
            生成されたデータセット
        """
        # RagDatasetGeneratorを使用してデータセット生成
        dataset_generator = RagDatasetGenerator.from_documents(
            documents=self.documents,
            num_questions_per_chunk=num_questions_per_chunk,
            question_gen_query=question_gen_query
        )
        
        # データセット生成
        eval_dataset = dataset_generator.generate_dataset_from_nodes()
        
        self.datasets['rag_dataset'] = eval_dataset
        return {
            'rag_dataset': eval_dataset,
            'questions': eval_dataset.questions,
            'qr_pairs': eval_dataset.qr_pairs
        }
    
    def create_custom_dataset(self, 
                             questions: List[str],
                             reference_answers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        カスタムデータセットを生成
        
        Args:
            questions: 質問のリスト
            reference_answers: 参照回答のリスト（オプション）
            
        Returns:
            生成されたデータセット
        """
        from llama_index.core.evaluation import QueryResponseDataset
        
        # QueryResponseDatasetを作成
        if reference_answers:
            qr_pairs = list(zip(questions, reference_answers))
            dataset = QueryResponseDataset(questions=questions, responses=reference_answers)
        else:
            dataset = QueryResponseDataset(questions=questions)
        
        self.datasets['custom_dataset'] = dataset
        return {
            'custom_dataset': dataset,
            'questions': questions,
            'reference_answers': reference_answers
        }
    
    async def run_evaluation(self, 
                           dataset_name: str,
                           evaluator_names: Optional[List[str]] = None,
                           show_progress: bool = True) -> Dict[str, List[EvaluationResult]]:
        """
        評価を実行
        
        Args:
            dataset_name: 使用するデータセット名
            evaluator_names: 使用するEvaluator名のリスト（Noneの場合は全て）
            show_progress: 進捗表示するか
            
        Returns:
            評価結果の辞書
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.datasets.keys())}")
        
        dataset = self.datasets[dataset_name]
        
        # 使用するEvaluatorを決定
        if evaluator_names is None:
            evaluators_to_use = self.evaluators
        else:
            evaluators_to_use = {name: self.evaluators[name] 
                               for name in evaluator_names 
                               if name in self.evaluators}
        
        # BatchEvalRunnerを使用して評価実行
        runner = BatchEvalRunner(
            evaluators=evaluators_to_use,
            workers=4,
            show_progress=show_progress
        )
        
        # 評価実行
        eval_results = await runner.aevaluate_dataset(
            dataset=dataset,
            query_engine=self.query_engine
        )
        
        return eval_results
    
    def run_evaluation_sync(self, 
                           dataset_name: str,
                           evaluator_names: Optional[List[str]] = None,
                           show_progress: bool = True) -> Dict[str, List[EvaluationResult]]:
        """
        評価を同期実行
        
        Args:
            dataset_name: 使用するデータセット名
            evaluator_names: 使用するEvaluator名のリスト
            show_progress: 進捗表示するか
            
        Returns:
            評価結果の辞書
        """
        return asyncio.run(self.run_evaluation(dataset_name, evaluator_names, show_progress))
    
    def get_evaluation_summary(self, eval_results: Dict[str, List[EvaluationResult]]) -> Dict[str, Dict[str, float]]:
        """
        評価結果のサマリーを取得
        
        Args:
            eval_results: 評価結果
            
        Returns:
            サマリー統計
        """
        summary = {}
        
        for evaluator_name, results in eval_results.items():
            scores = [result.score for result in results if result.score is not None]
            
            if scores:
                summary[evaluator_name] = {
                    'mean_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'num_evaluated': len(scores),
                    'total_questions': len(results)
                }
            else:
                summary[evaluator_name] = {
                    'mean_score': 0.0,
                    'min_score': 0.0,
                    'max_score': 0.0,
                    'num_evaluated': 0,
                    'total_questions': len(results)
                }
        
        return summary
    
    def export_results(self, 
                      eval_results: Dict[str, List[EvaluationResult]], 
                      output_path: str) -> None:
        """
        評価結果をファイルに出力
        
        Args:
            eval_results: 評価結果
            output_path: 出力ファイルパス
        """
        import json
        
        # 結果を辞書形式に変換
        exportable_results = {}
        for evaluator_name, results in eval_results.items():
            exportable_results[evaluator_name] = [
                {
                    'query': result.query,
                    'response': result.response,
                    'score': result.score,
                    'feedback': result.feedback,
                    'passing': result.passing
                }
                for result in results
            ]
        
        # JSONファイルとして出力
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(exportable_results, f, ensure_ascii=False, indent=2)
```

## 使用例

```python
# 使用例
# index, documents, query_engineは外部で作成済みと仮定

# EvaluationServiceの初期化
eval_service = EvaluationService(
    index=index,
    documents=documents,
    query_engine=query_engine
)

# Evaluatorの生成
evaluators = eval_service.create_evaluators(
    include_faithfulness=True,
    include_relevancy=True,
    include_correctness=True,
    include_semantic_similarity=True
)

# RAGデータセットの生成
rag_dataset = eval_service.create_rag_dataset(
    num_questions_per_chunk=3
)

# カスタムデータセットの生成
custom_questions = [
    "この文書の主要なポイントは何ですか？",
    "著者の結論は何ですか？"
]
custom_dataset = eval_service.create_custom_dataset(
    questions=custom_questions
)

# 評価の実行
eval_results = eval_service.run_evaluation_sync(
    dataset_name='rag_dataset',
    show_progress=True
)

# 結果のサマリー取得
summary = eval_service.get_evaluation_summary(eval_results)
print("評価結果サマリー:")
for evaluator, stats in summary.items():
    print(f"{evaluator}: 平均スコア = {stats['mean_score']:.3f}")

# 結果の出力
eval_service.export_results(eval_results, "evaluation_results.json")
```

この`EvaluationService`クラスの特徴：

1. **Evaluator生成**: 主要な4つのEvaluator（Faithfulness, Relevancy, Correctness, SemanticSimilarity）を生成
2. **Dataset生成**: RAGデータセットとカスタムデータセットの両方に対応
3. **評価実行**: 非同期・同期両方の評価実行をサポート
4. **結果分析**: 評価結果のサマリー統計を提供
5. **結果出力**: JSON形式での結果出力機能

外部で作成済みのindex、documents、query_engineを受け取り、それらを使用して包括的な評価システムを提供します。