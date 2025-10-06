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




RAG (Retrieval-Augmented Generation) の高精度化のためにLlamaIndexでできることについて説明します。

## 1. データの前処理と準備

### チャンクサイズの最適化
```python
from llama_index.core.node_parser import SentenceSplitter

# チャンクサイズを調整
splitter = SentenceSplitter(
    chunk_size=512,  # 適切なサイズに調整
    chunk_overlap=50  # オーバーラップを設定
)
```

### メタデータの活用
```python
from llama_index.core import Document

# 豊富なメタデータを付与
documents = [
    Document(
        text=text,
        metadata={
            "source": "document_name.pdf",
            "section": "章番号",
            "date": "作成日時",
            "category": "カテゴリ"
        }
    )
]
```

## 2. 埋め込みモデルの選択と調整

### 多言語対応モデルの使用
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 日本語に最適化されたモデル
embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)
```

### カスタム埋め込みの実装
```python
from llama_index.core import Settings

Settings.embed_model = embed_model
```

## 3. 検索手法の改善

### ハイブリッド検索の実装
```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# BM25とベクトル検索の組み合わせ
vector_retriever = VectorIndexRetriever(index=vector_index)
bm25_retriever = BM25Retriever.from_defaults(docstore=docstore)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=4
)
```

### リランキングの追加
```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

# リランキングでより関連性の高い結果を選択
rerank = CohereRerank(top_n=5, api_key="your_api_key")
query_engine = index.as_query_engine(
    node_postprocessors=[rerank]
)
```

## 4. クエリの拡張と変換

### クエリ変換の実装
```python
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

# HyDE (Hypothetical Document Embeddings)
hyde = HyDEQueryTransform(include_original=True)
query_engine = TransformQueryEngine(base_query_engine, query_transform=hyde)
```

### 複数クエリ生成
```python
from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever],
    similarity_top_k=10,
    num_queries=4,  # 複数のクエリを生成
)
```

## 5. コンテキストの最適化

### コンテキスト圧縮
```python
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor

# 不要な情報を削除してコンテキストを圧縮
compressor = LongLLMLinguaPostprocessor(
    instruction_str="Given the context, please answer the question",
    target_token=300,
    rank_method="longllmlingua",
    additional_compress_kwargs={
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort"
    }
)
```

## 6. 評価とモニタリング

### 評価メトリクスの実装
```python
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator

# 関連性と忠実性の評価
relevancy_evaluator = RelevancyEvaluator()
faithfulness_evaluator = FaithfulnessEvaluator()

# 評価の実行
for query, response in test_cases:
    relevancy = relevancy_evaluator.evaluate(query, response)
    faithfulness = faithfulness_evaluator.evaluate(query, response)
```

### A/Bテストの実装
```python
# 異なる設定での性能比較
def compare_configurations():
    config_a = {"chunk_size": 512, "top_k": 5}
    config_b = {"chunk_size": 256, "top_k": 10}
    
    # 各設定でのパフォーマンスを測定
    results_a = evaluate_with_config(config_a)
    results_b = evaluate_with_config(config_b)
    
    return results_a, results_b
```

## 7. 高度な手法

### グラフRAGの実装
```python
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# グラフベースの検索
graph_store = Neo4jGraphStore(username="neo4j", password="password")
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(graph_store=graph_store)
)
```

### 階層的検索
```python
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine

# 複数のインデックスを組み合わせ
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_index.as_query_engine(),
    description="サマリーレベルの情報用"
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(),
    description="詳細情報用"
)

router_query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[summary_tool, vector_tool]
)
```

## 8. パフォーマンス最適化

### キャッシング
```python
from llama_index.core.storage.cache import SimpleCache

# クエリ結果のキャッシング
cache = SimpleCache()
query_engine = index.as_query_engine(cache=cache)
```

### 非同期処理
```python
import asyncio
from llama_index.core import AsyncQueryEngine

# 非同期でのクエリ処理
async def async_query(questions):
    async_engine = AsyncQueryEngine(query_engine)
    tasks = [async_engine.aquery(q) for q in questions]
    return await asyncio.gather(*tasks)
```

これらの手法を組み合わせることで、RAGシステムの精度を大幅に向上させることができます。重要なのは、データの特性とユースケースに応じて適切な手法を選択し、継続的に評価・改善することです。


まず、指定されたファイル内容を確認して、llama_indexでRAGの高精度化について詳しく見てみましょう。上記のファイルを見ると、LlamaIndexのRAG高精度化に関する詳細な実装とEvaluationServiceクラスが記載されています。この内容に基づいて、**評価とメトリクス**の部分を詳細に解説します。

## 評価とメトリクスの詳細解説

### 1. 主要な評価指標

LlamaIndexでは以下の4つの主要なEvaluatorが提供されています：

#### **Faithfulness（忠実性）評価**
```python
from llama_index.core.evaluation import FaithfulnessEvaluator

# 生成された回答が与えられたコンテキストに忠実かを評価
faithfulness_evaluator = FaithfulnessEvaluator()

# 評価実行
result = faithfulness_evaluator.evaluate(
    query="質問文",
    response="生成された回答",
    contexts=["コンテキスト1", "コンテキスト2"]
)
```
- **目的**: 回答が検索されたドキュメントの内容に基づいているかを測定
- **スコア範囲**: 0.0-1.0（1.0が最高）
- **重要性**: 幻覚（hallucination）を防ぐために重要

#### **Relevancy（関連性）評価**
```python
from llama_index.core.evaluation import RelevancyEvaluator

# 回答が質問に対してどれだけ関連性があるかを評価
relevancy_evaluator = RelevancyEvaluator()

result = relevancy_evaluator.evaluate(
    query="質問文",
    response="生成された回答"
)
```
- **目的**: 生成された回答が質問にどれだけ適切に答えているかを測定
- **スコア範囲**: 0.0-1.0（1.0が最高）

#### **Correctness（正確性）評価**
```python
from llama_index.core.evaluation import CorrectnessEvaluator

# 参照回答と比較して正確性を評価
correctness_evaluator = CorrectnessEvaluator()

result = correctness_evaluator.evaluate(
    query="質問文",
    response="生成された回答",
    reference="参照回答"
)
```
- **目的**: 生成された回答が正解と比較してどれだけ正確かを測定
- **必要条件**: 参照回答（正解）が必要

#### **Semantic Similarity（意味的類似性）評価**
```python
from llama_index.core.evaluation import SemanticSimilarityEvaluator

# 意味的な類似性を評価
similarity_evaluator = SemanticSimilarityEvaluator()

result = similarity_evaluator.evaluate(
    query="質問文",
    response="生成された回答",
    reference="参照回答"
)
```
- **目的**: 生成された回答と参照回答の意味的な類似度を測定

### 2. バッチ評価システム

```python
from llama_index.core.evaluation import BatchEvalRunner

# 複数のEvaluatorを一括実行
evaluators = {
    'faithfulness': FaithfulnessEvaluator(),
    'relevancy': RelevancyEvaluator(),
    'correctness': CorrectnessEvaluator(),
    'semantic_similarity': SemanticSimilarityEvaluator()
}

runner = BatchEvalRunner(
    evaluators=evaluators,
    workers=4,  # 並列処理数
    show_progress=True
)

# 非同期での評価実行
eval_results = await runner.aevaluate_dataset(
    dataset=evaluation_dataset,
    query_engine=query_engine
)
```

### 3. データセット生成

#### **自動質問生成**
```python
from llama_index.core.evaluation.dataset_generation import RagDatasetGenerator

# ドキュメントから自動的に質問を生成
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    num_questions_per_chunk=3,  # チャンクあたりの質問数
    question_gen_query="このテキストに基づいて、理解度を測るための質問を作成してください。"
)

eval_dataset = dataset_generator.generate_dataset_from_nodes()
```

#### **カスタムデータセット**
```python
from llama_index.core.evaluation import QueryResponseDataset

# 手動で作成した質問・回答ペア
questions = [
    "この文書の主要なテーマは何ですか？",
    "著者の結論は何ですか？",
    "提案されている解決策は何ですか？"
]

reference_answers = [
    "主要なテーマは...",
    "著者の結論は...",
    "提案されている解決策は..."
]

dataset = QueryResponseDataset(
    questions=questions,
    responses=reference_answers
)
```

### 4. 詳細な評価メトリクス

#### **検索品質の評価**
```python
# 検索された文書の関連性評価
def evaluate_retrieval_quality(queries, ground_truth_docs, retriever):
    precision_scores = []
    recall_scores = []
    
    for query, relevant_docs in zip(queries, ground_truth_docs):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_doc_ids = {node.node_id for node in retrieved_nodes}
        relevant_doc_ids = set(relevant_docs)
        
        # Precision = 関連文書 ∩ 検索文書 / 検索文書
        precision = len(retrieved_doc_ids & relevant_doc_ids) / len(retrieved_doc_ids)
        
        # Recall = 関連文書 ∩ 検索文書 / 関連文書  
        recall = len(retrieved_doc_ids & relevant_doc_ids) / len(relevant_doc_ids)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    return {
        'avg_precision': sum(precision_scores) / len(precision_scores),
        'avg_recall': sum(recall_scores) / len(recall_scores)
    }
```

#### **レスポンス時間の測定**
```python
import time

def measure_response_time(query_engine, queries):
    response_times = []
    
    for query in queries:
        start_time = time.time()
        response = query_engine.query(query)
        end_time = time.time()
        
        response_times.append(end_time - start_time)
    
    return {
        'avg_response_time': sum(response_times) / len(response_times),
        'min_response_time': min(response_times),
        'max_response_time': max(response_times)
    }
```

### 5. 包括的な評価システム

```python
class ComprehensiveEvaluator:
    def __init__(self, query_engine, retriever):
        self.query_engine = query_engine
        self.retriever = retriever
        self.evaluators = {
            'faithfulness': FaithfulnessEvaluator(),
            'relevancy': RelevancyEvaluator(),
            'correctness': CorrectnessEvaluator(),
            'semantic_similarity': SemanticSimilarityEvaluator()
        }
    
    def full_evaluation(self, test_queries, reference_answers=None, ground_truth_docs=None):
        results = {
            'quality_metrics': {},
            'performance_metrics': {},
            'retrieval_metrics': {}
        }
        
        # 1. 品質メトリクス
        for evaluator_name, evaluator in self.evaluators.items():
            scores = []
            for i, query in enumerate(test_queries):
                response = self.query_engine.query(query)
                
                if evaluator_name in ['correctness', 'semantic_similarity'] and reference_answers:
                    result = evaluator.evaluate(
                        query=query,
                        response=str(response),
                        reference=reference_answers[i]
                    )
                else:
                    result = evaluator.evaluate(
                        query=query,
                        response=str(response)
                    )
                scores.append(result.score)
            
            results['quality_metrics'][evaluator_name] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'std': np.std(scores)
            }
        
        # 2. パフォーマンスメトリクス
        results['performance_metrics'] = self.measure_response_time(test_queries)
        
        # 3. 検索メトリクス
        if ground_truth_docs:
            results['retrieval_metrics'] = self.evaluate_retrieval_quality(
                test_queries, ground_truth_docs, self.retriever
            )
        
        return results
    
    def generate_report(self, results):
        """評価結果のレポート生成"""
        report = []
        report.append("=== RAGシステム評価レポート ===\n")
        
        # 品質メトリクス
        report.append("## 品質メトリクス")
        for metric, stats in results['quality_metrics'].items():
            report.append(f"- {metric.capitalize()}:")
            report.append(f"  平均: {stats['mean']:.3f}")
            report.append(f"  最小: {stats['min']:.3f}")
            report.append(f"  最大: {stats['max']:.3f}")
            report.append(f"  標準偏差: {stats['std']:.3f}\n")
        
        # パフォーマンスメトリクス
        report.append("## パフォーマンスメトリクス")
        perf = results['performance_metrics']
        report.append(f"- 平均応答時間: {perf['avg_response_time']:.3f}秒")
        report.append(f"- 最小応答時間: {perf['min_response_time']:.3f}秒")
        report.append(f"- 最大応答時間: {perf['max_response_time']:.3f}秒\n")
        
        # 検索メトリクス
        if 'retrieval_metrics' in results:
            report.append("## 検索メトリクス")
            retr = results['retrieval_metrics']
            report.append(f"- 平均Precision: {retr['avg_precision']:.3f}")
            report.append(f"- 平均Recall: {retr['avg_recall']:.3f}")
        
        return '\n'.join(report)
```

### 6. 継続的な評価とモニタリング

```python
import matplotlib.pyplot as plt
import pandas as pd

def create_evaluation_dashboard(eval_history):
    """評価結果の可視化ダッシュボード"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 時系列での品質変化
    df = pd.DataFrame(eval_history)
    
    # Faithfulnessの変化
    axes[0, 0].plot(df['timestamp'], df['faithfulness'])
    axes[0, 0].set_title('Faithfulness Score Over Time')
    axes[0, 0].set_ylabel('Score')
    
    # Relevancyの変化  
    axes[0, 1].plot(df['timestamp'], df['relevancy'])
    axes[0, 1].set_title('Relevancy Score Over Time')
    axes[0, 1].set_ylabel('Score')
    
    # 応答時間の変化
    axes[1, 0].plot(df['timestamp'], df['response_time'])
    axes[1, 0].set_title('Response Time Over Time')
    axes[1, 0].set_ylabel('Seconds')
    
    # 総合スコアの分布
    axes[1, 1].hist(df['overall_score'], bins=20)
    axes[1, 1].set_title('Overall Score Distribution')
    axes[1, 1].set_xlabel('Score')
    
    plt.tight_layout()
    plt.show()

# アラートシステム
def check_performance_degradation(current_scores, baseline_scores, threshold=0.1):
    """性能低下の検出"""
    alerts = []
    
    for metric, current_score in current_scores.items():
        baseline_score = baseline_scores.get(metric, 0)
        if baseline_score - current_score > threshold:
            alerts.append(f"警告: {metric}が{baseline_score:.3f}から{current_score:.3f}に低下")
    
    return alerts
```

この評価システムを使用することで、RAGシステムの性能を定量的に測定し、継続的な改善を行うことができます。重要なのは、複数の指標を組み合わせて総合的に評価し、実際のユースケースに応じて重要視する指標を調整することです。