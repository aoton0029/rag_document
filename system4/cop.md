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


製品の説明書をRAGする具体的なサンプルを作成します。製品マニュアルに特化した実装例を提示します。

## 製品説明書RAGシステムのサンプル実装

### 1. 基本的な製品マニュアルRAGシステム

```python
import os
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.readers.file import PDFReader
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator
)

class ProductManualRAG:
    """製品説明書に特化したRAGシステム"""
    
    def __init__(self):
        # LLMとEmbeddingの設定
        Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        self.index = None
        self.query_engine = None
        self.documents = []
        
    def load_product_manuals(self, manual_paths: List[str]) -> List[Document]:
        """製品マニュアルを読み込み"""
        documents = []
        reader = PDFReader()
        
        for path in manual_paths:
            # ファイル名から製品名を抽出
            product_name = os.path.splitext(os.path.basename(path))[0]
            
            # PDFを読み込み
            docs = reader.load_data(file=path)
            
            for i, doc in enumerate(docs):
                # 製品マニュアル特有のメタデータを付与
                doc.metadata.update({
                    "product_name": product_name,
                    "document_type": "manual",
                    "page_number": i + 1,
                    "source_file": path,
                    "manual_section": self._extract_section_from_content(doc.text),
                    "manual_category": self._classify_manual_content(doc.text)
                })
                documents.append(doc)
        
        self.documents = documents
        return documents
    
    def _extract_section_from_content(self, content: str) -> str:
        """コンテンツからセクションを抽出"""
        # 一般的なマニュアルセクションパターンを検出
        section_keywords = {
            "setup": ["セットアップ", "設置", "初期設定", "インストール"],
            "operation": ["操作方法", "使用方法", "操作手順", "使い方"],
            "troubleshooting": ["トラブルシューティング", "故障診断", "問題解決", "エラー"],
            "maintenance": ["メンテナンス", "保守", "清掃", "点検"],
            "specifications": ["仕様", "スペック", "技術仕様", "性能"],
            "safety": ["安全", "注意事項", "警告", "危険"]
        }
        
        content_lower = content.lower()
        for section, keywords in section_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return section
        return "general"
    
    def _classify_manual_content(self, content: str) -> str:
        """マニュアルコンテンツの分類"""
        # 手順書、仕様書、注意事項などに分類
        if any(word in content.lower() for word in ["手順", "ステップ", "方法"]):
            return "procedure"
        elif any(word in content.lower() for word in ["仕様", "スペック", "性能"]):
            return "specification"
        elif any(word in content.lower() for word in ["注意", "警告", "危険"]):
            return "warning"
        else:
            return "description"
    
    def create_enhanced_index(self) -> VectorStoreIndex:
        """強化されたインデックスを作成"""
        # 製品マニュアル用のノードパーサー
        splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=100,  # マニュアルでは前後の文脈が重要
            separator=" "
        )
        
        # メタデータ抽出器を設定
        extractors = [
            TitleExtractor(nodes=5),
            QuestionsAnsweredExtractor(questions=3),
            SummaryExtractor(summaries=["prev", "self", "next"]),
            KeywordExtractor(keywords=10)
        ]
        
        # ノードを作成
        nodes = splitter.get_nodes_from_documents(self.documents, show_progress=True)
        
        # メタデータを抽出
        for extractor in extractors:
            nodes = extractor.extract(nodes, show_progress=True)
        
        # インデックスを作成
        self.index = VectorStoreIndex(nodes, show_progress=True)
        return self.index
    
    def create_specialized_query_engine(self):
        """製品マニュアル特化のクエリエンジンを作成"""
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor
        from llama_index.core.postprocessor import SimilarityPostprocessor
        
        # レトリーバーを設定
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
        
        # ポストプロセッサーを設定
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.7),
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ]
        
        # クエリエンジンを作成
        self.query_engine = self.index.as_query_engine(
            retriever=retriever,
            node_postprocessors=postprocessors,
            response_mode="compact"
        )
        
        return self.query_engine
    
    def query_manual(self, question: str) -> Dict[str, Any]:
        """マニュアルに質問"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call create_specialized_query_engine first.")
        
        # 製品マニュアル用のプロンプトテンプレートを適用
        enhanced_prompt = self._enhance_manual_prompt(question)
        
        # クエリを実行
        response = self.query_engine.query(enhanced_prompt)
        
        # レスポンスを整理
        return {
            "answer": response.response,
            "source_nodes": [
                {
                    "content": node.node.text,
                    "product_name": node.node.metadata.get("product_name", "Unknown"),
                    "section": node.node.metadata.get("manual_section", "Unknown"),
                    "page": node.node.metadata.get("page_number", "Unknown"),
                    "score": node.score
                }
                for node in response.source_nodes
            ],
            "metadata": {
                "query": question,
                "enhanced_prompt": enhanced_prompt,
                "num_sources": len(response.source_nodes)
            }
        }
    
    def _enhance_manual_prompt(self, question: str) -> str:
        """マニュアル質問用のプロンプト強化"""
        base_prompt = f"""
以下は製品マニュアルに関する質問です。正確で実用的な回答を提供してください。

質問: {question}

回答する際は以下の点に注意してください：
1. 安全に関する情報がある場合は必ず含める
2. 手順がある場合は番号付きで明確に示す
3. 該当するページ番号や章を参照する
4. 不明な点がある場合は「マニュアルに記載なし」と明記する
5. 製品固有の注意事項があれば強調する

回答:
"""
        return base_prompt

# 評価用のクラス
class ManualRAGEvaluator:
    """製品マニュアルRAGの評価システム"""
    
    def __init__(self, rag_system: ProductManualRAG):
        self.rag_system = rag_system
        self.evaluators = {
            'faithfulness': FaithfulnessEvaluator(),
            'relevancy': RelevancyEvaluator(),
            'correctness': CorrectnessEvaluator()
        }
    
    def create_manual_test_questions(self) -> List[Dict[str, str]]:
        """製品マニュアル用のテスト質問を生成"""
        return [
            {
                "question": "この製品の初期設定方法を教えてください",
                "expected_category": "setup"
            },
            {
                "question": "エラーが発生した場合の対処法は？",
                "expected_category": "troubleshooting"
            },
            {
                "question": "定期的なメンテナンス方法について",
                "expected_category": "maintenance"
            },
            {
                "question": "製品の技術仕様を教えて",
                "expected_category": "specifications"
            },
            {
                "question": "安全に使用するための注意事項は？",
                "expected_category": "safety"
            }
        ]
    
    def evaluate_manual_responses(self, test_questions: List[Dict[str, str]]) -> Dict[str, Any]:
        """マニュアル回答の評価"""
        results = []
        
        for test_case in test_questions:
            question = test_case["question"]
            response = self.rag_system.query_manual(question)
            
            # 各評価指標でスコア算出
            evaluation_scores = {}
            for eval_name, evaluator in self.evaluators.items():
                try:
                    result = evaluator.evaluate(question, response["answer"])
                    evaluation_scores[eval_name] = result.score
                except Exception as e:
                    evaluation_scores[eval_name] = None
                    print(f"Evaluation error for {eval_name}: {e}")
            
            results.append({
                "question": question,
                "answer": response["answer"],
                "expected_category": test_case["expected_category"],
                "scores": evaluation_scores,
                "source_info": response["source_nodes"]
            })
        
        return {
            "individual_results": results,
            "summary": self._calculate_summary_scores(results)
        }
    
    def _calculate_summary_scores(self, results: List[Dict]) -> Dict[str, float]:
        """サマリースコアを計算"""
        summary = {}
        for eval_name in self.evaluators.keys():
            scores = [r["scores"][eval_name] for r in results if r["scores"][eval_name] is not None]
            if scores:
                summary[eval_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores)
                }
        return summary

# 使用例
def main():
    # 製品マニュアルRAGシステムの初期化
    manual_rag = ProductManualRAG()
    
    # マニュアルファイルを読み込み（例）
    manual_files = [
        "product_a_manual.pdf",
        "product_b_manual.pdf",
        "safety_guidelines.pdf"
    ]
    
    # ドキュメントを読み込み
    documents = manual_rag.load_product_manuals(manual_files)
    print(f"読み込んだドキュメント数: {len(documents)}")
    
    # インデックスを作成
    index = manual_rag.create_enhanced_index()
    
    # クエリエンジンを作成
    query_engine = manual_rag.create_specialized_query_engine()
    
    # 質問例
    questions = [
        "製品Aの初期設定手順を教えてください",
        "エラーコード E001 が表示された場合の対処法は？",
        "月次メンテナンスで確認すべき項目は何ですか？",
        "この製品の最大動作温度は何度ですか？"
    ]
    
    # 各質問に回答
    for question in questions:
        print(f"\n質問: {question}")
        print("-" * 50)
        
        response = manual_rag.query_manual(question)
        print(f"回答: {response['answer']}")
        
        print("\n参照したソース:")
        for i, source in enumerate(response['source_nodes'][:3]):  # 上位3つを表示
            print(f"{i+1}. {source['product_name']} - {source['section']} (ページ{source['page']})")
    
    # 評価の実行
    evaluator = ManualRAGEvaluator(manual_rag)
    test_questions = evaluator.create_manual_test_questions()
    evaluation_results = evaluator.evaluate_manual_responses(test_questions)
    
    print("\n=== 評価結果 ===")
    for eval_name, scores in evaluation_results["summary"].items():
        print(f"{eval_name}: 平均={scores['mean']:.3f}, 最小={scores['min']:.3f}, 最大={scores['max']:.3f}")

if __name__ == "__main__":
    main()
```

### 2. より高度な製品マニュアル特化機能

```python
# 追加の特化機能
class AdvancedManualRAG(ProductManualRAG):
    """より高度な製品マニュアルRAG機能"""
    
    def create_product_specific_retrievers(self) -> Dict[str, Any]:
        """製品別の特化レトリーバーを作成"""
        from llama_index.core.retrievers import VectorIndexRetriever
        
        retrievers = {}
        
        # 製品ごとにレトリーバーを作成
        products = set(doc.metadata.get("product_name") for doc in self.documents)
        
        for product in products:
            # 製品固有のフィルター
            def product_filter(nodes, product_name=product):
                return [node for node in nodes 
                       if node.metadata.get("product_name") == product_name]
            
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5,
                node_postprocessors=[product_filter]
            )
            retrievers[product] = retriever
        
        return retrievers
    
    def create_section_specific_engines(self) -> Dict[str, Any]:
        """セクション特化のクエリエンジンを作成"""
        engines = {}
        sections = ["setup", "operation", "troubleshooting", "maintenance", "specifications"]
        
        for section in sections:
            # セクション固有のプロンプト
            section_prompts = {
                "setup": "初期設定や設置に関する質問です。手順を明確に、安全事項を含めて回答してください。",
                "operation": "操作方法に関する質問です。具体的な手順と注意点を含めて回答してください。",
                "troubleshooting": "トラブルシューティングに関する質問です。問題の診断方法と解決手順を段階的に説明してください。",
                "maintenance": "メンテナンスに関する質問です。定期的な点検項目と手順を詳細に説明してください。",
                "specifications": "技術仕様に関する質問です。正確な数値と単位を含めて回答してください。"
            }
            
            # セクション特化のクエリエンジン作成
            engines[section] = self._create_section_engine(section, section_prompts[section])
        
        return engines
    
    def _create_section_engine(self, section: str, prompt_template: str):
        """セクション特化のエンジンを作成"""
        from llama_index.core.query_engine import CustomQueryEngine
        
        # セクションフィルタリング機能付きのカスタムエンジン
        def section_query_fn(query_str: str):
            # セクション情報を含む強化プロンプト
            enhanced_query = f"{prompt_template}\n\n質問: {query_str}"
            return self.query_engine.query(enhanced_query)
        
        return section_query_fn
    
    def smart_route_query(self, question: str) -> Dict[str, Any]:
        """質問を適切なセクションに自動ルーティング"""
        # 質問内容からセクションを推定
        section_keywords = {
            "setup": ["設定", "設置", "インストール", "初期", "セットアップ"],
            "operation": ["操作", "使用", "使い方", "動作", "機能"],
            "troubleshooting": ["エラー", "故障", "問題", "トラブル", "不具合"],
            "maintenance": ["メンテナンス", "保守", "清掃", "点検", "交換"],
            "specifications": ["仕様", "スペック", "性能", "容量", "サイズ"]
        }
        
        question_lower = question.lower()
        detected_section = "general"
        max_matches = 0
        
        for section, keywords in section_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            if matches > max_matches:
                max_matches = matches
                detected_section = section
        
        # 検出されたセクションに応じた処理
        if detected_section != "general":
            section_engines = self.create_section_specific_engines()
            response = section_engines[detected_section](question)
            
            return {
                "answer": response.response,
                "detected_section": detected_section,
                "confidence": max_matches,
                "routing": "section_specific"
            }
        else:
            # 一般的な質問として処理
            response = self.query_manual(question)
            response["detected_section"] = "general"
            response["routing"] = "general"
            return response

# 使用例（高度機能）
def advanced_usage_example():
    advanced_rag = AdvancedManualRAG()
    
    # 基本設定は同じ
    manual_files = ["product_manual.pdf"]
    advanced_rag.load_product_manuals(manual_files)
    advanced_rag.create_enhanced_index()
    advanced_rag.create_specialized_query_engine()
    
    # スマートルーティングの使用例
    questions = [
        "初期設定の手順を教えて",  # setup にルーティング
        "エラーコード404の対処法",  # troubleshooting にルーティング
        "最大消費電力は？",  # specifications にルーティング
    ]
    
    for question in questions:
        result = advanced_rag.smart_route_query(question)
        print(f"質問: {question}")
        print(f"検出されたセクション: {result['detected_section']}")
        print(f"回答: {result['answer'][:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    advanced_usage_example()
```

この実装の特徴：

1. **製品マニュアル特化**: セットアップ、操作、トラブルシューティングなど、マニュアル特有のセクション分類
2. **メタデータ豊富化**: 製品名、ページ番号、セクション情報の自動抽出
3. **スマートルーティング**: 質問内容に応じて適切なセクションに自動振り分け
4. **安全重視**: 安全情報の優先表示機能
5. **手順明確化**: ステップバイステップの回答生成
6. **評価システム**: マニュアル特化の評価指標

このサンプルを参考に、具体的な製品マニュアルに合わせてカスタマイズしてください。

製品説明書をRAGで処理する際の高精度化サンプルを作成しました。データ保存、クエリ処理、評価の各段階で最適化を図るコードです。

## 製品説明書RAG高精度化サンプル

```python
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext,
    Settings,
    ServiceContext
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser
)
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
    MetadataReplacementPostProcessor
)
from llama_index.core.query_engine import (
    TransformQueryEngine,
    RouterQueryEngine,
    RetrieverQueryEngine
)
from llama_index.core.indices.query.query_transform import (
    HyDEQueryTransform,
    StepDecomposeQueryTransform
)
from llama_index.core.tools import QueryEngineTool
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

@dataclass
class RAGConfiguration:
    """RAG設定クラス"""
    # データ保存設定
    chunk_size: int = 512
    chunk_overlap: int = 50
    use_semantic_chunking: bool = True
    use_hierarchical_chunking: bool = False
    
    # 埋め込み設定
    embedding_model: str = "intfloat/multilingual-e5-large"
    
    # 検索設定
    similarity_top_k: int = 10
    use_hybrid_search: bool = True
    use_reranking: bool = True
    rerank_top_n: int = 5
    
    # クエリ変換設定
    use_hyde: bool = True
    use_query_decomposition: bool = False
    num_fusion_queries: int = 4
    
    # 評価設定
    eval_batch_size: int = 10

class ProductManualRAGSystem:
    """製品説明書専用RAGシステム"""
    
    def __init__(self, config: RAGConfiguration):
        self.config = config
        self.setup_models()
        self.documents = []
        self.index = None
        self.query_engine = None
        self.evaluation_results = {}
        
    def setup_models(self):
        """モデルとサービスの初期化"""
        # 埋め込みモデルの設定
        embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding_model
        )
        Settings.embed_model = embed_model
        
    def load_product_documents(self, file_paths: List[str]) -> List[Document]:
        """製品説明書の読み込み"""
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # ファイル名から製品情報を抽出
                filename = os.path.basename(file_path)
                product_name = filename.split('.')[0]
                
                # メタデータの設定（製品説明書特有）
                metadata = {
                    "source": file_path,
                    "product_name": product_name,
                    "document_type": "manual",
                    "language": "ja",
                    "processed_date": datetime.now().isoformat(),
                    # 製品説明書特有のメタデータ
                    "sections": self._extract_sections(content),
                    "product_category": self._infer_product_category(content),
                    "technical_level": self._assess_technical_level(content)
                }
                
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        self.documents = documents
        return documents
    
    def _extract_sections(self, content: str) -> List[str]:
        """セクション情報の抽出"""
        import re
        # 一般的な説明書のセクションパターン
        patterns = [
            r'第\d+章\s*(.+)',  # 第1章 概要
            r'\d+\.\s*(.+)',    # 1. はじめに
            r'■\s*(.+)',        # ■ 安全上の注意
            r'【(.+)】',         # 【仕様】
        ]
        
        sections = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            sections.extend(matches)
        
        return sections[:10]  # 最大10セクション
    
    def _infer_product_category(self, content: str) -> str:
        """製品カテゴリの推定"""
        categories = {
            "electronics": ["電子機器", "回路", "電源", "バッテリー"],
            "software": ["ソフトウェア", "アプリ", "プログラム", "システム"],
            "mechanical": ["機械", "エンジン", "モーター", "ギア"],
            "medical": ["医療", "診断", "治療", "薬品"],
            "household": ["家電", "調理", "清掃", "生活"]
        }
        
        content_lower = content.lower()
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _assess_technical_level(self, content: str) -> str:
        """技術レベルの評価"""
        technical_terms = ["仕様", "パラメータ", "設定", "構成", "技術"]
        advanced_terms = ["アルゴリズム", "プロトコル", "API", "フレームワーク"]
        
        tech_count = sum(1 for term in technical_terms if term in content)
        advanced_count = sum(1 for term in advanced_terms if term in content)
        
        if advanced_count > 3:
            return "advanced"
        elif tech_count > 5:
            return "intermediate"
        else:
            return "basic"
    
    def create_optimized_storage(self) -> VectorStoreIndex:
        """最適化されたデータ保存システム"""
        
        # ノードパーサーの設定
        if self.config.use_semantic_chunking:
            # セマンティック分割（意味的な境界で分割）
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model
            )
        elif self.config.use_hierarchical_chunking:
            # 階層的分割
            node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            )
        else:
            # 通常の文分割
            node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        # メタデータ抽出器の設定
        extractors = [
            TitleExtractor(nodes=5),
            KeywordExtractor(keywords=10),
            SummaryExtractor(summaries=["prev", "self", "next"]),
            QuestionsAnsweredExtractor(questions=3)
        ]
        
        # インジェストパイプラインの作成
        pipeline = IngestionPipeline(
            transformations=[node_parser] + extractors
        )
        
        # ドキュメントの処理
        nodes = pipeline.run(documents=self.documents, show_progress=True)
        
        # インデックスの作成
        self.index = VectorStoreIndex(nodes)
        
        return self.index
    
    def create_hybrid_retriever(self):
        """ハイブリッド検索システムの作成"""
        # ベクトル検索
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.config.similarity_top_k
        )
        
        if self.config.use_hybrid_search:
            # BM25検索の追加
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.index.docstore,
                similarity_top_k=self.config.similarity_top_k
            )
            
            # ハイブリッド検索
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=self.config.similarity_top_k,
                num_queries=self.config.num_fusion_queries,
                mode="reciprocal_rerank",
                use_async=True
            )
        else:
            retriever = vector_retriever
            
        return retriever
    
    def create_advanced_query_engine(self):
        """高度なクエリエンジンの作成"""
        retriever = self.create_hybrid_retriever()
        
        # ポストプロセッサーの設定
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.7),
            KeywordNodePostprocessor(
                keywords=["製品", "仕様", "手順", "注意", "警告"],
                exclude_keywords=["無関係", "削除"]
            )
        ]
        
        # リランキングの追加
        if self.config.use_reranking:
            # Note: Cohere API keyが必要
            try:
                rerank = CohereRerank(
                    top_n=self.config.rerank_top_n,
                    api_key=os.getenv("COHERE_API_KEY")
                )
                postprocessors.append(rerank)
            except:
                print("Cohere reranking not available")
        
        # 基本クエリエンジン
        base_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=postprocessors
        )
        
        # クエリ変換の追加
        if self.config.use_hyde:
            # HyDE変換
            hyde_transform = HyDEQueryTransform(include_original=True)
            query_engine = TransformQueryEngine(
                base_query_engine, 
                query_transform=hyde_transform
            )
        elif self.config.use_query_decomposition:
            # ステップ分解
            step_transform = StepDecomposeQueryTransform()
            query_engine = TransformQueryEngine(
                base_query_engine,
                query_transform=step_transform
            )
        else:
            query_engine = base_query_engine
        
        self.query_engine = query_engine
        return query_engine

class ProductManualEvaluator:
    """製品説明書RAG専用評価システム"""
    
    def __init__(self, rag_system: ProductManualRAGSystem):
        self.rag_system = rag_system
        self.evaluators = {}
        self.test_datasets = {}
        self.setup_evaluators()
        
    def setup_evaluators(self):
        """評価器の初期化"""
        self.evaluators = {
            'faithfulness': FaithfulnessEvaluator(),
            'relevancy': RelevancyEvaluator(), 
            'correctness': CorrectnessEvaluator(),
            'semantic_similarity': SemanticSimilarityEvaluator()
        }
    
    def create_product_manual_dataset(self) -> Dict[str, Any]:
        """製品説明書専用のテストデータセット作成"""
        
        # 製品説明書特有の質問パターン
        manual_questions = [
            # 基本情報
            "この製品の主な機能は何ですか？",
            "製品の仕様を教えてください",
            "対応しているOSは何ですか？",
            
            # 操作・手順
            "初期設定の手順を教えてください",
            "電源の入れ方を教えてください",
            "リセット方法を教えてください",
            
            # トラブルシューティング
            "電源が入らない場合はどうすれば良いですか？",
            "エラーが発生した時の対処法は？",
            "動作が不安定な場合の対処法は？",
            
            # 安全・メンテナンス
            "使用上の注意点は何ですか？",
            "定期メンテナンスの方法を教えてください",
            "保証期間はどのくらいですか？"
        ]
        
        # RAGデータセット生成
        dataset_generator = RagDatasetGenerator.from_documents(
            documents=self.rag_system.documents,
            num_questions_per_chunk=2
        )
        
        auto_dataset = dataset_generator.generate_dataset_from_nodes()
        
        # マニュアル質問との結合
        from llama_index.core.evaluation import QueryResponseDataset
        
        combined_questions = manual_questions + auto_dataset.questions
        
        manual_dataset = QueryResponseDataset(questions=combined_questions)
        
        self.test_datasets['product_manual'] = manual_dataset
        
        return {
            'dataset': manual_dataset,
            'manual_questions': manual_questions,
            'auto_questions': auto_dataset.questions,
            'total_questions': len(combined_questions)
        }
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """包括的評価の実行"""
        
        if 'product_manual' not in self.test_datasets:
            self.create_product_manual_dataset()
        
        dataset = self.test_datasets['product_manual']
        
        # バッチ評価の実行
        runner = BatchEvalRunner(
            evaluators=self.evaluators,
            workers=4,
            show_progress=True
        )
        
        eval_results = await runner.aevaluate_dataset(
            dataset=dataset,
            query_engine=self.rag_system.query_engine
        )
        
        # 結果の分析
        analysis = self.analyze_evaluation_results(eval_results)
        
        return {
            'raw_results': eval_results,
            'analysis': analysis,
            'recommendations': self.generate_recommendations(analysis)
        }
    
    def analyze_evaluation_results(self, eval_results: Dict[str, List[EvaluationResult]]) -> Dict[str, Any]:
        """評価結果の詳細分析"""
        analysis = {}
        
        for evaluator_name, results in eval_results.items():
            scores = [r.score for r in results if r.score is not None]
            
            if scores:
                analysis[evaluator_name] = {
                    'mean_score': sum(scores) / len(scores),
                    'median_score': sorted(scores)[len(scores)//2],
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'std_deviation': self._calculate_std(scores),
                    'score_distribution': self._get_score_distribution(scores),
                    'low_performing_queries': self._get_low_performing_queries(results, threshold=0.7)
                }
        
        # 総合評価スコア
        overall_scores = []
        for evaluator_results in eval_results.values():
            scores = [r.score for r in evaluator_results if r.score is not None]
            if scores:
                overall_scores.extend(scores)
        
        if overall_scores:
            analysis['overall'] = {
                'mean_score': sum(overall_scores) / len(overall_scores),
                'total_evaluations': len(overall_scores),
                'passing_rate': len([s for s in overall_scores if s >= 0.7]) / len(overall_scores)
            }
        
        return analysis
    
    def _calculate_std(self, scores: List[float]) -> float:
        """標準偏差の計算"""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
        return variance ** 0.5
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """スコア分布の取得"""
        distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        for score in scores:
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _get_low_performing_queries(self, results: List[EvaluationResult], threshold: float = 0.7) -> List[Dict]:
        """低性能クエリの特定"""
        low_performing = []
        
        for result in results:
            if result.score is not None and result.score < threshold:
                low_performing.append({
                    'query': result.query,
                    'score': result.score,
                    'feedback': result.feedback
                })
        
        return low_performing[:5]  # 上位5件
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """改善提案の生成"""
        recommendations = []
        
        if 'overall' in analysis:
            overall_score = analysis['overall']['mean_score']
            
            if overall_score < 0.6:
                recommendations.extend([
                    "チャンクサイズを調整してコンテキストの精度を向上させる",
                    "より高性能な埋め込みモデルを使用する",
                    "メタデータ抽出を強化する"
                ])
            elif overall_score < 0.8:
                recommendations.extend([
                    "ハイブリッド検索の重み調整を行う",
                    "リランキングモデルを導入する",
                    "クエリ拡張手法を適用する"
                ])
        
        # 評価項目別の推奨事項
        if 'faithfulness' in analysis and analysis['faithfulness']['mean_score'] < 0.7:
            recommendations.append("回答の忠実性向上のため、コンテキスト圧縮を検討する")
        
        if 'relevancy' in analysis and analysis['relevancy']['mean_score'] < 0.7:
            recommendations.append("検索精度向上のため、セマンティック検索の調整を行う")
        
        if 'correctness' in analysis and analysis['correctness']['mean_score'] < 0.7:
            recommendations.append("回答精度向上のため、プロンプトエンジニアリングを改善する")
        
        return recommendations

class RAGOptimizationSuite:
    """RAG最適化統合システム"""
    
    def __init__(self):
        self.configurations = []
        self.results = []
        
    def add_configuration(self, name: str, config: RAGConfiguration):
        """設定の追加"""
        self.configurations.append((name, config))
    
    async def run_optimization_experiments(self, file_paths: List[str]) -> Dict[str, Any]:
        """最適化実験の実行"""
        
        results = {}
        
        for config_name, config in self.configurations:
            print(f"\n=== 実験: {config_name} ===")
            
            # RAGシステムの作成
            rag_system = ProductManualRAGSystem(config)
            
            # データの読み込み
            rag_system.load_product_documents(file_paths)
            
            # インデックスとクエリエンジンの作成
            rag_system.create_optimized_storage()
            rag_system.create_advanced_query_engine()
            
            # 評価の実行
            evaluator = ProductManualEvaluator(rag_system)
            eval_results = await evaluator.run_comprehensive_evaluation()
            
            results[config_name] = {
                'configuration': config,
                'evaluation_results': eval_results,
                'timestamp': datetime.now().isoformat()
            }
            
        return results
    
    def compare_configurations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """設定の比較分析"""
        
        comparison = {
            'best_overall': None,
            'best_faithfulness': None,
            'best_relevancy': None,
            'best_correctness': None,
            'configuration_ranking': []
        }
        
        config_scores = {}
        
        for config_name, result in results.items():
            analysis = result['evaluation_results']['analysis']
            
            if 'overall' in analysis:
                overall_score = analysis['overall']['mean_score']
                config_scores[config_name] = overall_score
                
                # 各項目での最高スコア記録
                if comparison['best_overall'] is None or overall_score > config_scores.get(comparison['best_overall'], 0):
                    comparison['best_overall'] = config_name
        
        # ランキング作成
        ranking = sorted(config_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['configuration_ranking'] = ranking
        
        return comparison

# 使用例とメイン実行部分
async def main():
    """メイン実行関数"""
    
    # 最適化スイートの作成
    optimizer = RAGOptimizationSuite()
    
    # 異なる設定の定義
    configs = [
        ("baseline", RAGConfiguration(
            chunk_size=512,
            use_hybrid_search=False,
            use_reranking=False,
            use_hyde=False
        )),
        ("hybrid_search", RAGConfiguration(
            chunk_size=512,
            use_hybrid_search=True,
            use_reranking=False,
            use_hyde=False
        )),
        ("full_optimization", RAGConfiguration(
            chunk_size=256,
            us