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