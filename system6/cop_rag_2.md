aoton0029: llama_indexでragの高精度化のためにすべきこと

GitHub Copilot: LlamaIndexでRAGシステムの高精度化を図るための主要な手法をまとめます：

## 1. データ前処理・インデックス最適化

### チャンクサイズとオーバーラップの調整
````python
from llama_index.core.node_parser import SentenceSplitter

# チャンクサイズとオーバーラップを調整
text_splitter = SentenceSplitter(
    chunk_size=512,  # 適切なサイズに調整
    chunk_overlap=50,  # オーバーラップを設定
    paragraph_separator="\n\n\n"
)
````

### 階層インデックス構造
````python
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.indices.composability import ComposableGraph

# 階層構造でインデックスを作成
summary_index = SummaryIndex.from_documents(documents)
vector_index = VectorStoreIndex.from_documents(documents)

# 複合グラフで組み合わせ
graph = ComposableGraph.from_indices(
    SummaryIndex,
    [summary_index, vector_index],
    index_summaries=["summary", "vector"]
)
````

## 2. 埋め込みモデルの最適化

### 高性能埋め込みモデルの使用
````python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 多言語対応の高性能モデル
embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)
````

## 3. リトリーバルの改善

### ハイブリッド検索
````python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# ベクトル検索とキーワード検索を組み合わせ
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=10)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=2  # クエリ拡張
)
````

### リランキング
````python
from llama_index.postprocessor.cohere_rerank import CohereRerank

# リランキングで精度向上
rerank = CohereRerank(api_key="your-api-key", top_n=3)

query_engine = index.as_query_engine(
    node_postprocessors=[rerank],
    similarity_top_k=10
)
````

## 4. プロンプトエンジニアリング

### カスタムプロンプトテンプレート
````python
from llama_index.core import PromptTemplate

custom_prompt = PromptTemplate(
    """
    以下の文脈情報を使用して質問に答えてください。
    文脈に含まれていない情報については「情報が不足しています」と答えてください。
    
    文脈情報:
    {context_str}
    
    質問: {query_str}
    
    回答は具体的で正確に、根拠を示して答えてください。
    """
)

query_engine = index.as_query_engine(
    text_qa_template=custom_prompt
)
````

## 5. メタデータフィルタリング

````python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# メタデータによるフィルタリング
filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="document_type", value="technical_manual"),
        ExactMatchFilter(key="language", value="japanese")
    ]
)

query_engine = index.as_query_engine(
    filters=filters,
    similarity_top_k=5
)
````

## 6. 評価とモニタリング

### 評価メトリクスの実装
````python
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator

# 関連性と忠実性を評価
relevancy_evaluator = RelevancyEvaluator()
faithfulness_evaluator = FaithfulnessEvaluator()

# 評価実行
relevancy_result = relevancy_evaluator.evaluate_response(
    query=query, response=response
)
faithfulness_result = faithfulness_evaluator.evaluate_response(
    query=query, response=response
)
````

## 7. キャッシュとパフォーマンス最適化

````python
from llama_index.core.storage.storage_context import StorageContext
from llama_index.storage.storage_context import SimpleStorageContext

# ストレージコンテキストでキャッシュ
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# インデックスの永続化
index.storage_context.persist(persist_dir="./storage")
````

これらの手法を組み合わせることで、RAGシステムの精度を大幅に向上させることができます。特に、ドメイン固有のデータに対しては、チャンクサイズの調整とカスタムプロンプトの最適化が効果的です。

aoton0029: 製品マニュアルをRAGするサンプルはどうなる

GitHub Copilot: 製品マニュアル専用のRAGシステムのサンプルコードを示します：

## 基本的な製品マニュアルRAGシステム

````python
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

class ProductManualRAG:
    def __init__(self, manual_directory: str):
        self.manual_directory = manual_directory
        self.index = None
        self.query_engine = None
        self._setup_models()
        self._create_index()
        
    def _setup_models(self):
        # 埋め込みモデルの設定（日本語対応）
        embed_model = HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-large"
        )
        
        # LLMの設定
        llm = OpenAI(model="gpt-4", temperature=0.1)
        
        Settings.embed_model = embed_model
        Settings.llm = llm
        
    def _create_index(self):
        # マニュアルファイルの読み込み
        reader = SimpleDirectoryReader(
            input_dir=self.manual_directory,
            file_extractor={
                ".pdf": "PDFReader",
                ".docx": "DocxReader", 
                ".txt": "SimpleReader"
            }
        )
        documents = reader.load_data()
        
        # メタデータの追加
        for doc in documents:
            filename = Path(doc.metadata.get('file_name', '')).stem
            doc.metadata.update({
                'document_type': 'product_manual',
                'manual_name': filename,
                'section': self._extract_section(doc.text[:200])
            })
        
        # チャンクサイズの最適化（マニュアル用）
        text_splitter = SentenceSplitter(
            chunk_size=800,  # マニュアルに適したサイズ
            chunk_overlap=100,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[。！？]"  # 日本語の文区切り
        )
        
        # インデックス作成
        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=[text_splitter]
        )
        
    def _extract_section(self, text: str) -> str:
        """テキストからセクション名を抽出"""
        section_keywords = [
            "仕様", "設定", "トラブルシューティング", "インストール",
            "使用方法", "メンテナンス", "安全上の注意", "概要"
        ]
        
        for keyword in section_keywords:
            if keyword in text:
                return keyword
        return "その他"
    
    def setup_query_engine(self):
        # マニュアル専用プロンプトテンプレート
        manual_prompt = PromptTemplate(
            """
            あなたは製品マニュアルの専門アシスタントです。
            以下のマニュアル情報を基に、正確で実用的な回答を提供してください。
            
            回答の際は以下のルールに従ってください：
            1. 手順がある場合は、番号付きリストで明確に示す
            2. 安全上の注意事項がある場合は必ず言及する
            3. 不確実な情報は「マニュアルに記載がありません」と明記する
            4. 関連するページや章がある場合は参照先を示す
            
            マニュアル情報:
            {context_str}
            
            質問: {query_str}
            
            回答:
            """
        )
        
        # リトリーバーの設定
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        
        # 後処理の設定
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.7  # 関連度の閾値
        )
        
        # クエリエンジンの作成
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            text_qa_template=manual_prompt
        )
        
    def query(self, question: str) -> str:
        """質問に対する回答を取得"""
        if not self.query_engine:
            self.setup_query_engine()
            
        response = self.query_engine.query(question)
        return str(response)
    
    def search_by_section(self, question: str, section: str) -> str:
        """セクション別検索"""
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="section", value=section)]
        )
        
        filtered_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3,
            filters=filters
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=filtered_retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )
        
        response = query_engine.query(question)
        return str(response)

# 使用例
def main():
    # RAGシステムの初期化
    manual_rag = ProductManualRAG("./product_manuals")
    
    # 基本的な質問応答
    questions = [
        "初期設定の手順を教えてください",
        "エラーコード E001 の対処法は？",
        "定期メンテナンスの方法は？",
        "製品の仕様を教えてください"
    ]
    
    print("=== 製品マニュアル RAG システム ===\n")
    
    for question in questions:
        print(f"質問: {question}")
        answer = manual_rag.query(question)
        print(f"回答: {answer}\n")
        print("-" * 50)
    
    # セクション別検索の例
    print("\n=== セクション別検索 ===")
    troubleshooting_answer = manual_rag.search_by_section(
        "画面が表示されない場合の対処法", "トラブルシューティング"
    )
    print(f"トラブルシューティング回答: {troubleshooting_answer}")

if __name__ == "__main__":
    main()
````

## 高度な製品マニュアルRAG（構造化データ対応）

````python
from typing import Dict, List, Optional
import re
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

class AdvancedManualRAG(ProductManualRAG):
    def __init__(self, manual_directory: str):
        super().__init__(manual_directory)
        self.structured_sections = {}
        
    def _process_structured_content(self, documents: List[Document]):
        """構造化されたマニュアルコンテンツの処理"""
        processed_docs = []
        
        for doc in documents:
            # セクションの抽出
            sections = self._extract_sections(doc.text)
            
            for section_title, content in sections.items():
                # 手順の抽出
                steps = self._extract_steps(content)
                
                # 警告・注意事項の抽出
                warnings = self._extract_warnings(content)
                
                # 新しいドキュメントの作成
                processed_doc = Document(
                    text=content,
                    metadata={
                        **doc.metadata,
                        'section_title': section_title,
                        'has_steps': len(steps) > 0,
                        'has_warnings': len(warnings) > 0,
                        'steps_count': len(steps),
                        'warnings': warnings
                    }
                )
                processed_docs.append(processed_doc)
                
        return processed_docs
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """マニュアルからセクションを抽出"""
        sections = {}
        
        # 見出しパターンの検出
        section_pattern = r'^(第?\d+[章節]?\.?\s*[^\n]+|[0-9]+\.[0-9]+\.?\s*[^\n]+|\d+\.\s*[^\n]+)'
        lines = text.split('\n')
        current_section = "概要"
        current_content = []
        
        for line in lines:
            if re.match(section_pattern, line.strip()):
                # 前のセクションを保存
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 最後のセクションを保存
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def _extract_steps(self, text: str) -> List[str]:
        """手順の抽出"""
        step_patterns = [
            r'^\d+\.\s+(.+)$',  # 1. ステップ
            r'^手順\s*\d+[:：]\s*(.+)$',  # 手順1: ステップ
            r'^ステップ\s*\d+[:：]\s*(.+)$'  # ステップ1: 内容
        ]
        
        steps = []
        lines = text.split('\n')
        
        for line in lines:
            for pattern in step_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    steps.append(match.group(1))
                    break
                    
        return steps
    
    def _extract_warnings(self, text: str) -> List[str]:
        """警告・注意事項の抽出"""
        warning_patterns = [
            r'(?:警告|注意|危険)[:：]\s*(.+)',
            r'※\s*(.+)',
            r'⚠\s*(.+)'
        ]
        
        warnings = []
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            warnings.extend(matches)
            
        return warnings
    
    def get_troubleshooting_guide(self, error_code: str = None, symptom: str = None) -> str:
        """トラブルシューティングガイドの専用検索"""
        query_parts = []
        
        if error_code:
            query_parts.append(f"エラーコード {error_code}")
        if symptom:
            query_parts.append(symptom)
            
        query = " ".join(query_parts) if query_parts else "トラブルシューティング"
        
        # トラブルシューティングセクションに特化した検索
        return self.search_by_section(query, "トラブルシューティング")
    
    def get_installation_steps(self, product_type: str = None) -> str:
        """インストール手順の専用検索"""
        query = f"{product_type} インストール" if product_type else "インストール手順"
        return self.search_by_section(query, "インストール")

# 使用例とテスト
def demo_advanced_manual_rag():
    print("=== 高度な製品マニュアル RAG デモ ===\n")
    
    # システム初期化
    rag = AdvancedManualRAG("./product_manuals")
    
    # 様々なクエリタイプのテスト
    test_cases = [
        {
            "type": "基本質問",
            "query": "この製品の主な機能は何ですか？"
        },
        {
            "type": "トラブルシューティング",
            "query": "電源が入らない場合の対処法",
            "method": "get_troubleshooting_guide",
            "params": {"symptom": "電源が入らない"}
        },
        {
            "type": "インストール",
            "query": "ソフトウェアのインストール手順",
            "method": "get_installation_steps",
            "params": {"product_type": "ソフトウェア"}
        }
    ]
    
    for case in test_cases:
        print(f"【{case['type']}】")
        print(f"質問: {case['query']}")
        
        if case.get("method"):
            method = getattr(rag, case["method"])
            answer = method(**case.get("params", {}))
        else:
            answer = rag.query(case["query"])
            
        print(f"回答: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    demo_advanced_manual_rag()
````

## 設定ファイル例

````yaml
rag_settings:
  chunk_size: 800
  chunk_overlap: 100
  similarity_top_k: 5
  similarity_cutoff: 0.7

embedding:
  model_name: "intfloat/multilingual-e5-large"
  
llm:
  model: "gpt-4"
  temperature: 0.1

manual_processing:
  supported_formats: [".pdf", ".docx", ".txt", ".md"]
  section_keywords:
    - "仕様"
    - "設定" 
    - "トラブルシューティング"
    - "インストール"
    - "使用方法"
    - "メンテナンス"
    - "安全上の注意"
````

このサンプルでは、製品マニュアルに特化した以下の機能を実装しています：

1. **構造化データ処理**: セクション、手順、警告の自動抽出
2. **専用プロンプト**: マニュアル回答に最適化されたテンプレート
3. **セクション別検索**: トラブルシューティングやインストール手順の専用検索
4. **メタデータ活用**: 文書の種類や構造情報を活用した精密検索

製品マニュアルの特性（手順の多さ、安全情報、構造化された内容）に対応した実用的なRAGシステムになっています。

aoton0029: RAGの高精度化のために、チャンキング、インデクシングの設定と手法、クエリの設定と手法、を評価する仕組みにしたい。

GitHub Copilot: RAGシステムの高精度化を図るための包括的な評価システムを実装します：

## 評価フレームワークの基盤システム

````python
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.evaluation import (
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    CorrectnessEvaluator,
    ContextRelevancyEvaluator
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

@dataclass
class ChunkingConfig:
    """チャンキング設定"""
    strategy: str  # "sentence", "semantic", "fixed_size"
    chunk_size: int
    chunk_overlap: int
    separator: Optional[str] = None
    
@dataclass
class IndexingConfig:
    """インデクシング設定"""
    index_type: str  # "vector", "summary", "tree", "keyword"
    embedding_model: str
    similarity_top_k: int
    
@dataclass
class QueryConfig:
    """クエリ設定"""
    retrieval_strategy: str  # "vector", "bm25", "hybrid", "fusion"
    reranking: bool
    postprocessing: List[str]
    similarity_cutoff: float
    
@dataclass
class EvaluationResult:
    """評価結果"""
    config_id: str
    chunking_config: ChunkingConfig
    indexing_config: IndexingConfig
    query_config: QueryConfig
    relevancy_score: float
    faithfulness_score: float
    correctness_score: float
    context_relevancy_score: float
    retrieval_time: float
    query_time: float
    cost_estimate: float
    
class RAGEvaluationFramework:
    def __init__(self, documents: List[Document], test_queries: List[Dict[str, str]]):
        self.documents = documents
        self.test_queries = test_queries  # {"query": "質問", "expected_answer": "期待される回答"}
        self.evaluation_results = []
        
        # 評価器の初期化
        self.relevancy_evaluator = RelevancyEvaluator()
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.correctness_evaluator = CorrectnessEvaluator()
        self.context_relevancy_evaluator = ContextRelevancyEvaluator()
        
    def create_chunking_strategy(self, config: ChunkingConfig):
        """チャンキング戦略の作成"""
        if config.strategy == "sentence":
            return SentenceSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separator=config.separator or "\n\n"
            )
        elif config.strategy == "semantic":
            return SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95
            )
        else:  # fixed_size
            return SentenceSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
    
    def create_index(self, chunking_config: ChunkingConfig, indexing_config: IndexingConfig):
        """インデックスの作成"""
        # チャンキング
        node_parser = self.create_chunking_strategy(chunking_config)
        
        # インデックス作成
        start_time = time.time()
        
        if indexing_config.index_type == "vector":
            index = VectorStoreIndex.from_documents(
                self.documents,
                transformations=[node_parser]
            )
        # 他のインデックスタイプもここに追加可能
        
        indexing_time = time.time() - start_time
        return index, indexing_time
    
    def create_query_engine(self, index, query_config: QueryConfig):
        """クエリエンジンの作成"""
        if query_config.retrieval_strategy == "vector":
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=query_config.similarity_cut_score
            )
        elif query_config.retrieval_strategy == "hybrid":
            # ハイブリッド検索の実装
            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=query_config.similarity_top_k * 2
            )
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=index.docstore,
                similarity_top_k=query_config.similarity_top_k * 2
            )
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=query_config.similarity_top_k
            )
        else:
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=query_config.similarity_top_k
            )
        
        # 後処理の設定
        postprocessors = []
        if query_config.similarity_cutoff > 0:
            postprocessors.append(
                SimilarityPostprocessor(similarity_cutoff=query_config.similarity_cutoff)
            )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=postprocessors
        )
    
    def evaluate_configuration(
        self, 
        chunking_config: ChunkingConfig,
        indexing_config: IndexingConfig,
        query_config: QueryConfig,
        config_id: str
    ) -> EvaluationResult:
        """設定の評価"""
        print(f"評価中: {config_id}")
        
        # インデックス作成
        index, indexing_time = self.create_index(chunking_config, indexing_config)
        
        # クエリエンジン作成
        query_engine = self.create_query_engine(index, query_config)
        
        # 評価メトリクス
        relevancy_scores = []
        faithfulness_scores = []
        correctness_scores = []
        context_relevancy_scores = []
        query_times = []
        
        for test_case in self.test_queries:
            query = test_case["query"]
            expected_answer = test_case.get("expected_answer", "")
            
            # クエリ実行時間測定
            start_time = time.time()
            response = query_engine.query(query)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # 各種評価
            try:
                relevancy_result = self.relevancy_evaluator.evaluate_response(
                    query=query, response=response
                )
                relevancy_scores.append(relevancy_result.score)
                
                faithfulness_result = self.faithfulness_evaluator.evaluate_response(
                    query=query, response=response
                )
                faithfulness_scores.append(faithfulness_result.score)
                
                if expected_answer:
                    correctness_result = self.correctness_evaluator.evaluate_response(
                        query=query, response=response, reference=expected_answer
                    )
                    correctness_scores.append(correctness_result.score)
                
                context_relevancy_result = self.context_relevancy_evaluator.evaluate_response(
                    query=query, response=response
                )
                context_relevancy_scores.append(context_relevancy_result.score)
                
            except Exception as e:
                print(f"評価エラー: {e}")
                relevancy_scores.append(0.0)
                faithfulness_scores.append(0.0)
                correctness_scores.append(0.0)
                context_relevancy_scores.append(0.0)
        
        # 平均スコア計算
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        avg_context_relevancy = sum(context_relevancy_scores) / len(context_relevancy_scores) if context_relevancy_scores else 0.0
        avg_query_time = sum(query_times) / len(query_times) if query_times else 0.0
        
        return EvaluationResult(
            config_id=config_id,
            chunking_config=chunking_config,
            indexing_config=indexing_config,
            query_config=query_config,
            relevancy_score=avg_relevancy,
            faithfulness_score=avg_faithfulness,
            correctness_score=avg_correctness,
            context_relevancy_score=avg_context_relevancy,
            retrieval_time=indexing_time,
            query_time=avg_query_time,
            cost_estimate=self._estimate_cost(chunking_config, indexing_config, query_config)
        )
    
    def _estimate_cost(self, chunking_config: ChunkingConfig, indexing_config: IndexingConfig, query_config: QueryConfig) -> float:
        """コスト推定（簡易版）"""
        base_cost = 1.0
        
        # チャンクサイズによるコスト影響
        chunk_factor = chunking_config.chunk_size / 512  # 512を基準
        
        # 検索戦略によるコスト影響
        strategy_factor = 1.5 if query_config.retrieval_strategy == "hybrid" else 1.0
        
        # 後処理によるコスト影響
        postprocess_factor = 1.2 if query_config.reranking else 1.0
        
        return base_cost * chunk_factor * strategy_factor * postprocess_factor
    
    def run_comprehensive_evaluation(self) -> List[EvaluationResult]:
        """包括的評価の実行"""
        print("包括的評価を開始します...")
        
        # 評価する設定パターンの定義
        chunking_configs = [
            ChunkingConfig("sentence", 256, 50),
            ChunkingConfig("sentence", 512, 100),
            ChunkingConfig("sentence", 1024, 200),
            ChunkingConfig("semantic", 0, 0),  # セマンティック分割はサイズ不要
        ]
        
        indexing_configs = [
            IndexingConfig("vector", "text-embedding-ada-002", 5),
            IndexingConfig("vector", "text-embedding-ada-002", 10),
            IndexingConfig("vector", "text-embedding-ada-002", 15),
        ]
        
        query_configs = [
            QueryConfig("vector", False, [], 0.7),
            QueryConfig("vector", True, ["similarity"], 0.8),
            QueryConfig("hybrid", False, [], 0.7),
            QueryConfig("hybrid", True, ["similarity"], 0.8),
        ]
        
        # 全組み合わせの評価
        config_id = 0
        for chunking_config in chunking_configs:
            for indexing_config in indexing_configs:
                for query_config in query_configs:
                    config_id += 1
                    
                    try:
                        result = self.evaluate_configuration(
                            chunking_config,
                            indexing_config, 
                            query_config,
                            f"config_{config_id:03d}"
                        )
                        self.evaluation_results.append(result)
                        
                    except Exception as e:
                        print(f"設定 {config_id} の評価でエラー: {e}")
        
        return self.evaluation_results
    
    def get_best_configuration(self, metric: str = "overall") -> EvaluationResult:
        """最適設定の取得"""
        if not self.evaluation_results:
            raise ValueError("評価結果がありません。先に評価を実行してください。")
        
        if metric == "overall":
            # 総合スコア（重み付き平均）
            def overall_score(result: EvaluationResult) -> float:
                return (
                    result.relevancy_score * 0.3 +
                    result.faithfulness_score * 0.3 +
                    result.correctness_score * 0.2 +
                    result.context_relevancy_score * 0.2
                )
            return max(self.evaluation_results, key=overall_score)
        
        elif metric == "relevancy":
            return max(self.evaluation_results, key=lambda x: x.relevancy_score)
        elif metric == "faithfulness":
            return max(self.evaluation_results, key=lambda x: x.faithfulness_score)
        elif metric == "speed":
            return min(self.evaluation_results, key=lambda x: x.query_time)
        elif metric == "cost":
            return min(self.evaluation_results, key=lambda x: x.cost_estimate)
        else:
            raise ValueError(f"未対応のメトリック: {metric}")
    
    def save_results(self, filepath: str):
        """結果の保存"""
        results_data = []
        for result in self.evaluation_results:
            result_dict = asdict(result)
            results_data.append(result_dict)
        
        df = pd.DataFrame(results_data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"評価結果を保存しました: {filepath}")
    
    def generate_report(self) -> str:
        """評価レポートの生成"""
        if not self.evaluation_results:
            return "評価結果がありません。"
        
        best_overall = self.get_best_configuration("overall")
        best_speed = self.get_best_configuration("speed")
        best_cost = self.get_best_configuration("cost")
        
        report = f"""
=== RAG システム評価レポート ===

総評価設定数: {len(self.evaluation_results)}

【総合最優秀設定】
設定ID: {best_overall.config_id}
- チャンクサイズ: {best_overall.chunking_config.chunk_size}
- チャンク戦略: {best_overall.chunking_config.strategy}
- 検索戦略: {best_overall.query_config.retrieval_strategy}
- 関連性スコア: {best_overall.relevancy_score:.3f}
- 忠実性スコア: {best_overall.faithfulness_score:.3f}
- 正確性スコア: {best_overall.correctness_score:.3f}

【最高速設定】
設定ID: {best_speed.config_id}
- 平均クエリ時間: {best_speed.query_time:.3f}秒
- チャンクサイズ: {best_speed.chunking_config.chunk_size}

【最低コスト設定】
設定ID: {best_cost.config_id}
- 推定コスト: {best_cost.cost_estimate:.3f}
- チャンクサイズ: {best_cost.chunking_config.chunk_size}

【推奨事項】
1. 精度重視の場合: {best_overall.config_id} を使用
2. 速度重視の場合: {best_speed.config_id} を使用
3. コスト重視の場合: {best_cost.config_id} を使用
        """
        
        return report
````

## 実際の使用例とテストケース

````python
from rag_evaluation_framework import RAGEvaluationFramework, ChunkingConfig, IndexingConfig, QueryConfig
from llama_index.core import Document

def create_test_documents():
    """テスト用ドキュメントの作成"""
    documents = [
        Document(
            text="""
            製品Aの初期設定手順
            1. 電源コードを接続してください
            2. 電源ボタンを押してください
            3. 初期設定画面が表示されます
            4. 言語設定を日本語に変更してください
            5. ネットワーク設定を行ってください
            
            注意: 初期設定中は電源を切らないでください
            """,
            metadata={"source": "setup_manual.pdf", "section": "初期設定"}
        ),
        Document(
            text="""
            トラブルシューティング
            
            Q: 電源が入らない場合
            A: 以下の手順を確認してください
            1. 電源コードが正しく接続されているか確認
            2. 電源ボタンを5秒間長押し
            3. それでも起動しない場合はサポートにお問い合わせください
            
            Q: 画面が表示されない場合  
            A: HDMIケーブルの接続を確認してください
            """,
            metadata={"source": "troubleshooting.pdf", "section": "トラブルシューティング"}
        ),
        Document(
            text="""
            製品仕様
            - CPU: Intel Core i7
            - メモリ: 16GB RAM
            - ストレージ: 512GB SSD
            - OS: Windows 11
            - 保証期間: 3年間
            - サポート: 24時間365日
            """,
            metadata={"source": "specs.pdf", "section": "仕様"}
        )
    ]
    return documents

def create_test_queries():
    """テスト用クエリの作成"""
    return [
        {
            "query": "初期設定の手順を教えてください",
            "expected_answer": "電源コードを接続し、電源ボタンを押して、言語設定とネットワーク設定を行います"
        },
        {
            "query": "電源が入らない時の対処法は？",
            "expected_answer": "電源コードの接続を確認し、電源ボタンを5秒間長押ししてください"
        },
        {
            "query": "この製品のメモリ容量は？",
            "expected_answer": "16GB RAMです"
        },
        {
            "query": "保証期間はどれくらいですか？",
            "expected_answer": "3年間です"
        }
    ]

def main():
    print("=== RAG評価システム デモ ===\n")
    
    # テストデータの準備
    documents = create_test_documents()
    test_queries = create_test_queries()
    
    # 評価フレームワークの初期化
    evaluator = RAGEvaluationFramework(documents, test_queries)
    
    # 包括的評価の実行
    results = evaluator.run_comprehensive_evaluation()
    
    # 結果の保存
    evaluator.save_results("rag_evaluation_results.csv")
    
    # レポート生成
    report = evaluator.generate_report()
    print(report)
    
    # 詳細な結果表示
    print("\n=== 詳細結果 ===")
    for result in results[:5]:  # 上位5つの結果を表示
        print(f"\n設定ID: {result.config_id}")
        print(f"チャンク設定: {result.chunking_config.strategy}, サイズ={result.chunking_config.chunk_size}")
        print(f"検索戦略: {result.query_config.retrieval_strategy}")
        print(f"関連性: {result.relevancy_score:.3f}")
        print(f"忠実性: {result.faithfulness_score:.3f}")
        print(f"クエリ時間: {result.query_time:.3f}秒")

if __name__ == "__main__":
    main()
````

## 評価結果の可視化

````python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
from rag_evaluation_framework import EvaluationResult

class EvaluationVisualizer:
    def __init__(self, results: List[EvaluationResult]):
        self.results = results
        self.df = pd.DataFrame([vars(result) for result in results])
        
    def plot_performance_comparison(self):
        """性能比較グラフ"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 関連性スコア
        axes[0, 0].bar(range(len(self.results)), 
                      [r.relevancy_score for r in self.results])
        axes[0, 0].set_title('関連性スコア')
        axes[0, 0].set_xlabel('設定ID')
        
        # 忠実性スコア
        axes[0, 1].bar(range(len(self.results)), 
                      [r.faithfulness_score for r in self.results])
        axes[0, 1].set_title('忠実性スコア')
        axes[0, 1].set_xlabel('設定ID')
        
        # クエリ時間
        axes[1, 0].bar(range(len(self.results)), 
                      [r.query_time for r in self.results])
        axes[1, 0].set_title('クエリ時間 (秒)')
        axes[1, 0].set_xlabel('設定ID')
        
        # コスト推定
        axes[1, 1].bar(range(len(self.results)), 
                      [r.cost_estimate for r in self.results])
        axes[1, 1].set_title('コスト推定')
        axes[1, 1].set_xlabel('設定ID')
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self):
        """相関行列の可視化"""
        metrics = ['relevancy_score', 'faithfulness_score', 'correctness_score', 
                  'context_relevancy_score', 'query_time', 'cost_estimate']
        
        correlation_data = self.df[metrics].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
        plt.title('評価メトリクス相関行列')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
````

この評価フレームワークにより、RAGシステムの以下の要素を体系的に評価できます：

1. **チャンキング**: サイズ、戦略、オーバーラップの最適化
2. **インデクシング**: 埋め込みモデル、検索パラメータの調整
3. **クエリ処理**: 検索戦略、後処理、リランキングの効果測定
4. **総合評価**: 精度、速度、コストのバランス評価

結果はCSVファイルに保存され、可視化も可能で、実運用における最適な設定選択を支援します。