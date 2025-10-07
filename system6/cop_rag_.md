aoton0029: llama_index、langchainを使ったragの精度向上のためにすべきこと

GitHub Copilot: RAGの精度向上のために取り組むべき主要なポイントをまとめました：

## 1. データ前処理の改善

````python
# 文書の適切なチャンキング
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# チャンクサイズとオーバーラップの最適化
node_parser = SentenceSplitter(
    chunk_size=512,  # トークン数を調整
    chunk_overlap=50,  # オーバーラップを設定
    separator=" "
)
````

## 2. エンベッディングモデルの選択と最適化

````python
# より良いエンベッディングモデルの使用
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 日本語に特化したモデル
embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)
````

## 3. 検索戦略の改善

````python
# ハイブリッド検索の実装
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# ベクトル検索とキーワード検索の組み合わせ
vector_retriever = VectorIndexRetriever(index=vector_index)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=3
)
````

## 4. リランキングの実装

````python
# コサイン類似度によるリランキング
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.sentence_transformer_rerank import SentenceTransformerRerank

postprocessor = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=3
)
````

## 5. プロンプトエンジニアリング

````python
# カスタムプロンプトテンプレートの作成
from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    """コンテキスト情報は以下の通りです：
    {context_str}
    
    この情報を基に、以下の質問に正確に答えてください。
    情報が不足している場合は、「提供された情報では回答できません」と答えてください。
    
    質問: {query_str}
    回答: """
)
````

## 6. メタデータフィルタリング

````python
# メタデータによる検索の絞り込み
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="document_type", value="technical"),
        ExactMatchFilter(key="language", value="japanese")
    ]
)
````

## 7. 評価とモニタリング

````python
# RAGの評価指標の実装
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()

# 評価の実行
faithfulness_result = faithfulness_evaluator.evaluate_response(
    query=query, response=response
)
````

## 8. 知識グラフの活用

````python
# 知識グラフインデックスの作成
from llama_index.core import KnowledgeGraphIndex

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    include_embeddings=True
)
````

これらの手法を組み合わせることで、RAGシステムの精度を大幅に向上させることができます。特に日本語文書の場合は、適切なエンベッディングモデルの選択とチャンキング戦略が重要です。

aoton0029: 例えば、製品説明書をRAGするサンプルはどうなる

GitHub Copilot: 製品説明書をRAGするための具体的なサンプルコードを示します：

## 完全なRAGシステムサンプル

````python
import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

class ProductManualRAG:
    def __init__(self, manual_directory: str):
        self.manual_directory = manual_directory
        self.setup_models()
        self.index = None
        
    def setup_models(self):
        """モデルとパーサーの設定"""
        # LLMの設定
        Settings.llm = OpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=1000
        )
        
        # エンベッディングモデル（日本語対応）
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        # チャンクサイズの最適化（製品説明書用）
        Settings.node_parser = SentenceSplitter(
            chunk_size=300,  # 説明書の段落サイズに適したサイズ
            chunk_overlap=50,
            separator="\n\n"
        )
    
    def load_and_process_documents(self):
        """製品説明書の読み込みと処理"""
        # PDF、Word、テキストファイルを読み込み
        reader = SimpleDirectoryReader(
            input_dir=self.manual_directory,
            file_extractor={
                ".pdf": "PyPDFReader",
                ".docx": "DocxReader",
                ".txt": "SimpleDirectoryReader"
            }
        )
        
        documents = reader.load_data()
        
        # メタデータの追加
        for doc in documents:
            filename = os.path.basename(doc.metadata.get('file_name', ''))
            doc.metadata.update({
                'document_type': 'product_manual',
                'product_name': self.extract_product_name(filename),
                'section': self.extract_section(doc.text[:200])
            })
        
        return documents
    
    def extract_product_name(self, filename: str) -> str:
        """ファイル名から製品名を抽出"""
        # 例: "washing_machine_manual.pdf" -> "washing_machine"
        return filename.split('_manual')[0].replace('_', ' ')
    
    def extract_section(self, text: str) -> str:
        """テキストからセクションを推定"""
        sections = {
            '安全': ['安全', '注意', '警告', '危険'],
            '設置': ['設置', 'インストール', '配置'],
            '操作': ['操作', '使用方法', '使い方'],
            'メンテナンス': ['メンテナンス', '清掃', 'お手入れ'],
            'トラブルシューティング': ['故障', 'トラブル', '問題', 'エラー']
        }
        
        text_lower = text.lower()
        for section, keywords in sections.items():
            if any(keyword in text_lower for keyword in keywords):
                return section
        return 'その他'
    
    def create_index(self):
        """インデックスの作成"""
        documents = self.load_and_process_documents()
        self.index = VectorStoreIndex.from_documents(documents)
        logging.info(f"インデックス作成完了: {len(documents)}個の文書を処理")
    
    def create_custom_prompt(self):
        """製品説明書専用のプロンプトテンプレート"""
        return PromptTemplate(
            """あなたは製品サポートの専門家です。以下の製品説明書の情報を基に、ユーザーの質問に正確かつ分かりやすく回答してください。

コンテキスト情報:
{context_str}

回答ガイドライン:
1. 安全に関する情報は必ず強調してください
2. 手順がある場合は、番号付きで整理してください  
3. 該当する製品名やモデル名があれば明記してください
4. 情報が不十分な場合は、「説明書に記載がありません」と伝えてください
5. 追加で確認すべき点があれば提案してください

質問: {query_str}

回答:"""
        )
    
    def setup_query_engine(self):
        """クエリエンジンの設定"""
        if not self.index:
            raise ValueError("インデックスが作成されていません")
        
        # 検索設定
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        
        # 後処理（類似度によるフィルタリング）
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.7
        )
        
        # クエリエンジンの作成
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            text_qa_template=self.create_custom_prompt()
        )
        
        return query_engine
    
    def query(self, question: str):
        """質問に対する回答"""
        if not self.index:
            self.create_index()
        
        query_engine = self.setup_query_engine()
        response = query_engine.query(question)
        
        return {
            'answer': str(response),
            'source_nodes': [
                {
                    'text': node.text[:200] + '...',
                    'metadata': node.metadata,
                    'score': node.score if hasattr(node, 'score') else None
                }
                for node in response.source_nodes
            ]
        }

# 使用例
def main():
    # 環境変数設定
    os.environ['OPENAI_API_KEY'] = 'your-api-key'
    
    # RAGシステムの初期化
    rag_system = ProductManualRAG("./product_manuals/")
    
    # 質問例
    questions = [
        "洗濯機の設置方法を教えてください",
        "エラーコードE01の対処法は何ですか？",
        "フィルターの清掃頻度はどのくらいですか？",
        "安全上の注意点を教えてください"
    ]
    
    for question in questions:
        print(f"\n質問: {question}")
        print("-" * 50)
        
        result = rag_system.query(question)
        print(f"回答: {result['answer']}")
        
        print("\n参照元:")
        for i, source in enumerate(result['source_nodes'], 1):
            print(f"{i}. {source['metadata'].get('product_name', 'Unknown')}")
            print(f"   セクション: {source['metadata'].get('section', 'Unknown')}")
            print(f"   内容: {source['text']}")

if __name__ == "__main__":
    main()
````

## 設定ファイル例

````yaml
rag_settings:
  chunk_size: 300
  chunk_overlap: 50
  similarity_cutoff: 0.7
  top_k: 5

models:
  llm:
    name: "gpt-4-turbo-preview"
    temperature: 0.1
    max_tokens: 1000
  embedding:
    name: "text-embedding-3-large"
    dimensions: 1024

file_types:
  supported: [".pdf", ".docx", ".txt", ".md"]
  
product_categories:
  - "washing_machine"
  - "refrigerator"
  - "air_conditioner"
  - "microwave"
````

## 評価スクリプト

````python
from typing import List, Dict
import json

class RAGEvaluator:
    def __init__(self, rag_system: ProductManualRAG):
        self.rag_system = rag_system
        
    def evaluate_test_cases(self, test_cases: List[Dict]):
        """テストケースによる評価"""
        results = []
        
        for case in test_cases:
            question = case['question']
            expected_keywords = case['expected_keywords']
            
            result = self.rag_system.query(question)
            answer = result['answer'].lower()
            
            # キーワード一致率
            matched_keywords = sum(1 for kw in expected_keywords if kw.lower() in answer)
            keyword_score = matched_keywords / len(expected_keywords)
            
            results.append({
                'question': question,
                'answer': result['answer'],
                'keyword_score': keyword_score,
                'source_count': len(result['source_nodes'])
            })
        
        return results

# テストケース例
test_cases = [
    {
        'question': '洗濯機の水漏れの原因は？',
        'expected_keywords': ['ホース', '接続', '排水', '点検']
    },
    {
        'question': '冷蔵庫の温度設定方法は？',
        'expected_keywords': ['温度', '設定', 'ボタン', '操作パネル']
    }
]
````

このサンプルでは、製品説明書に特化した以下の機能を実装しています：

1. **製品別メタデータ管理**
2. **セクション自動分類**
3. **安全情報の強調**
4. **手順の構造化出力**
5. **評価・テスト機能**

製品説明書の特性に合わせてチャンクサイズやプロンプトを最適化しているのがポイントです。

aoton0029: より高度な製品マニュアル特価機能

GitHub Copilot: より高度な製品マニュアル特化機能を実装したサンプルを示します：

## 高度な製品マニュアルRAGシステム

````python
import os
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    PromptTemplate,
    KnowledgeGraphIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KGTableRetriever,
    QueryFusionRetriever
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
    MetadataReplacementPostProcessor
)
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore
import spacy

class QueryType(Enum):
    """クエリタイプの分類"""
    TROUBLESHOOTING = "troubleshooting"
    OPERATION = "operation"
    INSTALLATION = "installation"
    MAINTENANCE = "maintenance"
    SAFETY = "safety"
    SPECIFICATION = "specification"
    WARRANTY = "warranty"

@dataclass
class ProductContext:
    """製品コンテキスト情報"""
    product_name: str
    model_number: Optional[str]
    category: str
    version: str
    language: str

class AdvancedProductManualRAG:
    def __init__(self, manual_directory: str):
        self.manual_directory = manual_directory
        self.setup_models()
        self.vector_index = None
        self.kg_index = None
        self.product_contexts = {}
        self.load_nlp_model()
        self.setup_domain_knowledge()
        
    def load_nlp_model(self):
        """日本語NLPモデルの読み込み"""
        try:
            self.nlp = spacy.load("ja_core_news_sm")
        except OSError:
            print("日本語モデルが見つかりません。英語モデルを使用します。")
            self.nlp = spacy.load("en_core_web_sm")
    
    def setup_domain_knowledge(self):
        """製品ドメイン知識の設定"""
        self.error_code_patterns = {
            r'E\d{2,3}': 'error_code',
            r'エラー[コード]*\s*[：:]\s*[A-Z]\d+': 'error_code_jp',
            r'故障[コード]*\s*[：:]\s*\d+': 'failure_code'
        }
        
        self.safety_keywords = [
            '危険', '警告', '注意', '禁止', '感電', '火災', '爆発',
            'danger', 'warning', 'caution', 'prohibited'
        ]
        
        self.operation_keywords = [
            '操作', '使用方法', '手順', '設定', 'ボタン', 'スイッチ',
            'operation', 'usage', 'procedure', 'setting'
        ]
    
    def setup_models(self):
        """高度なモデル設定"""
        Settings.llm = OpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.05,  # より一貫した回答のため低温度
            max_tokens=1500
        )
        
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        # 階層的チャンキング
        Settings.node_parser = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=64,
            separator="\n\n"
        )
    
    def extract_product_info(self, text: str, filename: str) -> ProductContext:
        """高度な製品情報抽出"""
        # モデル番号の抽出
        model_patterns = [
            r'[Mm]odel[:\s]+([A-Z0-9\-]+)',
            r'型番[:\s]+([A-Z0-9\-]+)',
            r'品番[:\s]+([A-Z0-9\-]+)'
        ]
        
        model_number = None
        for pattern in model_patterns:
            match = re.search(pattern, text[:1000])
            if match:
                model_number = match.group(1)
                break
        
        # 製品カテゴリの判定
        category_keywords = {
            'washing_machine': ['洗濯', 'washing', '脱水'],
            'refrigerator': ['冷蔵', 'refrigerator', '冷凍'],
            'air_conditioner': ['エアコン', 'air conditioner', '冷暖房'],
            'microwave': ['電子レンジ', 'microwave', '加熱'],
            'dishwasher': ['食洗機', 'dishwasher', '食器洗い']
        }
        
        category = 'unknown'
        for cat, keywords in category_keywords.items():
            if any(keyword in text.lower() or keyword in filename.lower() 
                   for keyword in keywords):
                category = cat
                break
        
        return ProductContext(
            product_name=self.extract_product_name_advanced(filename, text),
            model_number=model_number,
            category=category,
            version=self.extract_version(text),
            language='ja' if any(c for c in text if ord(c) > 127) else 'en'
        )
    
    def extract_product_name_advanced(self, filename: str, text: str) -> str:
        """高度な製品名抽出"""
        # ファイル名から基本的な製品名を抽出
        base_name = filename.split('_manual')[0].replace('_', ' ')
        
        # テキストから正確な製品名を探す
        product_patterns = [
            r'製品名[:\s]+([^\n]+)',
            r'Product Name[:\s]+([^\n]+)',
            r'機種[:\s]+([^\n]+)'
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, text[:500])
            if match:
                return match.group(1).strip()
        
        return base_name
    
    def extract_version(self, text: str) -> str:
        """バージョン情報の抽出"""
        version_patterns = [
            r'Version[:\s]+([0-9\.]+)',
            r'版[:\s]+([0-9\.]+)',
            r'Ver[:\s]*([0-9\.]+)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text[:500])
            if match:
                return match.group(1)
        
        return "1.0"
    
    def classify_query_type(self, query: str) -> QueryType:
        """クエリタイプの自動分類"""
        query_lower = query.lower()
        
        # トラブルシューティング
        if any(word in query_lower for word in 
               ['エラー', 'error', '故障', '動かない', 'トラブル', '問題']):
            return QueryType.TROUBLESHOOTING
        
        # 安全関連
        if any(word in query_lower for word in self.safety_keywords):
            return QueryType.SAFETY
        
        # 設置・インストール
        if any(word in query_lower for word in 
               ['設置', '取り付け', 'install', '配置']):
            return QueryType.INSTALLATION
        
        # メンテナンス
        if any(word in query_lower for word in 
               ['メンテナンス', '清掃', 'お手入れ', 'maintenance']):
            return QueryType.MAINTENANCE
        
        # 仕様・スペック
        if any(word in query_lower for word in 
               ['仕様', 'スペック', 'specification', '寸法', 'サイズ']):
            return QueryType.SPECIFICATION
        
        # 保証・warranty
        if any(word in query_lower for word in 
               ['保証', 'warranty', '修理', 'repair']):
            return QueryType.WARRANTY
        
        return QueryType.OPERATION
    
    def create_advanced_nodes(self, documents):
        """高度なノード作成とメタデータ付与"""
        enhanced_nodes = []
        
        for doc in documents:
            filename = os.path.basename(doc.metadata.get('file_name', ''))
            product_context = self.extract_product_info(doc.text, filename)
            
            # 基本的なノード分割
            node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=64)
            nodes = node_parser.get_nodes_from_documents([doc])
            
            for node in nodes:
                # 高度なメタデータ付与
                node.metadata.update({
                    'product_name': product_context.product_name,
                    'model_number': product_context.model_number,
                    'category': product_context.category,
                    'version': product_context.version,
                    'language': product_context.language,
                    'section_type': self.classify_section(node.text),
                    'has_error_code': self.contains_error_code(node.text),
                    'safety_level': self.assess_safety_level(node.text),
                    'step_sequence': self.extract_step_sequence(node.text),
                    'created_at': datetime.now().isoformat()
                })
                
                enhanced_nodes.append(node)
        
        return enhanced_nodes
    
    def classify_section(self, text: str) -> str:
        """セクションタイプの詳細分類"""
        text_lower = text.lower()
        
        section_patterns = {
            'safety': ['安全', '危険', '警告', '注意', 'safety', 'warning'],
            'installation': ['設置', '取り付け', 'installation', 'setup'],
            'operation': ['操作', '使用', 'operation', 'usage', '手順'],
            'maintenance': ['メンテナンス', '清掃', 'maintenance', 'cleaning'],
            'troubleshooting': ['故障', 'トラブル', 'エラー', 'troubleshooting', 'error'],
            'specification': ['仕様', 'スペック', 'specification', '寸法'],
            'warranty': ['保証', 'warranty', '修理', 'repair']
        }
        
        for section, keywords in section_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return section
        
        return 'general'
    
    def contains_error_code(self, text: str) -> bool:
        """エラーコードの有無をチェック"""
        for pattern in self.error_code_patterns.keys():
            if re.search(pattern, text):
                return True
        return False
    
    def assess_safety_level(self, text: str) -> int:
        """安全レベルの評価（0-3）"""
        text_lower = text.lower()
        
        danger_words = ['危険', 'danger', '感電', '火災']
        warning_words = ['警告', 'warning', '注意', 'caution']
        
        if any(word in text_lower for word in danger_words):
            return 3  # 危険
        elif any(word in text_lower for word in warning_words):
            return 2  # 警告
        elif any(word in text_lower for word in self.safety_keywords):
            return 1  # 注意
        
        return 0  # 通常
    
    def extract_step_sequence(self, text: str) -> Optional[List[int]]:
        """手順番号の抽出"""
        # 番号付きリストを検出
        step_patterns = [
            r'(\d+)[.\)]\s+',
            r'手順\s*(\d+)',
            r'ステップ\s*(\d+)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text)
            steps.extend([int(match) for match in matches])
        
        return sorted(list(set(steps))) if steps else None
    
    def create_indexes(self):
        """複数のインデックス作成"""
        documents = SimpleDirectoryReader(self.manual_directory).load_data()
        enhanced_nodes = self.create_advanced_nodes(documents)
        
        # ベクトルインデックス
        self.vector_index = VectorStoreIndex(enhanced_nodes)
        
        # 知識グラフインデックス
        self.kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=3,
            include_embeddings=True
        )
        
        # BM25インデックス用のノード準備
        self.bm25_nodes = enhanced_nodes
        
        print(f"インデックス作成完了: {len(enhanced_nodes)}個のノードを処理")
    
    def create_hybrid_retriever(self, query_type: QueryType, filters: Optional[MetadataFilters] = None):
        """クエリタイプに応じたハイブリッド検索"""
        # ベクトル検索
        vector_retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=8,
            filters=filters
        )
        
        # BM25検索
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.bm25_nodes,
            similarity_top_k=5
        )
        
        # 知識グラフ検索（構造化情報が重要な場合）
        kg_retriever = KGTableRetriever(
            index=self.kg_index,
            similarity_top_k=3
        )
        
        # クエリタイプに応じて検索手法を組み合わせ
        if query_type == QueryType.TROUBLESHOOTING:
            # トラブルシューティングではキーワード検索を重視
            return QueryFusionRetriever(
                [bm25_retriever, vector_retriever],
                similarity_top_k=6,
                num_queries=2
            )
        elif query_type == QueryType.OPERATION:
            # 操作手順では知識グラフも活用
            return QueryFusionRetriever(
                [vector_retriever, kg_retriever, bm25_retriever],
                similarity_top_k=8,
                num_queries=3
            )
        else:
            # 一般的なケースではベクトル検索を中心に
            return QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=6,
                num_queries=2
            )
    
    def create_advanced_postprocessors(self, query_type: QueryType) -> List:
        """高度な後処理パイプライン"""
        postprocessors = []
        
        # 基本的な類似度フィルタ
        postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.6))
        
        # クエリタイプ別の特別処理
        if query_type == QueryType.SAFETY:
            # 安全関連では高い安全レベルを優先
            postprocessors.append(SafetyPriorityPostprocessor())
        elif query_type == QueryType.TROUBLESHOOTING:
            # トラブルシューティングではエラーコード含有を優先
            postprocessors.append(ErrorCodePriorityPostprocessor())
        
        return postprocessors
    
    def create_context_aware_prompt(self, query_type: QueryType, product_context: Optional[ProductContext] = None):
        """コンテキスト認識プロンプト"""
        base_context = """あなたは製品マニュアルの専門アシスタントです。以下の情報を基に、正確で実用的な回答を提供してください。

コンテキスト情報:
{context_str}"""
        
        if product_context:
            base_context += f"""
            
製品情報:
- 製品名: {product_context.product_name}
- モデル番号: {product_context.model_number or '不明'}
- カテゴリ: {product_context.category}
- バージョン: {product_context.version}"""
        
        # クエリタイプ別の指示
        type_specific_instructions = {
            QueryType.TROUBLESHOOTING: """
トラブルシューティング回答の指針:
1. 症状を明確に確認してください
2. 考えられる原因を優先度順に列挙してください
3. 具体的な対処手順を番号付きで説明してください
4. 安全上の注意点があれば必ず言及してください
5. 解決しない場合の次のステップを提示してください""",
            
            QueryType.SAFETY: """
安全情報回答の指針:
1. 安全上の重要事項は必ず最初に記載してください
2. 危険度レベルを明確に示してください
3. 具体的な予防措置を説明してください
4. 緊急時の対応方法も含めてください""",
            
            QueryType.OPERATION: """
操作説明回答の指針:
1. 手順を明確な番号付きリストで示してください
2. 各ステップで確認すべき点を記載してください
3. 画面やボタンの名称は正確に記載してください
4. よくある間違いがあれば予防策も説明してください"""
        }
        
        instruction = type_specific_instructions.get(query_type, """
一般的な回答の指針:
1. 正確で分かりやすい説明を心がけてください
2. 必要に応じて手順を番号付きで整理してください
3. 関連する注意事項があれば含めてください""")
        
        return PromptTemplate(base_context + instruction + """

質問: {query_str}

回答:""")
    
    def query_with_context(self, question: str, product_filter: Optional[str] = None):
        """コンテキスト認識クエリ処理"""
        if not self.vector_index:
            self.create_indexes()
        
        # クエリタイプの分類
        query_type = self.classify_query_type(question)
        
        # フィルタの作成
        filters = None
        if product_filter:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="product_name", value=product_filter)]
            )
        
        # ハイブリッド検索の設定
        retriever = self.create_hybrid_retriever(query_type, filters)
        
        # 後処理パイプライン
        postprocessors = self.create_advanced_postprocessors(query_type)
        
        # プロンプトの作成
        product_context = self.product_contexts.get(product_filter) if product_filter else None
        prompt = self.create_context_aware_prompt(query_type, product_context)
        
        # クエリエンジンの作成
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            text_qa_template=prompt
        )
        
        # 実行と結果の拡張
        response = query_engine.query(question)
        
        return {
            'answer': str(response),
            'query_type': query_type.value,
            'confidence_score': self.calculate_confidence(response),
            'source_nodes': self.format_source_nodes(response.source_nodes),
            'suggested_followup': self.suggest_followup_questions(question, query_type)
        }

    def calculate_confidence(self, response) -> float:
        """回答の信頼度計算"""
        if not hasattr(response, 'source_nodes') or not response.source_nodes:
            return 0.0
        
        scores = []
        for node in response.source_nodes:
            if hasattr(node, 'score') and node.score is not None:
                scores.append(node.score)
        
        if not scores:
            return 0.5  # デフォルト値
        
        # 上位ノードの平均スコア
        top_scores = sorted(scores, reverse=True)[:3]
        return sum(top_scores) / len(top_scores)
    
    def format_source_nodes(self, source_nodes) -> List[Dict]:
        """ソースノードの整形"""
        formatted_nodes = []
        
        for i, node in enumerate(source_nodes, 1):
            formatted_nodes.append({
                'rank': i,
                'text_preview': node.text[:150] + '...' if len(node.text) > 150 else node.text,
                'metadata': {
                    'product_name': node.metadata.get('product_name', 'Unknown'),
                    'section_type': node.metadata.get('section_type', 'general'),
                    'safety_level': node.metadata.get('safety_level', 0),
                    'has_error_code': node.metadata.get('has_error_code', False)
                },
                'relevance_score': getattr(node, 'score', None)
            })
        
        return formatted_nodes
    
    def suggest_followup_questions(self, original_question: str, query_type: QueryType) -> List[str]:
        """フォローアップ質問の提案"""
        suggestions = []
        
        if query_type == QueryType.TROUBLESHOOTING:
            suggestions = [
                "この問題が解決しない場合の次のステップは？",
                "類似の症状で考えられる他の原因は？",
                "予防策はありますか？"
            ]
        elif query_type == QueryType.OPERATION:
            suggestions = [
                "この操作で注意すべき点は？",
                "操作がうまくいかない場合の対処法は？",
                "関連する設定項目はありますか？"
            ]
        elif query_type == QueryType.SAFETY:
            suggestions = [
                "緊急時の対応方法は？",
                "定期的な安全点検項目は？",
                "この安全対策が必要な理由は？"
            ]
        
        return suggestions

# カスタム後処理クラス
class SafetyPriorityPostprocessor:
    """安全レベルを考慮した後処理"""
    def postprocess_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        def safety_score(node):
            base_score = getattr(node, 'score', 0.5)
            safety_level = node.metadata.get('safety_level', 0)
            return base_score + (safety_level * 0.1)  # 安全レベルでブースト
        
        return sorted(nodes, key=safety_score, reverse=True)

class ErrorCodePriorityPostprocessor:
    """エラーコード含有を優先する後処理"""
    def postprocess_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        def error_code_score(node):
            base_score = getattr(node, 'score', 0.5)
            has_error_code = node.metadata.get('has_error_code', False)
            return base_score + (0.2 if has_error_code else 0)
        
        return sorted(nodes, key=error_code_score, reverse=True)

# 使用例
def main():
    os.environ['OPENAI_API_KEY'] = 'your-api-key'
    
    rag_system = AdvancedProductManualRAG("./product_manuals/")
    
    # 高度なクエリ例
    advanced_questions = [
        "エラーコードE02が表示されました。どう対処すればよいですか？",
        "洗濯機の設置時の安全上の注意点を教えてください",
        "型番ABC-123の操作パネルの使い方を詳しく説明してください",
        "フィルターの清掃手順と頻度について教えてください"
    ]
    
    for question in advanced_questions:
        print(f"\n{'='*60}")
        print(f"質問: {question}")
        print('='*60)
        
        result = rag_system.query_with_context(question)
        
        print(f"クエリタイプ: {result['query_type']}")
        print(f"信頼度: {result['confidence_score']:.2f}")
        print(f"\n回答:\n{result['answer']}")
        
        print(f"\n参照元 ({len(result['source_nodes'])}件):")
        for source in result['source_nodes'][:3]:  # 上位3件のみ表示
            print(f"  {source['rank']}. [{source['metadata']['section_type']}] "
                  f"{source['text_preview']}")
        
        if result['suggested_followup']:
            print(f"\n関連質問の提案:")
            for suggestion in result['suggested_followup']:
                print(f"  • {suggestion}")

if __name__ == "__main__":
    main()
````

## 評価・分析ダッシュボード

````python
import streamlit as st
import plotly.express as px
import pandas as pd
from collections import Counter

class RAGAnalyticsDashboard:
    def __init__(self, rag_system: AdvancedProductManualRAG):
        self.rag_system = rag_system
    
    def create_dashboard(self):
        st.title("🔧 製品マニュアルRAG分析ダッシュボード")
        
        # サイドバーでの製品選択
        st.sidebar.header("フィルター設定")
        selected_product = st.sidebar.selectbox(
            "製品を選択:",
            ["全て"] + list(self.rag_system.product_contexts.keys())
        )
        
        # メイン分析エリア
        self.show_query_analytics()
        self.show_performance_metrics()
        self.show_content_coverage()
    
    def show_query_analytics(self):
        st.header("📊 クエリ分析")
        
        # クエリタイプ分布の可視化
        query_types = [
            "troubleshooting", "operation", "installation", 
            "maintenance", "safety", "specification"
        ]
        type_counts = [50, 35, 20, 25, 15, 10]  # サンプルデータ
        
        fig = px.pie(
            values=type_counts, 
            names=query_types,
            title="クエリタイプ分布"
        )
        st.plotly_chart(fig)
    
    def show_performance_metrics(self):
        st.header("⚡ パフォーマンス指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("平均応答時間", "1.2秒", "▼0.3秒")
        with col2:
            st.metric("平均信頼度", "0.85", "▲0.05")
        with col3:
            st.metric("解決率", "92%", "▲3%")
        with col4:
            st.metric("ユーザー満足度", "4.6/5", "▲0.2")
````

この高度なシステムでは以下の特化機能を実装しています：

1. **インテリジェントクエリ分類**: 自動的にクエリタイプを判定
2. **ハイブリッド検索**: ベクトル、BM25、知識グラフを組み合わせ
3. **コンテキスト認識**: 製品情報を考慮した回答生成
4. **安全レベル評価**: 安全関連情報の優先度付け
5. **エラーコード検出**: トラブルシューティング用の特別処理
6. **信頼度スコアリング**: 回答の確信度を定量化
7. **フォローアップ提案**: 関連質問の自動生成
8. **分析ダッシュボード**: システムパフォーマンスの可視化

製品マニュアル特有のニーズに対応した、実用的で高精度なRAGシステムとなっています。

aoton0029: ragの精度向上のため、チャンキング、インデクシング及びデータ保存の設定と手法、クエリの設定と手法、を評価する仕組みにしたい。RAGの最適化を目指す

GitHub Copilot: RAGシステムの最適化を目的とした包括的な評価・調整システムを作成しました：

## RAG最適化フレームワーク

````python
import os
import json
import time
import itertools
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    PromptTemplate
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
    HierarchicalNodeParser
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator
)
from sklearn.metrics.pairwise import cosine_similarity

# 設定データクラス
@dataclass
class ChunkingConfig:
    """チャンキング設定"""
    method: str  # "sentence", "semantic", "token", "hierarchical"
    chunk_size: int
    chunk_overlap: int
    separator: str = "\n\n"
    
@dataclass
class EmbeddingConfig:
    """埋め込み設定"""
    model_name: str
    dimensions: Optional[int] = None
    
@dataclass
class IndexingConfig:
    """インデクシング設定"""
    similarity_top_k: int = 5
    similarity_cutoff: float = 0.0
    
@dataclass
class QueryConfig:
    """クエリ設定"""
    retrieval_method: str  # "vector", "bm25", "hybrid"
    llm_model: str
    temperature: float
    max_tokens: int
    
@dataclass
class RAGConfig:
    """RAG全体設定"""
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    indexing: IndexingConfig
    query: QueryConfig
    
@dataclass
class EvaluationMetrics:
    """評価指標"""
    faithfulness: float
    relevancy: float
    correctness: float
    semantic_similarity: float
    response_time: float
    retrieval_precision: float
    retrieval_recall: float
    answer_length: int
    
class RAGOptimizer:
    def __init__(self, documents_path: str, test_cases_path: str):
        self.documents_path = documents_path
        self.test_cases_path = test_cases_path
        self.documents = None
        self.test_cases = []
        self.evaluation_results = []
        self.best_config = None
        self.best_score = 0.0
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.load_test_cases()
        
    def load_test_cases(self):
        """テストケースの読み込み"""
        try:
            with open(self.test_cases_path, 'r', encoding='utf-8') as f:
                self.test_cases = json.load(f)
            self.logger.info(f"テストケース読み込み完了: {len(self.test_cases)}件")
        except FileNotFoundError:
            self.logger.warning("テストケースファイルが見つかりません。サンプルを作成します。")
            self.create_sample_test_cases()
    
    def create_sample_test_cases(self):
        """サンプルテストケースの作成"""
        self.test_cases = [
            {
                "question": "製品の設置方法を教えてください",
                "expected_answer": "設置手順は以下の通りです...",
                "expected_keywords": ["設置", "手順", "場所", "注意"],
                "category": "installation"
            },
            {
                "question": "エラーが発生した場合の対処法は？",
                "expected_answer": "エラーコードを確認し...",
                "expected_keywords": ["エラー", "対処", "確認", "コード"],
                "category": "troubleshooting"
            }
        ]
        
        # テストケースを保存
        with open(self.test_cases_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_cases, f, ensure_ascii=False, indent=2)
    
    def generate_config_variations(self) -> List[RAGConfig]:
        """設定のバリエーション生成"""
        # チャンキング設定のバリエーション
        chunking_configs = [
            ChunkingConfig("sentence", 256, 32),
            ChunkingConfig("sentence", 512, 64),
            ChunkingConfig("sentence", 1024, 128),
            ChunkingConfig("semantic", 256, 32),
            ChunkingConfig("token", 300, 50),
            ChunkingConfig("hierarchical", 512, 64)
        ]
        
        # 埋め込みモデルのバリエーション
        embedding_configs = [
            EmbeddingConfig("text-embedding-3-large", 1024),
            EmbeddingConfig("text-embedding-3-small", 1536),
            EmbeddingConfig("intfloat/multilingual-e5-large", None)
        ]
        
        # インデクシング設定のバリエーション
        indexing_configs = [
            IndexingConfig(similarity_top_k=3, similarity_cutoff=0.6),
            IndexingConfig(similarity_top_k=5, similarity_cutoff=0.7),
            IndexingConfig(similarity_top_k=8, similarity_cutoff=0.5)
        ]
        
        # クエリ設定のバリエーション
        query_configs = [
            QueryConfig("vector", "gpt-3.5-turbo", 0.1, 1000),
            QueryConfig("vector", "gpt-4-turbo-preview", 0.0, 1500),
            QueryConfig("hybrid", "gpt-4-turbo-preview", 0.1, 1200)
        ]
        
        # 全ての組み合わせを生成（計算量を考慮して制限）
        configs = []
        for chunking, embedding, indexing, query in itertools.product(
            chunking_configs[:3],  # 上位3つに制限
            embedding_configs[:2],  # 上位2つに制限
            indexing_configs[:2],   # 上位2つに制限
            query_configs[:2]       # 上位2つに制限
        ):
            configs.append(RAGConfig(chunking, embedding, indexing, query))
        
        return configs
    
    def create_node_parser(self, config: ChunkingConfig):
        """ノードパーサーの作成"""
        if config.method == "sentence":
            return SentenceSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separator=config.separator
            )
        elif config.method == "semantic":
            return SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95
            )
        elif config.method == "token":
            return TokenTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
        elif config.method == "hierarchical":
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=[config.chunk_size, config.chunk_size // 2]
            )
        else:
            raise ValueError(f"Unknown chunking method: {config.method}")
    
    def create_embedding_model(self, config: EmbeddingConfig):
        """埋め込みモデルの作成"""
        if "text-embedding" in config.model_name:
            return OpenAIEmbedding(
                model=config.model_name,
                dimensions=config.dimensions
            )
        else:
            return HuggingFaceEmbedding(
                model_name=config.model_name
            )
    
    def build_rag_system(self, config: RAGConfig):
        """RAGシステムの構築"""
        # ドキュメント読み込み
        if not self.documents:
            reader = SimpleDirectoryReader(self.documents_path)
            self.documents = reader.load_data()
        
        # 設定の適用
        Settings.node_parser = self.create_node_parser(config.chunking)
        Settings.embed_model = self.create_embedding_model(config.embedding)
        Settings.llm = OpenAI(
            model=config.query.llm_model,
            temperature=config.query.temperature,
            max_tokens=config.query.max_tokens
        )
        
        # インデックス作成
        index = VectorStoreIndex.from_documents(self.documents)
        
        # 検索エンジンの設定
        if config.query.retrieval_method == "vector":
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=config.indexing.similarity_top_k
            )
        elif config.query.retrieval_method == "hybrid":
            # ハイブリッド検索の実装
            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=config.indexing.similarity_top_k
            )
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=index.docstore.docs.values(),
                similarity_top_k=config.indexing.similarity_top_k
            )
            # 簡易的なハイブリッド実装
            retriever = vector_retriever  # 実際の実装では両方を組み合わせ
        
        # クエリエンジン作成
        query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
        
        return query_engine, index
    
    def evaluate_rag_system(self, query_engine, index, config: RAGConfig) -> EvaluationMetrics:
        """RAGシステムの評価"""
        # 評価器の設定
        faithfulness_evaluator = FaithfulnessEvaluator()
        relevancy_evaluator = RelevancyEvaluator()
        correctness_evaluator = CorrectnessEvaluator()
        semantic_evaluator = SemanticSimilarityEvaluator()
        
        metrics = {
            'faithfulness': [],
            'relevancy': [],
            'correctness': [],
            'semantic_similarity': [],
            'response_time': [],
            'answer_length': [],
            'retrieval_precision': [],
            'retrieval_recall': []
        }
        
        for test_case in self.test_cases:
            question = test_case['question']
            expected_answer = test_case.get('expected_answer', '')
            expected_keywords = test_case.get('expected_keywords', [])
            
            # 応答時間測定
            start_time = time.time()
            response = query_engine.query(question)
            response_time = time.time() - start_time
            
            answer = str(response)
            
            # 各評価指標の計算
            try:
                # Faithfulness評価
                faithfulness_result = faithfulness_evaluator.evaluate_response(
                    query=question, response=response
                )
                metrics['faithfulness'].append(faithfulness_result.score)
                
                # Relevancy評価
                relevancy_result = relevancy_evaluator.evaluate_response(
                    query=question, response=response
                )
                metrics['relevancy'].append(relevancy_result.score)
                
                # Correctness評価（期待回答がある場合）
                if expected_answer:
                    correctness_result = correctness_evaluator.evaluate(
                        query=question,
                        response=answer,
                        reference=expected_answer
                    )
                    metrics['correctness'].append(correctness_result.score)
                else:
                    metrics['correctness'].append(0.5)  # デフォルト値
                
                # 意味的類似度（期待回答がある場合）
                if expected_answer:
                    semantic_result = semantic_evaluator.evaluate(
                        query=question,
                        response=answer,
                        reference=expected_answer
                    )
                    metrics['semantic_similarity'].append(semantic_result.score)
                else:
                    metrics['semantic_similarity'].append(0.5)
                
            except Exception as e:
                self.logger.warning(f"評価エラー: {e}")
                # デフォルト値を設定
                metrics['faithfulness'].append(0.5)
                metrics['relevancy'].append(0.5)
                metrics['correctness'].append(0.5)
                metrics['semantic_similarity'].append(0.5)
            
            # その他のメトリクス
            metrics['response_time'].append(response_time)
            metrics['answer_length'].append(len(answer))
            
            # キーワード精度・再現率の計算
            precision, recall = self.calculate_keyword_metrics(answer, expected_keywords)
            metrics['retrieval_precision'].append(precision)
            metrics['retrieval_recall'].append(recall)
        
        # 平均値を計算
        return EvaluationMetrics(
            faithfulness=np.mean(metrics['faithfulness']),
            relevancy=np.mean(metrics['relevancy']),
            correctness=np.mean(metrics['correctness']),
            semantic_similarity=np.mean(metrics['semantic_similarity']),
            response_time=np.mean(metrics['response_time']),
            retrieval_precision=np.mean(metrics['retrieval_precision']),
            retrieval_recall=np.mean(metrics['retrieval_recall']),
            answer_length=int(np.mean(metrics['answer_length']))
        )
    
    def calculate_keyword_metrics(self, answer: str, expected_keywords: List[str]) -> Tuple[float, float]:
        """キーワードベースの精度・再現率計算"""
        if not expected_keywords:
            return 1.0, 1.0
        
        answer_lower = answer.lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        
        precision = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
        recall = precision  # 簡易実装
        
        return precision, recall
    
    def calculate_composite_score(self, metrics: EvaluationMetrics) -> float:
        """複合スコアの計算"""
        # 重み付き平均で総合スコアを計算
        weights = {
            'faithfulness': 0.25,
            'relevancy': 0.25,
            'correctness': 0.20,
            'semantic_similarity': 0.15,
            'retrieval_precision': 0.10,
            'retrieval_recall': 0.05
        }
        
        score = (
            metrics.faithfulness * weights['faithfulness'] +
            metrics.relevancy * weights['relevancy'] +
            metrics.correctness * weights['correctness'] +
            metrics.semantic_similarity * weights['semantic_similarity'] +
            metrics.retrieval_precision * weights['retrieval_precision'] +
            metrics.retrieval_recall * weights['retrieval_recall']
        )
        
        # 応答時間によるペナルティ（3秒以上で減点）
        time_penalty = max(0, (metrics.response_time - 3.0) * 0.1)
        score = max(0, score - time_penalty)
        
        return score
    
    def optimize_rag(self) -> Tuple[RAGConfig, EvaluationMetrics]:
        """RAGシステムの最適化実行"""
        configs = self.generate_config_variations()
        self.logger.info(f"最適化開始: {len(configs)}個の設定を評価")
        
        best_config = None
        best_metrics = None
        best_score = 0.0
        
        results = []
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"設定 {i}/{len(configs)} を評価中...")
            
            try:
                # RAGシステム構築
                query_engine, index = self.build_rag_system(config)
                
                # 評価実行
                metrics = self.evaluate_rag_system(query_engine, index, config)
                
                # 複合スコア計算
                composite_score = self.calculate_composite_score(metrics)
                
                # 結果記録
                result = {
                    'config': asdict(config),
                    'metrics': asdict(metrics),
                    'composite_score': composite_score,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                # ベスト更新チェック
                if composite_score > best_score:
                    best_score = composite_score
                    best_config = config
                    best_metrics = metrics
                    self.logger.info(f"新しいベストスコア: {best_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"設定評価エラー: {e}")
                continue
        
        # 結果保存
        self.save_optimization_results(results)
        
        self.best_config = best_config
        self.best_score = best_score
        
        return best_config, best_metrics
    
    def save_optimization_results(self, results: List[Dict]):
        """最適化結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_optimization_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"最適化結果を保存: {filename}")
    
    def generate_optimization_report(self) -> str:
        """最適化レポートの生成"""
        if not self.best_config:
            return "最適化が実行されていません。"
        
        report = f"""
# RAG最適化レポート

## 最適設定
- **総合スコア**: {self.best_score:.3f}
- **チャンキング**: {self.best_config.chunking.method} (サイズ: {self.best_config.chunking.chunk_size}, オーバーラップ: {self.best_config.chunking.chunk_overlap})
- **埋め込みモデル**: {self.best_config.embedding.model_name}
- **検索手法**: {self.best_config.query.retrieval_method}
- **LLM**: {self.best_config.query.llm_model}
- **温度**: {self.best_config.query.temperature}

## 推奨事項
1. 特定された最適設定を本番環境に適用
2. 定期的な再評価による継続的な改善
3. 新しいテストケースの追加による評価精度向上
"""
        
        return report

# 評価結果可視化クラス
class RAGVisualization:
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.load_results()
    
    def load_results(self):
        """結果ファイルの読み込み"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
    
    def create_performance_dashboard(self):
        """パフォーマンスダッシュボードの作成"""
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        
        st.title("🎯 RAG最適化ダッシュボード")
        
        # データ準備
        df = pd.DataFrame([
            {
                **result['metrics'],
                'composite_score': result['composite_score'],
                'chunking_method': result['config']['chunking']['method'],
                'chunk_size': result['config']['chunking']['chunk_size'],
                'embedding_model': result['config']['embedding']['model_name'],
                'llm_model': result['config']['query']['llm_model']
            }
            for result in self.results
        ])
        
        # 総合スコア分布
        fig1 = px.histogram(
            df, x='composite_score',
            title="総合スコア分布",
            nbins=20
        )
        st.plotly_chart(fig1)
        
        # 設定別パフォーマンス
        fig2 = px.box(
            df, x='chunking_method', y='composite_score',
            title="チャンキング手法別パフォーマンス"
        )
        st.plotly_chart(fig2)
        
        # 散布図マトリックス
        metrics_cols = ['faithfulness', 'relevancy', 'correctness', 'response_time']
        fig3 = px.scatter_matrix(
            df, dimensions=metrics_cols,
            title="評価指標間の相関"
        )
        st.plotly_chart(fig3)
        
        # トップ10設定
        st.subheader("🏆 トップ10設定")
        top10 = df.nlargest(10, 'composite_score')[
            ['composite_score', 'chunking_method', 'chunk_size', 
             'embedding_model', 'faithfulness', 'relevancy']
        ]
        st.dataframe(top10)

# 使用例
def main():
    # 環境設定
    os.environ['OPENAI_API_KEY'] = 'your-api-key'
    
    # 最適化実行
    optimizer = RAGOptimizer(
        documents_path="./product_manuals/",
        test_cases_path="./test_cases.json"
    )
    
    print("RAG最適化を開始します...")
    best_config, best_metrics = optimizer.optimize_rag()
    
    print("\n" + "="*60)
    print("最適化完了!")
    print("="*60)
    print(f"最適スコア: {optimizer.best_score:.3f}")
    print(f"Faithfulness: {best_metrics.faithfulness:.3f}")
    print(f"Relevancy: {best_metrics.relevancy:.3f}")
    print(f"応答時間: {best_metrics.response_time:.2f}秒")
    
    # レポート生成
    report = optimizer.generate_optimization_report()
    print(report)
    
    # 結果保存
    with open("optimization_report.md", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()
````

## 設定ファイルテンプレート

````yaml
optimization:
  # 評価重み
  evaluation_weights:
    faithfulness: 0.25
    relevancy: 0.25
    correctness: 0.20
    semantic_similarity: 0.15
    retrieval_precision: 0.10
    retrieval_recall: 0.05
  
  # 最適化範囲
  chunking_methods:
    - sentence
    - semantic
    - token
    - hierarchical
  
  chunk_sizes: [128, 256, 512, 1024]
  chunk_overlaps: [16, 32, 64, 128]
  
  embedding_models:
    - "text-embedding-3-large"
    - "text-embedding-3-small" 
    - "intfloat/multilingual-e5-large"
  
  llm_models:
    - "gpt-3.5-turbo"
    - "gpt-4-turbo-preview"
  
  temperature_range: [0.0, 0.1, 0.2]
  top_k_range: [3, 5, 8, 10]

# テストケース自動生成設定
test_case_generation:
  categories:
    - installation
    - operation
    - troubleshooting
    - maintenance
    - safety
  
  questions_per_category: 5
  use_synthetic_data: true
````

## バッチ最適化スクリプト

````python
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def optimize_subset(config_subset, optimizer_params):
    """設定サブセットの最適化"""
    optimizer = RAGOptimizer(**optimizer_params)
    results = []
    
    for config in config_subset:
        try:
            query_engine, index = optimizer.build_rag_system(config)
            metrics = optimizer.evaluate_rag_system(query_engine, index, config)
            score = optimizer.calculate_composite_score(metrics)
            
            results.append({
                'config': asdict(config),
                'metrics': asdict(metrics),
                'composite_score': score
            })
        except Exception as e:
            print(f"設定評価エラー: {e}")
            continue
    
    return results

def parallel_optimization(documents_path: str, test_cases_path: str, num_processes: int = None):
    """並列最適化実行"""
    if num_processes is None:
        num_processes = mp.cpu_count() - 1
    
    optimizer = RAGOptimizer(documents_path, test_cases_path)
    configs = optimizer.generate_config_variations()
    
    # 設定を分割
    chunk_size = len(configs) // num_processes
    config_chunks = [
        configs[i:i + chunk_size] 
        for i in range(0, len(configs), chunk_size)
    ]
    
    optimizer_params = {
        'documents_path': documents_path,
        'test_cases_path': test_cases_path
    }
    
    # 並列実行
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(optimize_subset, chunk, optimizer_params)
            for chunk in config_chunks
        ]
        
        all_results = []
        for future in futures:
            all_results.extend(future.result())
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG並列最適化")
    parser.add_argument("--docs", required=True, help="ドキュメントパス")
    parser.add_argument("--tests", required=True, help="テストケースパス")
    parser.add_argument("--processes", type=int, help="プロセス数")
    
    args = parser.parse_args()
    
    results = parallel_optimization(args.docs, args.tests, args.processes)
    
    # ベスト設定の特定
    best_result = max(results, key=lambda x: x['composite_score'])
    print(f"ベストスコア: {best_result['composite_score']:.3f}")
````

この最適化フレームワークの特徴：

1. **包括的評価**: 複数の評価指標を組み合わせた総合評価
2. **自動設定生成**: チャンキング、埋め込み、クエリ設定の自動バリエーション
3. **並列処理対応**: 大規模な最適化を効率的に実行
4. **可視化機能**: 結果の分析とレポート生成
5. **継続的改善**: 新しいテストケースによる再評価機能

RAGシステムの性能を体系的に改善できる包括的なツールセットです。
