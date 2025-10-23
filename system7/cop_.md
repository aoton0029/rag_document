提示したRAG評価フレームワークについて、手法の適切性とベストプラクティスを分析します。

## 現在のアプローチの評価

### ✅ 適切な点
1. **体系的な比較**: 複数の手法を統一環境で比較
2. **包括的なメトリクス**: 検索精度、生成品質、レスポンス時間を評価
3. **再現可能性**: 設定ファイルによる実験管理
4. **可視化**: 結果の分析とレポート生成

### ⚠️ 改善が必要な点
1. **評価データの質**: 人工的なground truthの限界
2. **メトリクスの選択**: ドメイン特化の評価指標不足
3. **統計的検証**: 有意性テストの不備
4. **実際のユースケースとの乖離**: 合成データでの評価

## ベストプラクティスな手法

### 1. **RAGAS Framework** (推奨)
```python
# より実践的な評価フレームワーク例
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)

# RAGASを使用した評価
def evaluate_with_ragas(dataset):
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_relevancy,
        ],
    )
    return result
```

### 2. **階層的評価アプローチ**
```yaml
# より体系的な評価設定
evaluation_levels:
  component_level:
    - chunk_quality_assessment
    - embedding_similarity_analysis
    - retrieval_coverage_analysis
  
  system_level:
    - end_to_end_accuracy
    - user_satisfaction_simulation
    - domain_specific_benchmarks
  
  production_level:
    - a_b_testing
    - user_feedback_analysis
    - business_metric_correlation
```

### 3. **ドメイン適応型評価**
```python
class DomainSpecificEvaluator:
    def __init__(self, domain: str):
        self.domain = domain
        self.domain_metrics = self.load_domain_metrics()
    
    def load_domain_metrics(self):
        domain_configs = {
            "legal": ["citation_accuracy", "legal_reasoning", "precedent_relevance"],
            "medical": ["clinical_accuracy", "safety_assessment", "evidence_quality"],
            "technical": ["code_correctness", "api_completeness", "technical_depth"]
        }
        return domain_configs.get(self.domain, [])
```

### 4. **LLM-as-a-Judge評価**
```python
class LLMJudgeEvaluator:
    def __init__(self, judge_model="gpt-4"):
        self.judge_model = judge_model
    
    def evaluate_response_quality(self, query, context, response, reference=None):
        prompt = f"""
        Query: {query}
        Retrieved Context: {context}
        Generated Response: {response}
        Reference Answer: {reference}
        
        Please evaluate the response on:
        1. Factual Accuracy (1-5)
        2. Relevance to Query (1-5) 
        3. Use of Context (1-5)
        4. Completeness (1-5)
        5. Overall Quality (1-5)
        
        Provide scores and brief justification.
        """
        return self.judge_model.generate(prompt)
```

## 推奨する改良版フレームワーク

### 1. **マルチレベル評価構造**
```python
class ComprehensiveRAGEvaluator:
    def __init__(self):
        self.component_evaluator = ComponentLevelEvaluator()
        self.system_evaluator = SystemLevelEvaluator()
        self.human_evaluator = HumanEvaluator()
        self.production_evaluator = ProductionEvaluator()
    
    def run_full_evaluation(self, rag_system, test_data):
        results = {}
        
        # コンポーネントレベル
        results['component'] = self.component_evaluator.evaluate(rag_system)
        
        # システムレベル
        results['system'] = self.system_evaluator.evaluate(rag_system, test_data)
        
        # 人間評価（サンプリング）
        results['human'] = self.human_evaluator.evaluate_sample(rag_system, test_data)
        
        # プロダクション指標
        results['production'] = self.production_evaluator.evaluate(rag_system)
        
        return results
```

### 2. **適応的テストセット生成**
```python
class AdaptiveTestSetGenerator:
    def generate_diverse_queries(self, documents, difficulty_levels=['easy', 'medium', 'hard']):
        """実際の文書から多様な難易度のクエリを生成"""
        test_queries = []
        
        for doc in documents:
            for level in difficulty_levels:
                queries = self.generate_queries_by_difficulty(doc, level)
                test_queries.extend(queries)
        
        return test_queries
    
    def generate_adversarial_cases(self, rag_system):
        """システムの弱点を特定する対抗例生成"""
        pass
```

### 3. **統計的検証強化**
```python
class StatisticalValidator:
    def compare_configurations(self, results_a, results_b):
        """統計的有意性テスト"""
        from scipy import stats
        
        # Wilcoxon符号順位検定
        statistic, p_value = stats.wilcoxon(results_a, results_b)
        
        # 効果量の計算
        effect_size = self.calculate_effect_size(results_a, results_b)
        
        return {
            'statistically_significant': p_value < 0.05,
            'p_value': p_value,
            'effect_size': effect_size,
            'practical_significance': effect_size > 0.2
        }
```

### 4. **継続的評価パイプライン**
```python
class ContinuousEvaluationPipeline:
    def setup_monitoring(self, rag_system):
        """プロダクション環境での継続的評価"""
        metrics = [
            'user_satisfaction_score',
            'task_completion_rate', 
            'fact_checking_accuracy',
            'latency_percentiles',
            'cost_per_query'
        ]
        
        for metric in metrics:
            self.setup_metric_monitoring(metric)
    
    def trigger_revaluation(self, threshold_violations):
        """閾値違反時の自動再評価"""
        if threshold_violations:
            self.run_comprehensive_evaluation()
            self.suggest_configuration_updates()
```

## 実践的な推奨事項

### 1. **段階的評価アプローチ**
```
Phase 1: オフライン評価 (提示したフレームワーク)
Phase 2: LLM-as-a-Judge評価
Phase 3: 人間評価 (サンプリング)
Phase 4: A/Bテスト (プロダクション)
```

### 2. **メトリクス選択指針**
- **検索品質**: Precision@K, Recall@K, NDCG, MRR
- **生成品質**: RAGAS metrics, BERTScore, LLM-judge scores
- **実用性**: Latency, Cost, User satisfaction
- **安全性**: Hallucination rate, Bias detection, Fact-checking

### 3. **データセット構築**
- **実際のユーザークエリ**: プロダクションログから収集
- **ドメインエキスパート**: 専門家による評価データ作成
- **合成データ**: LLMによる多様なクエリ生成
- **対抗例**: システムの限界をテストするケース

### 結論

提示したフレームワークは**基本的なアプローチとしては適切**ですが、より実践的で包括的な評価のためには：

1. **RAGAS**などの専門フレームワークの採用
2. **マルチレベル評価**の実装
3. **統計的検証**の強化
4. **ドメイン特化指標**の追加
5. **継続的評価パイプライン**の構築

が推奨されます。



RAGASやマルチレベル評価を導入した改良版プロジェクトを提示します：

## 改良版プロジェクト構造
```
rag-evaluation-framework-v2/
├── README.md
├── requirements.txt
├── config/
│   ├── chunking_configs.yaml
│   ├── embedding_configs.yaml
│   ├── evaluation_configs.yaml
│   └── domain_configs.yaml
├── src/
│   ├── __init__.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── strategies.py
│   │   └── quality_assessor.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── similarity_analyzer.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── strategies.py
│   │   └── coverage_analyzer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── component_evaluator.py
│   │   ├── system_evaluator.py
│   │   ├── human_evaluator.py
│   │   ├── production_evaluator.py
│   │   ├── statistical_validator.py
│   │   ├── llm_judge.py
│   │   └── comprehensive_runner.py
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── adaptive_generator.py
│   │   └── adversarial_generator.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── continuous_pipeline.py
│   │   └── alert_system.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── ragas_integration.py
│       └── domain_adapter.py
├── tests/
├── data/
├── results/
└── notebooks/
```

## 改良版主要ファイル

### `requirements.txt`
```txt
llama-index==0.10.57
llama-index-embeddings-openai
llama-index-embeddings-huggingface
llama-index-llms-openai
ragas==0.1.7
datasets==2.14.7
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
scipy==1.11.4
pyyaml==6.0.1
pytest==7.4.3
jupyter==1.0.0
tqdm==4.66.1
evaluate==0.4.1
rouge-score==0.1.2
bert-score==0.3.13
openai==1.3.0
tiktoken==0.5.2
langchain==0.1.0
nest-asyncio==1.5.8
```




GitHub URLを確認して、改良版プロジェクトの詳細な設定ファイルを提示します。既存のドキュメントを確認しました。改良版プロジェクト構造の各設定ファイル（コンフィグ）を詳細に提示します：

## 改良版プロジェクト構造の各コンフィグファイル

### 1. `config/chunking_configs.yaml`
```yaml
# チャンキング戦略の設定
chunking_strategies:
  # 固定サイズチャンキング（小）
  fixed_size_256:
    type: "fixed_size"
    chunk_size: 256
    chunk_overlap: 25
    separator: "\n\n"
    
  # 固定サイズチャンキング（中）
  fixed_size_512:
    type: "fixed_size"
    chunk_size: 512
    chunk_overlap: 50
    separator: "\n\n"
    
  # 固定サイズチャンキング（大）
  fixed_size_1024:
    type: "fixed_size"
    chunk_size: 1024
    chunk_overlap: 100
    separator: "\n\n"
    
  # 固定サイズチャンキング（特大）
  fixed_size_2048:
    type: "fixed_size"
    chunk_size: 2048
    chunk_overlap: 200
    separator: "\n\n"
  
  # セマンティックチャンキング
  semantic_llama:
    type: "semantic"
    embed_model: "sentence-transformers/all-MiniLM-L6-v2"
    breakpoint_percentile_threshold: 95
    buffer_size: 1
    
  # セマンティックチャンキング（厳格）
  semantic_strict:
    type: "semantic"
    embed_model: "sentence-transformers/all-MiniLM-L6-v2"
    breakpoint_percentile_threshold: 85
    buffer_size: 0
    
  # 階層的チャンキング
  hierarchical_standard:
    type: "hierarchical"
    chunk_sizes: [2048, 512, 128]
    chunk_overlap: 20
    
  # 階層的チャンキング（詳細）
  hierarchical_detailed:
    type: "hierarchical"
    chunk_sizes: [4096, 1024, 256, 64]
    chunk_overlap: 50
    
  # 文単位チャンキング
  sentence_based:
    type: "sentence"
    chunk_size: 3
    chunk_overlap: 1
    paragraph_separator: "\n\n"
    
  # トークンベースチャンキング
  token_based_512:
    type: "token"
    chunk_size: 512
    chunk_overlap: 50
    tokenizer: "cl100k_base"
    
  # トークンベースチャンキング（大）
  token_based_1024:
    type: "token"
    chunk_size: 1024
    chunk_overlap: 100
    tokenizer: "cl100k_base"

# チャンキング品質評価設定
chunk_quality_settings:
  semantic_coherence:
    enabled: true
    threshold: 0.7
    
  information_density:
    enabled: true
    min_density: 0.3
    
  overlap_analysis:
    enabled: true
    max_overlap: 0.3
    
  size_distribution:
    enabled: true
    target_variance: 0.2
```

### 2. `config/embedding_configs.yaml`
```yaml
# 埋め込みモデルの設定
embedding_models:
  # OpenAI Embeddings
  openai_ada_002:
    type: "openai"
    model_name: "text-embedding-ada-002"
    dimensions: 1536
    api_key_env: "OPENAI_API_KEY"
    
  openai_ada_003:
    type: "openai"
    model_name: "text-embedding-3-small"
    dimensions: 1536
    api_key_env: "OPENAI_API_KEY"
    
  openai_large:
    type: "openai"
    model_name: "text-embedding-3-large"
    dimensions: 3072
    api_key_env: "OPENAI_API_KEY"
  
  # Sentence Transformers
  sentence_transformers_mini:
    type: "huggingface"
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: 384
    device: "cpu"
    trust_remote_code: false
    
  sentence_transformers_mpnet:
    type: "huggingface"
    model_name: "sentence-transformers/all-mpnet-base-v2"
    dimensions: 768
    device: "cpu"
    trust_remote_code: false
    
  # BGE Embeddings
  bge_small:
    type: "huggingface"
    model_name: "BAAI/bge-small-en-v1.5"
    dimensions: 384
    device: "cpu"
    trust_remote_code: false
    
  bge_base:
    type: "huggingface"
    model_name: "BAAI/bge-base-en-v1.5"
    dimensions: 768
    device: "cpu"
    trust_remote_code: false
    
  bge_large:
    type: "huggingface"
    model_name: "BAAI/bge-large-en-v1.5"
    dimensions: 1024
    device: "cpu"
    trust_remote_code: false
    
  # E5 Embeddings
  e5_small:
    type: "huggingface"
    model_name: "intfloat/e5-small-v2"
    dimensions: 384
    device: "cpu"
    trust_remote_code: false
    prefix: "passage: "
    
  e5_base:
    type: "huggingface"
    model_name: "intfloat/e5-base-v2"
    dimensions: 768
    device: "cpu"
    trust_remote_code: false
    prefix: "passage: "
    
  # 日本語対応モデル
  japanese_sentence_bert:
    type: "huggingface"
    model_name: "colorfulscoop/sbert-base-ja"
    dimensions: 768
    device: "cpu"
    trust_remote_code: false
    
  multilingual_e5:
    type: "huggingface"
    model_name: "intfloat/multilingual-e5-base"
    dimensions: 768
    device: "cpu"
    trust_remote_code: false
    prefix: "passage: "

# 埋め込み品質評価設定
embedding_quality_settings:
  cluster_analysis:
    enabled: true
    n_clusters_range: [2, 10]
    
  dimensionality_analysis:
    enabled: true
    pca_variance_threshold: 0.95
    
  semantic_preservation:
    enabled: true
    sample_size: 100
    
  retrieval_performance:
    enabled: true
    top_k_values: [1, 3, 5, 10]
```

### 3. `config/retrieval_configs.yaml`
```yaml
# 検索戦略の設定
retrieval_strategies:
  # ベクター検索
  vector_search_basic:
    type: "vector"
    similarity_top_k: 5
    similarity_cutoff: 0.0
    
  vector_search_precise:
    type: "vector"
    similarity_top_k: 10
    similarity_cutoff: 0.7
    
  # ハイブリッド検索
  hybrid_search:
    type: "hybrid" 
    vector_similarity_top_k: 8
    keyword_similarity_top_k: 8
    alpha: 0.5  # ベクター検索の重み
    
  # キーワード検索
  keyword_search:
    type: "keyword"
    similarity_top_k: 5
    
  # リランキング付き検索
  rerank_search:
    type: "rerank"
    initial_top_k: 20
    final_top_k: 5
    rerank_model: "cross-encoder"
    
  # 階層検索
  hierarchical_retrieval:
    type: "hierarchical"
    parent_top_k: 3
    child_top_k: 5
    
  # セマンティック検索
  semantic_search:
    type: "semantic"
    similarity_top_k: 5
    semantic_threshold: 0.8
    
  # マルチクエリ検索
  multi_query:
    type: "multi_query"
    num_queries: 3
    similarity_top_k: 5

# ベクターストア設定
vector_store_configs:
  # FAISS設定
  faiss:
    type: "faiss"
    index_type: "flat"
    metric: "cosine"
    
  # Chroma設定
  chroma:
    type: "chroma"
    persist_directory: "./chroma_db"
    collection_name: "rag_documents"
    
  # Qdrant設定
  qdrant:
    type: "qdrant"
    location: ":memory:"
    collection_name: "rag_documents"
    vector_size: 1536
    
  # Pinecone設定
  pinecone:
    type: "pinecone"
    api_key_env: "PINECONE_API_KEY"
    environment: "us-west1-gcp"
    index_name: "rag-index"

# 検索品質評価設定
retrieval_quality_settings:
  coverage_analysis:
    enabled: true
    minimum_coverage: 0.8
    
  relevance_distribution:
    enabled: true
    relevance_threshold: 0.7
    
  diversity_assessment:
    enabled: true
    diversity_threshold: 0.5
    
  latency_analysis:
    enabled: true
    max_latency_ms: 1000
```

### 4. `config/evaluation_configs.yaml`
```yaml
# 評価フレームワークの設定
evaluation_framework:
  # 評価レベル
  levels:
    - component_level
    - system_level
    - human_level
    - production_level
  
  # コンポーネントレベル評価
  component_level:
    chunk_quality:
      enabled: true
      metrics:
        - semantic_coherence
        - information_density
        - overlap_analysis
        - size_distribution
      weights:
        semantic_coherence: 0.3
        information_density: 0.3
        overlap_analysis: 0.2
        size_distribution: 0.2
    
    embedding_quality:
      enabled: true
      metrics:
        - cluster_quality
        - semantic_preservation
        - dimensionality_analysis
        - retrieval_performance
      weights:
        cluster_quality: 0.25
        semantic_preservation: 0.3
        dimensionality_analysis: 0.2
        retrieval_performance: 0.25
    
    retrieval_quality:
      enabled: true
      metrics:
        - coverage_analysis
        - relevance_distribution
        - diversity_assessment
        - latency_analysis
      weights:
        coverage_analysis: 0.3
        relevance_distribution: 0.3
        diversity_assessment: 0.2
        latency_analysis: 0.2

  # システムレベル評価
  system_level:
    # RAGAS メトリクス
    ragas_metrics:
      enabled: true
      metrics:
        - faithfulness
        - answer_relevancy
        - context_precision
        - context_recall
        - context_relevancy
        - answer_correctness
        - answer_similarity
      weights:
        faithfulness: 0.2
        answer_relevancy: 0.2
        context_precision: 0.15
        context_recall: 0.15
        context_relevancy: 0.1
        answer_correctness: 0.1
        answer_similarity: 0.1
    
    # 従来メトリクス
    traditional_metrics:
      enabled: true
      metrics:
        - precision_at_k
        - recall_at_k
        - ndcg_at_k
        - mrr
        - map
      k_values: [1, 3, 5, 10]
      weights:
        precision_at_k: 0.25
        recall_at_k: 0.25
        ndcg_at_k: 0.25
        mrr: 0.15
        map: 0.1
    
    # LLM Judge評価
    llm_judge:
      enabled: true
      judge_model: "gpt-4-turbo"
      temperature: 0.1
      max_tokens: 1000
      evaluation_aspects:
        - factual_accuracy
        - relevance_to_query
        - context_utilization
        - completeness
        - overall_quality
      weights:
        factual_accuracy: 0.25
        relevance_to_query: 0.25
        context_utilization: 0.2
        completeness: 0.15
        overall_quality: 0.15
      
    # BERTScore評価
    bert_score:
      enabled: true
      model_type: "microsoft/deberta-xlarge-mnli"
      lang: "en"
      
    # ROUGE評価
    rouge_score:
      enabled: true
      rouge_types: ["rouge1", "rouge2", "rougeL"]

  # 人間評価レベル
  human_level:
    enabled: false  # デフォルトでは無効
    sample_size: 50
    evaluators: 3
    inter_rater_agreement: true
    aspects:
      - accuracy
      - helpfulness
      - clarity
      - completeness
      - fluency
    weights:
      accuracy: 0.3
      helpfulness: 0.25
      clarity: 0.2
      completeness: 0.15
      fluency: 0.1

  # プロダクションレベル評価
  production_level:
    enabled: false  # デフォルトでは無効
    metrics:
      - user_satisfaction
      - task_completion_rate
      - latency_percentiles
      - cost_per_query
      - error_rate
      - throughput
    monitoring_interval: "1h"
    alert_thresholds:
      user_satisfaction: 0.8
      task_completion_rate: 0.9
      latency_p95: 5000  # ms
      error_rate: 0.05

# 統計的検証設定
statistical_validation:
  enabled: true
  significance_threshold: 0.05
  effect_size_threshold: 0.2
  multiple_comparison_correction: "bonferroni"
  bootstrap_iterations: 1000
  confidence_level: 0.95
  
  # 検定手法の設定
  test_methods:
    auto_select: true
    preferred_parametric: "t_test"
    preferred_nonparametric: "wilcoxon"
    normality_test: "shapiro"
    
# レポート生成設定
reporting:
  enabled: true
  output_formats: ["html", "pdf", "json"]
  include_visualizations: true
  save_raw_data: true
  
  # 可視化設定
  visualizations:
    performance_comparison: true
    statistical_significance: true
    correlation_analysis: true
    time_series_analysis: false
    
  # 自動推奨機能
  recommendations:
    enabled: true
    top_n_configurations: 3
    include_reasoning: true
    confidence_threshold: 0.8
```

### 5. `config/domain_configs.yaml`
```yaml
# ドメイン特化設定
domains:
  # 技術ドメイン
  technical:
    description: "プログラミング、API、技術文書"
    specific_metrics:
      - code_correctness
      - api_completeness
      - technical_depth
      - example_quality
      - implementation_accuracy
    
    test_patterns:
      - programming_concepts
      - api_documentation
      - troubleshooting_guides
      - code_examples
      - technical_specifications
      
    evaluation_criteria:
      code_correctness:
        weight: 0.3
        description: "コードの正確性と実行可能性"
      api_completeness:
        weight: 0.25
        description: "API情報の完全性"
      technical_depth:
        weight: 0.2
        description: "技術的な詳細度"
      example_quality:
        weight: 0.15
        description: "実例の品質と適切性"
      implementation_accuracy:
        weight: 0.1
        description: "実装手順の正確性"
    
    query_templates:
      - "How to implement {concept} in {language}?"
      - "What is the difference between {tech_a} and {tech_b}?"
      - "Best practices for {technology}"
      - "Troubleshoot {error_type} in {framework}"
  
  # 医学ドメイン
  medical:
    description: "医学、臨床、薬学情報"
    specific_metrics:
      - clinical_accuracy
      - safety_assessment
      - evidence_quality
      - contraindication_awareness
      - dosage_precision
    
    test_patterns:
      - clinical_cases
      - drug_interactions
      - diagnostic_procedures
      - treatment_protocols
      - medical_guidelines
      
    evaluation_criteria:
      clinical_accuracy:
        weight: 0.35
        description: "臨床情報の正確性"
      safety_assessment:
        weight: 0.25
        description: "安全性に関する評価"
      evidence_quality:
        weight: 0.2
        description: "エビデンスの質と信頼性"
      contraindication_awareness:
        weight: 0.15
        description: "禁忌事項の認識"
      dosage_precision:
        weight: 0.05
        description: "投与量の正確性"
    
    safety_checks:
      enabled: true
      critical_keywords: ["dosage", "contraindication", "side effects", "allergy"]
      require_citation: true
      
    query_templates:
      - "What are the symptoms of {condition}?"
      - "Treatment options for {diagnosis}"
      - "Drug interactions with {medication}"
      - "Diagnostic criteria for {disease}"
  
  # 法律ドメイン
  legal:
    description: "法律、判例、規制情報"
    specific_metrics:
      - citation_accuracy
      - legal_reasoning
      - precedent_relevance
      - statute_interpretation
      - jurisdiction_awareness
    
    test_patterns:
      - case_law_analysis
      - statute_interpretation
      - contract_analysis
      - regulatory_compliance
      - legal_precedents
      
    evaluation_criteria:
      citation_accuracy:
        weight: 0.3
        description: "引用の正確性"
      legal_reasoning:
        weight: 0.25
        description: "法的推論の妥当性"
      precedent_relevance:
        weight: 0.2
        description: "判例の関連性"
      statute_interpretation:
        weight: 0.15
        description: "法令解釈の適切性"
      jurisdiction_awareness:
        weight: 0.1
        description: "管轄権の認識"
    
    citation_requirements:
      enabled: true
      required_formats: ["case_citation", "statute_reference"]
      verify_authenticity: true
      
    query_templates:
      - "Legal precedent for {legal_issue}"
      - "Statute requirements for {regulation}"
      - "Case law analysis of {legal_concept}"
      - "Compliance requirements for {industry}"
  
  # 金融ドメイン
  financial:
    description: "金融、投資、経済情報"
    specific_metrics:
      - financial_accuracy
      - risk_assessment
      - regulatory_compliance
      - market_relevance
      - numerical_precision
    
    test_patterns:
      - investment_analysis
      - risk_assessment
      - regulatory_compliance
      - market_analysis
      - financial_calculations
      
    evaluation_criteria:
      financial_accuracy:
        weight: 0.3
        description: "金融情報の正確性"
      risk_assessment:
        weight: 0.25
        description: "リスク評価の適切性"
      regulatory_compliance:
        weight: 0.2
        description: "規制遵守の認識"
      market_relevance:
        weight: 0.15
        description: "市場関連性"
      numerical_precision:
        weight: 0.1
        description: "数値の精度"
    
    risk_warnings:
      enabled: true
      required_disclaimers: true
      
    query_templates:
      - "Investment risks of {asset_class}"
      - "Regulatory requirements for {financial_product}"
      - "Market analysis of {sector}"
      - "Financial ratios for {company_analysis}"
  
  # 一般ドメイン
  general:
    description: "一般的な知識、事実情報"
    specific_metrics:
      - factual_accuracy
      - logical_consistency
      - comprehensiveness
      - source_reliability
      - information_freshness
    
    test_patterns:
      - factual_questions
      - reasoning_tasks
      - summarization_requests
      - comparison_analysis
      - explanatory_content
      
    evaluation_criteria:
      factual_accuracy:
        weight: 0.3
        description: "事実の正確性"
      logical_consistency:
        weight: 0.25
        description: "論理的一貫性"
      comprehensiveness:
        weight: 0.2
        description: "包括性"
      source_reliability:
        weight: 0.15
        description: "情報源の信頼性"
      information_freshness:
        weight: 0.1
        description: "情報の新しさ"
    
    query_templates:
      - "What is {concept}?"
      - "Explain the difference between {item_a} and {item_b}"
      - "History of {topic}"
      - "Current status of {subject}"

# ドメイン適応設定
domain_adaptation:
  automatic_detection:
    enabled: true
    confidence_threshold: 0.8
    keywords_based: true
    context_based: true
    
  cross_domain_evaluation:
    enabled: true
    penalty_factor: 0.1
    
  domain_specific_preprocessing:
    enabled: true
    custom_tokenizers: true
    domain_stopwords: true
    
  specialized_models:
    technical:
      embedding_model: "microsoft/codebert-base"
      rerank_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
    medical:
      embedding_model: "dmis-lab/biobert-base-cased-v1.1"
      rerank_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
    legal:
      embedding_model: "nlpaueb/legal-bert-base-uncased"
      rerank_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

### 6. `config/test_patterns.yaml`
```yaml
# テストパターン設定
test_patterns:
  # ベースライン設定
  - name: "baseline_small"
    description: "小さいチャンクでのベースライン"
    chunking: "fixed_size_512"
    embedding: "sentence_transformers_mini"
    retrieval: "vector_search_basic"
    top_k: 5
    domain: "general"
    
  - name: "baseline_large"
    description: "大きいチャンクでのベースライン"
    chunking: "fixed_size_1024"
    embedding: "sentence_transformers_mpnet"
    retrieval: "vector_search_basic"
    top_k: 5
    domain: "general"
    
  # OpenAI設定
  - name: "openai_standard"
    description: "OpenAI標準設定"  
    chunking: "fixed_size_512"
    embedding: "openai_ada_002"
    retrieval: "vector_search_basic"
    top_k: 5
    domain: "general"
    
  - name: "openai_premium"
    description: "OpenAI高性能設定"
    chunking: "fixed_size_1024"
    embedding: "openai_large"
    retrieval: "hybrid_search"
    top_k: 8
    domain: "general"
    
  # セマンティックチャンキング
  - name: "semantic_chunking"
    description: "セマンティックチャンキング"
    chunking: "semantic_llama"
    embedding: "bge_large"
    retrieval: "vector_search_precise"
    top_k: 8
    domain: "general"
    
  - name: "semantic_strict"
    description: "厳格なセマンティックチャンキング"
    chunking: "semantic_strict"
    embedding: "bge_large"
    retrieval: "rerank_search"
    top_k: 10
    domain: "general"
    
  # 階層的アプローチ
  - name: "hierarchical_standard"
    description: "階層的チャンキング標準"
    chunking: "hierarchical_standard"
    embedding: "bge_base"
    retrieval: "hierarchical_retrieval"
    top_k: 8
    domain: "general"
    
  - name: "hierarchical_detailed"
    description: "詳細階層的アプローチ"
    chunking: "hierarchical_detailed"
    embedding: "openai_large"
    retrieval: "hierarchical_retrieval"
    top_k: 10
    domain: "general"
    
  # ハイブリッド検索
  - name: "hybrid_approach"
    description: "ハイブリッド検索アプローチ"
    chunking: "fixed_size_512"
    embedding: "bge_large"
    retrieval: "hybrid_search"
    top_k: 10
    domain: "general"
    
  # 高性能設定
  - name: "high_performance"
    description: "高性能設定"
    chunking: "token_based_1024"
    embedding: "openai_large"
    retrieval: "rerank_search"
    top_k: 20
    domain: "general"
    
  # 多言語対応
  - name: "multilingual"
    description: "多言語対応設定"
    chunking: "fixed_size_512"
    embedding: "multilingual_e5"
    retrieval: "vector_search_basic"
    top_k: 5
    domain: "general"
    
  # ドメイン特化設定
  - name: "technical_optimized"
    description: "技術ドメイン最適化"
    chunking: "fixed_size_1024"
    embedding: "bge_large"
    retrieval: "hybrid_search"
    top_k: 8
    domain: "technical"
    
  - name: "medical_safe"
    description: "医学ドメイン安全設定"
    chunking: "fixed_size_512"
    embedding: "bge_base"
    retrieval: "rerank_search"
    top_k: 5
    domain: "medical"
    
  - name: "legal_precise"
    description: "法律ドメイン精密設定"
    chunking: "semantic_strict"
    embedding: "bge_large"
    retrieval: "rerank_search"
    top_k: 10
    domain: "legal"

# 評価設定
evaluation_settings:
  # データパス
  documents_path: "data/documents/"
  test_queries_path: "data/test_queries.json"
  ground_truth_path: "data/ground_truth.json"
  
  # 評価メトリクス
  metrics:
    - "component_le




改良版プロジェクト構造のchunking機能の詳細実装を提示します：

## chunking機能の詳細実装

### 1. `src/chunking/strategies.py`
```python
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import re
import tiktoken
from llama_index.core.node_parser import (
    SimpleNodeParser,
    SentenceSplitter,
    SemanticSplitterNodeParser,
    HierarchicalNodeParser,
    TokenTextSplitter
)
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans

class ChunkingStrategy(ABC):
    """チャンキング戦略の基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_type = config.get("type")
        
    @abstractmethod
    def create_parser(self):
        """パーサーの作成"""
        pass
    
    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """ドキュメントのチャンク分割"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """戦略のメタデータ取得"""
        return {
            "strategy_type": self.strategy_type,
            "config": self.config
        }

class FixedSizeChunkingStrategy(ChunkingStrategy):
    """固定サイズチャンキング戦略"""
    
    def create_parser(self):
        return SentenceSplitter(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            separator=self.config.get("separator", "\n\n")
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        parser = self.create_parser()
        nodes = parser.get_nodes_from_documents(documents)
        
        # メタデータの追加
        for i, node in enumerate(nodes):
            node.metadata.update({
                "chunking_strategy": "fixed_size",
                "chunk_size": self.config.get("chunk_size"),
                "chunk_overlap": self.config.get("chunk_overlap"),
                "chunk_index": i,
                "chunk_length": len(node.text)
            })
        
        return nodes

class TokenBasedChunkingStrategy(ChunkingStrategy):
    """トークンベースチャンキング戦略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tokenizer_name = config.get("tokenizer", "cl100k_base")
        try:
            self.tokenizer = tiktoken.get_encoding(self.tokenizer_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def create_parser(self):
        return TokenTextSplitter(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            tokenizer=self.tokenizer.encode
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        parser = self.create_parser()
        nodes = parser.get_nodes_from_documents(documents)
        
        for i, node in enumerate(nodes):
            token_count = len(self.tokenizer.encode(node.text))
            node.metadata.update({
                "chunking_strategy": "token_based",
                "chunk_size": self.config.get("chunk_size"),
                "chunk_overlap": self.config.get("chunk_overlap"),
                "chunk_index": i,
                "token_count": token_count,
                "tokenizer": self.tokenizer_name
            })
        
        return nodes

class SemanticChunkingStrategy(ChunkingStrategy):
    """セマンティックチャンキング戦略"""
    
    def create_parser(self):
        embed_model_name = self.config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        
        if embed_model_name.startswith("text-embedding"):
            embed_model = OpenAIEmbedding(model=embed_model_name)
        else:
            embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        
        return SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=self.config.get("breakpoint_percentile_threshold", 95),
            buffer_size=self.config.get("buffer_size", 1)
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        parser = self.create_parser()
        nodes = parser.get_nodes_from_documents(documents)
        
        for i, node in enumerate(nodes):
            node.metadata.update({
                "chunking_strategy": "semantic",
                "embed_model": self.config.get("embed_model"),
                "breakpoint_threshold": self.config.get("breakpoint_percentile_threshold"),
                "chunk_index": i,
                "chunk_length": len(node.text)
            })
        
        return nodes

class HierarchicalChunkingStrategy(ChunkingStrategy):
    """階層的チャンキング戦略"""
    
    def create_parser(self):
        chunk_sizes = self.config.get("chunk_sizes", [2048, 512])
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=self.config.get("chunk_overlap", 20)
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        parser = self.create_parser()
        nodes = parser.get_nodes_from_documents(documents)
        
        for i, node in enumerate(nodes):
            node.metadata.update({
                "chunking_strategy": "hierarchical",
                "chunk_sizes": self.config.get("chunk_sizes"),
                "chunk_overlap": self.config.get("chunk_overlap"),
                "chunk_index": i,
                "hierarchy_level": self._determine_hierarchy_level(node),
                "chunk_length": len(node.text)
            })
        
        return nodes
    
    def _determine_hierarchy_level(self, node: TextNode) -> int:
        """ノードの階層レベルを決定"""
        chunk_length = len(node.text)
        chunk_sizes = self.config.get("chunk_sizes", [2048, 512])
        
        for level, size in enumerate(chunk_sizes):
            if chunk_length >= size * 0.8:  # 80%以上なら該当レベル
                return level
        
        return len(chunk_sizes) - 1  # 最下位レベル

class SentenceBasedChunkingStrategy(ChunkingStrategy):
    """文単位チャンキング戦略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sentence_splitter = re.compile(r'[.!?]+\s+')
    
    def create_parser(self):
        # 文単位での分割なので、カスタム実装
        return None
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        nodes = []
        chunk_size = self.config.get("chunk_size", 3)  # 文の数
        chunk_overlap = self.config.get("chunk_overlap", 1)
        
        for doc_idx, document in enumerate(documents):
            sentences = self._split_into_sentences(document.text)
            
            for i in range(0, len(sentences), chunk_size - chunk_overlap):
                chunk_sentences = sentences[i:i + chunk_size]
                chunk_text = ' '.join(chunk_sentences)
                
                if chunk_text.strip():
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunking_strategy": "sentence_based",
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "chunk_index": i // (chunk_size - chunk_overlap) if chunk_overlap > 0 else i // chunk_size,
                            "sentence_count": len(chunk_sentences),
                            "document_index": doc_idx
                        }
                    )
                    nodes.append(node)
        
        return nodes
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        sentences = self.sentence_splitter.split(text)
        return [s.strip() for s in sentences if s.strip()]

class AdaptiveChunkingStrategy(ChunkingStrategy):
    """適応的チャンキング戦略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.semantic_model = SentenceTransformer(
            config.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
    
    def create_parser(self):
        return None  # カスタム実装
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        nodes = []
        
        for doc_idx, document in enumerate(documents):
            doc_nodes = self._adaptive_chunk_document(document, doc_idx)
            nodes.extend(doc_nodes)
        
        return nodes
    
    def _adaptive_chunk_document(self, document: Document, doc_idx: int) -> List[TextNode]:
        """文書を適応的にチャンク分割"""
        text = document.text
        paragraphs = text.split('\n\n')
        
        nodes = []
        current_chunk = ""
        current_embeddings = []
        chunk_index = 0
        
        min_chunk_size = self.config.get("min_chunk_size", 100)
        max_chunk_size = self.config.get("max_chunk_size", 1000)
        similarity_threshold = self.config.get("similarity_threshold", 0.8)
        
        for para in paragraphs:
            if not para.strip():
                continue
            
            para_embedding = self.semantic_model.encode([para])[0]
            
            if not current_chunk:
                current_chunk = para
                current_embeddings = [para_embedding]
            else:
                # 現在のチャンクとの類似度を計算
                avg_embedding = np.mean(current_embeddings, axis=0)
                similarity = np.dot(avg_embedding, para_embedding) / (
                    np.linalg.norm(avg_embedding) * np.linalg.norm(para_embedding)
                )
                
                potential_chunk = current_chunk + "\n\n" + para
                
                # チャンクを追加するかどうかの判定
                should_add = (
                    similarity >= similarity_threshold and
                    len(potential_chunk) <= max_chunk_size
                )
                
                if should_add:
                    current_chunk = potential_chunk
                    current_embeddings.append(para_embedding)
                else:
                    # 現在のチャンクを確定
                    if len(current_chunk) >= min_chunk_size:
                        node = TextNode(
                            text=current_chunk,
                            metadata={
                                **document.metadata,
                                "chunking_strategy": "adaptive",
                                "chunk_index": chunk_index,
                                "chunk_length": len(current_chunk),
                                "semantic_coherence": np.mean([
                                    np.dot(avg_embedding, emb) / (
                                        np.linalg.norm(avg_embedding) * np.linalg.norm(emb)
                                    ) for emb in current_embeddings
                                ]),
                                "document_index": doc_idx
                            }
                        )
                        nodes.append(node)
                        chunk_index += 1
                    
                    # 新しいチャンクを開始
                    current_chunk = para
                    current_embeddings = [para_embedding]
        
        # 最後のチャンクを処理
        if current_chunk and len(current_chunk) >= min_chunk_size:
            avg_embedding = np.mean(current_embeddings, axis=0)
            node = TextNode(
                text=current_chunk,
                metadata={
                    **document.metadata,
                    "chunking_strategy": "adaptive",
                    "chunk_index": chunk_index,
                    "chunk_length": len(current_chunk),
                    "semantic_coherence": np.mean([
                        np.dot(avg_embedding, emb) / (
                            np.linalg.norm(avg_embedding) * np.linalg.norm(emb)
                        ) for emb in current_embeddings
                    ]),
                    "document_index": doc_idx
                }
            )
            nodes.append(node)
        
        return nodes

class ChunkingStrategyFactory:
    """チャンキング戦略のファクトリークラス"""
    
    _strategies = {
        "fixed_size": FixedSizeChunkingStrategy,
        "token": TokenBasedChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
        "hierarchical": HierarchicalChunkingStrategy,
        "sentence": SentenceBasedChunkingStrategy,
        "adaptive": AdaptiveChunkingStrategy
    }
    
    @classmethod
    def create_strategy(cls, config: Dict[str, Any]) -> ChunkingStrategy:
        """設定に基づいてチャンキング戦略を作成"""
        strategy_type = config.get("type")
        
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy_type}")
        
        return cls._strategies[strategy_type](config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """利用可能な戦略のリストを取得"""
        return list(cls._strategies.keys())

class ChunkingPipeline:
    """チャンキングパイプライン"""
    
    def __init__(self, strategies_config: Dict[str, Dict[str, Any]]):
        self.strategies_config = strategies_config
        self.strategies = {}
        
        # 戦略を事前に初期化
        for name, config in strategies_config.items():
            self.strategies[name] = ChunkingStrategyFactory.create_strategy(config)
    
    def process_documents(self, documents: List[Document], 
                         strategy_name: str) -> List[TextNode]:
        """指定された戦略でドキュメントを処理"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        return strategy.chunk_documents(documents)
    
    def process_multiple_strategies(self, documents: List[Document], 
                                  strategy_names: List[str]) -> Dict[str, List[TextNode]]:
        """複数の戦略でドキュメントを処理"""
        results = {}
        
        for strategy_name in strategy_names:
            results[strategy_name] = self.process_documents(documents, strategy_name)
        
        return results
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """戦略の情報を取得"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        return self.strategies[strategy_name].get_metadata()
```

### 2. `src/chunking/quality_assessor.py`
```python
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from llama_index.core.schema import TextNode

class ChunkQualityAssessor:
    """チャンク品質評価クラス"""
    
    def __init__(self, semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def assess_chunk_quality(self, chunks: List[TextNode]) -> Dict[str, Any]:
        """チャンクの品質を総合評価"""
        if not chunks:
            return {"error": "No chunks provided"}
        
        chunk_texts = [chunk.text for chunk in chunks]
        
        assessment = {
            "total_chunks": len(chunks),
            "semantic_coherence": self.calculate_semantic_coherence(chunk_texts),
            "information_density": self.calculate_information_density(chunk_texts),
            "size_consistency": self.calculate_size_consistency(chunk_texts),
            "overlap_analysis": self.calculate_overlap_analysis(chunk_texts),
            "content_coverage": self.calculate_content_coverage(chunk_texts),
            "readability_scores": self.calculate_readability_scores(chunk_texts),
            "chunk_statistics": self.calculate_chunk_statistics(chunk_texts)
        }
        
        # 総合スコアの計算
        assessment["overall_quality_score"] = self._calculate_overall_score(assessment)
        
        return assessment
    
    def calculate_semantic_coherence(self, chunk_texts: List[str]) -> Dict[str, float]:
        """セマンティック一貫性の計算"""
        if len(chunk_texts) < 2:
            return {"coherence_score": 1.0, "std_coherence": 0.0}
        
        embeddings = self.semantic_model.encode(chunk_texts)
        
        # チャンク間の類似度を計算
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        # クラスタリングによる一貫性評価
        n_clusters = min(max(2, len(chunk_texts) // 3), 10)
        if len(chunk_texts) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        else:
            silhouette_avg = 0.5
        
        return {
            "coherence_score": np.mean(similarities),
            "std_coherence": np.std(similarities),
            "silhouette_score": max(0, silhouette_avg),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities)
        }
    
    def calculate_information_density(self, chunk_texts: List[str]) -> Dict[str, float]:
        """情報密度の計算"""
        densities = []
        unique_word_ratios = []
        
        for chunk in chunk_texts:
            words = re.findall(r'\b\w+\b', chunk.lower())
            if not words:
                densities.append(0.0)
                unique_word_ratios.append(0.0)
                continue
            
            # 語彙の多様性
            unique_words = len(set(words))
            total_words = len(words)
            unique_ratio = unique_words / total_words
            unique_word_ratios.append(unique_ratio)
            
            # 情報密度（単語あたりの文字数）
            char_count = len(re.sub(r'\s+', '', chunk))
            density = char_count / total_words if total_words > 0 else 0
            densities.append(density)
        
        # TF-IDFによる情報価値の評価
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
            tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=1)
        except ValueError:
            tfidf_scores = np.zeros(len(chunk_texts))
        
        return {
            "avg_density": np.mean(densities),
            "std_density": np.std(densities),
            "avg_unique_word_ratio": np.mean(unique_word_ratios),
            "std_unique_word_ratio": np.std(unique_word_ratios),
            "avg_tfidf_score": np.mean(tfidf_scores),
            "min_density": np.min(densities),
            "max_density": np.max(densities)
        }
    
    def calculate_size_consistency(self, chunk_texts: List[str]) -> Dict[str, float]:
        """サイズ一貫性の計算"""
        char_lengths = [len(chunk) for chunk in chunk_texts]
        word_lengths = [len(chunk.split()) for chunk in chunk_texts]
        
        return {
            "char_length_mean": np.mean(char_lengths),
            "char_length_std": np.std(char_lengths),
            "char_length_cv": np.std(char_lengths) / np.mean(char_lengths) if np.mean(char_lengths) > 0 else 0,
            "word_length_mean": np.mean(word_lengths),
            "word_length_std": np.std(word_lengths),
            "word_length_cv": np.std(word_lengths) / np.mean(word_lengths) if np.mean(word_lengths) 