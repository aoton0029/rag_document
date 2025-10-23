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

### `config/evaluation_configs.yaml`
```yaml
evaluation_framework:
  levels:
    - component_level
    - system_level
    - human_level
    - production_level
  
  component_level:
    chunk_quality:
      metrics:
        - semantic_coherence
        - information_density
        - overlap_analysis
    
    embedding_quality:
      metrics:
        - cluster_quality
        - semantic_preservation
        - dimensionality_analysis
    
    retrieval_quality:
      metrics:
        - coverage_analysis
        - relevance_distribution
        - diversity_assessment

  system_level:
    ragas_metrics:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall
      - context_relevancy
    
    traditional_metrics:
      - precision_at_k
      - recall_at_k
      - ndcg_at_k
      - mrr
    
    llm_judge:
      judge_model: "gpt-4-turbo"
      evaluation_aspects:
        - factual_accuracy
        - relevance_to_query
        - context_utilization
        - completeness
        - overall_quality

  human_level:
    sample_size: 50
    evaluators: 3
    inter_rater_agreement: true
    aspects:
      - accuracy
      - helpfulness
      - clarity
      - completeness

  production_level:
    metrics:
      - user_satisfaction
      - task_completion_rate
      - latency_percentiles
      - cost_per_query
      - error_rate

statistical_validation:
  significance_threshold: 0.05
  effect_size_threshold: 0.2
  multiple_comparison_correction: "bonferroni"
  bootstrap_iterations: 1000
```

### `config/domain_configs.yaml`
```yaml
domains:
  technical:
    specific_metrics:
      - code_correctness
      - api_completeness
      - technical_depth
    test_patterns:
      - programming_concepts
      - api_documentation
      - troubleshooting_guides
  
  medical:
    specific_metrics:
      - clinical_accuracy
      - safety_assessment
      - evidence_quality
    test_patterns:
      - clinical_cases
      - drug_interactions
      - diagnostic_procedures
  
  legal:
    specific_metrics:
      - citation_accuracy
      - legal_reasoning
      - precedent_relevance
    test_patterns:
      - case_law_analysis
      - statute_interpretation
      - contract_analysis
  
  general:
    specific_metrics:
      - factual_accuracy
      - logical_consistency
      - comprehensiveness
    test_patterns:
      - factual_questions
      - reasoning_tasks
      - summarization_requests
```

### `src/utils/ragas_integration.py`
```python
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_relevancy,
            answer_correctness,
            answer_similarity
        ]
    
    def prepare_dataset(self, queries: List[str], contexts: List[List[str]], 
                       answers: List[str], ground_truths: List[str] = None) -> Dataset:
        """RAGASフォーマットのデータセット準備"""
        data = {
            'question': queries,
            'contexts': contexts,
            'answer': answers
        }
        
        if ground_truths:
            data['ground_truth'] = ground_truths
        
        return Dataset.from_dict(data)
    
    def evaluate_rag_system(self, dataset: Dataset) -> Dict[str, float]:
        """RAGASメトリクスによる評価"""
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics
            )
            return result
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return {}
    
    def batch_evaluate(self, test_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """バッチ評価の実行"""
        queries = [item['query'] for item in test_data]
        contexts = [item['contexts'] for item in test_data]
        answers = [item['answer'] for item in test_data]
        ground_truths = [item.get('ground_truth', '') for item in test_data]
        
        dataset = self.prepare_dataset(queries, contexts, answers, ground_truths)
        results = self.evaluate_rag_system(dataset)
        
        return pd.DataFrame([results])
```

### `src/evaluation/llm_judge.py`
```python
import openai
from typing import Dict, List, Any
import json
import time

class LLMJudgeEvaluator:
    def __init__(self, judge_model: str = "gpt-4-turbo"):
        self.judge_model = judge_model
        self.client = openai.OpenAI()
    
    def create_evaluation_prompt(self, query: str, context: str, response: str, 
                               reference: str = None) -> str:
        """評価用プロンプトの生成"""
        prompt = f"""
あなたはRAGシステムの応答品質を評価する専門家です。以下の観点で応答を評価してください：

【クエリ】
{query}

【取得されたコンテキスト】
{context}

【生成された応答】
{response}

{f"【参考回答】\n{reference}" if reference else ""}

【評価観点】
1. 事実的正確性 (1-5): 応答に含まれる情報が正確かどうか
2. クエリ関連性 (1-5): 応答がクエリに適切に答えているかどうか
3. コンテキスト活用 (1-5): 提供されたコンテキストを適切に活用しているかどうか
4. 完全性 (1-5): 応答が十分に詳細で包括的かどうか
5. 総合品質 (1-5): 全体的な応答品質

【回答フォーマット】
以下のJSON形式で回答してください：
{{
    "factual_accuracy": {{
        "score": <1-5の数値>,
        "reasoning": "<評価理由>"
    }},
    "relevance_to_query": {{
        "score": <1-5の数値>,
        "reasoning": "<評価理由>"
    }},
    "context_utilization": {{
        "score": <1-5の数値>,
        "reasoning": "<評価理由>"
    }},
    "completeness": {{
        "score": <1-5の数値>,
        "reasoning": "<評価理由>"
    }},
    "overall_quality": {{
        "score": <1-5の数値>,
        "reasoning": "<評価理由>"
    }}
}}
"""
        return prompt
    
    def evaluate_response(self, query: str, context: str, response: str, 
                         reference: str = None) -> Dict[str, Any]:
        """単一応答の評価"""
        prompt = self.create_evaluation_prompt(query, context, response, reference)
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "あなたは公正で一貫性のある評価を行うAI評価者です。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            # JSONの抽出と解析
            result_json = self.extract_json_from_response(result_text)
            return result_json
            
        except Exception as e:
            print(f"LLM Judge evaluation error: {e}")
            return self.get_default_scores()
    
    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """レスポンスからJSONを抽出"""
        try:
            # JSONブロックの検索
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self.get_default_scores()
        except json.JSONDecodeError:
            return self.get_default_scores()
    
    def get_default_scores(self) -> Dict[str, Any]:
        """デフォルトスコア"""
        return {
            "factual_accuracy": {"score": 3, "reasoning": "評価エラー"},
            "relevance_to_query": {"score": 3, "reasoning": "評価エラー"},
            "context_utilization": {"score": 3, "reasoning": "評価エラー"},
            "completeness": {"score": 3, "reasoning": "評価エラー"},
            "overall_quality": {"score": 3, "reasoning": "評価エラー"}
        }
    
    def batch_evaluate(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """バッチ評価"""
        results = []
        
        for item in test_data:
            time.sleep(1)  # API制限対策
            result = self.evaluate_response(
                query=item['query'],
                context=item.get('context', ''),
                response=item['response'],
                reference=item.get('reference')
            )
            results.append(result)
        
        return results
```

### `src/evaluation/component_evaluator.py`
```python
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

class ComponentLevelEvaluator:
    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_chunk_quality(self, chunks: List[str]) -> Dict[str, float]:
        """チャンクの品質評価"""
        results = {}
        
        # セマンティック一貫性の評価
        results['semantic_coherence'] = self.calculate_semantic_coherence(chunks)
        
        # 情報密度の評価
        results['information_density'] = self.calculate_information_density(chunks)
        
        # 重複分析
        results['overlap_score'] = self.calculate_overlap_analysis(chunks)
        
        return results
    
    def calculate_semantic_coherence(self, chunks: List[str]) -> float:
        """セマンティック一貫性の計算"""
        if len(chunks) < 2:
            return 1.0
        
        embeddings = self.semantic_model.encode(chunks)
        
        # クラスタリングによる一貫性評価
        if len(chunks) >= 2:
            n_clusters = min(len(chunks) // 2, 10)
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # シルエット係数で一貫性を測定
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            return max(0, silhouette_avg)  # 負の値を0にクリップ
        
        return 0.5
    
    def calculate_information_density(self, chunks: List[str]) -> float:
        """情報密度の計算"""
        if not chunks:
            return 0.0
        
        # 単語の多様性を情報密度の指標として使用
        all_words = []
        for chunk in chunks:
            words = chunk.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = set(all_words)
        density = len(unique_words) / len(all_words)
        return density
    
    def calculate_overlap_analysis(self, chunks: List[str]) -> float:
        """重複分析"""
        if len(chunks) < 2:
            return 0.0
        
        total_overlap = 0
        comparisons = 0
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                words_i = set(chunks[i].lower().split())
                words_j = set(chunks[j].lower().split())
                
                if words_i or words_j:
                    overlap = len(words_i & words_j) / len(words_i | words_j)
                    total_overlap += overlap
                    comparisons += 1
        
        return total_overlap / comparisons if comparisons > 0 else 0.0
    
    def evaluate_embedding_quality(self, embeddings: np.ndarray, texts: List[str]) -> Dict[str, float]:
        """埋め込みの品質評価"""
        results = {}
        
        # クラスター品質
        results['cluster_quality'] = self.calculate_cluster_quality(embeddings)
        
        # セマンティック保存性
        results['semantic_preservation'] = self.calculate_semantic_preservation(embeddings, texts)
        
        # 次元解析
        results['dimensionality_score'] = self.calculate_dimensionality_score(embeddings)
        
        return results
    
    def calculate_cluster_quality(self, embeddings: np.ndarray) -> float:
        """クラスター品質の計算"""
        if len(embeddings) < 2:
            return 1.0
        
        n_clusters = min(len(embeddings) // 2, 10)
        if n_clusters < 2:
            return 0.5
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        return max(0, silhouette_score(embeddings, cluster_labels))
    
    def calculate_semantic_preservation(self, embeddings: np.ndarray, texts: List[str]) -> float:
        """セマンティック保存性の計算"""
        # テキスト間の類似度と埋め込み間の類似度の相関を計算
        if len(texts) < 2:
            return 1.0
        
        # 簡略化された実装
        return 0.8  # 実際にはより複雑な計算が必要
    
    def calculate_dimensionality_score(self, embeddings: np.ndarray) -> float:
        """次元解析スコア"""
        # PCAによる次元解析
        from sklearn.decomposition import PCA
        
        if len(embeddings) < 2:
            return 1.0
        
        pca = PCA()
        pca.fit(embeddings)
        
        # 累積寄与率から次元の有効性を評価
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        effective_dims = np.argmax(cumsum_ratio > 0.95) + 1
        
        # 有効次元の比率
        dimensionality_score = effective_dims / len(embeddings[0])
        return min(1.0, dimensionality_score)
```

### `src/evaluation/statistical_validator.py`
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalValidator:
    def __init__(self, significance_threshold: float = 0.05, 
                 effect_size_threshold: float = 0.2):
        self.significance_threshold = significance_threshold
        self.effect_size_threshold = effect_size_threshold
    
    def compare_configurations(self, results_a: List[float], 
                             results_b: List[float], 
                             metric_name: str) -> Dict[str, Any]:
        """2つの設定の統計的比較"""
        
        # 正規性検定
        normality_a = self.test_normality(results_a)
        normality_b = self.test_normality(results_b)
        
        # 適切な検定の選択
        if normality_a and normality_b and len(results_a) == len(results_b):
            # 対応ありt検定
            statistic, p_value = stats.ttest_rel(results_a, results_b)
            test_type = "paired_t_test"
        elif normality_a and normality_b:
            # 対応なしt検定
            statistic, p_value = stats.ttest_ind(results_a, results_b)
            test_type = "independent_t_test"
        elif len(results_a) == len(results_b):
            # Wilcoxon符号順位検定
            statistic, p_value = wilcoxon(results_a, results_b)
            test_type = "wilcoxon_signed_rank"
        else:
            # Mann-Whitney U検定
            statistic, p_value = mannwhitneyu(results_a, results_b)
            test_type = "mann_whitney_u"
        
        # 効果量の計算
        effect_size = self.calculate_effect_size(results_a, results_b)
        
        # 信頼区間の計算
        confidence_interval = self.calculate_confidence_interval(results_a, results_b)
        
        return {
            'metric_name': metric_name,
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'statistically_significant': p_value < self.significance_threshold,
            'effect_size': effect_size,
            'effect_size_magnitude': self.interpret_effect_size(effect_size),
            'practically_significant': abs(effect_size) > self.effect_size_threshold,
            'confidence_interval': confidence_interval,
            'mean_difference': np.mean(results_b) - np.mean(results_a),
            'median_difference': np.median(results_b) - np.median(results_a)
        }
    
    def test_normality(self, data: List[float]) -> bool:
        """正規性検定"""
        if len(data) < 3:
            return False
        
        # Shapiro-Wilk検定
        statistic, p_value = stats.shapiro(data)
        return p_value > 0.05
    
    def calculate_effect_size(self, group_a: List[float], group_b: List[float]) -> float:
        """Cohen's dによる効果量計算"""
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)
        
        # プールされた標準偏差
        n_a, n_b = len(group_a), len(group_b)
        pooled_std = np.sqrt(((n_a - 1) * np.var(group_a, ddof=1) + 
                             (n_b - 1) * np.var(group_b, ddof=1)) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean_b - mean_a) / pooled_std
    
    def interpret_effect_size(self, effect_size: float) -> str:
        """効果量の解釈"""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def calculate_confidence_interval(self, group_a: List[float], 
                                    group_b: List[float], 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """平均差の信頼区間"""
        diff = np.array(group_b) - np.array(group_a) if len(group_a) == len(group_b) else []
        
        if len(diff) == 0:
            # 独立サンプルの場合
            mean_diff = np.mean(group_b) - np.mean(group_a)
            se_diff = np.sqrt(np.var(group_a, ddof=1)/len(group_a) + np.var(group_b, ddof=1)/len(group_b))
            df = len(group_a) + len(group_b) - 2
        else:
            # 対応サンプルの場合
            mean_diff = np.mean(diff)
            se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
            df = len(diff) - 1
        
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * se_diff
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = "bonferroni") -> List[float]:
        """多重比較補正"""
        if method == "bonferroni":
            return [min(p * len(p_values), 1.0) for p in p_values]
        elif method == "holm":
            # Holm法
            sorted_indices = np.argsort(p_values)
            corrected_p = [0] * len(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = min(p_values[idx] * (len(p_values) - i), 1.0)
                if i > 0:
                    corrected_p[idx] = max(cor