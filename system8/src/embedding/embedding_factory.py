"""
埋め込みモデルのファクトリークラスと評価機能
"""
from typing import Dict, Any, List, Union
from . import BaseEmbedding
from .huggingface_embedding import HuggingFaceEmbedding, TransformersEmbedding, JapaneseEmbedding
from .openai_embedding import OpenAIEmbedding, AzureOpenAIEmbedding
from .ollama_embedding import OllamaEmbedding, LlamaIndexOllamaEmbedding

class EmbeddingFactory:
    """埋め込みモデル作成のファクトリークラス"""
    
    @staticmethod
    def create_embedding(provider: str, model_name: str, config: Dict[str, Any]) -> BaseEmbedding:
        """指定されたプロバイダーとモデルで埋め込みを作成"""
        
        if provider == "openai":
            return OpenAIEmbedding(config)
        
        elif provider == "azure_openai":
            return AzureOpenAIEmbedding(config)
        
        elif provider == "huggingface":
            if "sentence-transformers" in model_name:
                return HuggingFaceEmbedding(config)
            else:
                return TransformersEmbedding(config)
        
        elif provider == "japanese":
            return JapaneseEmbedding(config)
        
        elif provider == "ollama":
            if config.get('use_llamaindex', False):
                return LlamaIndexOllamaEmbedding(config)
            else:
                return OllamaEmbedding(config)
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    @staticmethod
    def create_from_config_key(config_key: str, embedding_configs: Dict[str, Any]) -> BaseEmbedding:
        """設定キーから埋め込みモデルを作成"""
        # config_key形式: "provider/model_name"
        provider, model_path = config_key.split('/', 1)
        
        provider_config = embedding_configs.get(provider, {})
        model_config = provider_config.get(model_path, {})
        
        if not model_config:
            raise ValueError(f"Model config not found for: {config_key}")
        
        return EmbeddingFactory.create_embedding(provider, model_path, model_config)

class EmbeddingEnsemble:
    """複数の埋め込みモデルのアンサンブル"""
    
    def __init__(self, embeddings: List[BaseEmbedding], weights: List[float] = None):
        self.embeddings = embeddings
        self.weights = weights or [1.0 / len(embeddings)] * len(embeddings)
        
        if len(self.embeddings) != len(self.weights):
            raise ValueError("Number of embeddings must match number of weights")
        
        # 重みを正規化
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def embed_text(self, text: Union[str, List[str]]):
        """アンサンブル埋め込み"""
        import numpy as np
        
        results = []
        for embedding in self.embeddings:
            result = embedding.embed_text(text)
            results.append(result)
        
        # 重み付き平均を計算
        weighted_embeddings = []
        for i, result in enumerate(results):
            weighted_embeddings.append(result.embeddings * self.weights[i])
        
        ensemble_embeddings = np.sum(weighted_embeddings, axis=0)
        
        # 正規化
        from sklearn.preprocessing import normalize
        ensemble_embeddings = normalize(ensemble_embeddings, norm='l2')
        
        return {
            'embeddings': ensemble_embeddings,
            'individual_results': results,
            'weights': self.weights
        }

def evaluate_embedding_quality(embedding_model: BaseEmbedding, 
                             test_data: List[Dict[str, Any]], 
                             metrics: List[str]) -> Dict[str, float]:
    """埋め込み品質を評価"""
    results = {}
    
    if "semantic_textual_similarity" in metrics:
        results["semantic_textual_similarity"] = _evaluate_sts(embedding_model, test_data)
    
    if "information_retrieval" in metrics:
        results["information_retrieval"] = _evaluate_ir(embedding_model, test_data)
    
    if "clustering" in metrics:
        results["clustering"] = _evaluate_clustering(embedding_model, test_data)
    
    if "classification" in metrics:
        results["classification"] = _evaluate_classification(embedding_model, test_data)
    
    return results

def _evaluate_sts(embedding_model: BaseEmbedding, test_data: List[Dict[str, Any]]) -> float:
    """セマンティック類似度評価"""
    from scipy.stats import spearmanr
    import numpy as np
    
    predicted_similarities = []
    ground_truth_similarities = []
    
    for item in test_data:
        if 'text1' not in item or 'text2' not in item or 'similarity' not in item:
            continue
            
        # 埋め込み計算
        result1 = embedding_model.embed_text(item['text1'])
        result2 = embedding_model.embed_text(item['text2'])
        
        # コサイン類似度計算
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            result1.embeddings.reshape(1, -1),
            result2.embeddings.reshape(1, -1)
        )[0][0]
        
        predicted_similarities.append(similarity)
        ground_truth_similarities.append(item['similarity'])
    
    if len(predicted_similarities) == 0:
        return 0.0
    
    # スピアマン相関
    correlation, _ = spearmanr(predicted_similarities, ground_truth_similarities)
    return correlation if not np.isnan(correlation) else 0.0

def _evaluate_ir(embedding_model: BaseEmbedding, test_data: List[Dict[str, Any]]) -> float:
    """情報検索性能評価"""
    # 簡易実装：クエリと関連文書の類似度評価
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    scores = []
    
    for item in test_data:
        if 'query' not in item or 'documents' not in item or 'relevance' not in item:
            continue
            
        query_embedding = embedding_model.embed_text(item['query']).embeddings
        
        similarities = []
        for doc in item['documents']:
            doc_embedding = embedding_model.embed_text(doc).embeddings
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        # 関連度との相関を計算
        if len(similarities) > 1:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(similarities, item['relevance'])
            if not np.isnan(correlation):
                scores.append(correlation)
    
    return np.mean(scores) if scores else 0.0

def _evaluate_clustering(embedding_model: BaseEmbedding, test_data: List[Dict[str, Any]]) -> float:
    """クラスタリング性能評価"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    
    if not test_data:
        return 0.0
    
    texts = [item.get('text', '') for item in test_data if 'text' in item]
    labels = [item.get('label', 0) for item in test_data if 'label' in item]
    
    if len(texts) != len(labels) or len(set(labels)) < 2:
        return 0.0
    
    # 埋め込み計算
    embeddings = []
    for text in texts:
        result = embedding_model.embed_text(text)
        embeddings.append(result.embeddings[0])
    
    embeddings = np.array(embeddings)
    
    # K-means クラスタリング
    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    # ARI (Adjusted Rand Index) で評価
    return adjusted_rand_score(labels, predicted_labels)

def _evaluate_classification(embedding_model: BaseEmbedding, test_data: List[Dict[str, Any]]) -> float:
    """分類性能評価"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    if not test_data:
        return 0.0
    
    texts = [item.get('text', '') for item in test_data if 'text' in item and 'label' in item]
    labels = [item.get('label') for item in test_data if 'text' in item and 'label' in item]
    
    if len(texts) < 10 or len(set(labels)) < 2:  # 最小データ数チェック
        return 0.0
    
    # 埋め込み計算
    embeddings = []
    for text in texts:
        result = embedding_model.embed_text(text)
        embeddings.append(result.embeddings[0])
    
    embeddings = np.array(embeddings)
    
    # ラベルエンコーディング
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # ロジスティック回帰でクロスバリデーション
    clf = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(clf, embeddings, encoded_labels, cv=min(5, len(set(labels))))
    
    return np.mean(scores)