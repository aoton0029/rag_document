"""
埋め込みファインチューニング実装
ドメイン特化型埋め込みの学習とアダプテーション
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import json
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .base_embedder import BaseEmbedder, EmbeddingResult
from .providers import SentenceTransformerEmbedder, HuggingFaceEmbedder
from ..utils import get_logger, performance_monitor

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class ContrastiveLearningDataset(Dataset):
    """対比学習用データセット"""
    
    def __init__(self, triplets: List[Tuple[str, str, str]]):
        """
        Args:
            triplets: (anchor, positive, negative) のタプルのリスト
        """
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]


class EmbeddingFineTuner:
    """埋め込みファインチューニングクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("embedding_fine_tuner")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for fine-tuning")
        
        # 基底モデル設定
        self.base_model_config = config.get("base_model", {})
        self.base_model_type = self.base_model_config.get("type", "sentence_transformers")
        
        # 学習設定
        self.learning_rate = config.get("learning_rate", 2e-5)
        self.batch_size = config.get("batch_size", 16)
        self.epochs = config.get("epochs", 3)
        self.warmup_steps = config.get("warmup_steps", 100)
        self.evaluation_steps = config.get("evaluation_steps", 500)
        
        # 出力ディレクトリ
        self.output_dir = Path(config.get("output_dir", "fine_tuned_models"))
        self.output_dir.mkdir(exist_ok=True)
        
        # デバイス設定
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info("EmbeddingFineTuner initialized",
                        base_model_type=self.base_model_type,
                        device=self.device)
    
    def prepare_training_data(self, 
                            positive_pairs: List[Tuple[str, str]],
                            negative_pairs: Optional[List[Tuple[str, str]]] = None,
                            hard_negatives: Optional[Dict[str, List[str]]] = None) -> List[InputExample]:
        """学習データを準備"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for fine-tuning")
        
        examples = []
        
        # ポジティブペアから学習例を作成
        for text1, text2 in positive_pairs:
            examples.append(InputExample(texts=[text1, text2], label=1.0))
        
        # ネガティブペアから学習例を作成
        if negative_pairs:
            for text1, text2 in negative_pairs:
                examples.append(InputExample(texts=[text1, text2], label=0.0))
        
        # ハードネガティブを使用
        if hard_negatives:
            for query, neg_texts in hard_negatives.items():
                for neg_text in neg_texts:
                    examples.append(InputExample(texts=[query, neg_text], label=0.0))
        
        self.logger.info("Training data prepared", 
                        total_examples=len(examples),
                        positive_pairs=len(positive_pairs),
                        negative_pairs=len(negative_pairs) if negative_pairs else 0)
        
        return examples
    
    def fine_tune_sentence_transformer(self,
                                     training_examples: List[InputExample],
                                     validation_examples: Optional[List[InputExample]] = None,
                                     model_name: Optional[str] = None) -> str:
        """SentenceTransformerモデルをファインチューニング"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required")
        
        # ベースモデルを読み込み
        base_model_name = self.base_model_config.get("model_name", "all-MiniLM-L6-v2")
        model = SentenceTransformer(base_model_name, device=self.device)
        
        self.logger.info("Started fine-tuning SentenceTransformer",
                        base_model=base_model_name,
                        training_examples=len(training_examples))
        
        # データローダー作成
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
        
        # 損失関数設定
        train_loss = losses.CosineSimilarityLoss(model)
        
        # 評価器設定
        evaluator = None
        if validation_examples:
            # 評価用のペアを作成
            sentences1 = [example.texts[0] for example in validation_examples]
            sentences2 = [example.texts[1] for example in validation_examples]
            scores = [example.label for example in validation_examples]
            
            evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
        
        # モデル名を設定
        if model_name is None:
            model_name = f"fine_tuned_{base_model_name.replace('/', '_')}"
        
        output_path = str(self.output_dir / model_name)
        
        # ファインチューニング実行
        with performance_monitor("fine_tuning"):
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.epochs,
                warmup_steps=self.warmup_steps,
                evaluator=evaluator,
                evaluation_steps=self.evaluation_steps,
                output_path=output_path,
                save_best_model=True,
                show_progress_bar=True
            )
        
        self.logger.info("Fine-tuning completed", 
                        output_path=output_path,
                        epochs=self.epochs)
        
        return output_path
    
    def create_domain_adapter(self, 
                            domain_texts: List[str],
                            domain_labels: Optional[List[str]] = None,
                            adaptation_method: str = "linear_probe") -> "DomainAdapter":
        """ドメイン特化アダプターを作成"""
        
        # ベース埋め込みを生成
        base_embedder = self._create_base_embedder()
        base_embeddings = base_embedder.encode_texts(domain_texts)
        
        # アダプター作成
        adapter = DomainAdapter(
            base_embeddings=base_embeddings.embeddings,
            domain_texts=domain_texts,
            domain_labels=domain_labels,
            method=adaptation_method,
            config=self.config
        )
        
        adapter.fit()
        
        return adapter
    
    def _create_base_embedder(self) -> BaseEmbedder:
        """ベース埋め込み生成器を作成"""
        if self.base_model_type == "sentence_transformers":
            return SentenceTransformerEmbedder(self.base_model_config)
        elif self.base_model_type == "huggingface":
            return HuggingFaceEmbedder(self.base_model_config)
        else:
            raise ValueError(f"Unsupported base model type: {self.base_model_type}")
    
    def evaluate_fine_tuned_model(self,
                                 model_path: str,
                                 test_examples: List[InputExample]) -> Dict[str, float]:
        """ファインチューニングされたモデルを評価"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required")
        
        # ファインチューニング済みモデルを読み込み
        model = SentenceTransformer(model_path, device=self.device)
        
        # 評価データを準備
        sentences1 = [example.texts[0] for example in test_examples]
        sentences2 = [example.texts[1] for example in test_examples]
        true_scores = [example.label for example in test_examples]
        
        # 埋め込みを生成
        embeddings1 = model.encode(sentences1)
        embeddings2 = model.encode(sentences2)
        
        # コサイン類似度を計算
        predicted_scores = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            predicted_scores.append(similarity)
        
        # 評価指標を計算
        mse = np.mean((np.array(predicted_scores) - np.array(true_scores)) ** 2)
        correlation = np.corrcoef(predicted_scores, true_scores)[0, 1]
        
        # 分類性能（閾値0.5で二値分類として評価）
        threshold = 0.5
        predicted_binary = [1 if score > threshold else 0 for score in predicted_scores]
        true_binary = [1 if score > threshold else 0 for score in true_scores]
        
        accuracy = np.mean(np.array(predicted_binary) == np.array(true_binary))
        
        results = {
            "mse": float(mse),
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "accuracy": float(accuracy)
        }
        
        self.logger.info("Model evaluation completed", **results)
        
        return results
    
    def generate_hard_negatives(self,
                              queries: List[str],
                              candidate_pool: List[str],
                              base_embedder: BaseEmbedder,
                              top_k: int = 5) -> Dict[str, List[str]]:
        """ハードネガティブサンプルを生成"""
        
        # クエリと候補プールの埋め込みを生成
        query_embeddings = base_embedder.encode_texts(queries)
        candidate_embeddings = base_embedder.encode_texts(candidate_pool)
        
        query_matrix = np.array(query_embeddings.embeddings)
        candidate_matrix = np.array(candidate_embeddings.embeddings)
        
        # 類似度計算
        similarities = cosine_similarity(query_matrix, candidate_matrix)
        
        hard_negatives = {}
        
        for i, query in enumerate(queries):
            # 類似度が高い（でも正解ではない）候補をハードネガティブとして選択
            sim_scores = similarities[i]
            
            # 類似度でソート（降順）
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            # 上位の類似候補をハードネガティブとして選択
            # 実際の実装では、正解候補を除外する必要がある
            hard_neg_indices = sorted_indices[:top_k]
            hard_negatives[query] = [candidate_pool[idx] for idx in hard_neg_indices]
        
        self.logger.info("Hard negatives generated",
                        query_count=len(queries),
                        avg_negatives_per_query=np.mean([len(negs) for negs in hard_negatives.values()]))
        
        return hard_negatives


class DomainAdapter:
    """ドメイン適応アダプタークラス"""
    
    def __init__(self,
                 base_embeddings: List[List[float]],
                 domain_texts: List[str],
                 domain_labels: Optional[List[str]] = None,
                 method: str = "linear_probe",
                 config: Dict[str, Any] = None):
        
        self.base_embeddings = np.array(base_embeddings)
        self.domain_texts = domain_texts
        self.domain_labels = domain_labels
        self.method = method
        self.config = config or {}
        
        self.logger = get_logger("domain_adapter")
        self.adapter_matrix = None
        self.is_fitted = False
    
    def fit(self) -> None:
        """アダプターを学習"""
        
        if self.method == "linear_probe":
            self._fit_linear_probe()
        elif self.method == "pca_adaptation":
            self._fit_pca_adaptation()
        elif self.method == "autoencoder":
            self._fit_autoencoder()
        else:
            raise ValueError(f"Unknown adaptation method: {self.method}")
        
        self.is_fitted = True
        self.logger.info(f"Domain adapter fitted using {self.method}")
    
    def _fit_linear_probe(self) -> None:
        """線形変換による適応"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # 標準化
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(self.base_embeddings)
        
        # PCAで主要な方向を学習
        n_components = min(self.config.get("adapter_dim", 256), normalized_embeddings.shape[1])
        pca = PCA(n_components=n_components)
        
        adapted_embeddings = pca.fit_transform(normalized_embeddings)
        
        # 線形変換行列を保存
        self.adapter_matrix = pca.components_.T
        self.scaler = scaler
        
    def _fit_pca_adaptation(self) -> None:
        """PCAベースの適応"""
        from sklearn.decomposition import PCA
        
        # ドメイン特化のPCA学習
        pca = PCA(n_components=self.config.get("pca_components", 256))
        adapted = pca.fit_transform(self.base_embeddings)
        
        self.adapter_matrix = pca.components_.T
        self.pca = pca
    
    def _fit_autoencoder(self) -> None:
        """オートエンコーダーベースの適応"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for autoencoder adaptation")
        
        # シンプルなオートエンコーダー
        input_dim = self.base_embeddings.shape[1]
        hidden_dim = self.config.get("hidden_dim", 256)
        
        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Linear(input_dim, hidden_dim)
                self.decoder = nn.Linear(hidden_dim, input_dim)
                
            def forward(self, x):
                encoded = torch.relu(self.encoder(x))
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleAutoencoder(input_dim, hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # データ準備
        X = torch.FloatTensor(self.base_embeddings).to(device)
        
        # 訓練
        epochs = self.config.get("autoencoder_epochs", 50)
        for epoch in range(epochs):
            encoded, decoded = model(X)
            loss = criterion(decoded, X)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.autoencoder = model
    
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """埋め込みを変換"""
        if not self.is_fitted:
            raise ValueError("Adapter must be fitted before transformation")
        
        embeddings_array = np.array(embeddings)
        
        if self.method == "linear_probe":
            normalized = self.scaler.transform(embeddings_array)
            transformed = normalized @ self.adapter_matrix
        
        elif self.method == "pca_adaptation":
            transformed = self.pca.transform(embeddings_array)
        
        elif self.method == "autoencoder":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X = torch.FloatTensor(embeddings_array).to(device)
            with torch.no_grad():
                transformed, _ = self.autoencoder(X)
            transformed = transformed.cpu().numpy()
        
        return transformed.tolist()
    
    def save(self, filepath: str) -> None:
        """アダプターを保存"""
        adapter_data = {
            "method": self.method,
            "config": self.config,
            "is_fitted": self.is_fitted
        }
        
        if self.method == "linear_probe":
            adapter_data["adapter_matrix"] = self.adapter_matrix.tolist()
            adapter_data["scaler_mean"] = self.scaler.mean_.tolist()
            adapter_data["scaler_scale"] = self.scaler.scale_.tolist()
        
        elif self.method == "pca_adaptation":
            adapter_data["components"] = self.pca.components_.tolist()
            adapter_data["mean"] = self.pca.mean_.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(adapter_data, f, indent=2)
        
        # PyTorchモデルは別途保存
        if self.method == "autoencoder":
            torch.save(self.autoencoder.state_dict(), 
                      filepath.replace('.json', '_model.pth'))
    
    def load(self, filepath: str) -> None:
        """アダプターを読み込み"""
        with open(filepath, 'r') as f:
            adapter_data = json.load(f)
        
        self.method = adapter_data["method"]
        self.config = adapter_data["config"]
        self.is_fitted = adapter_data["is_fitted"]
        
        if self.method == "linear_probe":
            self.adapter_matrix = np.array(adapter_data["adapter_matrix"])
            
            # スケーラーを復元
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(adapter_data["scaler_mean"])
            self.scaler.scale_ = np.array(adapter_data["scaler_scale"])
        
        elif self.method == "pca_adaptation":
            from sklearn.decomposition import PCA
            self.pca = PCA()
            self.pca.components_ = np.array(adapter_data["components"])
            self.pca.mean_ = np.array(adapter_data["mean"])


class FineTunedEmbedder(BaseEmbedder):
    """ファインチューニング済み埋め込み生成器"""
    
    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__(config)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required")
        
        self.model_path = model_path
        self.model = SentenceTransformer(model_path)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        
        # ドメインアダプター（オプション）
        self.domain_adapter = None
        adapter_path = config.get("domain_adapter_path")
        if adapter_path:
            self.domain_adapter = DomainAdapter([], [], method="linear_probe")
            self.domain_adapter.load(adapter_path)
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """ファインチューニング済みモデルでテキストを埋め込み"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings_list = embeddings.tolist()
        
        # ドメイン適応を適用
        if self.domain_adapter and self.domain_adapter.is_fitted:
            embeddings_list = self.domain_adapter.transform(embeddings_list)
        
        return embeddings_list
    
    def get_embedding_dimensions(self) -> int:
        return self.dimensions