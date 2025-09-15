import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging
from pathlib import Path
import json
from ..database.vector_db.milvus_client import MilvusClient
from ..database.keyvalue_db.redis_client import RedisClient


class TransformersEmbeddingService:
    """Transformersライブラリを使用した埋め込み生成サービス"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 milvus_client: MilvusClient = None,
                 redis_client: RedisClient = None):
        """
        初期化
        
        Args:
            model_name: 使用するモデル名
            device: 実行デバイス (cuda, cpu, auto)
            cache_dir: モデルキャッシュディレクトリ
            milvus_client: Milvusクライアント
            redis_client: Redisキライアント
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "./models/cache"
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.sentence_transformer = None
        self.embedding_dimension = None
        
        # Database clients
        self.milvus_client = milvus_client or MilvusClient()
        self.redis_client = redis_client or RedisClient()
        
        # ロギング設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # モデル初期化
        self._initialize_model()
    
    def _setup_device(self, device: Optional[str]) -> str:
        """デバイス設定"""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon Mac
            else:
                return "cpu"
        return device
    
    def _initialize_model(self):
        """モデルを初期化"""
        try:
            self.logger.info(f"モデル初期化開始: {self.model_name} on {self.device}")
            
            # sentence-transformersモデルを優先的に使用
            try:
                self.sentence_transformer = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_dir
                )
                self.embedding_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
                self.logger.info(f"SentenceTransformerモデル初期化完了: 次元数 {self.embedding_dimension}")
                return
            except Exception as e:
                self.logger.warning(f"SentenceTransformer初期化失敗: {e}")
            
            # transformersモデルにフォールバック
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            # 次元数を取得
            with torch.no_grad():
                dummy_input = self.tokenizer("test", return_tensors="pt", truncation=True, padding=True)
                dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
                outputs = self.model(**dummy_input)
                self.embedding_dimension = outputs.last_hidden_state.shape[-1]
            
            self.logger.info(f"Transformersモデル初期化完了: 次元数 {self.embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"モデル初期化エラー: {e}")
            raise
    
    def encode_texts(self, texts: List[str], 
                    batch_size: int = 32,
                    normalize_embeddings: bool = True,
                    show_progress: bool = False) -> np.ndarray:
        """
        テキストリストを埋め込みベクトルに変換
        
        Args:
            texts: 埋め込み対象のテキストリスト
            batch_size: バッチサイズ
            normalize_embeddings: ベクトル正規化フラグ
            show_progress: 進捗表示フラグ
            
        Returns:
            埋め込みベクトル配列 (shape: [len(texts), embedding_dimension])
        """
        if not texts:
            return np.array([])
        
        try:
            if self.sentence_transformer:
                embeddings = self.sentence_transformer.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
            else:
                embeddings = self._encode_with_transformers(
                    texts, batch_size, normalize_embeddings
                )
            
            self.logger.info(f"埋め込み生成完了: {len(texts)} texts -> {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"埋め込み生成エラー: {e}")
            raise
    
    def _encode_with_transformers(self, texts: List[str], 
                                 batch_size: int = 32,
                                 normalize_embeddings: bool = True) -> np.ndarray:
        """transformersライブラリで埋め込み生成"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # トークナイズ
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # [CLS]トークンの埋め込みを使用
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """単一テキストの埋め込み生成"""
        return self.encode_texts([text], normalize_embeddings=normalize)[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """2つのテキスト間の類似度計算"""
        embeddings = self.encode_texts([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """テキストリスト内の全ペア類似度行列を計算"""
        embeddings = self.encode_texts(texts)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        クエリテキストに最も類似するテキストを検索
        
        Args:
            query_text: 検索クエリ
            candidate_texts: 候補テキストリスト
            top_k: 上位k件を返す
            
        Returns:
            類似度順にソートされた結果リスト
        """
        if not candidate_texts:
            return []
        
        # 埋め込み生成
        all_texts = [query_text] + candidate_texts
        embeddings = self.encode_texts(all_texts)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # 類似度計算
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # 結果をソート
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in indices:
            results.append({
                "text": candidate_texts[idx],
                "similarity": float(similarities[idx]),
                "index": int(idx)
            })
        
        return results
    
    def chunk_and_embed(self, text: str, chunk_size: int = 512, 
                       overlap: int = 50) -> List[Dict[str, Any]]:
        """
        長いテキストをチャンクに分割して埋め込み生成
        
        Args:
            text: 対象テキスト
            chunk_size: チャンクサイズ（文字数）
            overlap: オーバーラップサイズ
            
        Returns:
            チャンク情報と埋め込みのリスト
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end,
                "chunk_id": len(chunks)
            })
            
            if end == len(text):
                break
                
            start = end - overlap
        
        # 埋め込み生成
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.encode_texts(chunk_texts)
        
        # 結果にembeddingを追加
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        return chunks
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       metadata: Dict[str, Any], 
                       output_path: str):
        """埋め込みとメタデータを保存"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "embeddings": embeddings.tolist(),
            "metadata": metadata,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"埋め込みデータ保存完了: {output_path}")
    
    def load_embeddings(self, input_path: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """保存された埋め込みとメタデータを読み込み"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embeddings = np.array(data["embeddings"])
        metadata = data["metadata"]
        
        self.logger.info(f"埋め込みデータ読み込み完了: {input_path}")
        return embeddings, metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "model_type": "sentence-transformer" if self.sentence_transformer else "transformers",
            "cache_dir": self.cache_dir
        }
    
    def cleanup(self):
        """リソースクリーンアップ"""
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'sentence_transformer') and self.sentence_transformer:
            del self.sentence_transformer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("リソースクリーンアップ完了")


# 使用例とテスト用関数
    
    # Database integration methods
    async def store_embeddings_to_milvus(self, 
                                       texts: List[str], 
                                       document_ids: List[str],
                                       metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """埋め込みをMilvusに保存"""
        try:
            embeddings = self.encode_texts(texts)
            
            for i, (text, doc_id, embedding) in enumerate(zip(texts, document_ids, embeddings)):
                meta = metadata[i] if metadata else {}
                success = await self.milvus_client.insert_embedding(
                    document_id=doc_id,
                    content=text,
                    embedding=embedding.tolist(),
                    metadata=meta
                )
                if not success:
                    self.logger.error(f"Failed to store embedding for document {doc_id}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing embeddings to Milvus: {e}")
            return False
    
    async def cache_embeddings_to_redis(self,
                                      document_id: str,
                                      embeddings: np.ndarray,
                                      expire_seconds: int = 3600) -> bool:
        """埋め込みをRedisにキャッシュ"""
        try:
            cache_key = f"embeddings:{document_id}"
            success = await self.redis_client.set_embeddings(
                key=cache_key,
                embeddings=embeddings.tolist(),
                expire_seconds=expire_seconds
            )
            return success
        except Exception as e:
            self.logger.error(f"Error caching embeddings to Redis: {e}")
            return False
    
    async def get_cached_embeddings(self, document_id: str) -> Optional[np.ndarray]:
        """Redisからキャッシュされた埋め込みを取得"""
        try:
            cache_key = f"embeddings:{document_id}"
            cached_data = await self.redis_client.get_embeddings(cache_key)
            if cached_data:
                return np.array(cached_data)
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving cached embeddings: {e}")
            return None


def test_embedding_service():
    """埋め込みサービスのテスト"""
    service = TransformersEmbeddingService()
    
    # テストテキスト
    texts = [
        "これは日本語のテストテキストです。",
        "This is an English test text.",
        "機械学習とAIについて学んでいます。",
        "I am studying machine learning and AI."
    ]
    
    print("=== モデル情報 ===")
    print(service.get_model_info())
    
    print("\n=== 埋め込み生成テスト ===")
    embeddings = service.encode_texts(texts)
    print(f"埋め込み形状: {embeddings.shape}")
    
    print("\n=== 類似度計算テスト ===")
    similarity = service.compute_similarity(texts[0], texts[2])
    print(f"日本語テキスト間類似度: {similarity:.4f}")
    
    print("\n=== 類似検索テスト ===")
    results = service.find_most_similar("機械学習", texts, top_k=3)
    for i, result in enumerate(results):
        print(f"{i+1}. 類似度: {result['similarity']:.4f} - {result['text'][:50]}...")
    
    # クリーンアップ
    service.cleanup()


if __name__ == "__main__":
    test_embedding_service()
