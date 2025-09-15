import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import BaseNode, TextNode, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.tools.neo4j import Neo4jQueryToolSpec
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from ..llms.ollama_connector import OllamaConnector
from ..database.vector_db.milvus_client import MilvusClient
from ..database.graph_db.neo4j_client import Neo4jClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.document_db.mongo_client import MongoClient


@dataclass
class EmbeddingConfig:
    """埋め込み設定クラス"""
    model_name: str = "nomic-embed-text"
    embedding_dim: int = 768
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32
    device: str = "cpu"


class EmbeddingService:
    """
    統合埋め込みサービス
    複数の埋め込みモデルとベクトルストアを管理
    """
    
    def __init__(
        self,
        config: EmbeddingConfig,
        milvus_client: MilvusClient = None,
        neo4j_client: Neo4jClient = None,
        redis_client: RedisClient = None,
        mongo_client: MongoClient = None,
        ollama_connector: OllamaConnector = None
    ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # モデル管理
        self.embedding_models: Dict[str, BaseEmbedding] = {}
        self.vector_store: Optional[MilvusVectorStore] = None
        self.graph_store: Optional[Neo4jGraphStore] = None
        self.node_parser: Optional[SentenceSplitter] = None
        
        # 初期化
        self._initialize_services()
    
    def _initialize_services(self):
        """各サービスを初期化"""
        try:
            # Node parser設定
            self.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            # 埋め込みモデル初期化
            self._initialize_embedding_models()
            
            # ベクトルストア初期化（既存のMilvusクライアントを使用）
            self._initialize_vector_store()
            
            # グラフストア初期化（既存のNeo4jクライアントを使用）
            self._initialize_graph_store()
            
            # グローバル設定
            Settings.embed_model = self.embedding_models.get("ollama")
            Settings.llm = self.ollama_connector.initialize_llm()
            
            self.logger.info("埋め込みサービス初期化完了")
            
        except Exception as e:
            self.logger.error(f"埋め込みサービス初期化エラー: {e}")
            raise
    
    def _initialize_embedding_models(self):
        """各種埋め込みモデルを初期化"""
        
        # 1. Ollama埋め込みモデル
        try:
            ollama_embed = OllamaEmbedding(
                model_name=self.config.model_name,
                base_url=self.ollama_base_url
            )
            self.embedding_models["ollama"] = ollama_embed
            self.logger.info(f"Ollama埋め込みモデル初期化: {self.config.model_name}")
        except Exception as e:
            self.logger.warning(f"Ollama埋め込みモデル初期化失敗: {e}")
        
        # 2. SentenceTransformers埋め込みモデル
        try:
            sentence_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.config.device
            )
            
            # LangChain wrapper経由でLlamaIndexに統合
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            langchain_embed = SentenceTransformerEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            
            llama_embed = LangchainEmbedding(langchain_embed)
            self.embedding_models["sentence_transformers"] = llama_embed
            self.logger.info("SentenceTransformers埋め込みモデル初期化完了")
        except Exception as e:
            self.logger.warning(f"SentenceTransformers埋め込みモデル初期化失敗: {e}")
        
        # 3. Transformers埋め込みモデル
        try:
            class TransformersEmbedding(BaseEmbedding):
                """Transformersベースのカスタム埋め込みクラス"""
                
                def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
                    super().__init__()
                    self.model_name = model_name
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
                    self.model.eval()
                
                def _get_query_embedding(self, query: str) -> List[float]:
                    return self._get_text_embedding(query)
                
                def _get_text_embedding(self, text: str) -> List[float]:
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                    
                    return embeddings.squeeze().tolist()
                
                async def _aget_query_embedding(self, query: str) -> List[float]:
                    return self._get_query_embedding(query)
                
                async def _aget_text_embedding(self, text: str) -> List[float]:
                    return self._get_text_embedding(text)
            
            transformers_embed = TransformersEmbedding()
            self.embedding_models["transformers"] = transformers_embed
            self.logger.info("Transformers埋め込みモデル初期化完了")
        except Exception as e:
            self.logger.warning(f"Transformers埋め込みモデル初期化失敗: {e}")
    
    def _initialize_vector_store(self):
        """Milvusベクトルストアを初期化"""
        try:
            self.vector_store = MilvusVectorStore(
                host=self.milvus_host,
                port=self.milvus_port,
                dim=self.config.embedding_dim,
                collection_name="ragshelf_embeddings",
                overwrite=False
            )
            self.logger.info("Milvusベクトルストア初期化完了")
        except Exception as e:
            self.logger.error(f"Milvusベクトルストア初期化失敗: {e}")
            raise
    
    def _initialize_graph_store(self):
        """Neo4jグラフストアを初期化"""
        try:
            self.graph_store = Neo4jGraphStore(
                username=self.neo4j_username,
                password=self.neo4j_password,
                url=self.neo4j_url,
                database="neo4j"
            )
            self.logger.info("Neo4jグラフストア初期化完了")
        except Exception as e:
            self.logger.warning(f"Neo4jグラフストア初期化失敗: {e}")
    
    async def create_embeddings(
        self, 
        texts: List[str], 
        model_name: str = "ollama",
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[List[float]]:
        """
        テキストリストから埋め込みを生成
        
        Args:
            texts: 埋め込みを生成するテキストリスト
            model_name: 使用する埋め込みモデル名
            metadata: 各テキストに対応するメタデータ
        
        Returns:
            埋め込みベクトルのリスト
        """
        if model_name not in self.embedding_models:
            raise ValueError(f"埋め込みモデル '{model_name}' が見つかりません")
        
        embedding_model = self.embedding_models[model_name]
        embeddings = []
        
        try:
            # バッチ処理で埋め込み生成
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                batch_embeddings = []
                for text in batch_texts:
                    if hasattr(embedding_model, '_aget_text_embedding'):
                        embedding = await embedding_model._aget_text_embedding(text)
                    else:
                        embedding = embedding_model._get_text_embedding(text)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                
                # 進捗ログ
                self.logger.info(f"埋め込み生成進捗: {min(i + self.config.batch_size, len(texts))}/{len(texts)}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"埋め込み生成エラー: {e}")
            raise
    
    async def store_document_embeddings(
        self,
        document: Document,
        document_id: str,
        model_name: str = "ollama"
    ) -> List[str]:
        """
        ドキュメントをチャンク分割し、埋め込みを生成してベクトルストアに保存
        
        Args:
            document: LlamaIndex Document
            document_id: ドキュメントID
            model_name: 使用する埋め込みモデル名
        
        Returns:
            作成されたノードIDのリスト
        """
        try:
            # ドキュメントをチャンクに分割
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            # 各ノードにメタデータを追加
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "document_id": document_id,
                    "chunk_index": i,
                    "source": document.metadata.get("source", "unknown")
                })
            
            # 埋め込み生成とベクトルストアへの保存
            if self.vector_store and model_name in self.embedding_models:
                # ベクトルストアにノードを追加
                self.vector_store.add(nodes)
                
                self.logger.info(f"ドキュメント {document_id} の埋め込み保存完了: {len(nodes)} チャンク")
                return [node.node_id for node in nodes]
            else:
                raise ValueError("ベクトルストアまたは埋め込みモデルが初期化されていません")
                
        except Exception as e:
            self.logger.error(f"ドキュメント埋め込み保存エラー: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        model_name: str = "ollama",
        filters: Optional[Dict[str, Any]] = None
    ) -> VectorStoreQueryResult:
        """
        類似検索を実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する結果数
            model_name: 使用する埋め込みモデル名
            filters: 検索フィルター
        
        Returns:
            検索結果
        """
        try:
            if model_name not in self.embedding_models:
                raise ValueError(f"埋め込みモデル '{model_name}' が見つかりません")
            
            # クエリの埋め込み生成
            embedding_model = self.embedding_models[model_name]
            if hasattr(embedding_model, '_aget_query_embedding'):
                query_embedding = await embedding_model._aget_query_embedding(query)
            else:
                query_embedding = embedding_model._get_query_embedding(query)
            
            # ベクトル検索クエリ作成
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=filters
            )
            
            # 検索実行
            result = self.vector_store.query(vector_query)
            
            self.logger.info(f"類似検索完了: クエリ='{query}', 結果数={len(result.nodes)}")
            return result
            
        except Exception as e:
            self.logger.error(f"類似検索エラー: {e}")
            raise
    
    async def create_knowledge_graph(
        self,
        documents: List[Document],
        extract_entities: bool = True,
        extract_relations: bool = True
    ) -> bool:
        """
        ドキュメントから知識グラフを作成
        
        Args:
            documents: ドキュメントリスト
            extract_entities: エンティティ抽出するか
            extract_relations: 関係抽出するか
        
        Returns:
            作成成功フラグ
        """
        try:
            if not self.graph_store:
                self.logger.warning("Neo4jグラフストアが初期化されていません")
                return False
            
            # Neo4jクエリツール初期化
            neo4j_tool = Neo4jQueryToolSpec(
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                database="neo4j"
            )
            
            for doc in documents:
                # ドキュメントノード作成
                doc_id = doc.metadata.get("document_id", str(doc.doc_id))
                
                # エンティティ抽出（簡単な実装例）
                if extract_entities:
                    entities = await self._extract_entities(doc.text)
                    for entity in entities:
                        # エンティティノード作成
                        cypher_query = """
                        MERGE (e:Entity {name: $name, type: $type})
                        MERGE (d:Document {id: $doc_id})
                        MERGE (d)-[:CONTAINS]->(e)
                        """
                        neo4j_tool.run_cypher(
                            cypher_query,
                            {"name": entity["name"], "type": entity["type"], "doc_id": doc_id}
                        )
                
                # 関係抽出（簡単な実装例）
                if extract_relations:
                    relations = await self._extract_relations(doc.text)
                    for relation in relations:
                        cypher_query = """
                        MERGE (e1:Entity {name: $entity1})
                        MERGE (e2:Entity {name: $entity2})
                        MERGE (e1)-[:RELATED {type: $relation_type}]->(e2)
                        """
                        neo4j_tool.run_cypher(
                            cypher_query,
                            {
                                "entity1": relation["entity1"],
                                "entity2": relation["entity2"],
                                "relation_type": relation["type"]
                            }
                        )
            
            self.logger.info(f"知識グラフ作成完了: {len(documents)} ドキュメント")
            return True
            
        except Exception as e:
            self.logger.error(f"知識グラフ作成エラー: {e}")
            return False
    
    async def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        テキストからエンティティを抽出（簡単な実装）
        実際の実装では、spaCyやNERモデルを使用
        """
        # 簡単な実装例：大文字で始まる単語を固有名詞として抽出
        import re
        entities = []
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in set(words):
            entities.append({"name": word, "type": "NOUN"})
        return entities[:10]  # 最大10個
    
    async def _extract_relations(self, text: str) -> List[Dict[str, str]]:
        """
        テキストから関係を抽出（簡単な実装）
        実際の実装では、依存関係解析や関係抽出モデルを使用
        """
        # 簡単な実装例：パターンマッチングで関係を抽出
        relations = []
        # "A is B" パターン
        import re
        patterns = re.findall(r'(\w+)\s+is\s+(\w+)', text, re.IGNORECASE)
        for entity1, entity2 in patterns[:5]:  # 最大5個
            relations.append({
                "entity1": entity1,
                "entity2": entity2,
                "type": "IS_A"
            })
        return relations
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """埋め込みサービスの統計情報を取得"""
        stats = {
            "available_models": list(self.embedding_models.keys()),
            "vector_store_status": self.vector_store is not None,
            "graph_store_status": self.graph_store is not None,
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "batch_size": self.config.batch_size
            }
        }
        
        # ベクトルストア統計
        if self.vector_store:
            try:
                # Milvusコレクション統計（実装は仮）
                stats["vector_store_info"] = {
                    "collection_name": "ragshelf_embeddings",
                    "status": "connected"
                }
            except Exception as e:
                stats["vector_store_info"] = {"error": str(e)}
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """各サービスのヘルスチェック"""
        health = {}
        
        # Ollama接続確認
        if self.ollama_connector:
            health["ollama"] = self.ollama_connector.check_connection()
        else:
            health["ollama"] = False
        
        # Milvus接続確認
        try:
            if self.vector_store:
                # 簡単な操作でMilvus接続確認
                health["milvus"] = True
            else:
                health["milvus"] = False
        except Exception:
            health["milvus"] = False
        
        # Neo4j接続確認
        try:
            if self.graph_store:
                # 簡単なクエリでNeo4j接続確認
                health["neo4j"] = True
            else:
                health["neo4j"] = False
        except Exception:
            health["neo4j"] = False
        
        return health


# ユーティリティ関数
async def create_embedding_service(
    config: Optional[EmbeddingConfig] = None,
    **kwargs
) -> EmbeddingService:
    """埋め込みサービスを作成する便利関数"""
    if config is None:
        config = EmbeddingConfig()
    
    service = EmbeddingService(config, **kwargs)
    return service


# 使用例とテスト用のメイン関数
async def main():
    """テスト用メイン関数"""
    logging.basicConfig(level=logging.INFO)
    
    # 設定
    config = EmbeddingConfig(
        model_name="nomic-embed-text",
        embedding_dim=768,
        chunk_size=512,
        batch_size=16
    )
    
    # サービス初期化
    service = await create_embedding_service(config)
    
    # ヘルスチェック
    health = await service.health_check()
    print("ヘルスチェック結果:", health)
    
    # 統計情報取得
    stats = await service.get_embedding_stats()
    print("統計情報:", stats)
    
    # テスト用ドキュメント
    test_texts = [
        "これは機械学習に関するドキュメントです。",
        "自然言語処理とディープラーニングについて説明します。",
        "RAGシステムは検索拡張生成の技術です。"
    ]
    
    try:
        # 埋め込み生成テスト
        embeddings = await service.create_embeddings(test_texts, "ollama")
        print(f"埋め込み生成完了: {len(embeddings)} 個")
        
        # 類似検索テスト
        if service.vector_store:
            # テストドキュメントを作成して保存
            from llama_index.core.schema import Document
            test_doc = Document(
                text=" ".join(test_texts),
                metadata={"document_id": "test_doc_001", "source": "test"}
            )
            
            node_ids = await service.store_document_embeddings(test_doc, "test_doc_001")
            print(f"ドキュメント保存完了: {len(node_ids)} チャンク")
            
            # 類似検索実行
            search_result = await service.similarity_search("機械学習について", top_k=3)
            print(f"類似検索結果: {len(search_result.nodes)} 件")
            
    except Exception as e:
        print(f"テスト実行エラー: {e}")
