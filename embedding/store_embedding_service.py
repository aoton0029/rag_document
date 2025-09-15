import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import json
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter
from ..database.document_db.mongo_client import MongoClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.vector_db.milvus_client import MilvusClient
from ..database.graph_db.neo4j_client import Neo4jClient
from ..llms.ollama_connector import OllamaConnector


@dataclass
class EmbeddingStoreConfig:
    """埋め込みストア設定"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_dim: int = 768
    batch_size: int = 32
    enable_cache: bool = True
    cache_expire_seconds: int = 3600
    enable_graph_extraction: bool = True
    similarity_threshold: float = 0.7


@dataclass
class StoredDocument:
    """保存されたドキュメント情報"""
    document_id: str
    mongo_id: str
    chunk_count: int
    embedding_count: int
    graph_nodes: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class StoreEmbeddingService:
    """
    統合埋め込みストレージサービス
    複数のデータベースクライアントを使用して包括的な埋め込み管理を提供
    """
    
    def __init__(
        self,
        config: EmbeddingStoreConfig = None,
        mongo_client: MongoClient = None,
        redis_client: RedisClient = None,
        milvus_client: MilvusClient = None,
        neo4j_client: Neo4jClient = None,
        ollama_connector: OllamaConnector = None
    ):
        self.config = config or EmbeddingStoreConfig()
        self.logger = logging.getLogger(__name__)
        
        # データベースクライアント初期化
        self.mongo_client = mongo_client or MongoClient()
        self.redis_client = redis_client or RedisClient()
        self.milvus_client = milvus_client or MilvusClient()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.ollama_connector = ollama_connector or OllamaConnector()
        
        # Milvusコレクション初期化
        if not self.milvus_client.collection:
            self.milvus_client.create_collection(self.config.embedding_dim)
        
        # ノードパーサー初期化
        self.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # 統計情報
        self.stats = {
            "documents_stored": 0,
            "embeddings_generated": 0,
            "cache_hits": 0,
            "graph_nodes_created": 0,
            "operations_count": 0,
            "last_operation": None
        }
        
        self.logger.info("StoreEmbeddingService初期化完了")
    
    async def store_document_with_embeddings(
        self,
        document: Document,
        document_id: Optional[str] = None,
        generate_embeddings: bool = True,
        extract_graph: bool = True
    ) -> StoredDocument:
        """
        ドキュメントを埋め込みと共に全データベースに保存
        
        Args:
            document: LlamaIndex Document
            document_id: ドキュメントID（None時は自動生成）
            generate_embeddings: 埋め込み生成フラグ
            extract_graph: グラフ抽出フラグ
        
        Returns:
            保存されたドキュメント情報
        """
        document_id = document_id or str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        self.logger.info(f"ドキュメント保存開始: {document_id}")
        
        try:
            # 1. ドキュメントチャンク分割
            chunks = await self._chunk_document(document, document_id)
            
            # 2. MongoDBにドキュメントとチャンク保存
            mongo_id = await self._store_to_mongodb(document, chunks, document_id)
            
            # 3. 埋め込み生成とMilvus保存
            embedding_count = 0
            if generate_embeddings:
                embedding_count = await self._generate_and_store_embeddings(
                    chunks, document_id
                )
            
            # 4. グラフ抽出とNeo4j保存
            graph_nodes = 0
            if extract_graph and self.config.enable_graph_extraction:
                graph_nodes = await self._extract_and_store_graph(
                    document, chunks, document_id
                )
            
            # 5. Redisキャッシュ更新
            if self.config.enable_cache:
                await self._update_document_cache(document_id, chunks)
            
            # 6. 結果作成
            stored_doc = StoredDocument(
                document_id=document_id,
                mongo_id=mongo_id,
                chunk_count=len(chunks),
                embedding_count=embedding_count,
                graph_nodes=graph_nodes,
                metadata=document.metadata or {},
                created_at=start_time,
                updated_at=datetime.utcnow()
            )
            
            # 統計更新
            self._update_stats("store_document", stored_doc)
            
            self.logger.info(f"ドキュメント保存完了: {document_id}, チャンク: {len(chunks)}, 埋め込み: {embedding_count}, グラフノード: {graph_nodes}")
            return stored_doc
            
        except Exception as e:
            self.logger.error(f"ドキュメント保存エラー: {document_id}, {e}")
            raise
    
    async def _chunk_document(self, document: Document, document_id: str) -> List[TextNode]:
        """ドキュメントをチャンクに分割"""
        nodes = self.node_parser.get_nodes_from_documents([document])
        
        # 各ノードにメタデータ追加
        for i, node in enumerate(nodes):
            if node.metadata is None:
                node.metadata = {}
            
            node.metadata.update({
                "document_id": document_id,
                "chunk_index": i,
                "chunk_id": f"{document_id}_chunk_{i}",
                "source_document": document_id,
                "word_count": len(node.text.split()),
                "char_count": len(node.text),
                "created_at": datetime.utcnow().isoformat()
            })
            
            # ノードIDを設定
            node.node_id = node.metadata["chunk_id"]
        
        return nodes
    
    async def _store_to_mongodb(
        self,
        document: Document,
        chunks: List[TextNode],
        document_id: str
    ) -> str:
        """MongoDBにドキュメントとチャンクを保存"""
        # メインドキュメント保存
        document_data = {
            "document_id": document_id,
            "content": document.text,
            "metadata": document.metadata or {},
            "chunk_count": len(chunks),
            "total_chars": len(document.text),
            "total_words": len(document.text.split()),
            "processing_info": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "processed_at": datetime.utcnow()
            }
        }
        
        mongo_id = self.mongo_client.save_document(
            document_id, document.text, document_data
        )
        
        # チャンク詳細をメタデータコレクションに保存
        chunk_metadata = {
            "document_id": document_id,
            "type": "chunks",
            "chunks": []
        }
        
        for chunk in chunks:
            chunk_info = {
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_index": chunk.metadata["chunk_index"],
                "content": chunk.text,
                "metadata": chunk.metadata,
                "mongo_id": None  # 個別チャンクは別途保存可能
            }
            chunk_metadata["chunks"].append(chunk_info)
        
        # チャンクメタデータ保存
        self.mongo_client.metadata.insert_one(chunk_metadata)
        
        return mongo_id
    
    async def _generate_and_store_embeddings(
        self,
        chunks: List[TextNode],
        document_id: str
    ) -> int:
        """埋め込み生成とMilvus保存"""
        if not chunks:
            return 0
        
        # キャッシュチェック
        if self.config.enable_cache:
            cached_embeddings = self.redis_client.get_document_embeddings(document_id)
            if cached_embeddings and len(cached_embeddings) == len(chunks):
                self.stats["cache_hits"] += 1
                self.logger.info(f"埋め込みキャッシュヒット: {document_id}")
                
                # キャッシュからMilvusに保存
                await self._store_embeddings_to_milvus(
                    chunks, cached_embeddings, document_id
                )
                return len(cached_embeddings)
        
        # 埋め込み生成
        texts = [chunk.text for chunk in chunks]
        embeddings = await self._generate_embeddings_batch(texts)
        
        if not embeddings:
            self.logger.warning(f"埋め込み生成失敗: {document_id}")
            return 0
        
        # Milvus保存
        await self._store_embeddings_to_milvus(chunks, embeddings, document_id)
        
        # Redisキャッシュ
        if self.config.enable_cache:
            self.redis_client.set_document_embeddings(
                document_id, embeddings, self.config.cache_expire_seconds
            )
        
        # MongoDBの埋め込み情報更新
        await self._update_mongodb_embedding_info(chunks, embeddings)
        
        return len(embeddings)
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """バッチで埋め込み生成"""
        try:
            embedding_model = self.ollama_connector.initialize_embedding()
            all_embeddings = []
            
            # バッチ処理
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                batch_embeddings = []
                for text in batch_texts:
                    embedding = await embedding_model._aget_text_embedding(text)
                    batch_embeddings.append(embedding)
                
                all_embeddings.extend(batch_embeddings)
                
                # 進捗ログ
                processed = min(i + self.config.batch_size, len(texts))
                self.logger.info(f"埋め込み生成進捗: {processed}/{len(texts)}")
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"埋め込み生成エラー: {e}")
            return []
    
    async def _store_embeddings_to_milvus(
        self,
        chunks: List[TextNode],
        embeddings: List[List[float]],
        document_id: str
    ):
        """Milvusに埋め込み保存"""
        chunk_texts = [chunk.text for chunk in chunks]
        
        self.milvus_client.insert_vectors(
            document_id, chunk_texts, embeddings
        )
        
        self.logger.info(f"Milvus埋め込み保存完了: {document_id}, {len(embeddings)} ベクトル")
    
    async def _update_mongodb_embedding_info(
        self,
        chunks: List[TextNode],
        embeddings: List[List[float]]
    ):
        """MongoDBのチャンクに埋め込み情報を追加"""
        for i, chunk in enumerate(chunks):
            embedding_info = {
                "has_embedding": True,
                "embedding_dim": len(embeddings[i]) if i < len(embeddings) else 0,
                "embedding_model": "ollama",
                "generated_at": datetime.utcnow()
            }
            
            # MongoDBでchunk情報を更新（実装に応じて調整）
            self.mongo_client.metadata.update_one(
                {
                    "document_id": chunk.metadata["document_id"],
                    "type": "chunks",
                    "chunks.chunk_id": chunk.metadata["chunk_id"]
                },
                {
                    "$set": {
                        "chunks.$.embedding_info": embedding_info
                    }
                }
            )
    
    async def _extract_and_store_graph(
        self,
        document: Document,
        chunks: List[TextNode],
        document_id: str
    ) -> int:
        """グラフ抽出とNeo4j保存"""
        graph_nodes_created = 0
        
        try:
            # ドキュメントノード作成
            title = document.metadata.get("title", "Unknown Document")
            if self.neo4j_client.create_document_node(document_id, title, document.metadata or {}):
                graph_nodes_created += 1
            
            # エンティティ抽出と保存
            entities = await self._extract_entities_from_chunks(chunks)
            
            for entity in entities:
                entity_id = f"{document_id}_{entity['name']}"
                if self.neo4j_client.create_entity_node(
                    entity_id, entity["type"], {"name": entity["name"], "confidence": entity.get("confidence", 1.0)}
                ):
                    graph_nodes_created += 1
                    
                    # ドキュメントとエンティティの関係作成
                    self.neo4j_client.create_relationship(
                        document_id, entity_id, "CONTAINS", {"extracted_from": "document"}
                    )
            
            # エンティティ間の関係抽出
            relations = await self._extract_relations_from_chunks(chunks)
            
            for relation in relations:
                entity1_id = f"{document_id}_{relation['entity1']}"
                entity2_id = f"{document_id}_{relation['entity2']}"
                
                self.neo4j_client.create_relationship(
                    entity1_id, entity2_id, relation["type"], 
                    {"confidence": relation.get("confidence", 1.0)}
                )
            
            self.logger.info(f"グラフ抽出完了: {document_id}, ノード: {graph_nodes_created}, 関係: {len(relations)}")
            
        except Exception as e:
            self.logger.error(f"グラフ抽出エラー: {document_id}, {e}")
        
        return graph_nodes_created
    
    async def _extract_entities_from_chunks(self, chunks: List[TextNode]) -> List[Dict[str, Any]]:
        """チャンクからエンティティを抽出"""
        entities = []
        
        for chunk in chunks:
            # 簡易エンティティ抽出（実際の実装ではNERモデルを使用）
            import re
            
            # 大文字で始まる単語を固有名詞として抽出
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', chunk.text)
            
            for noun in set(proper_nouns):
                if len(noun) > 2:  # 2文字以上
                    entities.append({
                        "name": noun,
                        "type": "ENTITY",
                        "confidence": 0.8,
                        "chunk_id": chunk.metadata["chunk_id"]
                    })
        
        # 重複除去
        unique_entities = {}
        for entity in entities:
            if entity["name"] not in unique_entities:
                unique_entities[entity["name"]] = entity
        
        return list(unique_entities.values())[:50]  # 最大50エンティティ
    
    async def _extract_relations_from_chunks(self, chunks: List[TextNode]) -> List[Dict[str, Any]]:
        """チャンクから関係を抽出"""
        relations = []
        
        for chunk in chunks:
            # 簡易関係抽出（実際の実装では関係抽出モデルを使用）
            import re
            
            # "A is B" パターン
            is_patterns = re.findall(r'([A-Z][a-z]+)\s+(?:is|are|was|were)\s+([A-Z][a-z]+)', chunk.text)
            for entity1, entity2 in is_patterns:
                relations.append({
                    "entity1": entity1,
                    "entity2": entity2,
                    "type": "IS_A",
                    "confidence": 0.7,
                    "chunk_id": chunk.metadata["chunk_id"]
                })
            
            # "A has B" パターン
            has_patterns = re.findall(r'([A-Z][a-z]+)\s+(?:has|have|had)\s+([A-Z][a-z]+)', chunk.text)
            for entity1, entity2 in has_patterns:
                relations.append({
                    "entity1": entity1,
                    "entity2": entity2,
                    "type": "HAS",
                    "confidence": 0.6,
                    "chunk_id": chunk.metadata["chunk_id"]
                })
        
        return relations[:20]  # 最大20関係
    
    async def _update_document_cache(self, document_id: str, chunks: List[TextNode]):
        """ドキュメントキャッシュ更新"""
        cache_data = {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.metadata["chunk_id"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "content_preview": chunk.text[:200],
                    "word_count": chunk.metadata.get("word_count", 0)
                }
                for chunk in chunks
            ],
            "cached_at": datetime.utcnow().isoformat()
        }
        
        self.redis_client.set_cache(
            f"document:{document_id}",
            cache_data,
            self.config.cache_expire_seconds
        )
    
    async def search_similar_documents(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = None,
        document_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """類似ドキュメント検索"""
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        try:
            # クエリ埋め込み生成
            embedding_model = self.ollama_connector.initialize_embedding()
            query_embedding = await embedding_model._aget_query_embedding(query)
            
            # Milvus検索
            milvus_results = self.milvus_client.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                document_filter=document_filter
            )
            
            # 結果フィルタリングと詳細情報追加
            filtered_results = []
            for result in milvus_results:
                if result["score"] >= similarity_threshold:
                    # MongoDBから詳細情報取得
                    doc_info = self.mongo_client.get_document(result["document_id"])
                    
                    enhanced_result = {
                        **result,
                        "document_metadata": doc_info.get("metadata", {}) if doc_info else {},
                        "document_title": doc_info.get("metadata", {}).get("title", "Unknown") if doc_info else "Unknown"
                    }
                    filtered_results.append(enhanced_result)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"類似検索エラー: {e}")
            return []
    
    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """ドキュメント詳細情報取得"""
        # キャッシュチェック
        if self.config.enable_cache:
            cached_info = self.redis_client.get_cache(f"document:{document_id}")
            if cached_info:
                self.stats["cache_hits"] += 1
                return cached_info
        
        # データベースから情報収集
        document_info = {}
        
        # MongoDB情報
        mongo_doc = self.mongo_client.get_document(document_id)
        if mongo_doc:
            document_info["mongodb"] = {
                "id": str(mongo_doc["_id"]),
                "content_length": len(mongo_doc.get("content", "")),
                "metadata": mongo_doc.get("metadata", {}),
                "created_at": mongo_doc.get("created_at"),
                "updated_at": mongo_doc.get("updated_at")
            }
        
        # Milvus統計
        milvus_stats = self.milvus_client.get_collection_stats()
        document_info["milvus"] = milvus_stats
        
        # Neo4j情報
        entities = self.neo4j_client.get_document_entities(document_id)
        document_info["neo4j"] = {
            "entity_count": len(entities),
            "entities": entities[:10]  # 最初の10エンティティ
        }
        
        # Redis情報
        embeddings = self.redis_client.get_document_embeddings(document_id)
        document_info["redis"] = {
            "has_cached_embeddings": embeddings is not None,
            "embedding_count": len(embeddings) if embeddings else 0
        }
        
        return document_info
    
    async def delete_document(self, document_id: str) -> bool:
        """ドキュメントを全データベースから削除"""
        success_count = 0
        total_operations = 4
        
        try:
            # MongoDB削除
            if self.mongo_client.delete_document(document_id):
                success_count += 1
                self.logger.info(f"MongoDB削除成功: {document_id}")
            
            # Milvus削除
            try:
                self.milvus_client.delete_document_vectors(document_id)
                success_count += 1
                self.logger.info(f"Milvus削除成功: {document_id}")
            except Exception as e:
                self.logger.warning(f"Milvus削除警告: {document_id}, {e}")
            
            # Neo4j削除
            if self.neo4j_client.delete_document_graph(document_id):
                success_count += 1
                self.logger.info(f"Neo4j削除成功: {document_id}")
            
            # Redis削除
            cache_keys = [
                f"document:{document_id}",
                f"embeddings:{document_id}"
            ]
            for key in cache_keys:
                self.redis_client.delete_cache(key)
            success_count += 1
            self.logger.info(f"Redis削除成功: {document_id}")
            
            deletion_success = success_count >= (total_operations - 1)  # 3/4以上成功で成功とみなす
            
            if deletion_success:
                self.logger.info(f"ドキュメント削除完了: {document_id}")
            else:
                self.logger.warning(f"ドキュメント削除部分失敗: {document_id}, 成功: {success_count}/{total_operations}")
            
            return deletion_success
            
        except Exception as e:
            self.logger.error(f"ドキュメント削除エラー: {document_id}, {e}")
            return False
    
    def _update_stats(self, operation: str, data: Any = None):
        """統計情報更新"""
        self.stats["operations_count"] += 1
        self.stats["last_operation"] = {
            "type": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "data": str(data)[:100] if data else None
        }
        
        if operation == "store_document" and isinstance(data, StoredDocument):
            self.stats["documents_stored"] += 1
            self.stats["embeddings_generated"] += data.embedding_count
            self.stats["graph_nodes_created"] += data.graph_nodes
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """サービス統計取得"""
        return {
            **self.stats,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "embedding_dim": self.config.embedding_dim,
                "enable_cache": self.config.enable_cache,
                "enable_graph_extraction": self.config.enable_graph_extraction
            },
            "database_status": await self._check_database_health()
        }
    
    async def _check_database_health(self) -> Dict[str, bool]:
        """データベース死活監視"""
        health = {}
        
        # MongoDB
        try:
            self.mongo_client.db.command("ping")
            health["mongodb"] = True
        except:
            health["mongodb"] = False
        
        # Redis
        try:
            self.redis_client.client.ping()
            health["redis"] = True
        except:
            health["redis"] = False
        
        # Milvus
        try:
            stats = self.milvus_client.get_collection_stats()
            health["milvus"] = bool(stats)
        except:
            health["milvus"] = False
        
        # Neo4j
        try:
            with self.neo4j_client.driver.session() as session:
                session.run("RETURN 1")
            health["neo4j"] = True
        except:
            health["neo4j"] = False
        
        # Ollama
        health["ollama"] = self.ollama_connector.check_connection()
        
        return health
    
    async def close_connections(self):
        """全データベース接続を閉じる"""
        try:
            self.mongo_client.close()
            self.redis_client.close()
            self.neo4j_client.close()
            self.logger.info("全データベース接続を閉じました")
        except Exception as e:
            self.logger.error(f"接続クローズエラー: {e}")


# 使用例とテスト用のメイン関数
async def main():
    """テスト用メイン関数"""
    logging.basicConfig(level=logging.INFO)
    
    # 設定
    config = EmbeddingStoreConfig(
        chunk_size=512,
        chunk_overlap=50,
        embedding_dim=768,
        enable_cache=True,
        enable_graph_extraction=True
    )
    
    # サービス初期化
    service = StoreEmbeddingService(config)
    
    # ヘルスチェック
    stats = await service.get_service_stats()
    print("サービス統計:", json.dumps(stats, indent=2, ensure_ascii=False))
    
    # テストドキュメント
    from llama_index.core.schema import Document
    
    test_document = Document(
        text="これは機械学習に関するテストドキュメントです。深層学習やニューラルネットワークについて説明しています。AIと自然言語処理の技術について詳しく解説します。",
        metadata={
            "title": "機械学習入門",
            "author": "テスト著者",
            "category": "技術文書",
            "tags": ["AI", "機械学習", "深層学習"]
        }
    )
    
    try:
        # ドキュメント保存
        print("\n=== ドキュメント保存テスト ===")
        stored_doc = await service.store_document_with_embeddings(
            test_document,
            generate_embeddings=True,
            extract_graph=True
        )
        print(f"保存完了: {stored_doc.document_id}")
        print(f"チャンク数: {stored_doc.chunk_count}")
        print(f"埋め込み数: {stored_doc.embedding_count}")
        print(f"グラフノード数: {stored_doc.graph_nodes}")
        
        # 類似検索テスト
        print("\n=== 類似検索テスト ===")
        search_results = await service.search_similar_documents(
            "機械学習について教えて",
            top_k=3
        )
        print(f"検索結果数: {len(search_results)}")
        for result in search_results:
            print(f"- スコア: {result['score']:.3f}, テキスト: {result['text'][:100]}...")
        
        # ドキュメント情報取得
        print("\n=== ドキュメント情報取得テスト ===")
        doc_info = await service.get_document_info(stored_doc.document_id)
        if doc_info:
            print("ドキュメント情報:")
            for db_name, info in doc_info.items():
                print(f"  {db_name}: {info}")
        
        # 統計情報
        final_stats = await service.get_service_stats()
        print(f"\n=== 最終統計 ===")
        print(json.dumps(final_stats, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"テスト実行エラー: {e}")
    
    finally:
        await service.close_connections()


if __name__ == "__main__":
    asyncio.run(main())
