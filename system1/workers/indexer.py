import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from ..database.document_db.mongo_client import MongoClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.vector_db.milvus_client import MilvusClient
from ..database.graph_db.neo4j_client import Neo4jClient
from ..database.relational_db.database import get_db
from ..database.relational_db.models import DocumentMetadata


@dataclass
class IndexConfig:
    """インデックス設定"""
    batch_size: int = 100
    sync_interval: int = 300  # 5分
    full_reindex_interval: int = 86400  # 24時間
    enable_auto_sync: bool = True
    enable_consistency_check: bool = True


class IndexerWorker:
    """
    インデックスワーカー
    複数のデータベース間でのインデックス同期と整合性管理
    """
    
    def __init__(
        self,
        config: IndexConfig = None,
        mongo_client: MongoClient = None,
        redis_client: RedisClient = None,
        milvus_client: MilvusClient = None,
        neo4j_client: Neo4jClient = None,
        worker_id: str = "indexer_001"
    ):
        self.config = config or IndexConfig()
        self.worker_id = worker_id
        self.logger = logging.getLogger(__name__)
        
        # データベースクライアント
        self.mongo_client = mongo_client or MongoClient()
        self.redis_client = redis_client or RedisClient()
        self.milvus_client = milvus_client or MilvusClient()
        self.neo4j_client = neo4j_client or Neo4jClient()
        
        # 状態管理
        self.last_sync_time = datetime.utcnow()
        self.last_full_reindex = datetime.utcnow()
        self.is_running = False
        
        # 統計情報
        self.stats = {
            "sync_operations": 0,
            "consistency_checks": 0,
            "inconsistencies_found": 0,
            "documents_indexed": 0,
            "start_time": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        self.logger.info(f"インデックスワーカー初期化完了: {self.worker_id}")
    
    async def start_indexer(self):
        """インデックスワーカー開始"""
        self.is_running = True
        self.logger.info(f"インデックスワーカー開始: {self.worker_id}")
        
        # 並行実行タスク
        tasks = []
        
        if self.config.enable_auto_sync:
            tasks.append(self._sync_loop())
        
        if self.config.enable_consistency_check:
            tasks.append(self._consistency_check_loop())
        
        tasks.append(self._stats_update_loop())
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("インデックスワーカー停止要求を受信")
        finally:
            self.is_running = False
            await self._cleanup()
        
        self.logger.info(f"インデックスワーカー終了: {self.worker_id}")
    
    async def _sync_loop(self):
        """同期ループ"""
        while self.is_running:
            try:
                await self._perform_sync()
                
                # フル再インデックスが必要か確認
                if self._need_full_reindex():
                    await self._perform_full_reindex()
                
                await asyncio.sleep(self.config.sync_interval)
                
            except Exception as e:
                self.logger.error(f"同期ループエラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機
    
    async def _consistency_check_loop(self):
        """整合性チェックループ"""
        while self.is_running:
            try:
                await self._check_database_consistency()
                await asyncio.sleep(self.config.sync_interval * 2)  # 同期の2倍の間隔
                
            except Exception as e:
                self.logger.error(f"整合性チェックエラー: {e}")
                await asyncio.sleep(120)  # エラー時は2分待機
    
    async def _stats_update_loop(self):
        """統計更新ループ"""
        while self.is_running:
            try:
                await self._update_stats()
                await asyncio.sleep(60)  # 1分間隔
                
            except Exception as e:
                self.logger.error(f"統計更新エラー: {e}")
                await asyncio.sleep(60)
    
    async def _perform_sync(self):
        """同期実行"""
        self.logger.info("データベース同期開始")
        
        # MongoDB の変更を検出
        recent_changes = await self._get_recent_changes()
        
        if not recent_changes:
            self.logger.info("同期対象の変更なし")
            return
        
        # 各データベースに同期
        for change in recent_changes:
            await self._sync_document_change(change)
        
        self.last_sync_time = datetime.utcnow()
        self.stats["sync_operations"] += 1
        self.stats["last_activity"] = datetime.utcnow()
        
        self.logger.info(f"データベース同期完了: {len(recent_changes)} 件")
    
    async def _get_recent_changes(self) -> List[Dict[str, Any]]:
        """最近の変更を取得"""
        cutoff_time = self.last_sync_time - timedelta(minutes=1)  # 1分のマージン
        
        # MongoDB から最近更新されたドキュメントを取得
        changes = list(self.mongo_client.documents.find({
            "updated_at": {"$gte": cutoff_time}
        }).sort("updated_at", 1))
        
        # メタデータコレクションからも取得
        metadata_changes = list(self.mongo_client.metadata.find({
            "processing_info.processed_at": {"$gte": cutoff_time}
        }))
        
        return changes + metadata_changes
    
    async def _sync_document_change(self, change: Dict[str, Any]):
        """ドキュメント変更の同期"""
        document_id = change.get("document_id")
        change_type = self._determine_change_type(change)
        
        self.logger.info(f"ドキュメント同期: {document_id}, タイプ: {change_type}")
        
        try:
            if change_type == "new_document":
                await self._index_new_document(change)
            elif change_type == "updated_document":
                await self._update_document_index(change)
            elif change_type == "deleted_document":
                await self._remove_document_index(change)
            
        except Exception as e:
            self.logger.error(f"ドキュメント同期エラー: {document_id}, {e}")
    
    def _determine_change_type(self, change: Dict[str, Any]) -> str:
        """変更タイプを判定"""
        if "chunk_index" in change:
            # チャンクの場合
            if change.get("deleted", False):
                return "deleted_document"
            elif change.get("created_at") == change.get("updated_at"):
                return "new_document"
            else:
                return "updated_document"
        else:
            # メタデータの場合
            return "new_document"
    
    async def _index_new_document(self, document: Dict[str, Any]):
        """新しいドキュメントのインデックス作成"""
        document_id = document["document_id"]
        
        # Neo4j にドキュメントノードを作成
        metadata = document.get("metadata", {})
        title = metadata.get("title", metadata.get("filename", "Unknown"))
        
        success = self.neo4j_client.create_document_node(
            document_id, title, metadata
        )
        
        if success:
            self.logger.info(f"Neo4j ドキュメントノード作成: {document_id}")
        
        # Redis にインデックス情報をキャッシュ
        index_info = {
            "document_id": document_id,
            "indexed_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        self.redis_client.set_cache(
            f"doc_index:{document_id}", index_info, 3600
        )
        
        self.stats["documents_indexed"] += 1
    
    async def _update_document_index(self, document: Dict[str, Any]):
        """ドキュメントインデックス更新"""
        document_id = document["document_id"]
        
        # メタデータ更新
        metadata = document.get("metadata", {})
        
        # Redis キャッシュ更新
        index_info = {
            "document_id": document_id,
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        self.redis_client.set_cache(
            f"doc_index:{document_id}", index_info, 3600
        )
        
        self.logger.info(f"ドキュメントインデックス更新: {document_id}")
    
    async def _remove_document_index(self, document: Dict[str, Any]):
        """ドキュメントインデックス削除"""
        document_id = document["document_id"]
        
        # Milvus からベクトル削除
        try:
            self.milvus_client.delete_document_vectors(document_id)
            self.logger.info(f"Milvus ベクトル削除: {document_id}")
        except Exception as e:
            self.logger.error(f"Milvus ベクトル削除エラー: {document_id}, {e}")
        
        # Neo4j からグラフ削除
        try:
            success = self.neo4j_client.delete_document_graph(document_id)
            if success:
                self.logger.info(f"Neo4j グラフ削除: {document_id}")
        except Exception as e:
            self.logger.error(f"Neo4j グラフ削除エラー: {document_id}, {e}")
        
        # Redis キャッシュ削除
        self.redis_client.delete_cache(f"doc_index:{document_id}")
        
        self.logger.info(f"ドキュメントインデックス削除完了: {document_id}")
    
    async def _check_database_consistency(self):
        """データベース間の整合性チェック"""
        self.logger.info("データベース整合性チェック開始")
        
        inconsistencies = []
        
        # MongoDB のドキュメント一覧取得
        mongo_docs = set()
        for doc in self.mongo_client.metadata.find({"type": "summary"}):
            mongo_docs.add(doc["document_id"])
        
        # Milvus のドキュメント一覧取得（簡易版）
        milvus_docs = set()
        try:
            # 実際の実装では、Milvusから全ドキュメントIDを取得
            # ここでは簡易的な実装
            milvus_stats = self.milvus_client.get_collection_stats()
            # Milvusからドキュメント一覧を取得する実装が必要
        except Exception as e:
            self.logger.warning(f"Milvus 整合性チェック失敗: {e}")
        
        # Neo4j のドキュメント一覧取得
        neo4j_docs = set()
        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN d.document_id as document_id")
                for record in result:
                    neo4j_docs.add(record["document_id"])
        except Exception as e:
            self.logger.warning(f"Neo4j 整合性チェック失敗: {e}")
        
        # 不整合検出
        mongo_only = mongo_docs - milvus_docs - neo4j_docs
        if mongo_only:
            inconsistencies.append({
                "type": "missing_in_vector_graph",
                "documents": list(mongo_only)
            })
        
        # 結果記録
        if inconsistencies:
            self.stats["inconsistencies_found"] += len(inconsistencies)
            await self._handle_inconsistencies(inconsistencies)
        
        self.stats["consistency_checks"] += 1
        self.logger.info(f"整合性チェック完了: {len(inconsistencies)} 件の不整合")
    
    async def _handle_inconsistencies(self, inconsistencies: List[Dict[str, Any]]):
        """不整合の処理"""
        for inconsistency in inconsistencies:
            inc_type = inconsistency["type"]
            documents = inconsistency["documents"]
            
            self.logger.warning(f"不整合検出: {inc_type}, {len(documents)} ドキュメント")
            
            # 自動修復を試行
            if inc_type == "missing_in_vector_graph":
                for doc_id in documents[:10]:  # 最大10件まで自動修復
                    await self._repair_missing_indexes(doc_id)
    
    async def _repair_missing_indexes(self, document_id: str):
        """不足しているインデックスの修復"""
        try:
            # MongoDB からドキュメント情報取得
            doc_summary = self.mongo_client.metadata.find_one({
                "document_id": document_id,
                "type": "summary"
            })
            
            if doc_summary:
                # インデックス再作成タスクをキューに追加
                repair_task = {
                    "task_type": "repair_index",
                    "document_id": document_id,
                    "repair_types": ["milvus", "neo4j"]
                }
                
                task_id = f"repair_{document_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                success = self.redis_client.add_to_processing_queue(task_id, repair_task)
                
                if success:
                    self.logger.info(f"インデックス修復タスクをキューに追加: {document_id}")
                
        except Exception as e:
            self.logger.error(f"インデックス修復エラー: {document_id}, {e}")
    
    def _need_full_reindex(self) -> bool:
        """フル再インデックスが必要かチェック"""
        elapsed = datetime.utcnow() - self.last_full_reindex
        return elapsed.total_seconds() > self.config.full_reindex_interval
    
    async def _perform_full_reindex(self):
        """フル再インデックス実行"""
        self.logger.info("フル再インデックス開始")
        
        try:
            # 全ドキュメントの再インデックス
            all_docs = list(self.mongo_client.metadata.find({"type": "summary"}))
            
            for doc in all_docs:
                document_id = doc["document_id"]
                
                # インデックス再構築タスクを作成
                reindex_task = {
                    "task_type": "full_reindex",
                    "document_id": document_id
                }
                
                task_id = f"fullreindex_{document_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                self.redis_client.add_to_processing_queue(task_id, reindex_task)
            
            self.last_full_reindex = datetime.utcnow()
            self.logger.info(f"フル再インデックス完了: {len(all_docs)} ドキュメント")
            
        except Exception as e:
            self.logger.error(f"フル再インデックスエラー: {e}")
    
    async def _update_stats(self):
        """統計情報更新"""
        current_stats = {
            **self.stats,
            "start_time": self.stats["start_time"].isoformat(),
            "last_activity": self.stats["last_activity"].isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.stats["start_time"]).total_seconds(),
            "last_sync": self.last_sync_time.isoformat(),
            "last_full_reindex": self.last_full_reindex.isoformat()
        }
        
        stats_key = f"indexer_stats:{self.worker_id}"
        self.redis_client.set_cache(stats_key, current_stats, 300)
    
    async def _cleanup(self):
        """クリーンアップ処理"""
        self.logger.info("インデックスワーカークリーンアップ開始")
        
        # 最終統計更新
        await self._update_stats()
        
        # データベース接続クローズ
        try:
            self.mongo_client.close()
            self.neo4j_client.close()
            self.redis_client.close()
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
        
        self.logger.info("インデックスワーカークリーンアップ完了")
    
    async def sync_with_sql_metadata(self):
        """SQL database メタデータとの同期"""
        try:
            with get_db() as session:
                # Get all documents from SQL metadata
                sql_documents = session.query(DocumentMetadata).all()
                
                for doc in sql_documents:
                    # Check if document exists in other databases
                    mongo_exists = self.mongo_client.get_document(doc.id) is not None
                    
                    # Sync metadata if missing or outdated
                    if not mongo_exists:
                        self.logger.warning(f"Document {doc.id} missing in MongoDB")
                        self.stats["inconsistent_documents"].add(doc.id)
                    
                    # Update embedding status if needed
                    if doc.metadata_json and doc.metadata_json.get("embedding_status") == "pending":
                        # Queue for embedding if not already processed
                        embedding_task = {
                            "document_id": doc.id,
                            "priority": "normal",
                            "created_at": datetime.utcnow().isoformat()
                        }
                        await self._queue_embedding_task(embedding_task)
                
                self.logger.info(f"SQL metadata sync completed: {len(sql_documents)} documents")
                
        except Exception as e:
            self.logger.error(f"SQL metadata sync error: {e}")

    async def update_sql_document_status(self, document_id: str, 
                                       index_status: str, 
                                       index_info: Dict[str, Any]):
        """SQL documentのインデックス状況を更新"""
        try:
            with get_db() as session:
                doc = session.query(DocumentMetadata).filter(
                    DocumentMetadata.id == document_id
                ).first()
                
                if doc:
                    if doc.metadata_json is None:
                        doc.metadata_json = {}
                    
                    doc.metadata_json.update({
                        "index_status": index_status,
                        "index_info": index_info,
                        "index_updated_at": datetime.utcnow().isoformat()
                    })
                    doc.updated_at = datetime.utcnow()
                    session.commit()
                    
                    self.logger.debug(f"Document index status updated: {document_id} -> {index_status}")
                    
        except Exception as e:
            self.logger.error(f"SQL document status update error: {e}")

    async def get_indexing_statistics(self) -> Dict[str, Any]:
        """インデックス統計情報を取得"""
        try:
            with get_db() as session:
                from sqlalchemy import func
                
                # Count documents by index status
                index_stats = session.query(
                    func.count(DocumentMetadata.id).label('total'),
                    func.sum(func.case(
                        (func.json_extract(DocumentMetadata.metadata_json, '$.index_status') == 'completed', 1),
                        else_=0
                    )).label('indexed'),
                    func.sum(func.case(
                        (func.json_extract(DocumentMetadata.metadata_json, '$.index_status') == 'processing', 1),
                        else_=0
                    )).label('processing'),
                    func.sum(func.case(
                        (func.json_extract(DocumentMetadata.metadata_json, '$.index_status') == 'failed', 1),
                        else_=0
                    )).label('failed')
                ).first()
                
                return {
                    "total_documents": index_stats.total or 0,
                    "indexed_documents": index_stats.indexed or 0,
                    "processing_documents": index_stats.processing or 0,
                    "failed_documents": index_stats.failed or 0,
                    "indexer_stats": self.stats,
                    "last_updated": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Indexing statistics error: {e}")
            return {"error": str(e)}

    async def get_indexer_stats(self) -> Dict[str, Any]:
        """インデックスワーカー統計取得"""
        stats_key = f"indexer_stats:{self.worker_id}"
        return self.redis_client.get_cache(stats_key)
    
    async def trigger_manual_sync(self) -> bool:
        """手動同期トリガー"""
        try:
            await self._perform_sync()
            # Also sync with SQL metadata
            await self.sync_with_sql_metadata()
            return True
        except Exception as e:
            self.logger.error(f"手動同期エラー: {e}")
            return False
    
    async def trigger_consistency_check(self) -> bool:
        """手動整合性チェックトリガー"""
        try:
            await self._check_database_consistency()
            return True
        except Exception as e:
            self.logger.error(f"手動整合性チェックエラー: {e}")
            return False


# インデックスワーカー実行用のメイン関数
async def main():
    """インデックスワーカーのメイン実行"""
    logging.basicConfig(level=logging.INFO)
    
    config = IndexConfig()
    indexer = IndexerWorker(config, worker_id="indexer_worker_001")
    
    try:
        await indexer.start_indexer()
    except KeyboardInterrupt:
        print("インデックスワーカー停止")


if __name__ == "__main__":
    asyncio.run(main())
