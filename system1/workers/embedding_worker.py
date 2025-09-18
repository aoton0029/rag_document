import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..database.document_db.mongo_client import MongoClient
from ..database.keyvalue_db.redis_client import RedisClient
from ..database.vector_db.milvus_client import MilvusClient
from ..database.relational_db.database import get_db
from ..database.relational_db.models import DocumentMetadata
from ..embedding.llamaindex_embedding_service import EmbeddingService, EmbeddingConfig


class EmbeddingWorker:
    """
    埋め込み生成ワーカー
    キューから埋め込み生成タスクを取得し、処理を実行
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        mongo_client: MongoClient = None,
        redis_client: RedisClient = None,
        milvus_client: MilvusClient = None,
        worker_id: str = "worker_001",
        batch_size: int = 32,
        max_retries: int = 3
    ):
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # サービス初期化
        self.embedding_service = embedding_service or self._create_embedding_service()
        self.mongo_client = mongo_client or MongoClient()
        self.redis_client = redis_client or RedisClient()
        self.milvus_client = milvus_client or MilvusClient()
        
        # 統計情報
        self.stats = {
            "processed_tasks": 0,
            "failed_tasks": 0,
            "total_embeddings": 0,
            "start_time": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        self.logger.info(f"埋め込みワーカー初期化完了: {self.worker_id}")
    
    def _create_embedding_service(self) -> EmbeddingService:
        """埋め込みサービス作成"""
        config = EmbeddingConfig(
            model_name="nomic-embed-text",
            embedding_dim=768,
            chunk_size=512,
            batch_size=self.batch_size
        )
        return EmbeddingService(config)
    
    async def start_worker(self):
        """ワーカー開始"""
        self.logger.info(f"埋め込みワーカー開始: {self.worker_id}")
        
        while True:
            try:
                # キューからタスク取得
                task = self.redis_client.get_from_processing_queue()
                
                if task:
                    await self._process_task(task)
                else:
                    # タスクがない場合は短時間待機
                    await asyncio.sleep(1)
                
                # 定期的な統計更新
                if self.stats["processed_tasks"] % 10 == 0:
                    await self._update_worker_stats()
                    
            except KeyboardInterrupt:
                self.logger.info("ワーカー停止要求を受信")
                break
            except Exception as e:
                self.logger.error(f"ワーカー実行エラー: {e}")
                await asyncio.sleep(5)  # エラー時は少し長めに待機
        
        self.logger.info(f"埋め込みワーカー終了: {self.worker_id}")
    
    async def _process_task(self, task: Dict[str, Any]):
        """タスク処理"""
        task_id = task.get("task_id")
        task_data = task.get("data", {})
        task_type = task_data.get("task_type")
        
        self.logger.info(f"タスク処理開始: {task_id}, タイプ: {task_type}")
        
        try:
            if task_type == "generate_embeddings":
                await self._generate_document_embeddings(task_data)
            elif task_type == "update_embeddings":
                await self._update_document_embeddings(task_data)
            elif task_type == "reindex_embeddings":
                await self._reindex_embeddings(task_data)
            else:
                raise ValueError(f"未知のタスクタイプ: {task_type}")
            
            self.stats["processed_tasks"] += 1
            self.stats["last_activity"] = datetime.utcnow()
            
            # 成功ログ
            self.logger.info(f"タスク処理完了: {task_id}")
            
            # 成功をRedisに記録
            await self._record_task_result(task_id, "completed")
            
        except Exception as e:
            self.stats["failed_tasks"] += 1
            self.logger.error(f"タスク処理失敗: {task_id}, {e}")
            
            # リトライ処理
            retry_count = task_data.get("retry_count", 0)
            if retry_count < self.max_retries:
                await self._retry_task(task, retry_count + 1)
            else:
                await self._record_task_result(task_id, "failed", str(e))
    
    async def _generate_document_embeddings(self, task_data: Dict[str, Any]):
        """ドキュメントの埋め込み生成"""
        document_id = task_data["document_id"]
        chunk_ids = task_data["chunk_ids"]
        model_name = task_data.get("model_name", "ollama")
        
        # チャンクデータ取得
        chunks = await self._get_chunks_from_mongo(chunk_ids)
        
        if not chunks:
            raise ValueError(f"チャンクが見つかりません: {document_id}")
        
        # テキストとメタデータを分離
        texts = [chunk["content"] for chunk in chunks]
        chunk_metadata = [chunk["metadata"] for chunk in chunks]
        
        # バッチ処理で埋め込み生成
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = await self.embedding_service.create_embeddings(
                batch_texts, model_name
            )
            all_embeddings.extend(batch_embeddings)
            
            # 進捗ログ
            processed = min(i + self.batch_size, len(texts))
            self.logger.info(f"埋め込み生成進捗: {processed}/{len(texts)} ({document_id})")
        
        # 埋め込みをMilvusに保存
        await self._store_embeddings_to_milvus(
            document_id, texts, all_embeddings, chunk_metadata
        )
        
        # 埋め込みをRedisにキャッシュ
        await self._cache_embeddings(document_id, all_embeddings)
        
        # MongoDB のチャンクに埋め込みIDを更新
        await self._update_chunk_embeddings_info(chunk_ids, all_embeddings)
        
        self.stats["total_embeddings"] += len(all_embeddings)
        self.logger.info(f"ドキュメント埋め込み生成完了: {document_id}, {len(all_embeddings)} 件")
    
    async def _update_document_embeddings(self, task_data: Dict[str, Any]):
        """既存ドキュメントの埋め込み更新"""
        document_id = task_data["document_id"]
        model_name = task_data.get("model_name", "ollama")
        
        # 既存埋め込みを削除
        self.milvus_client.delete_document_vectors(document_id)
        
        # 新しい埋め込みを生成
        await self._generate_document_embeddings(task_data)
        
        self.logger.info(f"ドキュメント埋め込み更新完了: {document_id}")
    
    async def _reindex_embeddings(self, task_data: Dict[str, Any]):
        """埋め込みの再インデックス"""
        model_name = task_data.get("model_name", "ollama")
        document_filter = task_data.get("document_filter")
        
        # 対象ドキュメント取得
        if document_filter:
            documents = self.mongo_client.search_by_metadata(document_filter)
        else:
            documents = self.mongo_client.get_all_documents()
        
        reindexed_count = 0
        for doc in documents:
            try:
                document_id = doc["document_id"]
                
                # ドキュメントのチャンク取得
                chunks = list(self.mongo_client.documents.find(
                    {"document_id": document_id, "chunk_index": {"$exists": True}}
                ))
                
                if chunks:
                    chunk_ids = [str(chunk["_id"]) for chunk in chunks]
                    
                    # 埋め込み再生成タスクを作成
                    reindex_task = {
                        "task_type": "update_embeddings",
                        "document_id": document_id,
                        "chunk_ids": chunk_ids,
                        "model_name": model_name
                    }
                    
                    await self._generate_document_embeddings(reindex_task)
                    reindexed_count += 1
                    
            except Exception as e:
                self.logger.error(f"再インデックス失敗: {doc.get('document_id')}, {e}")
        
        self.logger.info(f"再インデックス完了: {reindexed_count} ドキュメント")
    
    async def _get_chunks_from_mongo(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """MongoDBからチャンクデータを取得"""
        from bson import ObjectId
        
        object_ids = []
        for chunk_id in chunk_ids:
            try:
                object_ids.append(ObjectId(chunk_id))
            except:
                # ObjectIdでない場合はそのまま使用
                object_ids.append(chunk_id)
        
        chunks = list(self.mongo_client.documents.find(
            {"_id": {"$in": object_ids}}
        ))
        
        return chunks
    
    async def _store_embeddings_to_milvus(
        self,
        document_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ):
        """Milvusに埋め込みを保存"""
        try:
            # Milvusコレクションが存在しない場合は作成
            if not self.milvus_client.collection:
                self.milvus_client.create_collection(dimension=len(embeddings[0]))
            
            # データをMilvusに挿入
            self.milvus_client.insert_vectors(document_id, texts, embeddings)
            
            self.logger.info(f"Milvus保存完了: {document_id}, {len(embeddings)} ベクトル")
            
        except Exception as e:
            self.logger.error(f"Milvus保存エラー: {document_id}, {e}")
            raise
    
    async def _cache_embeddings(self, document_id: str, embeddings: List[List[float]]):
        """埋め込みをRedisにキャッシュ"""
        try:
            success = self.redis_client.set_document_embeddings(
                document_id, embeddings, expire_seconds=3600
            )
            
            if success:
                self.logger.info(f"埋め込みキャッシュ保存: {document_id}")
            else:
                self.logger.warning(f"埋め込みキャッシュ保存失敗: {document_id}")
                
        except Exception as e:
            self.logger.error(f"埋め込みキャッシュエラー: {document_id}, {e}")
    
    async def _update_chunk_embeddings_info(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]]
    ):
        """MongoDBのチャンクに埋め込み情報を更新"""
        from bson import ObjectId
        
        for i, chunk_id in enumerate(chunk_ids):
            try:
                object_id = ObjectId(chunk_id) if ObjectId.is_valid(chunk_id) else chunk_id
                
                self.mongo_client.documents.update_one(
                    {"_id": object_id},
                    {
                        "$set": {
                            "embedding_info": {
                                "has_embedding": True,
                                "embedding_model": "ollama",
                                "embedding_dim": len(embeddings[i]),
                                "generated_at": datetime.utcnow()
                            },
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
            except Exception as e:
                self.logger.error(f"チャンク埋め込み情報更新エラー: {chunk_id}, {e}")
    
    async def _retry_task(self, task: Dict[str, Any], retry_count: int):
        """タスクリトライ"""
        task_data = task.get("data", {})
        task_data["retry_count"] = retry_count
        
        # 指数バックオフでリトライ
        delay = min(60, 2 ** retry_count)  # 最大60秒
        await asyncio.sleep(delay)
        
        # キューに再追加
        retry_task_id = f"{task['task_id']}_retry_{retry_count}"
        success = self.redis_client.add_to_processing_queue(retry_task_id, task_data)
        
        if success:
            self.logger.info(f"タスクリトライキューイング: {retry_task_id}")
        else:
            self.logger.error(f"タスクリトライ失敗: {retry_task_id}")
    
    async def _record_task_result(
        self,
        task_id: str,
        status: str,
        error_message: str = None
    ):
        """タスク結果を記録"""
        result_data = {
            "task_id": task_id,
            "worker_id": self.worker_id,
            "status": status,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        if error_message:
            result_data["error"] = error_message
        
        result_key = f"task_result:{task_id}"
        self.redis_client.set_cache(result_key, result_data, 86400)  # 24時間保持
    
    async def update_document_embedding_status(self, document_id: str, 
                                           status: str, embedding_count: int = 0):
        """ドキュメントの埋め込み状況をSQL databaseに更新"""
        try:
            with get_db() as session:
                doc = session.query(DocumentMetadata).filter(
                    DocumentMetadata.id == document_id
                ).first()
                
                if doc:
                    if doc.metadata_json is None:
                        doc.metadata_json = {}
                    
                    doc.metadata_json.update({
                        "embedding_status": status,
                        "embedding_count": embedding_count,
                        "embedding_updated_at": datetime.utcnow().isoformat()
                    })
                    doc.updated_at = datetime.utcnow()
                    session.commit()
                    
                    self.logger.debug(f"埋め込み状況更新: {document_id} -> {status}")
                    
        except Exception as e:
            self.logger.error(f"埋め込み状況更新エラー: {e}")

    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """埋め込み統計情報を取得"""
        try:
            with get_db() as session:
                # Documents with embedding status
                from sqlalchemy import func, case
                
                embedding_stats = session.query(
                    func.count(DocumentMetadata.id).label('total'),
                    func.sum(case(
                        (func.json_extract(DocumentMetadata.metadata_json, '$.embedding_status') == 'completed', 1),
                        else_=0
                    )).label('completed'),
                    func.sum(case(
                        (func.json_extract(DocumentMetadata.metadata_json, '$.embedding_status') == 'processing', 1),
                        else_=0
                    )).label('processing'),
                    func.sum(case(
                        (func.json_extract(DocumentMetadata.metadata_json, '$.embedding_status') == 'failed', 1),
                        else_=0
                    )).label('failed')
                ).first()
                
                return {
                    "total_documents": embedding_stats.total or 0,
                    "completed_embeddings": embedding_stats.completed or 0,
                    "processing_embeddings": embedding_stats.processing or 0,
                    "failed_embeddings": embedding_stats.failed or 0,
                    "worker_stats": self.stats,
                    "last_updated": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"埋め込み統計取得エラー: {e}")
            return {"error": str(e)}

    async def _update_worker_stats(self):
        """ワーカー統計更新"""
        stats_data = {
            **self.stats,
            "start_time": self.stats["start_time"].isoformat(),
            "last_activity": self.stats["last_activity"].isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.stats["start_time"]).total_seconds()
        }
        
        stats_key = f"worker_stats:{self.worker_id}"
        self.redis_client.set_cache(stats_key, stats_data, 300)  # 5分間保持
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """ワーカー統計取得"""
        await self._update_worker_stats()
        stats_key = f"worker_stats:{self.worker_id}"
        return self.redis_client.get_cache(stats_key)
    
    async def stop_worker(self):
        """ワーカー停止"""
        self.logger.info(f"ワーカー停止: {self.worker_id}")
        # 必要であればクリーンアップ処理


# ワーカー実行用のメイン関数
async def main():
    """埋め込みワーカーのメイン実行"""
    logging.basicConfig(level=logging.INFO)
    
    worker = EmbeddingWorker(worker_id="embedding_worker_001")
    
    try:
        await worker.start_worker()
    except KeyboardInterrupt:
        await worker.stop_worker()


if __name__ == "__main__":
    asyncio.run(main())
