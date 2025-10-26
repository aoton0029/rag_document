"""
Redisクライアントクラス
インデックスストアとして使用
"""

import logging
from typing import Optional, Dict, Any, List
import redis
from redis.connection import ConnectionPool
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore


logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redisクライアントクラス
    llama_index RedisIndexStoreのラッパー
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        decode_responses: bool = True,
        socket_timeout: float = 30.0,
        socket_connect_timeout: float = 30.0,
        max_connections: int = 50,
        **kwargs
    ):
        """
        Redisクライアントの初期化
        
        Args:
            host: Redisホスト
            port: Redisポート
            password: パスワード
            db: データベース番号
            decode_responses: レスポンスをデコードするか
            socket_timeout: ソケットタイムアウト
            socket_connect_timeout: 接続タイムアウト
            max_connections: 最大接続数
            **kwargs: その他のRedis接続パラメータ
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.decode_responses = decode_responses
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.max_connections = max_connections
        self.connection_kwargs = kwargs
        
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._index_store: Optional[RedisIndexStore] = None
    
    def connect(self) -> None:
        """Redis接続を確立"""
        try:
            # 接続プール作成
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                max_connections=self.max_connections,
                **self.connection_kwargs
            )
            
            # Redisクライアント作成
            self._client = redis.Redis(connection_pool=self._pool)
            
            # 接続テスト
            self._client.ping()
            logger.info(f"Redis接続成功: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Redis接続エラー: {e}")
            raise
    
    def disconnect(self) -> None:
        """Redis接続を切断"""
        if self._client:
            self._client.close()
            self._client = None
        if self._pool:
            self._pool.disconnect()
            self._pool = None
        self._index_store = None
        logger.info("Redis接続を切断しました")
    
    def get_client(self) -> redis.Redis:
        """Redisクライアントを取得"""
        if self._client is None:
            self.connect()
        return self._client
    
    def get_index_store(
        self,
        namespace: str = "default",
        collection_suffix: str = "index_store"
    ) -> BaseIndexStore:
        """
        llama_index RedisIndexStoreを取得
        
        Args:
            namespace: インデックスの名前空間
            collection_suffix: コレクションサフィックス
            
        Returns:
            RedisIndexStore インスタンス
        """
        if self._index_store is None:
            client = self.get_client()
            self._index_store = RedisIndexStore(
                redis_client=client,
                namespace=namespace,
                collection_suffix=collection_suffix
            )
        
        return self._index_store
    
    def set_key(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """キーと値を設定"""
        client = self.get_client()
        return client.set(key, value, ex=ex)
    
    def get_key(self, key: str) -> Any:
        """キーの値を取得"""
        client = self.get_client()
        return client.get(key)
    
    def delete_key(self, key: str) -> int:
        """キーを削除"""
        client = self.get_client()
        return client.delete(key)
    
    def exists_key(self, key: str) -> bool:
        """キーが存在するかチェック"""
        client = self.get_client()
        return bool(client.exists(key))
    
    def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """パターンにマッチするキーを取得"""
        client = self.get_client()
        return client.keys(pattern)
    
    def flush_db(self) -> bool:
        """現在のDBをクリア"""
        client = self.get_client()
        return client.flushdb()
    
    def flush_all(self) -> bool:
        """全DBをクリア"""
        client = self.get_client()
        return client.flushall()
    
    def get_db_size(self) -> int:
        """DB内のキー数を取得"""
        client = self.get_client()
        return client.dbsize()
    
    def get_memory_usage(self, key: str) -> int:
        """キーのメモリ使用量を取得"""
        client = self.get_client()
        return client.memory_usage(key)
    
    def get_info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Redis情報を取得"""
        client = self.get_client()
        return client.info(section)
    
    def health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            client = self.get_client()
            client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ヘルスチェック失敗: {e}")
            return False
    
    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.disconnect()