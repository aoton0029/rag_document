"""
Neo4jクライアントクラス
グラフストアとして使用
"""

import logging
from typing import Optional, Dict, Any, List
from neo4j import GraphDatabase, Driver, Session
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.graph_stores.types import GraphStore


logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4jクライアントクラス
    llama_index Neo4jGraphStoreのラッパー
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60,
        **kwargs
    ):
        """
        Neo4jクライアントの初期化
        
        Args:
            uri: Neo4j接続URI
            username: ユーザー名
            password: パスワード
            database: データベース名
            max_connection_lifetime: 最大接続ライフタイム
            max_connection_pool_size: 最大接続プールサイズ
            connection_acquisition_timeout: 接続取得タイムアウト
            **kwargs: その他のNeo4j接続パラメータ
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_acquisition_timeout = connection_acquisition_timeout
        self.connection_kwargs = kwargs
        
        self._driver: Optional[Driver] = None
        self._graph_store: Optional[Neo4jGraphStore] = None
    
    def connect(self) -> None:
        """Neo4j接続を確立"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_acquisition_timeout=self.connection_acquisition_timeout,
                **self.connection_kwargs
            )
            
            # 接続テスト
            with self._driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            
            logger.info(f"Neo4j接続成功: {self.uri}")
            
        except Exception as e:
            logger.error(f"Neo4j接続エラー: {e}")
            raise
    
    def disconnect(self) -> None:
        """Neo4j接続を切断"""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._graph_store = None
            logger.info("Neo4j接続を切断しました")
    
    def get_driver(self) -> Driver:
        """Neo4jドライバーを取得"""
        if self._driver is None:
            self.connect()
        return self._driver
    
    def get_session(self) -> Session:
        """Neo4jセッションを取得"""
        driver = self.get_driver()
        return driver.session(database=self.database)
    
    def get_graph_store(
        self,
        node_label: str = "Entity",
        rel_type: str = "RELATED",
        **kwargs
    ) -> GraphStore:
        """
        llama_index Neo4jGraphStoreを取得
        
        Args:
            node_label: ノードラベル
            rel_type: リレーションシップタイプ
            **kwargs: Neo4jGraphStoreの追加パラメータ
            
        Returns:
            Neo4jGraphStore インスタンス
        """
        if self._graph_store is None:
            self._graph_store = Neo4jGraphStore(
                username=self.username,
                password=self.password,
                url=self.uri,
                database=self.database,
                node_label=node_label,
                rel_type=rel_type,
                **kwargs
            )
        
        return self._graph_store
    
    def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Cypherクエリを実行
        
        Args:
            query: Cypherクエリ
            parameters: クエリパラメータ
            
        Returns:
            クエリ結果
        """
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write_transaction(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        書き込みトランザクションを実行
        
        Args:
            query: Cypherクエリ
            parameters: クエリパラメータ
            
        Returns:
            クエリ結果
        """
        def _write_tx(tx, query, params):
            result = tx.run(query, params)
            return [record.data() for record in result]
        
        with self.get_session() as session:
            return session.execute_write(_write_tx, query, parameters or {})
    
    def create_index(
        self, 
        label: str, 
        property_name: str,
        index_type: str = "BTREE"
    ) -> None:
        """
        インデックスを作成
        
        Args:
            label: ノードラベル
            property_name: プロパティ名
            index_type: インデックスタイプ
        """
        query = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
        self.execute_write_transaction(query)
        logger.info(f"インデックスを作成しました: {label}.{property_name}")
    
    def create_constraint(
        self,
        label: str,
        property_name: str,
        constraint_type: str = "UNIQUE"
    ) -> None:
        """
        制約を作成
        
        Args:
            label: ノードラベル
            property_name: プロパティ名
            constraint_type: 制約タイプ
        """
        constraint_name = f"{label}_{property_name}_{constraint_type.lower()}"
        if constraint_type.upper() == "UNIQUE":
            query = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type}")
        
        self.execute_write_transaction(query)
        logger.info(f"制約を作成しました: {constraint_name}")
    
    def clear_database(self) -> None:
        """データベースをクリア（全ノード・リレーションシップ削除）"""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write_transaction(query)
        logger.info("データベースをクリアしました")
    
    def get_node_count(self, label: Optional[str] = None) -> int:
        """ノード数を取得"""
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            query = "MATCH (n) RETURN count(n) AS count"
        
        result = self.execute_query(query)
        return result[0]["count"] if result else 0
    
    def get_relationship_count(self, rel_type: Optional[str] = None) -> int:
        """リレーションシップ数を取得"""
        if rel_type:
            query = f"MATCH ()-[r:{rel_type}]-() RETURN count(r) AS count"
        else:
            query = "MATCH ()-[r]-() RETURN count(r) AS count"
        
        result = self.execute_query(query)
        return result[0]["count"] if result else 0
    
    def get_database_info(self) -> Dict[str, Any]:
        """データベース情報を取得"""
        node_count = self.get_node_count()
        relationship_count = self.get_relationship_count()
        
        # ラベル情報を取得
        labels_result = self.execute_query("CALL db.labels()")
        labels = [record["label"] for record in labels_result]
        
        # リレーションシップタイプを取得
        rel_types_result = self.execute_query("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in rel_types_result]
        
        return {
            "node_count": node_count,
            "relationship_count": relationship_count,
            "labels": labels,
            "relationship_types": rel_types
        }
    
    def health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            with self.get_session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            return True
        except Exception as e:
            logger.error(f"Neo4j ヘルスチェック失敗: {e}")
            return False
    
    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.disconnect()