import os
import re
from typing import Dict, Optional

class SQLLoader:
    def __init__(self, sql_dir="queries"):
        self.sql_dir = sql_dir
        self.queries: Dict[str, Dict[str, str]] = {}  # db_type -> query_name -> sql
        self._load_queries()
    
    def _load_queries(self):
        """SQLファイルを読み込み、名前付きクエリを辞書に格納"""
        # 各DBタイプのディレクトリを検索
        for db_type in os.listdir(self.sql_dir):
            db_dir = os.path.join(self.sql_dir, db_type)
            
            if not os.path.isdir(db_dir):
                continue
            
            self.queries[db_type] = {}
            
            # 各SQLファイルを読み込み
            for filename in os.listdir(db_dir):
                if filename.endswith('.sql'):
                    file_path = os.path.join(db_dir, filename)
                    module_name = filename[:-4]  # .sqlを除去
                    
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # 名前付きクエリを抽出 (-- query_name の形式)
                    queries = re.split(r'--\s+([a-zA-Z0-9_]+)\s*\n', content)[1:]
                    
                    # 奇数インデックスが名前、偶数インデックスがSQL
                    for i in range(0, len(queries), 2):
                        if i+1 < len(queries):
                            query_name = queries[i]
                            query_sql = queries[i+1].strip()
                            self.queries[db_type][f"{module_name}.{query_name}"] = query_sql
    
    def get_query(self, query_name: str, db_type: Optional[str] = None) -> Optional[str]:
        """名前からSQLクエリを取得"""
        from database import db_manager
        
        db_type = db_type or db_manager.default_db
        
        # 指定されたDBタイプにクエリがあればそれを返す
        if db_type in self.queries and query_name in self.queries[db_type]:
            return self.queries[db_type][query_name]
        
        # 共通クエリ（commonディレクトリ）をフォールバックとして使用
        if "common" in self.queries and query_name in self.queries["common"]:
            return self.queries["common"][query_name]
        
        return None

# シングルトンインスタンス
sql_loader = SQLLoader()