import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

class Config:
    """アプリケーション設定を一元管理するクラス"""
    
    # データベース設定
    DEFAULT_DB = os.getenv("DEFAULT_DB", "sqlite")
    # SQLite設定
    SQLITE_DATABASE_URL = os.getenv("SQLITE_DATABASE_URL", "sqlite:///./app.db")
    # SQL Server設定
    SQLSERVER_HOST = os.getenv("SQLSERVER_HOST", "localhost")
    SQLSERVER_PORT = int(os.getenv("SQLSERVER_PORT", "1433"))
    SQLSERVER_USER = os.getenv("SQLSERVER_USER", "sa")
    SQLSERVER_PASSWORD = os.getenv("SQLSERVER_PASSWORD", "")
    SQLSERVER_DATABASE = os.getenv("SQLSERVER_DATABASE", "appdb")
    
    @classmethod
    def get_db_config(cls, db_type: Optional[str] = None) -> Dict[str, Any]:
        """指定されたデータベース向けの設定を取得"""
        db_type = db_type or cls.DEFAULT_DB
        
        if db_type == "sqlite":
            return {
                "url": cls.SQLITE_DATABASE_URL,
                "connect_args": {"check_same_thread": False}
            }
        
        elif db_type == "mariadb":
            return {
                "host": cls.MDB_HOST,
                "port": cls.MDB_PORT,
                "user": cls.MDB_USER,
                "password": cls.MDB_PASSWORD,
                "database": cls.MDB_DATABASE,
                "url": f"mysql+pymysql://{cls.MDB_USER}:{cls.MDB_PASSWORD}@{cls.MDB_HOST}:{cls.MDB_PORT}/{cls.MDB_DATABASE}"
            }
        
        elif db_type == "sqlserver":
            return {
                "host": cls.SQLSERVER_HOST,
                "port": cls.SQLSERVER_PORT,
                "user": cls.SQLSERVER_USER,
                "password": cls.SQLSERVER_PASSWORD,
                "database": cls.SQLSERVER_DATABASE,
                "url": f"mssql+pyodbc://{cls.SQLSERVER_USER}:{cls.SQLSERVER_PASSWORD}@{cls.SQLSERVER_HOST}:{cls.SQLSERVER_PORT}/{cls.SQLSERVER_DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
            }
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

# 設定インスタンス
config = Config()