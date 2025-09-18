import asyncio
import uvicorn
from api.main import app
from core.database import db_manager
import logging

logger = logging.getLogger(__name__)

async def initialize_system():
    """システム初期化"""
    try:
        # データベース接続の初期化
        db_manager.initialize_connections()
        logger.info("System initialized successfully")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

def main():
    """メイン実行関数"""
    # システム初期化
    asyncio.run(initialize_system())
    
    # FastAPIアプリケーション起動
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
