import logging
import json
from datetime import datetime
from typing import Dict, Any
from core.database import db_manager

class RAGSystemLogger:
    def __init__(self):
        self.logger = logging.getLogger("rag_system")
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定の初期化"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def log_query_metrics(self, metrics: Dict[str, Any]):
        """クエリメトリクスのログ"""
        try:
            storage_context = db_manager.get_storage_context()
            kvstore = storage_context.kvstore
            
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "query_metrics",
                "data": metrics
            }
            
            await kvstore.aput(f"metrics:{metrics.get('query_id')}", json.dumps(log_entry))
            self.logger.info(f"Query metrics logged: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Failed to log query metrics: {e}")
    
    async def log_performance_metrics(self, component: str, metrics: Dict[str, Any]):
        """パフォーマンスメトリクスのログ"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "component": component,
                "metrics": metrics
            }
            
            self.logger.info(f"Performance metrics - {component}: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metrics: {e}")

# グローバルインスタンス
system_logger = RAGSystemLogger()
