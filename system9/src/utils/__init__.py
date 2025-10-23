"""
Utils パッケージの初期化
共通ユーティリティの公開インターフェース
"""

from .config_manager import ConfigManager
from .logger import StructuredLogger, get_logger, log_execution_time
from .performance import (
    PerformanceMonitor, 
    PerformanceMetrics, 
    performance_monitor, 
    benchmark_function,
    global_monitor
)
from .file_utils import FileManager, DocumentProcessor, batch_process_files

# バージョン情報
__version__ = "1.0.0"

# パッケージレベルの設定
__all__ = [
    # Config management
    "ConfigManager",
    
    # Logging
    "StructuredLogger", 
    "get_logger", 
    "log_execution_time",
    
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceMetrics",
    "performance_monitor",
    "benchmark_function", 
    "global_monitor",
    
    # File utilities
    "FileManager",
    "DocumentProcessor", 
    "batch_process_files"
]

# パッケージレベルの設定インスタンス（シングルトン）
_config_manager = None
_default_logger = None


def get_config_manager(config_dir: str = "config") -> ConfigManager:
    """グローバル設定マネージャーを取得"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def get_default_logger(name: str = "rag_system") -> StructuredLogger:
    """デフォルトロガーを取得"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger(name)
    return _default_logger


# パッケージ初期化時の処理
def initialize_package(config_dir: str = "config", log_dir: str = "logs"):
    """パッケージの初期設定"""
    global _config_manager, _default_logger
    
    # 設定マネージャー初期化
    _config_manager = ConfigManager(config_dir)
    
    # ロガー初期化
    _default_logger = get_logger("rag_system", log_dir)
    
    # 初期化完了をログに記録
    _default_logger.info("RAG evaluation framework utils package initialized",
                        config_dir=config_dir, log_dir=log_dir)


# 便利関数
def quick_benchmark(func, *args, **kwargs):
    """関数のクイックベンチマーク"""
    return benchmark_function(func, iterations=5, *args, **kwargs)


def log_and_monitor(operation_name: str):
    """ログとパフォーマンス監視を組み合わせたデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_default_logger()
            monitor = global_monitor
            
            logger.info(f"Starting operation: {operation_name}")
            
            try:
                result, metrics = monitor.measure_performance(
                    operation_name, func, *args, **kwargs
                )
                
                logger.log_performance(
                    operation_name,
                    metrics.duration,
                    memory_usage=metrics.memory_after - metrics.memory_before,
                    success=True
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Operation failed: {operation_name}", 
                           error=str(e))
                raise
        
        return wrapper
    return decorator