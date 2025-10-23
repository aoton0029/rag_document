"""
ログ設定モジュール
構造化ログとパフォーマンス監視を提供
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from functools import wraps
import json
import sys


class StructuredLogger:
    """構造化ログクラス"""
    
    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # ロガーの設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # ハンドラーが既に設定されている場合はクリア
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # ファイルハンドラー
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # フォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # メトリクス保存
        self.metrics = {}
    
    def log_structured(self, level: str, message: str, **kwargs) -> None:
        """構造化ログを出力"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "logger": self.name,
            **kwargs
        }
        
        # JSON形式でログ出力
        json_message = json.dumps(log_data, ensure_ascii=False, default=str)
        
        if level.upper() == "DEBUG":
            self.logger.debug(json_message)
        elif level.upper() == "INFO":
            self.logger.info(json_message)
        elif level.upper() == "WARNING":
            self.logger.warning(json_message)
        elif level.upper() == "ERROR":
            self.logger.error(json_message)
        elif level.upper() == "CRITICAL":
            self.logger.critical(json_message)
    
    def info(self, message: str, **kwargs) -> None:
        """INFO レベルログ"""
        self.log_structured("INFO", message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """DEBUG レベルログ"""
        self.log_structured("DEBUG", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """WARNING レベルログ"""
        self.log_structured("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """ERROR レベルログ"""
        self.log_structured("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """CRITICAL レベルログ"""
        self.log_structured("CRITICAL", message, **kwargs)
    
    def log_experiment(self, experiment_name: str, parameters: Dict[str, Any], results: Dict[str, Any]) -> None:
        """実験結果をログ"""
        self.log_structured(
            "INFO",
            "Experiment completed",
            experiment_name=experiment_name,
            parameters=parameters,
            results=results,
            event_type="experiment"
        )
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """パフォーマンスログ"""
        self.log_structured(
            "INFO",
            f"Performance: {operation}",
            operation=operation,
            duration_seconds=duration,
            event_type="performance",
            **kwargs
        )
    
    def add_metric(self, name: str, value: float) -> None:
        """メトリクス追加"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """メトリクス取得"""
        if name:
            return self.metrics.get(name, [])
        return self.metrics
    
    def save_metrics(self, filepath: Optional[str] = None) -> None:
        """メトリクスをファイルに保存"""
        if not filepath:
            filepath = self.log_dir / f"metrics_{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2, default=str)


def log_execution_time(logger: StructuredLogger, operation_name: str):
    """実行時間を測定するデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_performance(
                    operation=f"{operation_name}:{func.__name__}",
                    duration=duration,
                    success=True
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_performance(
                    operation=f"{operation_name}:{func.__name__}",
                    duration=duration,
                    success=False,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator


def get_logger(name: str, log_dir: str = "logs") -> StructuredLogger:
    """ロガーインスタンスを取得"""
    return StructuredLogger(name, log_dir)