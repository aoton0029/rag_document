"""
パフォーマンス監視ユーティリティ
実行時間、メモリ使用量、その他のメトリクス監視
"""

import time
import psutil
import gc
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass
from functools import wraps
import threading
import json


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標データクラス"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_usage: float
    success: bool
    error: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_operation: Optional[str] = None
        self._monitoring_active = False
        self._monitoring_thread = None
        self._peak_memory = 0
    
    def start_monitoring(self, operation_name: str) -> None:
        """監視開始"""
        self.current_operation = operation_name
        self._monitoring_active = True
        self._peak_memory = self._get_memory_usage()
        
        # バックグラウンドでメモリ使用量を監視
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """監視停止"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
    
    def _monitor_memory(self) -> None:
        """メモリ使用量監視（バックグラウンド）"""
        while self._monitoring_active:
            current_memory = self._get_memory_usage()
            if current_memory > self._peak_memory:
                self._peak_memory = current_memory
            time.sleep(0.1)  # 100ms間隔で監視
    
    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """CPU使用率を取得"""
        return psutil.cpu_percent(interval=0.1)
    
    def measure_performance(self, operation_name: str, func: Callable, *args, **kwargs) -> tuple:
        """関数の実行パフォーマンスを測定"""
        # ガベージコレクション実行
        gc.collect()
        
        # 開始時の状態記録
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # 監視開始
        self.start_monitoring(operation_name)
        
        try:
            # 関数実行
            result = func(*args, **kwargs)
            
            # 終了時の状態記録
            end_time = time.time()
            memory_after = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            # 監視停止
            self.stop_monitoring()
            
            # メトリクス作成
            metrics = PerformanceMetrics(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=self._peak_memory,
                cpu_usage=cpu_usage,
                success=True
            )
            
            self.metrics_history.append(metrics)
            return result, metrics
            
        except Exception as e:
            # エラー時の処理
            end_time = time.time()
            memory_after = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            self.stop_monitoring()
            
            metrics = PerformanceMetrics(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=self._peak_memory,
                cpu_usage=cpu_usage,
                success=False,
                error=str(e)
            )
            
            self.metrics_history.append(metrics)
            raise
    
    def get_metrics_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """メトリクス要約を取得"""
        if operation_name:
            filtered_metrics = [m for m in self.metrics_history if m.operation == operation_name]
        else:
            filtered_metrics = self.metrics_history
        
        if not filtered_metrics:
            return {}
        
        durations = [m.duration for m in filtered_metrics]
        memory_usages = [m.memory_after - m.memory_before for m in filtered_metrics]
        success_rate = sum(1 for m in filtered_metrics if m.success) / len(filtered_metrics)
        
        return {
            "total_operations": len(filtered_metrics),
            "success_rate": success_rate,
            "duration_stats": {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "total": sum(durations)
            },
            "memory_stats": {
                "mean_usage": sum(memory_usages) / len(memory_usages),
                "max_usage": max(memory_usages),
                "min_usage": min(memory_usages)
            },
            "avg_cpu_usage": sum(m.cpu_usage for m in filtered_metrics) / len(filtered_metrics)
        }
    
    def export_metrics(self, filepath: str) -> None:
        """メトリクスをJSONファイルにエクスポート"""
        metrics_data = []
        for m in self.metrics_history:
            metrics_data.append({
                "operation": m.operation,
                "start_time": datetime.fromtimestamp(m.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(m.end_time).isoformat(),
                "duration": m.duration,
                "memory_before": m.memory_before,
                "memory_after": m.memory_after,
                "memory_peak": m.memory_peak,
                "cpu_usage": m.cpu_usage,
                "success": m.success,
                "error": m.error,
                "additional_metrics": m.additional_metrics
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)


# グローバルパフォーマンスモニターインスタンス
global_monitor = PerformanceMonitor()


def performance_monitor(operation_name: str):
    """パフォーマンス監視デコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result, metrics = global_monitor.measure_performance(
                operation_name, func, *args, **kwargs
            )
            return result
        return wrapper
    return decorator


def benchmark_function(func: Callable, iterations: int = 10, *args, **kwargs) -> Dict[str, Any]:
    """関数のベンチマーク実行"""
    monitor = PerformanceMonitor()
    operation_name = f"benchmark_{func.__name__}"
    
    results = []
    for i in range(iterations):
        try:
            result, metrics = monitor.measure_performance(
                f"{operation_name}_iter_{i}",
                func,
                *args,
                **kwargs
            )
            results.append(metrics)
        except Exception as e:
            print(f"Benchmark iteration {i} failed: {e}")
            continue
    
    if not results:
        return {"error": "All benchmark iterations failed"}
    
    # 統計計算
    durations = [r.duration for r in results]
    memory_usages = [r.memory_after - r.memory_before for r in results]
    
    return {
        "function_name": func.__name__,
        "iterations": len(results),
        "duration_stats": {
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "std": (sum((d - sum(durations) / len(durations)) ** 2 for d in durations) / len(durations)) ** 0.5
        },
        "memory_stats": {
            "mean_usage": sum(memory_usages) / len(memory_usages),
            "max_usage": max(memory_usages),
            "min_usage": min(memory_usages)
        },
        "success_rate": sum(1 for r in results if r.success) / len(results)
    }