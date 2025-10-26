"""
Monitoring Module
実験ログ、メトリクス追跡、レポート生成
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentLog:
    """実験ログ"""
    run_id: str
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class ExperimentLogger:
    """
    実験ロガー
    実験設定と結果を記録
    """
    
    def __init__(self, log_dir: str = "results/logs"):
        """
        ExperimentLoggerの初期化
        
        Args:
            log_dir: ログディレクトリ
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_run_id = None
        self._current_log = []
    
    def start_run(self, run_id: str, config: Dict[str, Any]):
        """
        実験実行を開始
        
        Args:
            run_id: 実行ID
            config: 実験設定
        """
        self._current_run_id = run_id
        self._current_log = []
        
        # 設定を保存
        config_path = self.log_dir / f"{run_id}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"実験実行開始: {run_id}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        メトリクスを記録
        
        Args:
            metrics: メトリクス辞書
            metadata: メタデータ
        """
        if self._current_run_id is None:
            logger.warning("実験実行が開始されていません")
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        self._current_log.append(log_entry)
    
    def end_run(self):
        """実験実行を終了"""
        if self._current_run_id is None:
            logger.warning("実験実行が開始されていません")
            return
        
        # ログを保存
        log_path = self.log_dir / f"{self._current_run_id}_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self._current_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"実験実行終了: {self._current_run_id}")
        self._current_run_id = None
        self._current_log = []
    
    def load_run_log(self, run_id: str) -> List[Dict[str, Any]]:
        """
        実行ログを読み込む
        
        Args:
            run_id: 実行ID
            
        Returns:
            ログエントリのリスト
        """
        log_path = self.log_dir / f"{run_id}_log.json"
        
        if not log_path.exists():
            logger.warning(f"ログファイルが見つかりません: {log_path}")
            return []
        
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_run_config(self, run_id: str) -> Dict[str, Any]:
        """
        実行設定を読み込む
        
        Args:
            run_id: 実行ID
            
        Returns:
            設定辞書
        """
        config_path = self.log_dir / f"{run_id}_config.json"
        
        if not config_path.exists():
            logger.warning(f"設定ファイルが見つかりません: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class MetricsTracker:
    """
    メトリクス追跡クラス
    リアルタイムでメトリクスを収集
    """
    
    def __init__(self):
        """MetricsTrackerの初期化"""
        self._metrics: Dict[str, List[float]] = {}
        self._timestamps: List[str] = []
    
    def track(self, metric_name: str, value: float):
        """
        メトリクスを追跡
        
        Args:
            metric_name: メトリクス名
            value: 値
        """
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        
        self._metrics[metric_name].append(value)
        self._timestamps.append(datetime.now().isoformat())
    
    def get_metrics(self, metric_name: str) -> List[float]:
        """
        メトリクスを取得
        
        Args:
            metric_name: メトリクス名
            
        Returns:
            値のリスト
        """
        return self._metrics.get(metric_name, [])
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        最新のメトリクスを取得
        
        Args:
            metric_name: メトリクス名
            
        Returns:
            最新の値
        """
        metrics = self.get_metrics(metric_name)
        return metrics[-1] if metrics else None
    
    def get_average(self, metric_name: str) -> Optional[float]:
        """
        平均値を取得
        
        Args:
            metric_name: メトリクス名
            
        Returns:
            平均値
        """
        metrics = self.get_metrics(metric_name)
        return sum(metrics) / len(metrics) if metrics else None
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        すべてのメトリクスの要約を取得
        
        Returns:
            要約辞書
        """
        import statistics
        
        summary = {}
        for metric_name, values in self._metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return summary
    
    def reset(self):
        """メトリクスをリセット"""
        self._metrics.clear()
        self._timestamps.clear()


class ResultReporter:
    """
    結果レポーター
    実験結果をレポート生成
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        ResultReporterの初期化
        
        Args:
            results_dir: 結果ディレクトリ
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(
        self,
        run_id: str,
        results: Dict[str, Any],
        format: str = "json"
    ):
        """
        結果を保存
        
        Args:
            run_id: 実行ID
            results: 結果辞書
            format: 保存形式（json, csv）
        """
        if format == "json":
            result_path = self.results_dir / f"{run_id}_results.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            result_path = self.results_dir / f"{run_id}_results.csv"
            # 結果をDataFrameに変換
            df = pd.DataFrame([results])
            df.to_csv(result_path, index=False, encoding='utf-8')
        
        logger.info(f"結果を保存しました: {result_path}")
    
    def load_results(self, run_id: str, format: str = "json") -> Dict[str, Any]:
        """
        結果を読み込む
        
        Args:
            run_id: 実行ID
            format: 保存形式
            
        Returns:
            結果辞書
        """
        if format == "json":
            result_path = self.results_dir / f"{run_id}_results.json"
            
            if not result_path.exists():
                logger.warning(f"結果ファイルが見つかりません: {result_path}")
                return {}
            
            with open(result_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif format == "csv":
            result_path = self.results_dir / f"{run_id}_results.csv"
            
            if not result_path.exists():
                logger.warning(f"結果ファイルが見つかりません: {result_path}")
                return {}
            
            df = pd.read_csv(result_path, encoding='utf-8')
            return df.to_dict('records')[0]
        
        return {}
    
    def generate_comparison_report(
        self,
        run_ids: List[str],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        複数実行の比較レポートを生成
        
        Args:
            run_ids: 実行IDのリスト
            output_path: 出力パス
            
        Returns:
            比較DataFrame
        """
        results_list = []
        
        for run_id in run_ids:
            results = self.load_results(run_id)
            if results:
                results['run_id'] = run_id
                results_list.append(results)
        
        if not results_list:
            logger.warning("比較する結果がありません")
            return pd.DataFrame()
        
        df = pd.DataFrame(results_list)
        
        if output_path:
            output_path = Path(output_path)
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"比較レポートを保存しました: {output_path}")
        
        return df


class ResultVisualizer:
    """
    結果可視化クラス
    メトリクスのグラフ化
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        ResultVisualizerの初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_comparison(
        self,
        results_df: pd.DataFrame,
        metric_names: List[str],
        output_filename: Optional[str] = None
    ):
        """
        メトリクス比較プロット
        
        Args:
            results_df: 結果DataFrame
            metric_names: メトリクス名のリスト
            output_filename: 出力ファイル名
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, len(metric_names), figsize=(5*len(metric_names), 4))
            
            if len(metric_names) == 1:
                axes = [axes]
            
            for ax, metric_name in zip(axes, metric_names):
                if metric_name in results_df.columns:
                    results_df.plot(
                        x='run_id',
                        y=metric_name,
                        kind='bar',
                        ax=ax,
                        title=metric_name
                    )
            
            plt.tight_layout()
            
            if output_filename:
                output_path = self.output_dir / output_filename
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"グラフを保存しました: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlibが利用できません: pip install matplotlib")
        except Exception as e:
            logger.error(f"プロットエラー: {e}")


class MonitoringManager:
    """
    モニタリング管理クラス
    ロギング、追跡、レポート、可視化を統合
    """
    
    def __init__(
        self,
        log_dir: str = "results/logs",
        results_dir: str = "results"
    ):
        """
        MonitoringManagerの初期化
        
        Args:
            log_dir: ログディレクトリ
            results_dir: 結果ディレクトリ
        """
        self.logger = ExperimentLogger(log_dir)
        self.tracker = MetricsTracker()
        self.reporter = ResultReporter(results_dir)
        self.visualizer = ResultVisualizer(f"{results_dir}/figures")
    
    def start_experiment(self, run_id: str, config: Dict[str, Any]):
        """実験を開始"""
        self.logger.start_run(run_id, config)
        self.tracker.reset()
    
    def log_metrics(self, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        """メトリクスを記録"""
        self.logger.log_metrics(metrics, metadata)
        
        for metric_name, value in metrics.items():
            self.tracker.track(metric_name, value)
    
    def end_experiment(self, save_results: bool = True) -> Dict[str, Any]:
        """
        実験を終了
        
        Args:
            save_results: 結果を保存するか
            
        Returns:
            メトリクス要約
        """
        self.logger.end_run()
        
        summary = self.tracker.get_summary()
        
        if save_results and self.logger._current_run_id:
            self.reporter.save_results(
                self.logger._current_run_id,
                summary
            )
        
        return summary
