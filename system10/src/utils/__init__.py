"""
Utilities module
共通ユーティリティ、設定ロード、ログ設定等
"""

import logging
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import hashlib


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    ログ設定をセットアップ
    
    Args:
        log_level: ログレベル
        log_file: ログファイルパス
        log_format: ログフォーマット
        
    Returns:
        設定されたロガー
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ルートロガーを取得
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 既存のハンドラをクリア
    logger.handlers.clear()
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # ファイルハンドラ（オプション）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    設定ファイルを読み込む
    
    Args:
        config_path: 設定ファイルパス
        
    Returns:
        設定辞書
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"サポートされていないファイル形式: {config_path.suffix}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    設定をファイルに保存
    
    Args:
        config: 設定辞書
        config_path: 保存先パス
    """
    config_path = Path(config_path)
    os.makedirs(config_path.parent, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        elif config_path.suffix == '.json':
            json.dump(config, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"サポートされていないファイル形式: {config_path.suffix}")


def generate_run_id(prefix: str = "run", timestamp: bool = True) -> str:
    """
    実験ランIDを生成
    
    Args:
        prefix: プレフィックス
        timestamp: タイムスタンプを含めるか
        
    Returns:
        ランID
    """
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}"
    else:
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    ファイルのハッシュ値を計算
    
    Args:
        file_path: ファイルパス
        algorithm: ハッシュアルゴリズム
        
    Returns:
        ハッシュ値（16進数）
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    ディレクトリが存在することを確認し、なければ作成
    
    Args:
        dir_path: ディレクトリパス
        
    Returns:
        Pathオブジェクト
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    current = Path(__file__).resolve()
    # src/utils/__init__.py から2階層上がプロジェクトルート
    return current.parent.parent.parent


def format_bytes(size: int) -> str:
    """
    バイト数を人間が読みやすい形式に変換
    
    Args:
        size: バイトサイズ
        
    Returns:
        フォーマットされた文字列
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_time(seconds: float) -> str:
    """
    秒数を人間が読みやすい形式に変換
    
    Args:
        seconds: 秒数
        
    Returns:
        フォーマットされた文字列
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}時間"


class DictUtils:
    """辞書操作のユーティリティクラス"""
    
    @staticmethod
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        辞書を深くマージ
        
        Args:
            base: ベース辞書
            override: 上書き辞書
            
        Returns:
            マージされた辞書
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DictUtils.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def flatten(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        ネストされた辞書をフラット化
        
        Args:
            d: 辞書
            parent_key: 親キー
            sep: セパレータ
            
        Returns:
            フラット化された辞書
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DictUtils.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def unflatten(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
        """
        フラットな辞書をネスト化
        
        Args:
            d: フラットな辞書
            sep: セパレータ
            
        Returns:
            ネストされた辞書
        """
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result


class MetricsCollector:
    """メトリクス収集ユーティリティ"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        
    def start(self):
        """計測開始"""
        import time
        self.start_time = time.time()
        
    def stop(self) -> float:
        """計測終了"""
        import time
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed
    
    def add_metric(self, name: str, value: Any):
        """メトリクスを追加"""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """メトリクスを取得"""
        return self.metrics.copy()
    
    def clear(self):
        """メトリクスをクリア"""
        self.metrics.clear()
        self.start_time = None


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    設定内の環境変数を展開
    
    Args:
        config: 設定辞書
        
    Returns:
        環境変数が展開された設定辞書
    """
    def _expand_value(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        elif isinstance(value, dict):
            return {k: _expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_expand_value(item) for item in value]
        return value
    
    return {k: _expand_value(v) for k, v in config.items()}
