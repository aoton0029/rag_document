"""
監視・ログ出力モジュール
"""
import logging
import time
from typing import Dict, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class ExperimentLog:
    """実験ログのデータクラス"""
    experiment_id: str
    pattern_name: str
    chunking_strategy: str
    embedding_model: str
    llm_model: str
    retrieval_method: str
    start_time: float
    end_time: float = 0.0
    status: str = "running"
    metrics: Dict[str, float] = None
    error_message: str = ""

class ExperimentLogger:
    """実験ログ管理クラス"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.experiments = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """ログ設定をセットアップ"""
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.log_dir}/rag_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_experiment(self, experiment_id: str, pattern_config: Dict[str, Any]) -> ExperimentLog:
        """実験開始をログ"""
        log_entry = ExperimentLog(
            experiment_id=experiment_id,
            pattern_name=pattern_config.get('name', 'Unknown'),
            chunking_strategy=pattern_config.get('chunking', 'Unknown'),
            embedding_model=pattern_config.get('embedding', 'Unknown'),
            llm_model=pattern_config.get('llm', 'Unknown'),
            retrieval_method=pattern_config.get('retrieval', 'Unknown'),
            start_time=time.time()
        )
        
        self.experiments[experiment_id] = log_entry
        self.logger.info(f"Started experiment {experiment_id}: {log_entry.pattern_name}")
        return log_entry
    
    def end_experiment(self, experiment_id: str, metrics: Dict[str, float], status: str = "completed"):
        """実験終了をログ"""
        if experiment_id in self.experiments:
            log_entry = self.experiments[experiment_id]
            log_entry.end_time = time.time()
            log_entry.metrics = metrics
            log_entry.status = status
            
            duration = log_entry.end_time - log_entry.start_time
            self.logger.info(f"Completed experiment {experiment_id} in {duration:.2f}s")
            
            # 結果をファイルに保存
            self._save_experiment_log(log_entry)
        
    def _save_experiment_log(self, log_entry: ExperimentLog):
        """実験ログをファイルに保存"""
        log_file = f"{self.log_dir}/experiment_{log_entry.experiment_id}.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as file:
                json.dump(asdict(log_entry), file, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save experiment log: {e}")

def log_performance(func):
    """パフォーマンス測定デコレータ"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__name__)
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f}s")
        
        return result
    return wrapper