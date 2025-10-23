"""
設定ファイル管理モジュール
YAML設定ファイルの読み込みと管理を行う
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """設定ファイルマネージャー"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """指定された設定ファイルを読み込む"""
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        self._configs[config_name] = config
        return config
    
    def get_chunking_config(self, strategy: str) -> Dict[str, Any]:
        """チャンキング設定を取得"""
        config = self.load_config("chunking_configs")
        return config["chunking_strategies"].get(strategy, {})
    
    def get_embedding_config(self, model_name: str) -> Dict[str, Any]:
        """埋め込み設定を取得"""
        config = self.load_config("embedding_configs")
        return config["embedding_models"].get(model_name, {})
    
    def get_llm_config(self, model_name: str) -> Dict[str, Any]:
        """LLM設定を取得"""
        config = self.load_config("llm_configs")
        return config["llm_models"].get(model_name, {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """評価設定を取得"""
        return self.load_config("evaluation_configs")
    
    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """ドメイン設定を取得"""
        config = self.load_config("domain_configs")
        return config["domains"].get(domain, {})
    
    def get_test_patterns(self) -> Dict[str, Any]:
        """テストパターン設定を取得"""
        return self.load_config("test_patterns")
    
    def get_tokenizer_config(self, tokenizer_name: str) -> Dict[str, Any]:
        """トークナイザー設定を取得"""
        config = self.load_config("tokenizer")
        return config["tokenizers"].get(tokenizer_name, {})
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """設定を保存"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_data, file, default_flow_style=False, allow_unicode=True)
        
        self._configs[config_name] = config_data
    
    def validate_config(self, config_name: str) -> bool:
        """設定ファイルの妥当性をチェック"""
        try:
            config = self.load_config(config_name)
            return isinstance(config, dict) and len(config) > 0
        except Exception:
            return False
    
    def list_available_configs(self) -> list:
        """利用可能な設定ファイル一覧を取得"""
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.stem)
        return config_files
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """設定をマージ"""
        merged = base_config.copy()
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged