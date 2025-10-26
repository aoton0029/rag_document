"""
設定管理クラス
各種YAMLファイルの読み込み・管理・検証
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """設定ファイルパス管理"""
    chunking: str = "config/chunking_configs.yaml"
    embedding: str = "config/embedding_configs.yaml" 
    llm: str = "config/llm_configs.yaml"
    tokenizer: str = "config/tokenizer_configs.yaml"
    evaluation: str = "config/evaluation_configs.yaml"
    domain: str = "config/domain_configs.yaml"
    test_patterns: str = "config/test_patterns.yaml"


class ConfigManager:
    """
    設定管理クラス
    全ての設定ファイルの読み込み・管理・検証を行う
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        ConfigManagerの初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        # 設定ファイルパス
        self.paths = ConfigPaths()
        self._update_paths_with_config_dir()
        
        # キャッシュ
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
    def _update_paths_with_config_dir(self) -> None:
        """設定ディレクトリでパスを更新"""
        self.paths.chunking = str(self.config_dir / "chunking_configs.yaml")
        self.paths.embedding = str(self.config_dir / "embedding_configs.yaml")
        self.paths.llm = str(self.config_dir / "llm_configs.yaml")
        self.paths.tokenizer = str(self.config_dir / "tokenizer_configs.yaml")
        self.paths.evaluation = str(self.config_dir / "evaluation_configs.yaml")
        self.paths.domain = str(self.config_dir / "domain_configs.yaml")
        self.paths.test_patterns = str(self.config_dir / "test_patterns.yaml")
    
    def load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """
        YAMLファイルを読み込む
        
        Args:
            file_path: YAMLファイルパス
            
        Returns:
            読み込まれた設定辞書
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"設定ファイル読み込み成功: {file_path}")
            return config
        except FileNotFoundError:
            logger.error(f"設定ファイルが見つかりません: {file_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML解析エラー: {file_path} - {e}")
            return {}
    
    def save_yaml_file(self, config: Dict[str, Any], file_path: str) -> bool:
        """
        設定をYAMLファイルに保存
        
        Args:
            config: 保存する設定辞書
            file_path: 保存先ファイルパス
            
        Returns:
            保存成功可否
        """
        try:
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logger.info(f"設定ファイル保存成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"設定ファイル保存エラー: {file_path} - {e}")
            return False
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        全設定ファイルを読み込む
        
        Returns:
            全設定辞書
        """
        self.configs = {
            "chunking": self.load_yaml_file(self.paths.chunking),
            "embedding": self.load_yaml_file(self.paths.embedding),
            "llm": self.load_yaml_file(self.paths.llm),
            "tokenizer": self.load_yaml_file(self.paths.tokenizer),
            "evaluation": self.load_yaml_file(self.paths.evaluation),
            "domain": self.load_yaml_file(self.paths.domain),
            "test_patterns": self.load_yaml_file(self.paths.test_patterns)
        }
        return self.configs
    
    def get_config(self, config_type: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        指定された設定を取得
        
        Args:
            config_type: 設定タイプ (chunking, embedding, llm, etc.)
            use_cache: キャッシュを使用するか
            
        Returns:
            設定辞書
        """
        if use_cache and config_type in self._config_cache:
            return self._config_cache[config_type]
            
        if config_type not in self.configs:
            self.load_all_configs()
            
        config = self.configs.get(config_type, {})
        if use_cache:
            self._config_cache[config_type] = config
            
        return config
    
    def get_chunking_config(self, method: str = None) -> Dict[str, Any]:
        """チャンキング設定を取得"""
        config = self.get_config("chunking")
        
        if method:
            methods = config.get("chunking_methods", {})
            return methods.get(method, {})
        
        default_method = config.get("default_chunking", {}).get("method", "token_based")
        methods = config.get("chunking_methods", {})
        return methods.get(default_method, {})
    
    def get_embedding_config(self, model: str = None) -> Dict[str, Any]:
        """埋め込み設定を取得"""
        config = self.get_config("embedding")
        
        if model:
            models = config.get("embedding_models", {})
            return models.get(model, {})
        
        default_model = config.get("default_embedding", {}).get("model", "ollama_qwen3")
        models = config.get("embedding_models", {})
        return models.get(default_model, {})
    
    def get_llm_config(self, model: str = None) -> Dict[str, Any]:
        """LLM設定を取得"""
        config = self.get_config("llm")
        
        if model:
            models = config.get("llm_models", {})
            return models.get(model, {})
        
        default_model = config.get("default_llm", {}).get("model", "ollama_qwen3_32b")
        models = config.get("llm_models", {})
        return models.get(default_model, {})
    
    def get_experiment_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """実験パターンを取得"""
        test_patterns = self.get_config("test_patterns")
        
        # 基本パターンをチェック
        basic_patterns = test_patterns.get("basic_patterns", {})
        if pattern_name in basic_patterns:
            return basic_patterns[pattern_name]
        
        # AdvancedRAGパターンをチェック
        advanced_patterns = test_patterns.get("advanced_rag_patterns", {})
        if pattern_name in advanced_patterns:
            return advanced_patterns[pattern_name]
        
        logger.warning(f"実験パターンが見つかりません: {pattern_name}")
        return {}
    
    def create_experiment_config(
        self, 
        experiment_pattern: str = "standard_pattern",
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        実験設定を作成
        
        Args:
            experiment_pattern: 実験パターン名
            overrides: 上書き設定
            
        Returns:
            完整な実験設定
        """
        # ベースパターン取得
        base_config = self.get_experiment_pattern(experiment_pattern)
        config_dict = base_config.get("config", {})
        
        # 各コンポーネントの詳細設定を取得・マージ
        final_config = {}
        
        # チャンキング設定
        if "chunking" in config_dict:
            chunking_method = config_dict["chunking"].get("method")
            chunking_config = self.get_chunking_config(chunking_method)
            final_config["chunking"] = {**chunking_config, **config_dict["chunking"]}
        
        # 埋め込み設定
        if "embedding" in config_dict:
            embedding_model = config_dict["embedding"].get("model")
            embedding_config = self.get_embedding_config(embedding_model)
            final_config["embedding"] = {**embedding_config, **config_dict["embedding"]}
        
        # LLM設定
        if "llm" in config_dict:
            llm_model = config_dict["llm"].get("model")
            llm_config = self.get_llm_config(llm_model)
            final_config["llm"] = {**llm_config, **config_dict["llm"]}
        
        # その他の設定
        for key in ["retrieval", "evaluation", "tokenizer"]:
            if key in config_dict:
                final_config[key] = config_dict[key]
        
        # 上書き設定を適用
        if overrides:
            final_config = self._deep_merge_configs(final_config, overrides)
        
        return final_config
    
    def _deep_merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """設定を深くマージ"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        設定を検証
        
        Args:
            config: 検証する設定
            
        Returns:
            エラーメッセージリスト
        """
        errors = []
        
        # 必須フィールドチェック
        required_fields = ["chunking", "embedding", "llm"]
        for field in required_fields:
            if field not in config:
                errors.append(f"必須フィールドが不足: {field}")
        
        # チャンキング設定検証
        if "chunking" in config:
            chunking = config["chunking"]
            if "chunk_size" in chunking:
                if not isinstance(chunking["chunk_size"], int) or chunking["chunk_size"] <= 0:
                    errors.append("chunk_sizeは正の整数である必要があります")
        
        # 埋め込み設定検証
        if "embedding" in config:
            embedding = config["embedding"]
            if "dimensions" in embedding:
                if not isinstance(embedding["dimensions"], int) or embedding["dimensions"] <= 0:
                    errors.append("dimensionsは正の整数である必要があります")
        
        return errors
    
    def save_experiment_snapshot(
        self, 
        experiment_id: str, 
        config: Dict[str, Any],
        output_dir: str = "results"
    ) -> str:
        """
        実験設定のスナップショットを保存
        
        Args:
            experiment_id: 実験ID
            config: 実験設定
            output_dir: 出力ディレクトリ
            
        Returns:
            保存されたファイルパス
        """
        snapshot_path = os.path.join(output_dir, "configs", f"{experiment_id}_config.yaml")
        
        # メタデータを追加
        snapshot = {
            "experiment_id": experiment_id,
            "created_at": str(datetime.now()),
            "config_version": "1.0",
            "config": config
        }
        
        self.save_yaml_file(snapshot, snapshot_path)
        return snapshot_path
    
    def list_available_patterns(self) -> Dict[str, List[str]]:
        """利用可能なパターン一覧を取得"""
        test_patterns = self.get_config("test_patterns")
        
        return {
            "basic_patterns": list(test_patterns.get("basic_patterns", {}).keys()),
            "advanced_rag_patterns": list(test_patterns.get("advanced_rag_patterns", {}).keys()),
            "chunking_methods": list(self.get_config("chunking").get("chunking_methods", {}).keys()),
            "embedding_models": list(self.get_config("embedding").get("embedding_models", {}).keys()),
            "llm_models": list(self.get_config("llm").get("llm_models", {}).keys())
        }
    
    def clear_cache(self) -> None:
        """設定キャッシュをクリア"""
        self._config_cache.clear()
        logger.info("設定キャッシュをクリアしました")


# 使用例とヘルパー関数
def load_experiment_config(
    pattern_name: str = "standard_pattern",
    config_dir: str = "config",
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    実験設定を簡単に読み込むヘルパー関数
    
    Args:
        pattern_name: 実験パターン名
        config_dir: 設定ファイルディレクトリ
        overrides: 上書き設定
        
    Returns:
        実験設定
    """
    manager = ConfigManager(config_dir)
    return manager.create_experiment_config(pattern_name, overrides)


if __name__ == "__main__":
    # 使用例
    import datetime
    
    # ConfigManagerの作成
    config_manager = ConfigManager("./config")
    
    # 全設定の読み込み
    all_configs = config_manager.load_all_configs()
    print("読み込まれた設定:", list(all_configs.keys()))
    
    # 利用可能なパターンの表示
    patterns = config_manager.list_available_patterns()
    print("利用可能なパターン:", patterns)
    
    # 実験設定の作成
    experiment_config = config_manager.create_experiment_config("standard_pattern")
    print("実験設定作成完了")
    
    # 設定検証
    errors = config_manager.validate_config(experiment_config)
    if errors:
        print("設定エラー:", errors)
    else:
        print("設定検証成功")