#!/usr/bin/env python3
"""
RAG Evaluation Framework - Main Entry Point
===========================================

チャンキングから評価までを一つの試行として、チャンキング・インデクシング手法、RAG手法の
組み合わせによる試行パターンを作り評価する。

Usage:
    python main.py --config config/test_patterns.yaml --pattern basic_patterns.chunking_comparison
    python main.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from tqdm import tqdm

# プロジェクトのsrcディレクトリをパスに追加
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# ローカルモジュールのインポート
try:
    from src.utils.config_loader import ConfigLoader
    from src.utils.logger_setup import setup_logger
    from src.evaluation.experiment_runner import ExperimentRunner
    from src.monitoring.metrics_collector import MetricsCollector
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure all required dependencies are installed.")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパースする。"""
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework - Comprehensive RAG system evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本的なチャンキング比較を実行
  python main.py --pattern basic_patterns.chunking_comparison
  
  # 特定のドメインでの最適化を実行
  python main.py --pattern domain_patterns.academic_paper_optimization
  
  # カスタム設定ファイルを使用
  python main.py --config custom_config.yaml --pattern advanced_patterns.full_pipeline_optimization
  
  # デバッグモードで実行
  python main.py --pattern basic_patterns.embedding_comparison --debug
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/test_patterns.yaml",
        help="Test pattern configuration file path (default: config/test_patterns.yaml)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Test pattern to execute (e.g., basic_patterns.chunking_comparison)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Input data directory (default: data)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running experiments"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    
    return parser.parse_args()


def load_configuration(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込む。"""
    try:
        config_loader = ConfigLoader(project_root / "config")
        return config_loader.load_all_configs()
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def validate_pattern(pattern: str, config: Dict[str, Any]) -> None:
    """指定されたパターンが設定に存在することを確認する。"""
    pattern_parts = pattern.split('.')
    if len(pattern_parts) != 2:
        raise ValueError(f"Pattern must be in format 'category.name', got: {pattern}")
    
    category, name = pattern_parts
    test_patterns = config.get("test_patterns", {})
    
    if category not in test_patterns:
        available_categories = list(test_patterns.keys())
        raise ValueError(f"Pattern category '{category}' not found. Available: {available_categories}")
    
    if name not in test_patterns[category]:
        available_patterns = list(test_patterns[category].keys())
        raise ValueError(f"Pattern '{name}' not found in category '{category}'. Available: {available_patterns}")


def main():
    """メイン実行関数。"""
    args = parse_arguments()
    
    # ログ設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_level)
    
    logger.info("RAG Evaluation Framework starting...")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Test pattern: {args.pattern}")
    
    try:
        # 設定読み込み
        config = load_configuration(args.config)
        logger.info("Configuration loaded successfully")
        
        # パターン検証
        validate_pattern(args.pattern, config)
        logger.info(f"Pattern '{args.pattern}' validated")
        
        # 出力ディレクトリ作成
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # データディレクトリ確認
        data_path = Path(args.data_dir)
        if not data_path.exists():
            logger.warning(f"Data directory {data_path} does not exist. Please add your documents there.")
            data_path.mkdir(parents=True, exist_ok=True)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - Showing execution plan:")
            # ドライランモードの実装
            pattern_parts = args.pattern.split('.')
            category, name = pattern_parts
            pattern_config = config["test_patterns"][category][name]
            
            logger.info(f"Pattern: {pattern_config.get('description', 'No description')}")
            logger.info(f"Variables: {pattern_config.get('variables', {})}")
            logger.info(f"Fixed parameters: {pattern_config.get('fixed_parameters', {})}")
            
            return
        
        # 実験実行
        metrics_collector = MetricsCollector(output_path)
        experiment_runner = ExperimentRunner(
            config=config,
            output_dir=output_path,
            data_dir=data_path,
            metrics_collector=metrics_collector,
            parallel_workers=args.parallel
        )
        
        logger.info("Starting experiment execution...")
        results = experiment_runner.run_pattern(args.pattern)
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {output_path}")
        
        # 結果サマリー表示
        if results:
            logger.info("=== Experiment Results Summary ===")
            for exp_name, result in results.items():
                logger.info(f"{exp_name}: Score = {result.get('overall_score', 'N/A')}")
        
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        if args.debug:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()