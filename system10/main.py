"""
RAG評価フレームワーク メイン実行スクリプト
実験パターンを実行し、評価結果を保存する
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.llms.vllm import Vllm

from src.utils.config_manager import ConfigManager
from src.database.database_manager import DatabaseManager, DatabaseConfig
from src.data_generation.document_loader import PDFLoader, DirectoryLoader
from src.data_generation.metadata_extractor import MetadataExtractor
from src.chunking.chunking import ChunkerFactory
from src.embedding.embedding import EmbeddingFactory
from src.indexing.index_builder import IndexBuilderFactory
from src.retrieval.retriever import RetrieverFactory
from src.query.query_engine import QueryEngineFactory
from src.evaluation.evaluator import RAGEvaluator
from src.monitoring.monitoring import ExperimentLogger, MetricsCollector


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('results/logs/experiment.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    実験ランナー
    全体の実験フローを管理
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        results_dir: str = "results",
        data_dir: str = "data"
    ):
        """
        ExperimentRunnerの初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
            results_dir: 結果出力ディレクトリ
            data_dir: データディレクトリ
        """
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        
        # ディレクトリ作成
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "configs").mkdir(exist_ok=True)
        (self.results_dir / "metrics").mkdir(exist_ok=True)
        
        # 設定マネージャー
        self.config_manager = ConfigManager(config_dir=str(self.config_dir))
        
        # ロガーとメトリクス収集
        self.logger = ExperimentLogger(log_dir=str(self.results_dir / "logs"))
        self.metrics_collector = MetricsCollector(
            output_dir=str(self.results_dir / "metrics")
        )
        
        # データベースマネージャー
        self.db_manager = None
        
    def setup_llm(self, llm_config: Dict[str, Any]):
        """
        LLMをセットアップ
        
        Args:
            llm_config: LLM設定
        """
        model_name = llm_config.get("model", "qwen3:32b")
        backend = llm_config.get("backend", "ollama")
        temperature = llm_config.get("temperature", 0.0)
        max_tokens = llm_config.get("max_tokens", 1024)
        
        try:
            if backend == "ollama":
                llm = Ollama(
                    model=model_name,
                    temperature=temperature,
                    request_timeout=120.0,
                    additional_kwargs={"num_predict": max_tokens}
                )
            elif backend == "vllm":
                llm = Vllm(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM backend: {backend}")
            
            Settings.llm = llm
            logger.info(f"LLM設定完了: {backend}/{model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"LLMセットアップエラー: {e}")
            raise
    
    def setup_embedding(self, embedding_config: Dict[str, Any]):
        """
        埋め込みモデルをセットアップ
        
        Args:
            embedding_config: 埋め込み設定
        """
        backend = embedding_config.get("backend", "ollama")
        model_name = embedding_config.get("model", "qwen3-embedding:8b")
        
        try:
            embed_model = EmbeddingFactory.create(
                backend=backend,
                model_name=model_name,
                **embedding_config
            )
            Settings.embed_model = embed_model
            logger.info(f"埋め込みモデル設定完了: {backend}/{model_name}")
            return embed_model
            
        except Exception as e:
            logger.error(f"埋め込みモデルセットアップエラー: {e}")
            raise
    
    def setup_storage_context(self, storage_config: Dict[str, Any]):
        """
        StorageContextをセットアップ
        
        Args:
            storage_config: ストレージ設定
        """
        try:
            # データベース設定作成
            db_config = DatabaseConfig(
                mongodb_host=storage_config.get("mongodb", {}).get("host", "localhost"),
                mongodb_port=storage_config.get("mongodb", {}).get("port", 27017),
                mongodb_database=storage_config.get("mongodb", {}).get("database", "rag_system"),
                redis_host=storage_config.get("redis", {}).get("host", "localhost"),
                redis_port=storage_config.get("redis", {}).get("port", 6379),
                milvus_host=storage_config.get("milvus", {}).get("host", "localhost"),
                milvus_port=storage_config.get("milvus", {}).get("port", 19530),
                milvus_collection=storage_config.get("milvus", {}).get("collection", "rag_vectors"),
                neo4j_uri=storage_config.get("neo4j", {}).get("uri", "bolt://localhost:7687"),
            )
            
            # データベースマネージャー作成
            self.db_manager = DatabaseManager(db_config)
            
            # StorageContext構築
            storage_context = self.db_manager.build_storage_context(
                use_vector_store=storage_config.get("use_vector_store", True),
                use_docstore=storage_config.get("use_docstore", True),
                use_index_store=storage_config.get("use_index_store", True),
                use_graph_store=storage_config.get("use_graph_store", False)
            )
            
            logger.info("StorageContext構築完了")
            return storage_context
            
        except Exception as e:
            logger.error(f"StorageContextセットアップエラー: {e}")
            raise
    
    def load_documents(self, data_path: str) -> List:
        """
        ドキュメントを読み込む
        
        Args:
            data_path: データパス
            
        Returns:
            Document リスト
        """
        try:
            data_path = Path(data_path)
            
            if data_path.is_file():
                # 単一ファイル
                if data_path.suffix.lower() == ".pdf":
                    loader = PDFLoader()
                    documents = loader.load(str(data_path))
                else:
                    raise ValueError(f"Unsupported file type: {data_path.suffix}")
            elif data_path.is_dir():
                # ディレクトリ
                loader = DirectoryLoader(
                    input_dir=str(data_path),
                    recursive=True,
                    required_exts=[".pdf"]
                )
                documents = loader.load_data()
            else:
                raise ValueError(f"Invalid data path: {data_path}")
            
            logger.info(f"ドキュメント読み込み完了: {len(documents)}件")
            return documents
            
        except Exception as e:
            logger.error(f"ドキュメント読み込みエラー: {e}")
            raise
    
    def extract_metadata(
        self,
        documents: List,
        metadata_config: Dict[str, Any]
    ) -> List:
        """
        メタデータを抽出
        
        Args:
            documents: ドキュメントリスト
            metadata_config: メタデータ抽出設定
            
        Returns:
            メタデータ付きDocumentリスト
        """
        try:
            extractor = MetadataExtractor(
                extract_title=metadata_config.get("extract_title", True),
                extract_keywords=metadata_config.get("extract_keywords", True),
                extract_summary=metadata_config.get("extract_summary", False),
                llm=Settings.llm if metadata_config.get("use_llm", False) else None
            )
            
            # メタデータ抽出
            documents = extractor.extract(documents)
            
            logger.info("メタデータ抽出完了")
            return documents
            
        except Exception as e:
            logger.error(f"メタデータ抽出エラー: {e}")
            raise
    
    def chunk_documents(
        self,
        documents: List,
        chunking_config: Dict[str, Any]
    ) -> List:
        """
        ドキュメントをチャンキング
        
        Args:
            documents: ドキュメントリスト
            chunking_config: チャンキング設定
            
        Returns:
            ノードリスト
        """
        try:
            method = chunking_config.get("method", "token_based")
            
            chunker = ChunkerFactory.create(
                method=method,
                **chunking_config
            )
            
            nodes = chunker.chunk(documents)
            
            logger.info(f"チャンキング完了: {len(nodes)}ノード (method={method})")
            return nodes
            
        except Exception as e:
            logger.error(f"チャンキングエラー: {e}")
            raise
    
    def build_index(
        self,
        nodes: List,
        index_config: Dict[str, Any],
        storage_context
    ):
        """
        インデックスを構築
        
        Args:
            nodes: ノードリスト
            index_config: インデックス設定
            storage_context: StorageContext
            
        Returns:
            インデックス
        """
        try:
            index_type = index_config.get("type", "vector")
            
            index = IndexBuilderFactory.build(
                index_type=index_type,
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True,
                **index_config
            )
            
            logger.info(f"インデックス構築完了: type={index_type}")
            return index
            
        except Exception as e:
            logger.error(f"インデックス構築エラー: {e}")
            raise
    
    def create_query_engine(
        self,
        index,
        retrieval_config: Dict[str, Any],
        query_config: Dict[str, Any]
    ):
        """
        QueryEngineを作成
        
        Args:
            index: インデックス
            retrieval_config: 検索設定
            query_config: クエリ設定
            
        Returns:
            QueryEngine
        """
        try:
            # Retriever作成
            retriever = RetrieverFactory.create_retriever(
                index=index,
                retriever_type=retrieval_config.get("retriever_type", "vector"),
                **retrieval_config
            )
            
            # QueryEngine作成
            query_engine = QueryEngineFactory.create_query_engine(
                index=index,
                query_engine_type=query_config.get("type", "retriever"),
                retriever=retriever,
                response_mode=query_config.get("response_mode", "compact"),
                **query_config
            )
            
            logger.info("QueryEngine作成完了")
            return query_engine
            
        except Exception as e:
            logger.error(f"QueryEngine作成エラー: {e}")
            raise
    
    def run_experiment(
        self,
        pattern_name: str,
        data_path: str,
        queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        実験を実行
        
        Args:
            pattern_name: テストパターン名
            data_path: データパス
            queries: 評価クエリリスト
            
        Returns:
            実験結果
        """
        # 実験ID生成
        run_id = f"{pattern_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        logger.info(f"=== 実験開始: {run_id} ===")
        
        try:
            # 設定読み込み
            config = self.config_manager.get_experiment_pattern(pattern_name)
            
            # 実験ログ開始
            self.logger.start_run(run_id, config)
            
            # 設定スナップショット保存
            self.config_manager.save_experiment_snapshot(
                experiment_id=run_id,
                config=config,
                output_dir=str(self.results_dir)
            )
            
            # LLMセットアップ
            llm = self.setup_llm(config.get("llm", {}))
            
            # 埋め込みモデルセットアップ
            embed_model = self.setup_embedding(config.get("embedding", {}))
            
            # StorageContextセットアップ
            storage_context = self.setup_storage_context(
                config.get("storage", {})
            )
            
            # ドキュメント読み込み
            documents = self.load_documents(data_path)
            
            # メタデータ抽出
            if config.get("metadata", {}).get("enabled", True):
                documents = self.extract_metadata(
                    documents,
                    config.get("metadata", {})
                )
            
            # チャンキング
            nodes = self.chunk_documents(
                documents,
                config.get("chunking", {})
            )
            
            # インデックス構築
            index = self.build_index(
                nodes,
                config.get("indexing", {}),
                storage_context
            )
            
            # QueryEngine作成
            query_engine = self.create_query_engine(
                index,
                config.get("retrieval", {}),
                config.get("query", {})
            )
            
            # 評価
            results = {}
            if queries:
                logger.info(f"評価実行: {len(queries)}クエリ")
                
                evaluator = RAGEvaluator(
                    llm=llm,
                    embed_model=embed_model
                )
                
                # クエリ実行と評価
                eval_results = evaluator.evaluate_query_engine(
                    query_engine=query_engine,
                    queries=queries,
                    **config.get("evaluation", {})
                )
                
                results["evaluation"] = eval_results
                
                # メトリクス保存
                self.metrics_collector.save_metrics(
                    run_id=run_id,
                    metrics=eval_results
                )
            
            # 実験ログ終了
            self.logger.end_run(run_id, results)
            
            logger.info(f"=== 実験完了: {run_id} ===")
            
            return {
                "run_id": run_id,
                "pattern_name": pattern_name,
                "results": results,
                "config": config
            }
            
        except Exception as e:
            logger.error(f"実験実行エラー: {e}", exc_info=True)
            self.logger.log_error(run_id, str(e))
            raise
        
        finally:
            # クリーンアップ
            if self.db_manager:
                self.db_manager.disconnect_all()
    
    def run_all_experiments(
        self,
        data_path: str,
        queries: Optional[List[str]] = None,
        pattern_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        全ての実験パターンを実行
        
        Args:
            data_path: データパス
            queries: 評価クエリリスト
            pattern_filter: 実行するパターン名リスト（Noneの場合は全て）
            
        Returns:
            全実験結果のリスト
        """
        # テストパターン取得
        test_patterns_config = self.config_manager.load_yaml_file(
            self.config_manager.paths.test_patterns
        )
        patterns = test_patterns_config.get("test_patterns", {})
        
        results = []
        
        for pattern_name, pattern_config in patterns.items():
            # フィルタチェック
            if pattern_filter and pattern_name not in pattern_filter:
                continue
            
            # 有効チェック
            if not pattern_config.get("enabled", True):
                logger.info(f"スキップ（無効）: {pattern_name}")
                continue
            
            logger.info(f"実行: {pattern_name}")
            
            try:
                result = self.run_experiment(
                    pattern_name=pattern_name,
                    data_path=data_path,
                    queries=queries
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"パターン {pattern_name} 実行失敗: {e}")
                continue
        
        # 比較レポート生成
        self._generate_comparison_report(results)
        
        return results
    
    def _generate_comparison_report(self, results: List[Dict[str, Any]]):
        """
        比較レポートを生成
        
        Args:
            results: 実験結果リスト
        """
        try:
            report_path = self.results_dir / "comparison_report.json"
            
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"比較レポート保存: {report_path}")
            
        except Exception as e:
            logger.error(f"比較レポート生成エラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="RAG評価フレームワーク実験ランナー"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="データファイル/ディレクトリパス"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="実行するテストパターン名（指定しない場合は全パターン実行）"
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="評価クエリファイルパス（1行1クエリのテキストファイル）"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="設定ファイルディレクトリ"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="結果出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    # クエリ読み込み
    queries = None
    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    
    # 実験ランナー作成
    runner = ExperimentRunner(
        config_dir=args.config_dir,
        results_dir=args.results_dir
    )
    
    # 実験実行
    if args.pattern:
        # 単一パターン実行
        result = runner.run_experiment(
            pattern_name=args.pattern,
            data_path=args.data,
            queries=queries
        )
        print(f"\n実験完了: {result['run_id']}")
    else:
        # 全パターン実行
        results = runner.run_all_experiments(
            data_path=args.data,
            queries=queries
        )
        print(f"\n全実験完了: {len(results)}パターン")


if __name__ == "__main__":
    main()
