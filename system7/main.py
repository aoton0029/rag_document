"""
Advanced RAG System - Main Module
高度なRAGシステムのメインモジュール
"""
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio

from config.config import config
from src.storage_setup import storage_manager
from src.model_setup import model_manager
from src.rag_engine import rag_engine
from src.document_processor import document_processor

class AdvancedRAGSystem:
    """
    Advanced RAG Systemのメインクラス
    システム全体の初期化と操作を管理
    """
    
    def __init__(self):
        self.is_initialized = False
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定"""
        logger.add(
            "logs/advanced_rag_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time} | {level} | {module}:{function}:{line} | {message}"
        )
    
    async def initialize_system(self, check_connections: bool = True) -> Dict[str, Any]:
        """システムの初期化"""
        try:
            logger.info("Initializing Advanced RAG System...")
            
            # 1. Ollamaサービスの確認
            if not model_manager.check_ollama_availability():
                raise ConnectionError("Ollama service is not available")
            
            # 2. 必要なモデルの確認とダウンロード
            logger.info("Checking required models...")
            if not model_manager.pull_model_if_needed(config.ollama.llm_model):
                raise ValueError(f"Failed to load LLM model: {config.ollama.llm_model}")
            
            if not model_manager.pull_model_if_needed(config.ollama.embed_model):
                raise ValueError(f"Failed to load embedding model: {config.ollama.embed_model}")
            
            # 3. モデルの初期化
            model_manager.setup_global_settings()
            
            # 4. ストレージの初期化
            storage_context = storage_manager.setup_storage_context()
            
            # 5. 接続テスト（オプション）
            connection_status = {}
            if check_connections:
                connection_status = storage_manager.test_connections()
                logger.info(f"Storage connections: {connection_status}")
            
            # 6. RAGエンジンの初期化
            rag_engine.initialize()
            
            self.is_initialized = True
            
            initialization_result = {
                "status": "success",
                "llm_model": config.ollama.llm_model,
                "embed_model": config.ollama.embed_model,
                "storage_connections": connection_status,
                "available_engines": list(rag_engine.query_engines.keys())
            }
            
            logger.success("Advanced RAG System initialized successfully")
            return initialization_result
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            raise
    
    def load_documents(self, 
                      sources: List[str],
                      use_semantic_chunking: bool = False) -> Dict[str, Any]:
        """ドキュメントの読み込みとインデックス化"""
        try:
            if not self.is_initialized:
                raise RuntimeError("System not initialized. Call initialize_system() first.")
            
            logger.info(f"Loading documents from {len(sources)} source(s)...")
            
            all_results = []
            
            for source in sources:
                source_path = Path(source)
                
                if source_path.is_dir():
                    result = document_processor.process_and_index(
                        source_path, 
                        use_semantic_chunking=use_semantic_chunking,
                        is_directory=True
                    )
                elif source_path.is_file():
                    result = document_processor.process_and_index(
                        source_path,
                        use_semantic_chunking=use_semantic_chunking,
                        is_directory=False
                    )
                else:
                    logger.warning(f"Invalid source path: {source}")
                    continue
                
                all_results.append(result)
            
            # 結果の統合
            total_documents = sum(r['total_documents'] for r in all_results)
            total_chunks = sum(r['total_chunks'] for r in all_results)
            processed_files = []
            for r in all_results:
                processed_files.extend(r['processed_files'])
            
            summary = {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "processed_files": processed_files,
                "sources_processed": len(all_results),
                "indexing_completed": all(r['indexing_completed'] for r in all_results)
            }
            
            logger.success(f"Document loading completed: {total_documents} documents, {total_chunks} chunks")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            raise
    
    def query(self, 
             question: str, 
             engine_type: str = "hybrid",
             include_sources: bool = True) -> Dict[str, Any]:
        """質問応答の実行"""
        try:
            if not self.is_initialized:
                raise RuntimeError("System not initialized. Call initialize_system() first.")
            
            return rag_engine.query(question, engine_type, include_sources)
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            raise
    
    def start_interactive_session(self):
        """インタラクティブセッションの開始"""
        try:
            if not self.is_initialized:
                logger.error("System not initialized. Please initialize first.")
                return
            
            logger.info("Starting interactive session...")
            print("\n" + "="*60)
            print("Advanced RAG System - Interactive Session")
            print("="*60)
            print("Available engines:", list(rag_engine.query_engines.keys()))
            print("Type 'quit' or 'exit' to end session")
            print("Type 'status' to check system status")
            print("Type 'help' for more commands")
            print("-"*60)
            
            while True:
                try:
                    user_input = input("\nQ: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Ending session...")
                        break
                    
                    elif user_input.lower() == 'status':
                        status = rag_engine.get_engine_status()
                        print(f"\nSystem Status:")
                        for key, value in status.items():
                            print(f"  {key}: {value}")
                        continue
                    
                    elif user_input.lower() == 'help':
                        print("\nAvailable commands:")
                        print("  quit/exit - End session")
                        print("  status - Show system status")
                        print("  help - Show this help")
                        print("  Any other input - Ask a question")
                        continue
                    
                    elif not user_input:
                        continue
                    
                    # 質問の処理
                    logger.info(f"Processing question: {user_input[:50]}...")
                    response = self.query(user_input)
                    
                    print(f"\nA: {response['answer']}")
                    
                    if response.get('sources'):
                        print(f"\nSources ({len(response['sources'])}):")
                        for i, source in enumerate(response['sources'][:3], 1):
                            print(f"  {i}. Score: {source['score']:.3f}")
                            print(f"     Text: {source['text'][:100]}...")
                    
                except KeyboardInterrupt:
                    print("\nSession interrupted. Exiting...")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    logger.error(f"Session error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Failed to start interactive session: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータスの取得"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "rag_engine_status": rag_engine.get_engine_status(),
            "storage_connections": storage_manager.test_connections(),
            "model_status": {
                "llm_model": config.ollama.llm_model,
                "embed_model": config.ollama.embed_model,
                "ollama_available": model_manager.check_ollama_availability()
            }
        }
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            logger.info("Cleaning up resources...")
            storage_manager.cleanup()
            logger.success("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

# グローバルシステムインスタンス
advanced_rag_system = AdvancedRAGSystem()

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG System")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive session")
    parser.add_argument("--documents", "-d", nargs="+", help="Documents or directories to load")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--engine", "-e", type=str, default="hybrid", 
                       choices=["vector", "graph", "hybrid"], help="Query engine to use")
    parser.add_argument("--semantic-chunking", action="store_true", 
                       help="Use semantic chunking for documents")
    
    args = parser.parse_args()
    
    try:
        # システム初期化
        print("Initializing Advanced RAG System...")
        init_result = asyncio.run(advanced_rag_system.initialize_system())
        print(f"Initialization completed: {init_result['status']}")
        
        # ドキュメントの読み込み
        if args.documents:
            print(f"\nLoading documents from {len(args.documents)} source(s)...")
            load_result = advanced_rag_system.load_documents(
                args.documents, 
                use_semantic_chunking=args.semantic_chunking
            )
            print(f"Documents loaded: {load_result['total_documents']} documents, {load_result['total_chunks']} chunks")
        
        # クエリ処理
        if args.query:
            print(f"\nProcessing query: {args.query}")
            response = advanced_rag_system.query(args.query, args.engine)
            print(f"\nAnswer: {response['answer']}")
            
            if response.get('sources'):
                print(f"\nTop sources:")
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"{i}. {source['text'][:100]}... (Score: {source['score']:.3f})")
        
        # インタラクティブセッション
        if args.interactive:
            advanced_rag_system.start_interactive_session()
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    finally:
        advanced_rag_system.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())