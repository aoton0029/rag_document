"""
Advanced RAG System - Demo Script
システムの基本的な使用方法を示すデモスクリプト
"""
import asyncio
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import advanced_rag_system
from loguru import logger

async def run_demo():
    """デモの実行"""
    try:
        print("="*60)
        print("Advanced RAG System - Demo")
        print("="*60)
        
        # 1. システムの初期化
        print("\n1. Initializing system...")
        init_result = await advanced_rag_system.initialize_system()
        print(f"✓ Initialization: {init_result['status']}")
        print(f"✓ LLM Model: {init_result['llm_model']}")
        print(f"✓ Embedding Model: {init_result['embed_model']}")
        print(f"✓ Available Engines: {init_result['available_engines']}")
        
        # 2. サンプルドキュメントの作成
        print("\n2. Creating sample documents...")
        sample_docs_dir = Path("data/sample_docs")
        sample_docs_dir.mkdir(parents=True, exist_ok=True)
        
        # サンプルテキストファイルの作成
        sample_content = {
            "ai_introduction.txt": """
人工知能（AI）は、コンピューターシステムが人間の知能を模倣する技術です。
機械学習、深層学習、自然言語処理などの分野を含みます。

AIの主要な応用分野：
- 画像認識と画像生成
- 自然言語処理と文章生成
- 音声認識と音声合成
- 推薦システム
- 自動運転技術
- 医療診断支援

現在、大規模言語モデル（LLM）が注目を集めており、
ChatGPTやGPT-4などがその代表例です。
""",
            
            "machine_learning.txt": """
機械学習は、データからパターンを学習して予測を行う技術です。

主要な機械学習の種類：

1. 教師あり学習（Supervised Learning）
   - 分類問題
   - 回帰問題
   
2. 教師なし学習（Unsupervised Learning）
   - クラスタリング
   - 次元削減
   
3. 強化学習（Reinforcement Learning）
   - エージェントが環境との相互作用を通じて学習

深層学習（Deep Learning）は機械学習の一部で、
ニューラルネットワークを使用してより複雑なパターンを学習できます。
""",
            
            "rag_systems.txt": """
RAG（Retrieval-Augmented Generation）は、情報検索と生成を組み合わせた技術です。

RAGシステムの構成要素：

1. ドキュメント処理
   - テキストの分割（チャンク化）
   - ベクトル化（埋め込み）
   
2. ベクトルデータベース
   - Milvus、Pinecone、Chroma
   - 高速な類似性検索
   
3. 検索システム
   - セマンティック検索
   - キーワード検索
   
4. 言語モデル
   - 検索結果を基にした回答生成
   - コンテキストを考慮した応答

RAGの利点は、最新情報の利用と事実性の向上です。
"""
        }
        
        for filename, content in sample_content.items():
            file_path = sample_docs_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"✓ Created {len(sample_content)} sample documents in {sample_docs_dir}")
        
        # 3. ドキュメントの読み込み
        print("\n3. Loading documents...")
        load_result = advanced_rag_system.load_documents([str(sample_docs_dir)])
        print(f"✓ Loaded {load_result['total_documents']} documents")
        print(f"✓ Created {load_result['total_chunks']} chunks")
        print(f"✓ Files: {', '.join(load_result['processed_files'])}")
        
        # 4. サンプルクエリのテスト
        print("\n4. Testing sample queries...")
        
        sample_queries = [
            "AIとは何ですか？",
            "機械学習の種類を教えてください",
            "RAGシステムの構成要素は何ですか？",
            "深層学習について説明してください"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # ベクトル検索エンジンでテスト
            if 'vector' in init_result['available_engines']:
                response = advanced_rag_system.query(query, engine_type='vector')
                print(f"Vector Engine Answer: {response['answer'][:200]}...")
            
            # ハイブリッドエンジンでテスト（利用可能な場合）
            if 'hybrid' in init_result['available_engines']:
                response = advanced_rag_system.query(query, engine_type='hybrid')
                print(f"Hybrid Engine Answer: {response['answer'][:200]}...")
                
                # ソース情報の表示
                if response.get('sources'):
                    print(f"Sources ({len(response['sources'])}):")
                    for j, source in enumerate(response['sources'][:2], 1):
                        print(f"  {j}. Score: {source['score']:.3f} - {source['text'][:80]}...")
        
        # 5. システムステータスの表示
        print("\n5. System status:")
        status = advanced_rag_system.get_system_status()
        print(f"✓ Status: {status['status']}")
        
        if 'rag_engine_status' in status:
            rag_status = status['rag_engine_status']
            print(f"✓ Available engines: {rag_status['available_engines']}")
            print(f"✓ Vector index loaded: {rag_status['vector_index_loaded']}")
            print(f"✓ Graph index loaded: {rag_status['graph_index_loaded']}")
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
        # インタラクティブセッションの提案
        user_input = input("\nWould you like to start an interactive session? (y/n): ")
        if user_input.lower().startswith('y'):
            advanced_rag_system.start_interactive_session()
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"Error during demo: {str(e)}")
        return 1
    
    finally:
        advanced_rag_system.cleanup()
    
    return 0

async def quick_test():
    """クイックテスト関数"""
    try:
        print("Running quick connectivity test...")
        
        # 基本的な初期化のみ
        init_result = await advanced_rag_system.initialize_system(check_connections=True)
        print(f"✓ System initialized: {init_result['status']}")
        
        # 接続テスト結果の表示
        connections = init_result.get('storage_connections', {})
        print("Storage connections:")
        for store, status in connections.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {store.capitalize()}: {'OK' if status else 'Failed'}")
        
        return 0
        
    except Exception as e:
        print(f"Quick test failed: {str(e)}")
        return 1

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG System Demo")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick connectivity test only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        return asyncio.run(quick_test())
    else:
        return asyncio.run(run_demo())

if __name__ == "__main__":
    exit(main())