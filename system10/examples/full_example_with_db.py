"""
データベースを使用した完全な例
Milvus、MongoDB、Redisを使用
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.database.database_manager import DatabaseManager, DatabaseConfig
from src.data_generation.document_loader import PDFLoader
from src.chunking.chunking import ChunkerFactory
from src.indexing.index_builder import IndexBuilderFactory
from src.query.query_engine import QueryEngineFactory


def main():
    """データベースを使用した完全な例"""
    
    # 1. Settings設定
    print("1. LLMと埋め込みモデルを設定...")
    Settings.llm = Ollama(model="qwen3:32b", temperature=0.0)
    Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:8b")
    
    # 2. データベース設定
    print("2. データベース接続を設定...")
    db_config = DatabaseConfig(
        mongodb_host="localhost",
        mongodb_port=27017,
        mongodb_database="rag_example",
        redis_host="localhost",
        redis_port=6379,
        milvus_host="localhost",
        milvus_port=19530,
        milvus_collection="example_vectors",
        milvus_dim=1024  # qwen3-embedding:8bの次元数
    )
    
    db_manager = DatabaseManager(db_config)
    
    # 3. StorageContext構築
    print("3. StorageContextを構築...")
    storage_context = db_manager.build_storage_context(
        use_vector_store=True,   # Milvus使用
        use_docstore=True,        # MongoDB使用
        use_index_store=True,     # Redis使用
        use_graph_store=False     # Neo4j不使用
    )
    
    # 4. PDFを読み込む
    print("4. PDFを読み込む...")
    pdf_path = "data/sample.pdf"
    
    loader = PDFLoader()
    documents = loader.load(pdf_path)
    print(f"   読み込んだドキュメント数: {len(documents)}")
    
    # 5. チャンキング
    print("5. チャンキング...")
    chunker = ChunkerFactory.create(
        method="token_based",
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = chunker.chunk(documents)
    print(f"   作成したノード数: {len(nodes)}")
    
    # 6. インデックス構築（StorageContextを使用）
    print("6. インデックス構築（データベースに保存）...")
    index = IndexBuilderFactory.build(
        index_type="vector",
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    print("   インデックス構築完了（データベースに永続化）")
    
    # 7. QueryEngine作成
    print("7. QueryEngine作成...")
    query_engine = QueryEngineFactory.create_query_engine(
        index=index,
        query_engine_type="retriever",
        similarity_top_k=5,
        response_mode="compact"
    )
    
    # 8. クエリ実行
    print("\n8. クエリ実行...")
    queries = [
        "この文書の主題は何ですか？",
        "重要なポイントを教えてください。"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- クエリ {i}: {query} ---")
        response = query_engine.query(query)
        print(f"回答: {response.response}")
        
        # 検索されたノードの情報
        print(f"\n検索されたチャンク:")
        for j, node in enumerate(response.source_nodes[:3], 1):
            print(f"  {j}. スコア: {node.score:.4f}")
            print(f"     テキスト: {node.text[:100]}...")
    
    # 9. クリーンアップ
    print("\n9. データベース接続をクローズ...")
    db_manager.disconnect_all()
    print("完了！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nエラー: {e}")
        print("\n注意: Milvus、MongoDB、Redisが起動していることを確認してください")
        print("Docker使用の場合: docker-compose up -d")
