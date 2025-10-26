"""
簡単な使用例
単一のPDFファイルを処理してクエリを実行
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.data_generation.document_loader import PDFLoader
from src.chunking.chunking import ChunkerFactory
from src.embedding.embedding import EmbeddingFactory
from src.indexing.index_builder import IndexBuilderFactory
from src.query.query_engine import QueryEngineFactory


def main():
    """シンプルな使用例"""
    
    # 1. Settings設定
    print("1. LLMと埋め込みモデルを設定...")
    Settings.llm = Ollama(model="qwen3:32b", temperature=0.0)
    Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:8b")
    
    # 2. PDFを読み込む
    print("2. PDFを読み込む...")
    pdf_path = "data/sample.pdf"  # 実際のPDFパスに変更してください
    
    loader = PDFLoader()
    documents = loader.load(pdf_path)
    print(f"   読み込んだドキュメント数: {len(documents)}")
    
    # 3. チャンキング
    print("3. チャンキング...")
    chunker = ChunkerFactory.create(
        method="token_based",
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = chunker.chunk(documents)
    print(f"   作成したノード数: {len(nodes)}")
    
    # 4. インデックス構築
    print("4. インデックス構築...")
    index = IndexBuilderFactory.build(
        index_type="vector",
        nodes=nodes,
        show_progress=True
    )
    print("   インデックス構築完了")
    
    # 5. QueryEngine作成
    print("5. QueryEngine作成...")
    query_engine = QueryEngineFactory.create_query_engine(
        index=index,
        query_engine_type="retriever",
        similarity_top_k=5,
        response_mode="compact"
    )
    
    # 6. クエリ実行
    print("\n6. クエリ実行...")
    queries = [
        "この文書の主題は何ですか？",
        "重要なポイントを教えてください。",
        "結論は何ですか？"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- クエリ {i}: {query} ---")
        response = query_engine.query(query)
        print(f"回答: {response.response}")
        print(f"ソース数: {len(response.source_nodes)}")


if __name__ == "__main__":
    main()
