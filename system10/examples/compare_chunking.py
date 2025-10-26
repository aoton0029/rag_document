"""
複数のチャンキング手法を比較する例
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.data_generation.document_loader import PDFLoader
from src.chunking.chunking import ChunkerFactory
from src.indexing.index_builder import IndexBuilderFactory
from src.query.query_engine import QueryEngineFactory
from src.evaluation.evaluator import RAGEvaluator


def compare_chunking_methods():
    """複数のチャンキング手法を比較"""
    
    # Settings設定
    print("LLMと埋め込みモデルを設定...")
    llm = Ollama(model="qwen3:32b", temperature=0.0)
    embed_model = OllamaEmbedding(model_name="qwen3-embedding:8b")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # PDFを読み込む
    print("\nPDFを読み込む...")
    pdf_path = "data/sample.pdf"
    
    loader = PDFLoader()
    documents = loader.load(pdf_path)
    
    # 比較するチャンキング手法
    chunking_methods = {
        "token_based_512": {
            "method": "token_based",
            "chunk_size": 512,
            "chunk_overlap": 50
        },
        "token_based_1024": {
            "method": "token_based",
            "chunk_size": 1024,
            "chunk_overlap": 100
        },
        "sentence_splitter": {
            "method": "sentence_splitter",
            "chunk_size": 512,
            "chunk_overlap": 50
        }
    }
    
    # 評価クエリ
    queries = [
        "この文書の主題は何ですか？",
        "重要なポイントを3つ挙げてください。",
        "著者の結論を要約してください。"
    ]
    
    # 各手法で実験
    results = {}
    
    for method_name, config in chunking_methods.items():
        print(f"\n{'='*60}")
        print(f"チャンキング手法: {method_name}")
        print(f"{'='*60}")
        
        # チャンキング
        print(f"  チャンキング中...")
        chunker = ChunkerFactory.create(**config)
        nodes = chunker.chunk(documents)
        print(f"  ノード数: {len(nodes)}")
        
        # インデックス構築
        print(f"  インデックス構築中...")
        index = IndexBuilderFactory.build(
            index_type="vector",
            nodes=nodes,
            show_progress=False
        )
        
        # QueryEngine作成
        query_engine = QueryEngineFactory.create_query_engine(
            index=index,
            query_engine_type="retriever",
            similarity_top_k=5,
            response_mode="compact"
        )
        
        # 評価
        print(f"  評価中...")
        evaluator = RAGEvaluator(llm=llm, embed_model=embed_model)
        
        eval_results = evaluator.evaluate_query_engine(
            query_engine=query_engine,
            queries=queries
        )
        
        results[method_name] = {
            "num_nodes": len(nodes),
            "metrics": eval_results
        }
        
        print(f"  評価完了")
    
    # 結果表示
    print(f"\n{'='*60}")
    print("比較結果")
    print(f"{'='*60}")
    
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print(f"  ノード数: {result['num_nodes']}")
        if "metrics" in result and "average" in result["metrics"]:
            avg_metrics = result["metrics"]["average"]
            for metric_name, value in avg_metrics.items():
                print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    compare_chunking_methods()
