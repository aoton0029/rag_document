"""
チャンキング戦略のファクトリークラス
"""
from typing import Dict, Any
from . import BaseChunker
from .fixed_size_chunker import FixedSizeChunker
from .semantic_chunker import SemanticChunker
from .hierarchical_chunker import HierarchicalChunker

class ChunkerFactory:
    """チャンカー作成のファクトリークラス"""
    
    @staticmethod
    def create_chunker(strategy: str, config: Dict[str, Any], 
                      domain_config: Dict[str, Any] = None) -> BaseChunker:
        """指定された戦略でチャンカーを作成"""
        
        if strategy == "fixed_size":
            return FixedSizeChunker(config)
        
        elif strategy == "semantic":
            return SemanticChunker(config)
        
        elif strategy == "hierarchical":
            if domain_config is None:
                raise ValueError("Hierarchical chunker requires domain_config")
            return HierarchicalChunker(config, domain_config)
        
        elif strategy == "document_structure":
            # 文書構造チャンカーは階層チャンカーの特殊版
            if domain_config is None:
                raise ValueError("Document structure chunker requires domain_config")
            return HierarchicalChunker(config, domain_config)
        
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

def evaluate_chunking_quality(chunks, metrics):
    """チャンキング品質を評価"""
    results = {}
    
    if "coherence" in metrics:
        results["coherence"] = _calculate_coherence(chunks)
    
    if "coverage" in metrics:
        results["coverage"] = _calculate_coverage(chunks)
        
    if "overlap_ratio" in metrics:
        results["overlap_ratio"] = _calculate_overlap_ratio(chunks)
        
    if "chunk_size_distribution" in metrics:
        results["chunk_size_distribution"] = _calculate_size_distribution(chunks)
        
    return results

def _calculate_coherence(chunks):
    """チャンクの一貫性を計算"""
    # 簡単な実装：文の平均長さの分散を使用
    lengths = [len(chunk.content.split()) for chunk in chunks]
    if len(lengths) <= 1:
        return 1.0
    
    import statistics
    mean_length = statistics.mean(lengths)
    variance = statistics.variance(lengths)
    
    # 正規化された一貫性スコア
    coherence = 1.0 / (1.0 + variance / (mean_length + 1))
    return coherence

def _calculate_coverage(chunks):
    """チャンクのカバレッジを計算"""
    # 簡単な実装：空でないチャンクの割合
    non_empty_chunks = sum(1 for chunk in chunks if chunk.content.strip())
    total_chunks = len(chunks)
    return non_empty_chunks / total_chunks if total_chunks > 0 else 0.0

def _calculate_overlap_ratio(chunks):
    """チャンク間のオーバーラップ率を計算"""
    if len(chunks) <= 1:
        return 0.0
    
    overlaps = []
    for i in range(len(chunks) - 1):
        current_words = set(chunks[i].content.split())
        next_words = set(chunks[i + 1].content.split())
        
        if not current_words or not next_words:
            overlaps.append(0.0)
            continue
            
        intersection = current_words.intersection(next_words)
        union = current_words.union(next_words)
        overlap = len(intersection) / len(union)
        overlaps.append(overlap)
    
    return sum(overlaps) / len(overlaps)

def _calculate_size_distribution(chunks):
    """チャンクサイズの分布を計算"""
    sizes = [len(chunk.content) for chunk in chunks]
    
    if not sizes:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    import statistics
    return {
        "mean": statistics.mean(sizes),
        "std": statistics.stdev(sizes) if len(sizes) > 1 else 0,
        "min": min(sizes),
        "max": max(sizes),
        "count": len(sizes)
    }