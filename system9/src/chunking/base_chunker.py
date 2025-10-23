"""
チャンキング基底クラス
各種チャンキング戦略の共通インターフェース
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode, TextNode, Node
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor, SummaryExtractor, DocumentContextExtractor, QuestionsAnsweredExtractor
from llama_index.extractors.entity import EntityExtractor

from ..utils import get_logger, performance_monitor, TokenizerManager


class ChunkType(Enum):
    """チャンク種別"""
    TEXT = "text"
    TITLE = "title"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"


class ChunkingStrategy(Enum):
    """チャンキング戦略"""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    TOKEN_BASED = "token_based"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"


@dataclass
class Chunk:
    """チャンクデータクラス"""
    text: str
    start_idx: int
    end_idx: int
    chunk_type: ChunkType = ChunkType.TEXT
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # 基本統計を追加
        self.metadata.update({
            "char_count": len(self.text),
            "word_count": len(self.text.split()),
            "sentence_count": self.text.count('。') + self.text.count('？') + self.text.count('！')
        })
    
    def to_text_node(self, node_id: Optional[str] = None) -> TextNode:
        """LlamaIndexのTextNodeに変換"""
        return TextNode(
            text=self.text,
            metadata={
                **self.metadata,
                "start_idx": self.start_idx,
                "end_idx": self.end_idx,
                "chunk_type": self.chunk_type.value
            },
            id_=node_id
        )
    
    def get_token_count(self, tokenizer_config: Dict[str, Any]) -> int:
        """トークン数を取得"""
        manager = TokenizerManager()
        tokenizer = manager.get_tokenizer(tokenizer_config)
        return tokenizer.count_tokens(self.text)


@dataclass
class ChunkingResult:
    """チャンキング結果"""
    chunks: List[Chunk]
    original_text: str
    strategy: ChunkingStrategy
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # 統計情報を追加
        if self.chunks:
            chunk_lengths = [len(chunk.text) for chunk in self.chunks]
            self.metadata.update({
                "total_chunks": len(self.chunks),
                "avg_chunk_length": np.mean(chunk_lengths),
                "std_chunk_length": np.std(chunk_lengths),
                "min_chunk_length": min(chunk_lengths),
                "max_chunk_length": max(chunk_lengths),
                "total_chars": sum(chunk_lengths),
                "coverage_ratio": sum(chunk_lengths) / len(self.original_text) if self.original_text else 0
            })
    
    def to_nodes(self) -> List[TextNode]:
        """LlamaIndexのノードリストに変換"""
        return [chunk.to_text_node(f"chunk_{i}") for i, chunk in enumerate(self.chunks)]
    
    def get_overlaps(self) -> List[Tuple[int, int, int]]:
        """チャンク間の重複を計算 (chunk_idx1, chunk_idx2, overlap_length)"""
        overlaps = []
        
        for i in range(len(self.chunks)):
            for j in range(i + 1, len(self.chunks)):
                chunk1 = self.chunks[i]
                chunk2 = self.chunks[j]
                
                # 位置ベースの重複計算
                overlap_start = max(chunk1.start_idx, chunk2.start_idx)
                overlap_end = min(chunk1.end_idx, chunk2.end_idx)
                
                if overlap_start < overlap_end:
                    overlap_length = overlap_end - overlap_start
                    overlaps.append((i, j, overlap_length))
        
        return overlaps


class BaseChunker(ABC):
    """チャンキング基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = ChunkingStrategy(config.get("strategy", "fixed_size"))
        self.logger = get_logger(f"chunker_{self.__class__.__name__}")
        
        # 共通パラメータ
        self.chunk_size = config.get("chunk_size", 1024)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.min_chunk_size = config.get("min_chunk_size", 50)
        self.max_chunk_size = config.get("max_chunk_size", 4096)
        
        # トークナイザー設定
        self.tokenizer_config = config.get("tokenizer", {"type": "tiktoken", "encoding_name": "cl100k_base"})
        
        # メタデータ抽出器
        self.enable_metadata_extraction = config.get("enable_metadata_extraction", True)
        self.extractors = self._setup_extractors() if self.enable_metadata_extraction else []
    
    def _setup_extractors(self) -> List[Any]:
        """メタデータ抽出器を設定"""
        extractors = []
        
        extractor_config = self.config.get("extractors", {})
        
        if extractor_config.get("title", True):
            extractors.append(TitleExtractor(nodes=5))
        
        if extractor_config.get("keywords", True):
            extractors.append(KeywordExtractor(keywords=10))
        
        if extractor_config.get("summary", False):
            extractors.append(SummaryExtractor(summaries=["prev", "self", "next"]))
        
        if extractor_config.get("entities", False):
            extractors.append(EntityExtractor())
        
        return extractors
    
    @abstractmethod
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """内部的なチャンキング処理（実装必須）"""
        pass
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ChunkingResult:
        """テキストをチャンクに分割"""
        self.logger.info("Starting text chunking", 
                        text_length=len(text),
                        strategy=self.strategy.value)
        
        with performance_monitor(f"chunking_{self.strategy.value}"):
            # 前処理
            cleaned_text = self._preprocess_text(text)
            
            # チャンキング実行
            chunks = self._chunk_text_internal(cleaned_text)
            
            # 後処理
            chunks = self._postprocess_chunks(chunks, cleaned_text)
            
            # メタデータ抽出
            if self.extractors:
                chunks = self._extract_metadata(chunks)
        
        result = ChunkingResult(
            chunks=chunks,
            original_text=text,
            strategy=self.strategy,
            metadata=metadata or {}
        )
        
        self.logger.info("Chunking completed",
                        total_chunks=len(chunks),
                        avg_chunk_size=result.metadata.get("avg_chunk_length", 0))
        
        return result
    
    def chunk_document(self, document: Union[Document, str, Path]) -> ChunkingResult:
        """ドキュメントをチャンクに分割"""
        if isinstance(document, Path):
            with open(document, 'r', encoding='utf-8') as f:
                text = f.read()
            metadata = {"source": str(document)}
        elif isinstance(document, str):
            text = document
            metadata = {}
        else:  # Document
            text = document.text
            metadata = document.metadata or {}
        
        return self.chunk_text(text, metadata)
    
    def _preprocess_text(self, text: str) -> str:
        """テキスト前処理"""
        # 基本的なクリーニング
        import re
        
        # 連続する空白を単一スペースに
        text = re.sub(r'\s+', ' ', text)
        
        # 特殊な改行パターンを正規化
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # 先頭・末尾の空白を削除
        text = text.strip()
        
        return text
    
    def _postprocess_chunks(self, chunks: List[Chunk], original_text: str) -> List[Chunk]:
        """チャンク後処理"""
        processed_chunks = []
        
        for chunk in chunks:
            # サイズフィルタリング
            if len(chunk.text) < self.min_chunk_size:
                continue
            
            if len(chunk.text) > self.max_chunk_size:
                # 大きすぎるチャンクは分割
                sub_chunks = self._split_oversized_chunk(chunk, original_text)
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)
        
        # 重複除去（オプション）
        if self.config.get("remove_duplicates", False):
            processed_chunks = self._remove_duplicate_chunks(processed_chunks)
        
        return processed_chunks
    
    def _split_oversized_chunk(self, chunk: Chunk, original_text: str) -> List[Chunk]:
        """過大なチャンクを分割"""
        # 単純な文区切り分割
        sentences = chunk.text.split('。')
        sub_chunks = []
        current_text = ""
        current_start = chunk.start_idx
        
        for i, sentence in enumerate(sentences):
            if i < len(sentences) - 1:
                sentence += '。'
            
            if len(current_text + sentence) <= self.max_chunk_size:
                current_text += sentence
            else:
                if current_text:
                    # 現在のチャンクを完成
                    sub_chunks.append(Chunk(
                        text=current_text,
                        start_idx=current_start,
                        end_idx=current_start + len(current_text),
                        chunk_type=chunk.chunk_type,
                        metadata=chunk.metadata.copy()
                    ))
                
                # 新しいチャンクを開始
                current_start += len(current_text)
                current_text = sentence
        
        # 最後のチャンク
        if current_text:
            sub_chunks.append(Chunk(
                text=current_text,
                start_idx=current_start,
                end_idx=current_start + len(current_text),
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata.copy()
            ))
        
        return sub_chunks
    
    def _remove_duplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """重複チャンクを除去"""
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.text not in seen_texts:
                seen_texts.add(chunk.text)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _extract_metadata(self, chunks: List[Chunk]) -> List[Chunk]:
        """メタデータを抽出"""
        # TextNodeに変換してextractorを適用
        nodes = [chunk.to_text_node(f"temp_{i}") for i, chunk in enumerate(chunks)]
        
        for extractor in self.extractors:
            try:
                nodes = extractor.extract(nodes)
            except Exception as e:
                self.logger.warning(f"Metadata extraction failed with {extractor}: {e}")
        
        # チャンクにメタデータを追加
        for i, node in enumerate(nodes):
            if i < len(chunks):
                chunks[i].metadata.update(node.metadata)
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """チャンク統計を計算"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        token_counts = []
        
        # トークン数を計算（サンプリング）
        sample_size = min(10, len(chunks))
        sample_chunks = chunks[:sample_size]
        
        for chunk in sample_chunks:
            try:
                token_count = chunk.get_token_count(self.tokenizer_config)
                token_counts.append(token_count)
            except Exception:
                pass
        
        stats = {
            "total_chunks": len(chunks),
            "char_length": {
                "mean": np.mean(chunk_lengths),
                "std": np.std(chunk_lengths),
                "min": min(chunk_lengths),
                "max": max(chunk_lengths),
                "median": np.median(chunk_lengths)
            },
            "chunk_types": {}
        }
        
        if token_counts:
            stats["token_count"] = {
                "mean": np.mean(token_counts),
                "std": np.std(token_counts),
                "min": min(token_counts),
                "max": max(token_counts)
            }
        
        # チャンク種別の分布
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        stats["chunk_types"] = type_counts
        
        return stats


class ChunkEvaluator:
    """チャンク品質評価クラス"""
    
    def __init__(self, tokenizer_config: Optional[Dict[str, Any]] = None):
        self.tokenizer_config = tokenizer_config or {"type": "tiktoken", "encoding_name": "cl100k_base"}
        self.logger = get_logger("chunk_evaluator")
    
    def evaluate_chunks(self, chunking_result: ChunkingResult) -> Dict[str, Any]:
        """チャンキング結果を評価"""
        chunks = chunking_result.chunks
        original_text = chunking_result.original_text
        
        if not chunks:
            return {"error": "No chunks to evaluate"}
        
        evaluation = {
            "basic_stats": self._calculate_basic_stats(chunks),
            "coherence_score": self._calculate_coherence_score(chunks),
            "information_completeness": self._calculate_completeness(chunks, original_text),
            "overlap_analysis": self._analyze_overlaps(chunking_result),
            "size_distribution": self._analyze_size_distribution(chunks)
        }
        
        # 総合スコア計算
        evaluation["overall_score"] = self._calculate_overall_score(evaluation)
        
        return evaluation
    
    def _calculate_basic_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """基本統計を計算"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        word_counts = [chunk.metadata.get("word_count", 0) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(chunk_lengths),
            "avg_chunk_length": np.mean(chunk_lengths),
            "std_chunk_length": np.std(chunk_lengths),
            "avg_word_count": np.mean(word_counts) if word_counts else 0,
            "length_coefficient_variation": np.std(chunk_lengths) / np.mean(chunk_lengths) if np.mean(chunk_lengths) > 0 else 0
        }
    
    def _calculate_coherence_score(self, chunks: List[Chunk]) -> float:
        """チャンクの一貫性スコアを計算"""
        if len(chunks) < 2:
            return 1.0
        
        # 簡単な一貫性評価：隣接チャンク間のテキスト類似度
        coherence_scores = []
        
        for i in range(len(chunks) - 1):
            chunk1_words = set(chunks[i].text.lower().split())
            chunk2_words = set(chunks[i + 1].text.lower().split())
            
            if chunk1_words and chunk2_words:
                intersection = len(chunk1_words & chunk2_words)
                union = len(chunk1_words | chunk2_words)
                jaccard_similarity = intersection / union if union > 0 else 0
                coherence_scores.append(jaccard_similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_completeness(self, chunks: List[Chunk], original_text: str) -> float:
        """情報完全性を計算"""
        if not original_text:
            return 0.0
        
        # チャンクテキストの総文字数と元テキストの比率
        total_chunk_chars = sum(len(chunk.text) for chunk in chunks)
        completeness_ratio = total_chunk_chars / len(original_text)
        
        # 1.0を超える場合は重複があることを示す
        return min(completeness_ratio, 1.0)
    
    def _analyze_overlaps(self, chunking_result: ChunkingResult) -> Dict[str, Any]:
        """重複分析"""
        overlaps = chunking_result.get_overlaps()
        
        if not overlaps:
            return {"total_overlaps": 0, "avg_overlap_length": 0, "overlap_ratio": 0}
        
        overlap_lengths = [overlap[2] for overlap in overlaps]
        total_text_length = len(chunking_result.original_text)
        
        return {
            "total_overlaps": len(overlaps),
            "avg_overlap_length": np.mean(overlap_lengths),
            "max_overlap_length": max(overlap_lengths),
            "total_overlap_chars": sum(overlap_lengths),
            "overlap_ratio": sum(overlap_lengths) / total_text_length if total_text_length > 0 else 0
        }
    
    def _analyze_size_distribution(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """サイズ分布分析"""
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        if not chunk_lengths:
            return {}
        
        # パーセンタイル計算
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f"p{p}": np.percentile(chunk_lengths, p) for p in percentiles}
        
        return {
            **percentile_values,
            "mean": np.mean(chunk_lengths),
            "std": np.std(chunk_lengths),
            "min": min(chunk_lengths),
            "max": max(chunk_lengths),
            "range": max(chunk_lengths) - min(chunk_lengths)
        }
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """総合スコアを計算"""
        scores = []
        
        # 一貫性スコア（0-1）
        coherence = evaluation.get("coherence_score", 0)
        scores.append(coherence)
        
        # 完全性スコア（0-1）
        completeness = evaluation.get("information_completeness", 0)
        scores.append(completeness)
        
        # サイズ一様性スコア（変動係数の逆数）
        basic_stats = evaluation.get("basic_stats", {})
        cv = basic_stats.get("length_coefficient_variation", 1)
        size_uniformity = 1 / (1 + cv)  # 0-1の範囲に正規化
        scores.append(size_uniformity)
        
        # 重複ペナルティ
        overlap_analysis = evaluation.get("overlap_analysis", {})
        overlap_ratio = overlap_analysis.get("overlap_ratio", 0)
        overlap_penalty = max(0, 1 - overlap_ratio)  # 重複が多いほどペナルティ
        scores.append(overlap_penalty)
        
        return np.mean(scores) if scores else 0.0
    
    def compare_chunking_strategies(self, 
                                  results: Dict[str, ChunkingResult]) -> Dict[str, Any]:
        """複数のチャンキング戦略を比較"""
        comparisons = {}
        
        for strategy_name, result in results.items():
            evaluation = self.evaluate_chunks(result)
            comparisons[strategy_name] = {
                "overall_score": evaluation.get("overall_score", 0),
                "coherence_score": evaluation.get("coherence_score", 0),
                "completeness": evaluation.get("information_completeness", 0),
                "total_chunks": len(result.chunks),
                "avg_chunk_length": evaluation.get("basic_stats", {}).get("avg_chunk_length", 0)
            }
        
        # ベスト戦略を特定
        if comparisons:
            best_strategy = max(comparisons.keys(), 
                              key=lambda k: comparisons[k]["overall_score"])
            comparisons["best_strategy"] = best_strategy
        
        return comparisons
