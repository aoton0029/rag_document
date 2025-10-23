"""
適応的チャンカー
コンテンツに応じて動的にチャンキング戦略を調整するAdvanced RAG手法
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_chunker import BaseChunker, Chunk, ChunkType, ChunkingStrategy
from .traditional_chunkers import FixedSizeChunker, SentenceBasedChunker
from .semantic_chunker import SemanticChunker
from ..utils import get_logger, performance_monitor, TokenizerManager


class AdaptiveChunker(BaseChunker):
    """適応的チャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "adaptive"
        super().__init__(config)
        
        # 適応パラメータ
        self.min_chunk_size = config.get("min_chunk_size", 200)
        self.max_chunk_size = config.get("max_chunk_size", 1500)
        self.target_chunk_size = config.get("target_chunk_size", 800)
        
        # コンテンツ分析設定
        self.content_analysis_enabled = config.get("content_analysis_enabled", True)
        self.structure_detection_enabled = config.get("structure_detection_enabled", True)
        self.semantic_analysis_enabled = config.get("semantic_analysis_enabled", True)
        
        # 戦略選択の重み
        self.strategy_weights = config.get("strategy_weights", {
            "semantic": 0.4,
            "sentence_based": 0.3,
            "fixed_size": 0.2,
            "structure_based": 0.1
        })
        
        # トークナイザー
        self.tokenizer_manager = TokenizerManager()
        self.tokenizer = self.tokenizer_manager.get_tokenizer(self.tokenizer_config)
        
        # 内部チャンカー
        self._initialize_chunkers()
    
    def _initialize_chunkers(self):
        """内部チャンカーを初期化"""
        self.chunkers = {}
        
        # 固定サイズチャンカー
        self.chunkers["fixed_size"] = FixedSizeChunker({
            "chunk_size": self.target_chunk_size,
            "chunk_overlap": self.chunk_overlap
        })
        
        # 文ベースチャンカー
        self.chunkers["sentence_based"] = SentenceBasedChunker({
            "max_chunk_size": self.max_chunk_size,
            "min_chunk_size": self.min_chunk_size
        })
        
        # セマンティックチャンカー
        if self.semantic_analysis_enabled:
            try:
                self.chunkers["semantic"] = SemanticChunker({
                    "chunk_size": self.target_chunk_size,
                    "similarity_threshold": 0.7
                })
            except Exception as e:
                self.logger.warning(f"Failed to initialize semantic chunker: {e}")
                self.semantic_analysis_enabled = False
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """適応的チャンキング処理"""
        # コンテンツ分析
        content_analysis = self._analyze_content(text)
        
        # 最適な戦略を選択
        optimal_strategy = self._select_optimal_strategy(content_analysis)
        
        # 選択された戦略でチャンキング
        chunks = self._execute_chunking_strategy(text, optimal_strategy, content_analysis)
        
        # 後処理と最適化
        optimized_chunks = self._optimize_chunks(chunks, content_analysis)
        
        return optimized_chunks
    
    def _analyze_content(self, text: str) -> Dict[str, Any]:
        """コンテンツを分析して特徴を抽出"""
        analysis = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "sentence_count": text.count('。') + text.count('？') + text.count('！'),
            "paragraph_count": text.count('\n\n') + 1,
            "avg_sentence_length": 0,
            "lexical_diversity": 0,
            "structure_indicators": {},
            "content_type": "unknown"
        }
        
        # 基本統計
        if analysis["sentence_count"] > 0:
            analysis["avg_sentence_length"] = analysis["word_count"] / analysis["sentence_count"]
        
        # 語彙多様性（ユニーク単語数/総単語数）
        words = text.split()
        if words:
            unique_words = set(words)
            analysis["lexical_diversity"] = len(unique_words) / len(words)
        
        # 構造分析
        if self.structure_detection_enabled:
            analysis["structure_indicators"] = self._analyze_structure(text)
        
        # コンテンツタイプ推定
        analysis["content_type"] = self._identify_content_type(text, analysis)
        
        # トークン密度
        try:
            token_count = self.tokenizer.count_tokens(text)
            analysis["tokens_per_char"] = token_count / len(text) if len(text) > 0 else 0
            analysis["chars_per_token"] = len(text) / token_count if token_count > 0 else 0
        except Exception:
            analysis["tokens_per_char"] = 0.25  # 日本語の平均的な値
            analysis["chars_per_token"] = 4.0
        
        return analysis
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """テキスト構造を分析"""
        structure = {
            "has_headers": False,
            "header_count": 0,
            "has_lists": False,
            "list_count": 0,
            "has_code_blocks": False,
            "has_tables": False,
            "indentation_levels": 0,
            "formatting_complexity": "low"
        }
        
        lines = text.split('\n')
        
        # ヘッダー検出
        header_patterns = [
            r'^#+\s',  # Markdown
            r'^\d+\.\s',  # 番号付き
            r'^第\d+[章節]\s',  # 日本語章節
            r'^[■□▼▽]',  # 日本語の記号
        ]
        
        for line in lines:
            line = line.strip()
            
            # ヘッダーチェック
            for pattern in header_patterns:
                if re.match(pattern, line):
                    structure["has_headers"] = True
                    structure["header_count"] += 1
                    break
            
            # リストチェック
            if re.match(r'^[-*+]\s', line) or re.match(r'^\d+\.\s', line):
                structure["has_lists"] = True
                structure["list_count"] += 1
            
            # コードブロック
            if '```' in line or line.startswith('    '):
                structure["has_code_blocks"] = True
            
            # テーブル
            if '|' in line and line.count('|') >= 2:
                structure["has_tables"] = True
        
        # インデントレベル
        indent_levels = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.add(leading_spaces)
        structure["indentation_levels"] = len(indent_levels)
        
        # 複雑さ評価
        complexity_score = (
            structure["header_count"] * 0.3 +
            structure["list_count"] * 0.2 +
            structure["indentation_levels"] * 0.1 +
            (1 if structure["has_code_blocks"] else 0) * 0.2 +
            (1 if structure["has_tables"] else 0) * 0.2
        )
        
        if complexity_score > 2.0:
            structure["formatting_complexity"] = "high"
        elif complexity_score > 1.0:
            structure["formatting_complexity"] = "medium"
        
        return structure
    
    def _identify_content_type(self, text: str, analysis: Dict[str, Any]) -> str:
        """コンテンツタイプを推定"""
        # 学術論文の特徴
        academic_indicators = [
            "abstract", "introduction", "methodology", "results", "conclusion",
            "参考文献", "要約", "序論", "手法", "結果", "結論"
        ]
        
        # マニュアルの特徴
        manual_indicators = [
            "手順", "操作", "設定", "インストール", "使用方法", "注意",
            "step", "procedure", "installation", "setup", "usage"
        ]
        
        text_lower = text.lower()
        
        academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
        manual_score = sum(1 for indicator in manual_indicators if indicator in text_lower)
        
        # 構造的特徴も考慮
        structure = analysis.get("structure_indicators", {})
        
        if academic_score > manual_score and academic_score > 2:
            return "academic_paper"
        elif manual_score > academic_score and manual_score > 2:
            return "manual"
        elif structure.get("has_code_blocks", False):
            return "technical_documentation"
        elif structure.get("formatting_complexity") == "high":
            return "structured_document"
        else:
            return "general_text"
    
    def _select_optimal_strategy(self, analysis: Dict[str, Any]) -> str:
        """分析結果に基づいて最適な戦略を選択"""
        scores = {}
        
        # 各戦略のスコア計算
        
        # セマンティック戦略
        if self.semantic_analysis_enabled:
            semantic_score = 0
            if analysis["lexical_diversity"] > 0.3:  # 語彙が豊富
                semantic_score += 0.3
            if analysis["content_type"] in ["academic_paper", "technical_documentation"]:
                semantic_score += 0.4
            if analysis["avg_sentence_length"] > 20:  # 長い文
                semantic_score += 0.2
            if analysis["text_length"] > 2000:  # 長いテキスト
                semantic_score += 0.1
            
            scores["semantic"] = semantic_score
        
        # 文ベース戦略
        sentence_score = 0
        if analysis["sentence_count"] > 10:  # 適度な文数
            sentence_score += 0.3
        if analysis["structure_indicators"].get("formatting_complexity") == "low":
            sentence_score += 0.2
        if analysis["content_type"] == "general_text":
            sentence_score += 0.3
        if 10 < analysis["avg_sentence_length"] < 30:  # 適度な文長
            sentence_score += 0.2
        
        scores["sentence_based"] = sentence_score
        
        # 固定サイズ戦略
        fixed_score = 0
        if analysis["structure_indicators"].get("formatting_complexity") == "high":
            fixed_score += 0.4  # 構造が複雑な場合は固定サイズが安定
        if analysis["lexical_diversity"] < 0.2:  # 語彙が単調
            fixed_score += 0.2
        if analysis["content_type"] in ["manual", "structured_document"]:
            fixed_score += 0.2
        
        scores["fixed_size"] = fixed_score
        
        # 構造ベース戦略
        structure_score = 0
        if analysis["structure_indicators"].get("has_headers"):
            structure_score += 0.4
        if analysis["structure_indicators"].get("has_lists"):
            structure_score += 0.2
        if analysis["structure_indicators"].get("indentation_levels", 0) > 2:
            structure_score += 0.2
        
        scores["structure_based"] = structure_score
        
        # 重み付きスコアで最終決定
        weighted_scores = {}
        for strategy, score in scores.items():
            weight = self.strategy_weights.get(strategy, 0)
            weighted_scores[strategy] = score * weight
        
        # 最高スコアの戦略を選択
        if weighted_scores:
            optimal_strategy = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
            
            # 最小閾値チェック
            if weighted_scores[optimal_strategy] < 0.1:
                optimal_strategy = "sentence_based"  # デフォルト
        else:
            optimal_strategy = "sentence_based"
        
        self.logger.debug("Strategy selection", 
                         scores=scores, 
                         weighted_scores=weighted_scores,
                         selected=optimal_strategy,
                         content_type=analysis["content_type"])
        
        return optimal_strategy
    
    def _execute_chunking_strategy(self, text: str, strategy: str, analysis: Dict[str, Any]) -> List[Chunk]:
        """選択された戦略でチャンキングを実行"""
        
        if strategy == "structure_based":
            return self._structure_based_chunking(text, analysis)
        elif strategy in self.chunkers:
            chunker = self.chunkers[strategy]
            return chunker._chunk_text_internal(text)
        else:
            # フォールバック
            self.logger.warning(f"Unknown strategy {strategy}, falling back to sentence_based")
            return self.chunkers["sentence_based"]._chunk_text_internal(text)
    
    def _structure_based_chunking(self, text: str, analysis: Dict[str, Any]) -> List[Chunk]:
        """構造ベースチャンキング"""
        structure = analysis["structure_indicators"]
        
        if structure.get("has_headers"):
            return self._header_based_chunking(text)
        elif structure.get("has_lists"):
            return self._list_aware_chunking(text)
        else:
            # パラグラフベース
            return self._paragraph_based_chunking(text)
    
    def _header_based_chunking(self, text: str) -> List[Chunk]:
        """ヘッダーベースチャンキング"""
        lines = text.split('\n')
        chunks = []
        current_section = []
        current_start = 0
        chunk_id = 0
        
        header_patterns = [
            r'^#+\s',
            r'^\d+\.\s',
            r'^第\d+[章節]\s',
            r'^[■□▼▽]'
        ]
        
        position = 0
        
        for line in lines:
            line_with_newline = line + '\n'
            
            # ヘッダーかチェック
            is_header = any(re.match(pattern, line.strip()) for pattern in header_patterns)
            
            if is_header and current_section:
                # 前のセクションを完了
                section_text = ''.join(current_section)
                if section_text.strip():
                    chunks.append(Chunk(
                        text=section_text,
                        start_idx=current_start,
                        end_idx=position,
                        chunk_type=ChunkType.SECTION,
                        metadata={
                            "chunk_id": chunk_id,
                            "chunking_method": "header_based"
                        }
                    ))
                    chunk_id += 1
                
                # 新しいセクション開始
                current_section = [line_with_newline]
                current_start = position
            else:
                current_section.append(line_with_newline)
            
            position += len(line_with_newline)
        
        # 最後のセクション
        if current_section:
            section_text = ''.join(current_section)
            if section_text.strip():
                chunks.append(Chunk(
                    text=section_text,
                    start_idx=current_start,
                    end_idx=position,
                    chunk_type=ChunkType.SECTION,
                    metadata={
                        "chunk_id": chunk_id,
                        "chunking_method": "header_based"
                    }
                ))
        
        return chunks
    
    def _list_aware_chunking(self, text: str) -> List[Chunk]:
        """リスト構造を考慮したチャンキング"""
        # リスト項目を保持しながら文ベースチャンキングを実行
        return self.chunkers["sentence_based"]._chunk_text_internal(text)
    
    def _paragraph_based_chunking(self, text: str) -> List[Chunk]:
        """段落ベースチャンキング"""
        paragraphs = text.split('\n\n')
        chunks = []
        chunk_id = 0
        position = 0
        
        current_chunk_text = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph_with_newlines = paragraph + '\n\n'
            
            # チャンクサイズチェック
            if len(current_chunk_text + paragraph_with_newlines) <= self.target_chunk_size:
                if not current_chunk_text:
                    current_start = position
                current_chunk_text += paragraph_with_newlines
            else:
                # 現在のチャンクを完了
                if current_chunk_text.strip():
                    chunks.append(Chunk(
                        text=current_chunk_text,
                        start_idx=current_start,
                        end_idx=current_start + len(current_chunk_text),
                        chunk_type=ChunkType.PARAGRAPH,
                        metadata={
                            "chunk_id": chunk_id,
                            "chunking_method": "paragraph_based"
                        }
                    ))
                    chunk_id += 1
                
                # 新しいチャンクを開始
                current_chunk_text = paragraph_with_newlines
                current_start = position
            
            position += len(paragraph_with_newlines)
        
        # 最後のチャンク
        if current_chunk_text.strip():
            chunks.append(Chunk(
                text=current_chunk_text,
                start_idx=current_start,
                end_idx=current_start + len(current_chunk_text),
                chunk_type=ChunkType.PARAGRAPH,
                metadata={
                    "chunk_id": chunk_id,
                    "chunking_method": "paragraph_based"
                }
            ))
        
        return chunks
    
    def _optimize_chunks(self, chunks: List[Chunk], analysis: Dict[str, Any]) -> List[Chunk]:
        """チャンクを最適化"""
        if not chunks:
            return chunks
        
        optimized_chunks = []
        
        for chunk in chunks:
            # サイズチェックと調整
            if len(chunk.text) < self.min_chunk_size:
                # 小さすぎるチャンクをマージ候補にする
                optimized_chunks.append(chunk)
            elif len(chunk.text) > self.max_chunk_size:
                # 大きすぎるチャンクを分割
                sub_chunks = self._split_large_chunk(chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(chunk)
        
        # 小さなチャンクをマージ
        merged_chunks = self._merge_small_chunks(optimized_chunks)
        
        # 重複除去
        final_chunks = self._remove_duplicate_chunks(merged_chunks)
        
        return final_chunks
    
    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """大きなチャンクを分割"""
        # 文単位で分割
        sentences = re.split(r'[。！？\n]', chunk.text)
        
        sub_chunks = []
        current_text = ""
        current_start = chunk.start_idx
        sub_chunk_id = 0
        
        for sentence in sentences:
            if sentence.strip():
                sentence_with_delimiter = sentence + '。'  # 適当な区切り文字
                
                if len(current_text + sentence_with_delimiter) <= self.target_chunk_size:
                    current_text += sentence_with_delimiter
                else:
                    # 現在のサブチャンクを完了
                    if current_text.strip():
                        sub_chunk = Chunk(
                            text=current_text,
                            start_idx=current_start,
                            end_idx=current_start + len(current_text),
                            chunk_type=chunk.chunk_type,
                            metadata={
                                **chunk.metadata,
                                "sub_chunk_id": sub_chunk_id,
                                "is_split_chunk": True
                            }
                        )
                        sub_chunks.append(sub_chunk)
                        sub_chunk_id += 1
                    
                    # 新しいサブチャンク開始
                    current_start += len(current_text)
                    current_text = sentence_with_delimiter
        
        # 最後のサブチャンク
        if current_text.strip():
            sub_chunk = Chunk(
                text=current_text,
                start_idx=current_start,
                end_idx=current_start + len(current_text),
                chunk_type=chunk.chunk_type,
                metadata={
                    **chunk.metadata,
                    "sub_chunk_id": sub_chunk_id,
                    "is_split_chunk": True
                }
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """小さなチャンクをマージ"""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            if len(current_chunk.text) < self.min_chunk_size and i < len(chunks) - 1:
                # 次のチャンクとマージを試す
                next_chunk = chunks[i + 1]
                
                if len(current_chunk.text + next_chunk.text) <= self.max_chunk_size:
                    # マージ実行
                    merged_text = current_chunk.text + next_chunk.text
                    merged_chunk = Chunk(
                        text=merged_text,
                        start_idx=current_chunk.start_idx,
                        end_idx=next_chunk.end_idx,
                        chunk_type=current_chunk.chunk_type,
                        metadata={
                            **current_chunk.metadata,
                            "merged_with": next_chunk.metadata.get("chunk_id"),
                            "is_merged_chunk": True
                        }
                    )
                    merged.append(merged_chunk)
                    i += 2  # 2つのチャンクをスキップ
                    continue
            
            merged.append(current_chunk)
            i += 1
        
        return merged
    
    def _remove_duplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """重複チャンクを除去"""
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            # テキストの正規化
            normalized_text = re.sub(r'\s+', ' ', chunk.text.strip())
            
            if normalized_text not in seen_texts and len(normalized_text) > 10:
                seen_texts.add(normalized_text)
                unique_chunks.append(chunk)
        
        return unique_chunks