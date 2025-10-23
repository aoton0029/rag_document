"""
従来のチャンキング手法
固定サイズ、再帰的、文ベース、トークンベースのチャンカー実装
"""

from typing import List, Dict, Any, Optional, Tuple
import re

from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter

from .base_chunker import BaseChunker, Chunk, ChunkType, ChunkingStrategy
from ..utils import TokenizerManager


class FixedSizeChunker(BaseChunker):
    """固定サイズチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "fixed_size"
        super().__init__(config)
        
        # SimpleNodeParserを使用
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """固定サイズでテキストを分割"""
        chunks = []
        
        # オーバーラップを考慮したスライディングウィンドウ
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]
            
            # 空のチャンクはスキップ
            if chunk_text.strip():
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "chunk_id": chunk_id,
                        "chunking_method": "fixed_size"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # 次の開始位置を計算（オーバーラップを考慮）
            step_size = self.chunk_size - self.chunk_overlap
            start_idx += step_size
            
            # 無限ループを防ぐ
            if step_size <= 0:
                break
        
        return chunks


class SentenceBasedChunker(BaseChunker):
    """文ベースチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "sentence_based"
        super().__init__(config)
        
        # 文の最大長と最小長
        self.max_chunk_size = config.get("max_chunk_size", 800)
        self.min_chunk_size = config.get("min_chunk_size", 100)
        self.overlap_sentences = config.get("overlap_sentences", 1)
        
        # SentenceSplitterを使用
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """文単位でテキストを分割"""
        # 日本語の文分割パターン
        sentence_pattern = r'[。！？\n]'
        sentences = re.split(sentence_pattern, text)
        
        # 区切り文字を復元
        delimiters = re.findall(sentence_pattern, text)
        
        # 文と区切り文字を結合
        full_sentences = []
        for i, sentence in enumerate(sentences[:-1]):  # 最後の空要素を除く
            if sentence.strip():
                delimiter = delimiters[i] if i < len(delimiters) else ''
                full_sentences.append(sentence.strip() + delimiter)
        
        # 最後の文（区切り文字がない場合）
        if sentences[-1].strip():
            full_sentences.append(sentences[-1].strip())
        
        # 文をまとめてチャンクを作成
        chunks = []
        chunk_id = 0
        current_position = 0
        
        i = 0
        while i < len(full_sentences):
            chunk_sentences = []
            chunk_length = 0
            start_position = current_position
            
            # 制限内で文を追加
            while i < len(full_sentences) and chunk_length + len(full_sentences[i]) <= self.max_chunk_size:
                sentence = full_sentences[i]
                chunk_sentences.append(sentence)
                chunk_length += len(sentence)
                current_position += len(sentence)
                i += 1
            
            # 最小サイズチェック
            if chunk_length >= self.min_chunk_size and chunk_sentences:
                chunk_text = ''.join(chunk_sentences)
                
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=start_position,
                    end_idx=start_position + chunk_length,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "chunk_id": chunk_id,
                        "sentence_count": len(chunk_sentences),
                        "chunking_method": "sentence_based"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # オーバーラップのために位置を調整
                if self.overlap_sentences > 0 and i < len(full_sentences):
                    overlap_start = max(0, len(chunk_sentences) - self.overlap_sentences)
                    overlap_length = sum(len(chunk_sentences[j]) for j in range(overlap_start, len(chunk_sentences)))
                    current_position -= overlap_length
                    i -= self.overlap_sentences
            else:
                # 最小サイズに満たない場合は次の文に進む
                if i < len(full_sentences):
                    current_position += len(full_sentences[i])
                    i += 1
        
        return chunks


class RecursiveChunker(BaseChunker):
    """再帰的チャンカー（階層的区切り文字使用）"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "recursive"
        super().__init__(config)
        
        # 階層的区切り文字
        self.separators = config.get("separators", ["\n\n", "\n", "。", " ", ""])
        self.keep_separator = config.get("keep_separator", True)
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """再帰的にテキストを分割"""
        chunks = self._recursive_split(text, 0, self.separators, 0)
        return chunks
    
    def _recursive_split(self, text: str, start_offset: int, separators: List[str], depth: int) -> List[Chunk]:
        """再帰的分割の実装"""
        if not separators or len(text) <= self.chunk_size:
            # ベースケース：分割できないかサイズが適切
            if text.strip():
                return [Chunk(
                    text=text,
                    start_idx=start_offset,
                    end_idx=start_offset + len(text),
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "chunking_method": "recursive",
                        "depth": depth
                    }
                )]
            return []
        
        # 現在の区切り文字で分割
        current_separator = separators[0]
        parts = text.split(current_separator)
        
        chunks = []
        current_chunk = ""
        current_start = start_offset
        
        for i, part in enumerate(parts):
            # 区切り文字を保持する場合
            if self.keep_separator and i < len(parts) - 1:
                part_with_sep = part + current_separator
            else:
                part_with_sep = part
            
            # チャンクサイズをチェック
            if len(current_chunk + part_with_sep) <= self.chunk_size:
                current_chunk += part_with_sep
            else:
                # 現在のチャンクを完成
                if current_chunk.strip():
                    if len(current_chunk) > self.chunk_size:
                        # まだ大きすぎる場合は再帰的に分割
                        sub_chunks = self._recursive_split(
                            current_chunk, 
                            current_start, 
                            separators[1:], 
                            depth + 1
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(Chunk(
                            text=current_chunk,
                            start_idx=current_start,
                            end_idx=current_start + len(current_chunk),
                            chunk_type=ChunkType.TEXT,
                            metadata={
                                "chunking_method": "recursive",
                                "depth": depth,
                                "separator": current_separator
                            }
                        ))
                
                # 新しいチャンクを開始
                current_start += len(current_chunk)
                current_chunk = part_with_sep
        
        # 最後のチャンク
        if current_chunk.strip():
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._recursive_split(
                    current_chunk, 
                    current_start, 
                    separators[1:], 
                    depth + 1
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=current_chunk,
                    start_idx=current_start,
                    end_idx=current_start + len(current_chunk),
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "chunking_method": "recursive",
                        "depth": depth,
                        "separator": current_separator
                    }
                ))
        
        return chunks


class TokenBasedChunker(BaseChunker):
    """トークンベースチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "token_based"
        super().__init__(config)
        
        # トークナイザー設定
        self.tokenizer_manager = TokenizerManager()
        self.tokenizer = self.tokenizer_manager.get_tokenizer(self.tokenizer_config)
        
        # トークン単位のチャンクサイズ
        self.token_chunk_size = config.get("token_chunk_size", 256)
        self.token_overlap = config.get("token_overlap", 25)
        
        # TokenTextSplitterを使用
        try:
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.token_chunk_size,
                chunk_overlap=self.token_overlap,
                separator=" "
            )
        except Exception as e:
            self.logger.warning(f"Failed to create TokenTextSplitter: {e}")
            self.token_splitter = None
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """トークン単位でテキストを分割"""
        if self.token_splitter:
            return self._chunk_with_llama_index(text)
        else:
            return self._chunk_with_manual_tokenization(text)
    
    def _chunk_with_llama_index(self, text: str) -> List[Chunk]:
        """LlamaIndexのTokenTextSplitterを使用"""
        try:
            splits = self.token_splitter.split_text(text)
            chunks = []
            
            current_position = 0
            for i, split in enumerate(splits):
                # テキスト内での位置を特定
                split_start = text.find(split, current_position)
                if split_start == -1:
                    split_start = current_position
                
                chunk = Chunk(
                    text=split,
                    start_idx=split_start,
                    end_idx=split_start + len(split),
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "chunk_id": i,
                        "chunking_method": "token_based_llama",
                        "token_count": self.tokenizer.count_tokens(split)
                    }
                )
                chunks.append(chunk)
                current_position = split_start + len(split)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"LlamaIndex token splitting failed: {e}")
            return self._chunk_with_manual_tokenization(text)
    
    def _chunk_with_manual_tokenization(self, text: str) -> List[Chunk]:
        """手動トークン化による分割"""
        # テキスト全体をトークン化
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        chunk_id = 0
        
        # トークンをチャンクに分割
        i = 0
        while i < len(tokens):
            # チャンクサイズ分のトークンを取得
            chunk_tokens = tokens[i:i + self.token_chunk_size]
            
            # トークンをテキストに戻す
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # 元のテキスト内での位置を特定（近似）
            # これは完全ではないが、実用的な実装
            start_char_estimate = int((i / len(tokens)) * len(text))
            end_char_estimate = int(((i + len(chunk_tokens)) / len(tokens)) * len(text))
            
            chunk = Chunk(
                text=chunk_text,
                start_idx=start_char_estimate,
                end_idx=min(end_char_estimate, len(text)),
                chunk_type=ChunkType.TEXT,
                metadata={
                    "chunk_id": chunk_id,
                    "chunking_method": "token_based_manual",
                    "token_count": len(chunk_tokens),
                    "token_start": i,
                    "token_end": i + len(chunk_tokens)
                }
            )
            chunks.append(chunk)
            chunk_id += 1
            
            # 次の開始位置（オーバーラップ考慮）
            step = self.token_chunk_size - self.token_overlap
            i += max(1, step)  # 最低1トークンは進める
        
        return chunks


class ParagraphChunker(BaseChunker):
    """段落ベースチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "paragraph"
        super().__init__(config)
        
        self.paragraph_separators = config.get("paragraph_separators", ["\n\n", "　　"])
        self.merge_short_paragraphs = config.get("merge_short_paragraphs", True)
        self.min_paragraph_length = config.get("min_paragraph_length", 50)
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """段落単位でテキストを分割"""
        # 段落を特定
        paragraphs = self._identify_paragraphs(text)
        
        # 短い段落をマージ（オプション）
        if self.merge_short_paragraphs:
            paragraphs = self._merge_short_paragraphs(paragraphs)
        
        # パラグラフをチャンクに変換
        chunks = []
        for i, (paragraph_text, start_pos, end_pos) in enumerate(paragraphs):
            if paragraph_text.strip():
                chunk = Chunk(
                    text=paragraph_text,
                    start_idx=start_pos,
                    end_idx=end_pos,
                    chunk_type=ChunkType.PARAGRAPH,
                    metadata={
                        "chunk_id": i,
                        "chunking_method": "paragraph",
                        "paragraph_index": i
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _identify_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """段落を特定"""
        paragraphs = []
        
        # 最も適切な区切り文字を選択
        best_separator = None
        for separator in self.paragraph_separators:
            if separator in text:
                best_separator = separator
                break
        
        if best_separator:
            # 区切り文字で分割
            parts = text.split(best_separator)
            current_position = 0
            
            for part in parts:
                if part.strip():
                    start_pos = current_position
                    end_pos = current_position + len(part)
                    paragraphs.append((part, start_pos, end_pos))
                
                current_position += len(part) + len(best_separator)
        else:
            # 区切り文字が見つからない場合は全体を一つの段落として扱う
            paragraphs.append((text, 0, len(text)))
        
        return paragraphs
    
    def _merge_short_paragraphs(self, paragraphs: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """短い段落をマージ"""
        if not paragraphs:
            return paragraphs
        
        merged = []
        current_text = ""
        current_start = 0
        current_end = 0
        
        for i, (paragraph_text, start_pos, end_pos) in enumerate(paragraphs):
            if not current_text:
                # 最初の段落
                current_text = paragraph_text
                current_start = start_pos
                current_end = end_pos
            elif len(current_text) < self.min_paragraph_length:
                # 短い段落なので結合
                current_text += "\n\n" + paragraph_text
                current_end = end_pos
            else:
                # 現在の段落を確定して新しい段落を開始
                merged.append((current_text, current_start, current_end))
                current_text = paragraph_text
                current_start = start_pos
                current_end = end_pos
        
        # 最後の段落
        if current_text:
            merged.append((current_text, current_start, current_end))
        
        return merged