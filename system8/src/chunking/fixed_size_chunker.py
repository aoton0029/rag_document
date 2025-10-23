"""
固定サイズチャンキング実装
"""
from typing import List, Dict, Any, Optional
from . import BaseChunker, Chunk, ChunkType

class FixedSizeChunker(BaseChunker):
    """固定サイズでテキストを分割するチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chunk_size = config.get('chunk_size', 1024)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.separator = config.get('separator', '\n\n')
        
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """固定サイズでテキストを分割"""
        text = self._clean_text(text)
        chunks = []
        
        # セパレータで事前分割
        if self.separator in text:
            paragraphs = text.split(self.separator)
        else:
            paragraphs = [text]
        
        current_chunk = ""
        chunk_index = 0
        start_idx = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 現在のチャンクに追加可能かチェック
            if len(current_chunk) + len(paragraph) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 現在のチャンクを保存
                if current_chunk:
                    chunk = Chunk(
                        id=self._generate_chunk_id(chunk_index, metadata),
                        content=current_chunk,
                        metadata=metadata or {},
                        start_idx=start_idx,
                        end_idx=start_idx + len(current_chunk),
                        chunk_type=ChunkType.CONTENT
                    )
                    chunks.append(chunk)
                    
                    # オーバーラップの処理
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + self.separator + paragraph
                        start_idx = start_idx + len(current_chunk) - len(overlap_text) - len(self.separator)
                    else:
                        current_chunk = paragraph
                        start_idx = start_idx + len(current_chunk) + len(self.separator)
                    
                    chunk_index += 1
                else:
                    current_chunk = paragraph
                    
                # パラグラフが長すぎる場合の処理
                if len(paragraph) > self.chunk_size:
                    paragraph_chunks = self._split_long_paragraph(
                        paragraph, chunk_index, start_idx, metadata
                    )
                    chunks.extend(paragraph_chunks)
                    chunk_index += len(paragraph_chunks)
                    current_chunk = ""
                    start_idx += len(paragraph)
        
        # 最後のチャンクを追加
        if current_chunk:
            chunk = Chunk(
                id=self._generate_chunk_id(chunk_index, metadata),
                content=current_chunk,
                metadata=metadata or {},
                start_idx=start_idx,
                end_idx=start_idx + len(current_chunk),
                chunk_type=ChunkType.CONTENT
            )
            chunks.append(chunk)
            
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, start_index: int, 
                             start_pos: int, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """長いパラグラフを分割"""
        chunks = []
        sentences = self._split_into_sentences(paragraph)
        
        current_chunk = ""
        chunk_index = start_index
        current_start = start_pos
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunk = Chunk(
                        id=self._generate_chunk_id(chunk_index, metadata),
                        content=current_chunk,
                        metadata=metadata or {},
                        start_idx=current_start,
                        end_idx=current_start + len(current_chunk),
                        chunk_type=ChunkType.CONTENT
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_start += len(current_chunk)
                
                current_chunk = sentence
                
        # 最後のチャンクを追加
        if current_chunk:
            chunk = Chunk(
                id=self._generate_chunk_id(chunk_index, metadata),
                content=current_chunk,
                metadata=metadata or {},
                start_idx=current_start,
                end_idx=current_start + len(current_chunk),
                chunk_type=ChunkType.CONTENT
            )
            chunks.append(chunk)
            
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """文章を文に分割"""
        import re
        # 日本語と英語の文区切りパターン
        sentence_patterns = [
            r'[.!?]+\s+',  # 英語の文末
            r'[。！？]+\s*',  # 日本語の文末
        ]
        
        sentences = []
        current_pos = 0
        
        for pattern in sentence_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                sentence = text[current_pos:match.end()].strip()
                if sentence:
                    sentences.append(sentence)
                current_pos = match.end()
        
        # 残りのテキストを追加
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sentences.append(remaining)
                
        return sentences if sentences else [text]