"""
セマンティックチャンキング実装
"""
from typing import List, Dict, Any, Optional
from . import BaseChunker, Chunk, ChunkType
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker(BaseChunker):
    """セマンティック類似度を考慮したチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.threshold = config.get('threshold', 0.7)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        self.max_chunk_size = config.get('max_chunk_size', 2000)
        
        # モデルをロード
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}. Error: {e}")
            self.model = None
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """セマンティック類似度でテキストを分割"""
        if self.model is None:
            # モデルが利用できない場合はフォールバック
            return self._fallback_chunking(text, metadata)
            
        text = self._clean_text(text)
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            # 文が1つ以下の場合はそのまま返す
            chunk = Chunk(
                id=self._generate_chunk_id(0, metadata),
                content=text,
                metadata=metadata or {},
                chunk_type=ChunkType.CONTENT
            )
            return [chunk]
        
        # 文の埋め込みを計算
        embeddings = self.model.encode(sentences)
        
        # セマンティック境界を検出
        boundaries = self._detect_semantic_boundaries(embeddings)
        
        # 境界に基づいてチャンクを作成
        chunks = self._create_chunks_from_boundaries(sentences, boundaries, metadata)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        import re
        
        # 日本語と英語対応の文区切りパターン
        sentence_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # 英語文末
            r'(?<=[。！？])\s*',  # 日本語文末
            r'(?<=\n)\s*(?=\S)',  # 改行
        ]
        
        sentences = [text]
        
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                split_parts = re.split(pattern, sentence)
                new_sentences.extend([part.strip() for part in split_parts if part.strip()])
            sentences = new_sentences
        
        # 短すぎる文を結合
        merged_sentences = []
        current_sentence = ""
        
        for sentence in sentences:
            if len(current_sentence) + len(sentence) < self.min_chunk_size:
                if current_sentence:
                    current_sentence += " " + sentence
                else:
                    current_sentence = sentence
            else:
                if current_sentence:
                    merged_sentences.append(current_sentence)
                current_sentence = sentence
        
        if current_sentence:
            merged_sentences.append(current_sentence)
            
        return merged_sentences
    
    def _detect_semantic_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """セマンティック境界を検出"""
        boundaries = [0]  # 最初の境界
        
        for i in range(1, len(embeddings)):
            # 前の文との類似度を計算
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            # 閾値以下の場合は境界とする
            if similarity < self.threshold:
                boundaries.append(i)
        
        boundaries.append(len(embeddings))  # 最後の境界
        return boundaries
    
    def _create_chunks_from_boundaries(self, sentences: List[str], boundaries: List[int],
                                     metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """境界情報からチャンクを作成"""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # チャンクの内容を結合
            chunk_content = " ".join(sentences[start_idx:end_idx])
            
            # サイズ制限をチェック
            if len(chunk_content) > self.max_chunk_size:
                # 大きすぎる場合は分割
                sub_chunks = self._split_large_chunk(
                    sentences[start_idx:end_idx], i, metadata
                )
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    id=self._generate_chunk_id(i, metadata),
                    content=chunk_content,
                    metadata=metadata or {},
                    chunk_type=ChunkType.CONTENT
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_large_chunk(self, sentences: List[str], base_index: int,
                          metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """大きすぎるチャンクを分割"""
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunk = Chunk(
                        id=self._generate_chunk_id(f"{base_index}_{chunk_index}", metadata),
                        content=current_chunk,
                        metadata=metadata or {},
                        chunk_type=ChunkType.CONTENT
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = sentence
        
        # 最後のチャンクを追加
        if current_chunk:
            chunk = Chunk(
                id=self._generate_chunk_id(f"{base_index}_{chunk_index}", metadata),
                content=current_chunk,
                metadata=metadata or {},
                chunk_type=ChunkType.CONTENT
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_chunking(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """フォールバック用のチャンキング"""
        from .fixed_size_chunker import FixedSizeChunker
        
        fallback_config = {
            'chunk_size': 1024,
            'chunk_overlap': 50,
            'separator': '\n\n'
        }
        
        fallback_chunker = FixedSizeChunker(fallback_config)
        return fallback_chunker.chunk(text, metadata)