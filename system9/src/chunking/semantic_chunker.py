"""
セマンティックチャンカー
意味的類似性に基づいたAdvanced RAGチャンキング手法
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
import re

from .base_chunker import BaseChunker, Chunk, ChunkType, ChunkingStrategy
from ..utils import get_logger, performance_monitor
from ..embedding import create_embedder


class SemanticChunker(BaseChunker):
    """セマンティックチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "semantic"
        super().__init__(config)
        
        # セマンティック分析パラメータ
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.sentence_window = config.get("sentence_window", 3)  # 文の窓サイズ
        self.embedding_config = config.get("embedding", {
            "provider": "sentence_transformers",
            "model_name": "all-MiniLM-L6-v2"
        })
        
        # 埋め込みモデル初期化
        try:
            self.embedder = create_embedder(
                self.embedding_config["provider"],
                self.embedding_config
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embedder: {e}")
            self.embedder = None
        
        # クラスタリング手法
        self.clustering_method = config.get("clustering_method", "similarity_threshold")
        self.max_clusters = config.get("max_clusters", 10)
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """セマンティック類似性に基づいてチャンクに分割"""
        if not self.embedder:
            self.logger.warning("Embedder not available, falling back to sentence-based chunking")
            return self._fallback_chunking(text)
        
        # 文に分割
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < 2:
            # 文が少ない場合はそのまま返す
            return [Chunk(
                text=text,
                start_idx=0,
                end_idx=len(text),
                chunk_type=ChunkType.TEXT,
                metadata={"chunking_method": "semantic_single"}
            )]
        
        # 文の埋め込みを生成
        sentence_embeddings = self._get_sentence_embeddings(sentences)
        
        # セマンティック類似性に基づいてチャンクを作成
        if self.clustering_method == "similarity_threshold":
            chunk_boundaries = self._find_boundaries_by_similarity(sentence_embeddings)
        elif self.clustering_method == "kmeans":
            chunk_boundaries = self._find_boundaries_by_kmeans(sentence_embeddings, sentences)
        elif self.clustering_method == "hierarchical":
            chunk_boundaries = self._find_boundaries_by_hierarchical(sentence_embeddings)
        else:
            chunk_boundaries = self._find_boundaries_by_similarity(sentence_embeddings)
        
        # チャンクを構築
        chunks = self._build_chunks_from_boundaries(sentences, chunk_boundaries, text)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        # 日本語の文分割パターン
        sentence_pattern = r'[。！？\n]'
        sentences = re.split(sentence_pattern, text)
        
        # 区切り文字を復元
        delimiters = re.findall(sentence_pattern, text)
        
        full_sentences = []
        for i, sentence in enumerate(sentences[:-1]):
            if sentence.strip():
                delimiter = delimiters[i] if i < len(delimiters) else ''
                full_sentences.append(sentence.strip() + delimiter)
        
        # 最後の文
        if sentences[-1].strip():
            full_sentences.append(sentences[-1].strip())
        
        return [s for s in full_sentences if s.strip()]
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """文の埋め込みを取得"""
        try:
            embedding_result = self.embedder.encode_texts(sentences)
            return np.array(embedding_result.embeddings)
        except Exception as e:
            self.logger.error(f"Failed to get embeddings: {e}")
            # フォールバック：ランダムな埋め込み
            return np.random.rand(len(sentences), 384)
    
    def _find_boundaries_by_similarity(self, embeddings: np.ndarray) -> List[int]:
        """類似度閾値に基づいてチャンク境界を特定"""
        boundaries = [0]  # 最初は常に境界
        
        for i in range(1, len(embeddings)):
            # 窓内の文との類似度を計算
            window_start = max(0, i - self.sentence_window)
            window_embeddings = embeddings[window_start:i]
            current_embedding = embeddings[i].reshape(1, -1)
            
            # 平均類似度を計算
            similarities = cosine_similarity(current_embedding, window_embeddings)
            avg_similarity = np.mean(similarities)
            
            # 閾値以下なら境界
            if avg_similarity < self.similarity_threshold:
                boundaries.append(i)
        
        boundaries.append(len(embeddings))  # 最後も境界
        
        return boundaries
    
    def _find_boundaries_by_kmeans(self, embeddings: np.ndarray, sentences: List[str]) -> List[int]:
        """K-Meansクラスタリングでチャンク境界を特定"""
        # クラスター数を動的に決定
        n_clusters = min(self.max_clusters, max(2, len(sentences) // 5))
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # クラスター変更点を境界とする
            boundaries = [0]
            for i in range(1, len(clusters)):
                if clusters[i] != clusters[i-1]:
                    boundaries.append(i)
            boundaries.append(len(clusters))
            
            return boundaries
            
        except Exception as e:
            self.logger.error(f"K-means clustering failed: {e}")
            return self._find_boundaries_by_similarity(embeddings)
    
    def _find_boundaries_by_hierarchical(self, embeddings: np.ndarray) -> List[int]:
        """階層クラスタリングでチャンク境界を特定"""
        try:
            n_clusters = min(self.max_clusters, max(2, len(embeddings) // 4))
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            clusters = clustering.fit_predict(embeddings)
            
            # クラスター変更点を境界とする
            boundaries = [0]
            for i in range(1, len(clusters)):
                if clusters[i] != clusters[i-1]:
                    boundaries.append(i)
            boundaries.append(len(clusters))
            
            return boundaries
            
        except Exception as e:
            self.logger.error(f"Hierarchical clustering failed: {e}")
            return self._find_boundaries_by_similarity(embeddings)
    
    def _build_chunks_from_boundaries(self, sentences: List[str], 
                                    boundaries: List[int], 
                                    original_text: str) -> List[Chunk]:
        """境界から実際のチャンクを構築"""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_sentence_idx = boundaries[i]
            end_sentence_idx = boundaries[i + 1]
            
            # チャンクのテキストを構築
            chunk_sentences = sentences[start_sentence_idx:end_sentence_idx]
            chunk_text = ''.join(chunk_sentences)
            
            # 元テキスト内での位置を特定
            start_char = self._find_text_position(chunk_sentences[0], original_text, 0)
            end_char = start_char + len(chunk_text)
            
            if chunk_text.strip():
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=start_char,
                    end_idx=end_char,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "chunk_id": i,
                        "chunking_method": "semantic",
                        "sentence_count": len(chunk_sentences),
                        "sentence_start_idx": start_sentence_idx,
                        "sentence_end_idx": end_sentence_idx,
                        "clustering_method": self.clustering_method
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _find_text_position(self, sentence: str, text: str, start_search: int = 0) -> int:
        """文の元テキスト内での位置を特定"""
        # シンプルな検索
        position = text.find(sentence.strip(), start_search)
        return max(0, position)
    
    def _fallback_chunking(self, text: str) -> List[Chunk]:
        """埋め込みが利用できない場合のフォールバック"""
        sentences = self._split_into_sentences(text)
        
        # 文数ベースでチャンクを作成
        sentences_per_chunk = max(1, len(sentences) // 5)
        chunks = []
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = ''.join(chunk_sentences)
            
            if chunk_text.strip():
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=i * 100,  # 近似
                    end_idx=(i + len(chunk_sentences)) * 100,
                    chunk_type=ChunkType.TEXT,
                    metadata={
                        "chunking_method": "semantic_fallback",
                        "sentence_count": len(chunk_sentences)
                    }
                )
                chunks.append(chunk)
        
        return chunks


class HierarchicalSemanticChunker(SemanticChunker):
    """階層セマンティックチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hierarchy_levels = config.get("hierarchy_levels", [
            {"chunk_size": 2048, "similarity_threshold": 0.6},
            {"chunk_size": 1024, "similarity_threshold": 0.7},
            {"chunk_size": 512, "similarity_threshold": 0.8}
        ])
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """階層的にチャンキング"""
        # 最初のレベルでチャンキング
        current_chunks = super()._chunk_text_internal(text)
        
        # 各階層レベルで再チャンキング
        for level_idx, level_config in enumerate(self.hierarchy_levels[1:], 1):
            refined_chunks = []
            
            for chunk in current_chunks:
                if len(chunk.text) > level_config["chunk_size"]:
                    # このチャンクをさらに分割
                    old_threshold = self.similarity_threshold
                    self.similarity_threshold = level_config["similarity_threshold"]
                    
                    sub_chunks = super()._chunk_text_internal(chunk.text)
                    
                    # サブチャンクの位置を調整
                    for sub_chunk in sub_chunks:
                        sub_chunk.start_idx += chunk.start_idx
                        sub_chunk.end_idx += chunk.start_idx
                        sub_chunk.metadata["hierarchy_level"] = level_idx
                        sub_chunk.metadata["parent_chunk"] = chunk.metadata.get("chunk_id")
                    
                    refined_chunks.extend(sub_chunks)
                    self.similarity_threshold = old_threshold
                else:
                    # そのまま保持
                    chunk.metadata["hierarchy_level"] = level_idx
                    refined_chunks.append(chunk)
            
            current_chunks = refined_chunks
        
        return current_chunks


class TopicBasedChunker(SemanticChunker):
    """トピックベースチャンカー"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.topic_change_threshold = config.get("topic_change_threshold", 0.3)
        self.topic_window_size = config.get("topic_window_size", 5)
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """トピック変化に基づいてチャンキング"""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < self.topic_window_size:
            return super()._chunk_text_internal(text)
        
        # 文の埋め込みを取得
        embeddings = self._get_sentence_embeddings(sentences)
        
        # トピック変化点を検出
        topic_boundaries = self._detect_topic_changes(embeddings)
        
        # チャンクを構築
        chunks = self._build_chunks_from_boundaries(sentences, topic_boundaries, text)
        
        return chunks
    
    def _detect_topic_changes(self, embeddings: np.ndarray) -> List[int]:
        """トピック変化点を検出"""
        boundaries = [0]
        
        for i in range(self.topic_window_size, len(embeddings) - self.topic_window_size):
            # 前後の窓での平均埋め込みを計算
            before_window = embeddings[i - self.topic_window_size:i]
            after_window = embeddings[i:i + self.topic_window_size]
            
            before_mean = np.mean(before_window, axis=0).reshape(1, -1)
            after_mean = np.mean(after_window, axis=0).reshape(1, -1)
            
            # 類似度を計算
            similarity = cosine_similarity(before_mean, after_mean)[0][0]
            
            # トピック変化を検出
            if similarity < self.topic_change_threshold:
                boundaries.append(i)
        
        boundaries.append(len(embeddings))
        return boundaries


class SemanticSectionChunker(BaseChunker):
    """セマンティックセクションチャンカー（見出し考慮）"""
    
    def __init__(self, config: Dict[str, Any]):
        config["strategy"] = "semantic_section"
        super().__init__(config)
        
        self.section_patterns = config.get("section_patterns", [
            r'^#+\s',  # Markdown headers
            r'^\d+\.\s',  # 番号付きセクション
            r'^第\d+[章節]\s',  # 日本語章節
            r'^[IVXLCDM]+\.\s',  # ローマ数字
        ])
        
        self.respect_sections = config.get("respect_sections", True)
        self.max_section_size = config.get("max_section_size", 4096)
    
    def _chunk_text_internal(self, text: str) -> List[Chunk]:
        """セクション構造を考慮したセマンティックチャンキング"""
        # セクション境界を特定
        sections = self._identify_sections(text)
        
        chunks = []
        chunk_id = 0
        
        for section_text, start_pos, end_pos, section_level in sections:
            if len(section_text) <= self.max_section_size:
                # セクションサイズが適切
                chunk = Chunk(
                    text=section_text,
                    start_idx=start_pos,
                    end_idx=end_pos,
                    chunk_type=ChunkType.SECTION,
                    metadata={
                        "chunk_id": chunk_id,
                        "chunking_method": "semantic_section",
                        "section_level": section_level,
                        "is_complete_section": True
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 大きなセクションはセマンティック分割
                semantic_chunker = SemanticChunker(self.config)
                sub_chunks = semantic_chunker._chunk_text_internal(section_text)
                
                for sub_chunk in sub_chunks:
                    sub_chunk.start_idx += start_pos
                    sub_chunk.end_idx += start_pos
                    sub_chunk.metadata.update({
                        "chunk_id": chunk_id,
                        "parent_section_level": section_level,
                        "is_section_fragment": True
                    })
                    chunks.append(sub_chunk)
                    chunk_id += 1
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[Tuple[str, int, int, int]]:
        """セクション構造を特定"""
        sections = []
        lines = text.split('\n')
        
        current_section = []
        current_start = 0
        current_level = 0
        
        position = 0
        
        for line in lines:
            line_with_newline = line + '\n'
            
            # セクションヘッダーかチェック
            section_level = self._get_section_level(line)
            
            if section_level > 0:
                # 前のセクションを完了
                if current_section:
                    section_text = ''.join(current_section)
                    sections.append((
                        section_text,
                        current_start,
                        position,
                        current_level
                    ))
                
                # 新しいセクションを開始
                current_section = [line_with_newline]
                current_start = position
                current_level = section_level
            else:
                current_section.append(line_with_newline)
            
            position += len(line_with_newline)
        
        # 最後のセクション
        if current_section:
            section_text = ''.join(current_section)
            sections.append((
                section_text,
                current_start,
                position,
                current_level
            ))
        
        return sections
    
    def _get_section_level(self, line: str) -> int:
        """行のセクションレベルを取得"""
        line = line.strip()
        
        for level, pattern in enumerate(self.section_patterns, 1):
            if re.match(pattern, line):
                return level
        
        return 0  # セクションヘッダーではない