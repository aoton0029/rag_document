"""
階層的チャンキング実装（論文PDF対応）
"""
from typing import List, Dict, Any, Optional
from . import BaseChunker, Chunk, ChunkType, DocumentStructureDetector
import re

class HierarchicalChunker(BaseChunker):
    """論文の階層構造を考慮したチャンカー"""
    
    def __init__(self, config: Dict[str, Any], domain_config: Dict[str, Any]):
        super().__init__(config)
        self.sections_config = config.get('sections', {})
        self.domain_config = domain_config
        self.structure_detector = DocumentStructureDetector(domain_config)
        
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """階層構造を考慮してテキストを分割"""
        text = self._clean_text(text)
        chunks = []
        
        # 文書構造を検出
        sections = self.structure_detector.detect_sections(text)
        
        if not sections:
            # 構造が検出できない場合は通常のチャンキング
            return self._fallback_chunking(text, metadata)
            
        chunk_index = 0
        
        for section in sections:
            section_type = section['type']
            section_content = section['content']
            
            # セクション設定を取得
            section_config = self.sections_config.get(section_type, {
                'chunk_size': 1024,
                'overlap': 50,
                'importance': 1.0
            })
            
            # セクションをチャンクに分割
            section_chunks = self._chunk_section(
                section_content,
                section_type,
                section_config,
                chunk_index,
                metadata
            )
            
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
            
        return chunks
    
    def _chunk_section(self, content: str, section_type: str, 
                      section_config: Dict[str, Any], start_index: int,
                      metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """セクション内容をチャンクに分割"""
        chunks = []
        chunk_size = section_config.get('chunk_size', 1024)
        overlap = section_config.get('overlap', 50)
        importance = section_config.get('importance', 1.0)
        
        # セクション特有の処理
        if section_type == 'abstract':
            # 要約は通常1つのチャンクとして扱う
            chunk = Chunk(
                id=self._generate_chunk_id(start_index, metadata),
                content=content,
                metadata=self._create_section_metadata(metadata, section_type),
                chunk_type=ChunkType.ABSTRACT,
                importance_score=importance,
                section=section_type
            )
            chunks.append(chunk)
            
        elif section_type == 'title':
            # タイトルは単独チャンク
            chunk = Chunk(
                id=self._generate_chunk_id(start_index, metadata),
                content=content,
                metadata=self._create_section_metadata(metadata, section_type),
                chunk_type=ChunkType.TITLE,
                importance_score=importance,
                section=section_type
            )
            chunks.append(chunk)
            
        elif section_type == 'references':
            # 参考文献は個別エントリに分割
            ref_chunks = self._chunk_references(content, start_index, metadata, importance)
            chunks.extend(ref_chunks)
            
        else:
            # その他のセクションは設定に基づいて分割
            content_chunks = self._chunk_by_size(
                content, chunk_size, overlap, start_index, 
                section_type, importance, metadata
            )
            chunks.extend(content_chunks)
            
        return chunks
    
    def _chunk_references(self, content: str, start_index: int,
                         metadata: Optional[Dict[str, Any]], importance: float) -> List[Chunk]:
        """参考文献をエントリごとに分割"""
        chunks = []
        
        # 参考文献のパターンを検出
        ref_patterns = [
            r'\[\d+\].*?(?=\[\d+\]|$)',  # [1] ... [2] ...
            r'\d+\.\s+.*?(?=\d+\.\s+|$)',  # 1. ... 2. ...
            r'^\s*[A-Za-z].*?(?=^\s*[A-Za-z]|\Z)',  # 著者名で始まる
        ]
        
        references = []
        for pattern in ref_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                references = [ref.strip() for ref in matches if ref.strip()]
                break
        
        if not references:
            # パターンが見つからない場合は行ごとに分割
            references = [line.strip() for line in content.split('\n') if line.strip()]
        
        for i, ref in enumerate(references):
            chunk = Chunk(
                id=self._generate_chunk_id(start_index + i, metadata),
                content=ref,
                metadata=self._create_section_metadata(metadata, 'references'),
                chunk_type=ChunkType.REFERENCES,
                importance_score=importance,
                section='references'
            )
            chunks.append(chunk)
            
        return chunks
    
    def _chunk_by_size(self, content: str, chunk_size: int, overlap: int,
                       start_index: int, section_type: str, importance: float,
                       metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """サイズベースでチャンクに分割"""
        chunks = []
        
        # パラグラフで分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = start_index
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 現在のチャンクを保存
                if current_chunk:
                    chunk = Chunk(
                        id=self._generate_chunk_id(chunk_index, metadata),
                        content=current_chunk,
                        metadata=self._create_section_metadata(metadata, section_type),
                        chunk_type=self._get_chunk_type(section_type),
                        importance_score=importance,
                        section=section_type
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # オーバーラップ処理
                    if overlap > 0 and len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + '\n\n' + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph
        
        # 最後のチャンクを追加
        if current_chunk:
            chunk = Chunk(
                id=self._generate_chunk_id(chunk_index, metadata),
                content=current_chunk,
                metadata=self._create_section_metadata(metadata, section_type),
                chunk_type=self._get_chunk_type(section_type),
                importance_score=importance,
                section=section_type
            )
            chunks.append(chunk)
            
        return chunks
    
    def _create_section_metadata(self, base_metadata: Optional[Dict[str, Any]], 
                               section_type: str) -> Dict[str, Any]:
        """セクション情報を含むメタデータを作成"""
        metadata = base_metadata.copy() if base_metadata else {}
        metadata['section'] = section_type
        metadata['section_importance'] = self.sections_config.get(section_type, {}).get('importance', 1.0)
        return metadata
    
    def _get_chunk_type(self, section_type: str) -> ChunkType:
        """セクションタイプからチャンクタイプを決定"""
        section_to_chunk_type = {
            'title': ChunkType.TITLE,
            'abstract': ChunkType.ABSTRACT,
            'introduction': ChunkType.INTRODUCTION,
            'methodology': ChunkType.METHODOLOGY,
            'results': ChunkType.RESULTS,
            'conclusion': ChunkType.CONCLUSION,
            'references': ChunkType.REFERENCES,
        }
        return section_to_chunk_type.get(section_type, ChunkType.CONTENT)
    
    def _fallback_chunking(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """フォールバック用のシンプルチャンキング"""
        from .fixed_size_chunker import FixedSizeChunker
        
        fallback_config = {
            'chunk_size': 1024,
            'chunk_overlap': 50,
            'separator': '\n\n'
        }
        
        fallback_chunker = FixedSizeChunker(fallback_config)
        return fallback_chunker.chunk(text, metadata)