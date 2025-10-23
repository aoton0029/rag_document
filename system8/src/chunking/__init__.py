"""
チャンキングモジュールの基底クラスと共通インターフェース
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from enum import Enum

class ChunkType(Enum):
    """チャンクのタイプ"""
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    CONTENT = "content"
    HEADER = "header"
    PARAGRAPH = "paragraph"

@dataclass
class Chunk:
    """チャンクデータクラス"""
    id: str
    content: str
    metadata: Dict[str, Any]
    start_idx: int = 0
    end_idx: int = 0
    chunk_type: ChunkType = ChunkType.CONTENT
    importance_score: float = 1.0
    section: Optional[str] = None
    
class BaseChunker(ABC):
    """チャンカーの基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """テキストをチャンクに分割"""
        pass
    
    def _generate_chunk_id(self, index: int, metadata: Optional[Dict[str, Any]] = None) -> str:
        """チャンクIDを生成"""
        source = metadata.get('source', 'unknown') if metadata else 'unknown'
        return f"{source}_{index}"
    
    def _clean_text(self, text: str) -> str:
        """テキストをクリーニング"""
        # 連続する空白を単一のスペースに置換
        text = re.sub(r'\s+', ' ', text)
        # 先頭末尾の空白を除去
        text = text.strip()
        return text

class DocumentStructureDetector:
    """文書構造検出クラス"""
    
    def __init__(self, domain_config: Dict[str, Any]):
        self.domain_config = domain_config
        
    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """文書のセクションを検出"""
        sections = []
        lines = text.split('\n')
        
        # 論文のセクションパターンを取得
        section_patterns = self.domain_config.get('academic_papers', {}).get('section_patterns', {})
        
        current_section = None
        section_start = 0
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # セクションの検出
            for section_name, patterns in section_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in line_lower:
                        # 前のセクションを終了
                        if current_section:
                            sections.append({
                                'type': current_section,
                                'start': section_start,
                                'end': i,
                                'content': '\n'.join(lines[section_start:i])
                            })
                        
                        # 新しいセクション開始
                        current_section = section_name
                        section_start = i
                        break
        
        # 最後のセクション追加
        if current_section:
            sections.append({
                'type': current_section,
                'start': section_start,
                'end': len(lines),
                'content': '\n'.join(lines[section_start:])
            })
        
        return sections
    
    def extract_headers(self, text: str) -> List[Tuple[str, int, int]]:
        """見出しを抽出"""
        headers = []
        lines = text.split('\n')
        
        # 見出しのパターン
        header_patterns = [
            r'^#\s+(.+)$',  # Markdown H1
            r'^##\s+(.+)$',  # Markdown H2
            r'^###\s+(.+)$',  # Markdown H3
            r'^\d+\.\s+(.+)$',  # 1. 形式
            r'^\d+\.\d+\s+(.+)$',  # 1.1 形式
            r'^[A-Z][A-Z\s]+$',  # 全て大文字
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in header_patterns:
                match = re.match(pattern, line)
                if match:
                    headers.append((match.group(1) if match.groups() else line, i, i+1))
                    break
                    
        return headers