"""
Metadata Extractor Module
ドキュメントからメタデータを抽出
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from llama_index.core.schema import Document, BaseNode
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.extractors.entity import EntityExtractor

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    メタデータ抽出クラス
    llama_indexのExtractorをラップ
    """
    
    def __init__(
        self,
        extract_title: bool = True,
        extract_keywords: bool = True,
        extract_summary: bool = False,
        extract_entities: bool = False,
        extract_questions: bool = False,
        llm: Optional[Any] = None
    ):
        """
        MetadataExtractorの初期化
        
        Args:
            extract_title: タイトルを抽出するか
            extract_keywords: キーワードを抽出するか
            extract_summary: 要約を抽出するか
            extract_entities: エンティティを抽出するか
            extract_questions: 質問を抽出するか
            llm: LLMインスタンス（要約・質問抽出に必要）
        """
        self.extract_title = extract_title
        self.extract_keywords = extract_keywords
        self.extract_summary = extract_summary
        self.extract_entities = extract_entities
        self.extract_questions = extract_questions
        self.llm = llm
        
        self._extractors = []
        self._init_extractors()
    
    def _init_extractors(self):
        """Extractorを初期化"""
        if self.extract_title:
            try:
                extractor = TitleExtractor(llm=self.llm) if self.llm else TitleExtractor()
                self._extractors.append(extractor)
                logger.info("TitleExtractorを追加")
            except Exception as e:
                logger.warning(f"TitleExtractor初期化失敗: {e}")
        
        if self.extract_keywords:
            try:
                extractor = KeywordExtractor(llm=self.llm) if self.llm else KeywordExtractor()
                self._extractors.append(extractor)
                logger.info("KeywordExtractorを追加")
            except Exception as e:
                logger.warning(f"KeywordExtractor初期化失敗: {e}")
        
        if self.extract_summary and self.llm:
            try:
                extractor = SummaryExtractor(llm=self.llm)
                self._extractors.append(extractor)
                logger.info("SummaryExtractorを追加")
            except Exception as e:
                logger.warning(f"SummaryExtractor初期化失敗: {e}")
        
        if self.extract_entities:
            try:
                extractor = EntityExtractor()
                self._extractors.append(extractor)
                logger.info("EntityExtractorを追加")
            except Exception as e:
                logger.warning(f"EntityExtractor初期化失敗: {e}")
        
        if self.extract_questions and self.llm:
            try:
                extractor = QuestionsAnsweredExtractor(llm=self.llm)
                self._extractors.append(extractor)
                logger.info("QuestionsAnsweredExtractorを追加")
            except Exception as e:
                logger.warning(f"QuestionsAnsweredExtractor初期化失敗: {e}")
    
    def extract_metadata(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        ノードからメタデータを抽出
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            メタデータが追加されたノードのリスト
        """
        if not self._extractors:
            logger.warning("Extractorが設定されていません")
            return nodes
        
        try:
            # 各Extractorを順次適用
            for extractor in self._extractors:
                nodes = extractor.extract(nodes)
            
            logger.info(f"{len(nodes)}ノードにメタデータを抽出")
            return nodes
        except Exception as e:
            logger.error(f"メタデータ抽出エラー: {e}")
            return nodes
    
    def extract_document_metadata(self, document: Document) -> Document:
        """
        ドキュメントからメタデータを抽出
        
        Args:
            document: Document
            
        Returns:
            メタデータが追加されたDocument
        """
        # 基本的なメタデータを追加
        if "extracted_at" not in document.metadata:
            document.metadata["extracted_at"] = datetime.now().isoformat()
        
        # ファイルパスが存在する場合、ファイル情報を追加
        if "file_path" in document.metadata:
            file_path = Path(document.metadata["file_path"])
            document.metadata.update({
                "file_name": file_path.name,
                "file_extension": file_path.suffix,
                "file_size": file_path.stat().st_size if file_path.exists() else 0
            })
        
        # テキスト統計を追加
        text = document.get_content()
        document.metadata.update({
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split("\n"))
        })
        
        return document


class PDFMetadataExtractor(MetadataExtractor):
    """
    PDF専用メタデータ抽出クラス
    """
    
    def extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        PDFファイルからメタデータを抽出
        
        Args:
            file_path: PDFファイルパス
            
        Returns:
            メタデータ辞書
        """
        metadata = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_type": "pdf"
        }
        
        try:
            import pymupdf
            
            doc = pymupdf.open(file_path)
            
            # PDFメタデータを抽出
            pdf_metadata = doc.metadata
            metadata.update({
                "title": pdf_metadata.get("title", ""),
                "author": pdf_metadata.get("author", ""),
                "subject": pdf_metadata.get("subject", ""),
                "keywords": pdf_metadata.get("keywords", ""),
                "creator": pdf_metadata.get("creator", ""),
                "producer": pdf_metadata.get("producer", ""),
                "creation_date": pdf_metadata.get("creationDate", ""),
                "modification_date": pdf_metadata.get("modDate", ""),
                "page_count": doc.page_count
            })
            
            doc.close()
            logger.info(f"PDFメタデータ抽出: {file_path}")
            
        except ImportError:
            logger.warning("pymupdfが利用不可、基本メタデータのみ")
        except Exception as e:
            logger.error(f"PDFメタデータ抽出エラー: {e}")
        
        return metadata
