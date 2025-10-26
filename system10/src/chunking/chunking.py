"""
Chunking Module
NodeParser/TextSplitterのラッパー
要件定義に基づいた正確なインポート
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter, NodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.extractors.entity import EntityExtractor


logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """チャンキング設定"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = "\n\n"
    paragraph_separator: str = "\n"
    include_metadata: bool = True
    include_prev_next_rel: bool = True


class BaseChunker(ABC):
    """
    チャンカー基底クラス
    NodeParserをラップ
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        BaseChunkerの初期化
        
        Args:
            config: チャンキング設定
        """
        self.config = config or ChunkingConfig()
        self._parser = None
    
    @abstractmethod
    def _create_parser(self) -> NodeParser:
        """NodeParserを作成"""
        pass
    
    def get_parser(self) -> NodeParser:
        """NodeParserを取得"""
        if self._parser is None:
            self._parser = self._create_parser()
        return self._parser
    
    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        ドキュメントをチャンクに分割
        
        Args:
            documents: Documentのリスト
            
        Returns:
            BaseNodeのリスト
        """
        parser = self.get_parser()
        nodes = parser.get_nodes_from_documents(documents)
        
        logger.info(f"{len(documents)}ドキュメントから{len(nodes)}ノードを作成")
        return nodes
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[BaseNode]:
        """
        テキストをチャンクに分割
        
        Args:
            text: テキスト
            metadata: メタデータ
            
        Returns:
            BaseNodeのリスト
        """
        document = Document(text=text, metadata=metadata or {})
        return self.chunk_documents([document])


class SimpleNodeParserChunker(BaseChunker):
    """
    SimpleNodeParserを使用したチャンカー
    from llama_index.core.node_parser import SimpleNodeParser
    """
    
    def _create_parser(self) -> SimpleNodeParser:
        """SimpleNodeParserを作成"""
        try:
            return SimpleNodeParser.from_defaults(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                include_metadata=self.config.include_metadata,
                include_prev_next_rel=self.config.include_prev_next_rel
            )
        except Exception as e:
            logger.error(f"SimpleNodeParser作成エラー: {e}")
            raise


class SentenceSplitterChunker(BaseChunker):
    """
    SentenceSplitterを使用したチャンカー
    from llama_index.core.node_parser import SentenceSplitter
    """
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        paragraph_separator: str = "\n\n\n"
    ):
        """
        SentenceSplitterChunkerの初期化
        
        Args:
            config: チャンキング設定
            paragraph_separator: 段落区切り文字
        """
        super().__init__(config)
        self.paragraph_separator = paragraph_separator
    
    def _create_parser(self) -> SentenceSplitter:
        """SentenceSplitterを作成"""
        try:
            return SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                paragraph_separator=self.paragraph_separator,
                separator=self.config.separator
            )
        except Exception as e:
            logger.error(f"SentenceSplitter作成エラー: {e}")
            raise


class TokenBasedChunker(BaseChunker):
    """
    トークンベースのチャンカー
    from llama_index.core.text_splitter import TokenTextSplitter
    """
    
    def _create_parser(self) -> NodeParser:
        """TokenTextSplitterを使用したNodeParserを作成"""
        try:
            from llama_index.core.node_parser import SentenceSplitter
            
            # TokenTextSplitterを内部で使用するSentenceSplitterを作成
            return SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=self.config.separator
            )
        except Exception as e:
            logger.error(f"TokenBasedChunker作成エラー: {e}")
            raise


class SemanticChunker(BaseChunker):
    """
    セマンティックチャンカー
    意味的なまとまりでチャンク分割
    """
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embed_model: Optional[Any] = None,
        breakpoint_percentile_threshold: int = 95
    ):
        """
        SemanticChunkerの初期化
        
        Args:
            config: チャンキング設定
            embed_model: 埋め込みモデル
            breakpoint_percentile_threshold: ブレークポイント閾値
        """
        super().__init__(config)
        self.embed_model = embed_model
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
    
    def _create_parser(self) -> NodeParser:
        """SemanticSplitterNodeParserを作成"""
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            
            if self.embed_model is None:
                raise ValueError("SemanticChunkerには埋め込みモデルが必要です")
            
            return SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
                embed_model=self.embed_model
            )
        except ImportError:
            logger.warning("SemanticSplitterNodeParserが利用不可、SentenceSplitterにフォールバック")
            return SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        except Exception as e:
            logger.error(f"SemanticChunker作成エラー: {e}")
            raise


class MetadataAwareChunker(BaseChunker):
    """
    メタデータ抽出機能付きチャンカー
    extractorsを使用してメタデータを抽出
    """
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        extractors: Optional[List[Any]] = None,
        llm: Optional[Any] = None
    ):
        """
        MetadataAwareChunkerの初期化
        
        Args:
            config: チャンキング設定
            extractors: Extractorのリスト
            llm: LLMインスタンス
        """
        super().__init__(config)
        self.extractors = extractors or self._create_default_extractors(llm)
    
    def _create_default_extractors(self, llm: Optional[Any] = None) -> List[Any]:
        """デフォルトのExtractorを作成"""
        extractors = []
        
        try:
            # TitleExtractor
            if llm:
                extractors.append(TitleExtractor(llm=llm, nodes=5))
            else:
                extractors.append(TitleExtractor(nodes=5))
        except Exception as e:
            logger.warning(f"TitleExtractor作成失敗: {e}")
        
        try:
            # KeywordExtractor
            if llm:
                extractors.append(KeywordExtractor(llm=llm, keywords=10))
            else:
                extractors.append(KeywordExtractor(keywords=10))
        except Exception as e:
            logger.warning(f"KeywordExtractor作成失敗: {e}")
        
        try:
            # QuestionsAnsweredExtractor
            if llm:
                extractors.append(QuestionsAnsweredExtractor(llm=llm, questions=3))
        except Exception as e:
            logger.warning(f"QuestionsAnsweredExtractor作成失敗: {e}")
        
        try:
            # EntityExtractor
            extractors.append(EntityExtractor())
        except Exception as e:
            logger.warning(f"EntityExtractor作成失敗: {e}")
        
        return extractors
    
    def _create_parser(self) -> NodeParser:
        """メタデータ抽出機能付きNodeParserを作成"""
        try:
            return SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        except Exception as e:
            logger.error(f"MetadataAwareChunker作成エラー: {e}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        ドキュメントをチャンクに分割し、メタデータを抽出
        
        Args:
            documents: Documentのリスト
            
        Returns:
            BaseNodeのリスト（メタデータ付き）
        """
        # 基本的なチャンキング
        parser = self.get_parser()
        nodes = parser.get_nodes_from_documents(documents)
        
        # メタデータ抽出
        if self.extractors:
            for extractor in self.extractors:
                try:
                    nodes = extractor.extract(nodes)
                except Exception as e:
                    logger.warning(f"メタデータ抽出エラー: {e}")
        
        logger.info(f"{len(documents)}ドキュメントから{len(nodes)}ノード（メタデータ付き）を作成")
        return nodes


class ChunkerFactory:
    """
    チャンカーファクトリー
    """
    
    @staticmethod
    def create_chunker(
        chunker_type: str = "sentence",
        config: Optional[ChunkingConfig] = None,
        **kwargs
    ) -> BaseChunker:
        """
        チャンカーを作成
        
        Args:
            chunker_type: チャンカータイプ
                - "simple": SimpleNodeParser
                - "sentence": SentenceSplitter
                - "token": TokenTextSplitter
                - "semantic": SemanticChunker
                - "metadata": MetadataAwareChunker
            config: チャンキング設定
            **kwargs: 追加パラメータ
            
        Returns:
            BaseChunker
        """
        if chunker_type == "simple":
            return SimpleNodeParserChunker(config=config)
        elif chunker_type == "sentence":
            return SentenceSplitterChunker(config=config, **kwargs)
        elif chunker_type == "token":
            return TokenBasedChunker(config=config)
        elif chunker_type == "semantic":
            return SemanticChunker(config=config, **kwargs)
        elif chunker_type == "metadata":
            return MetadataAwareChunker(config=config, **kwargs)
        else:
            logger.warning(f"未対応のチャンカータイプ: {chunker_type}、SentenceSplitterを使用")
            return SentenceSplitterChunker(config=config)
