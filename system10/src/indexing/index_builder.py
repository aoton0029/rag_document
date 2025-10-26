"""
Index Builder Module
各種インデックスの構築
"""

import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from llama_index.core.schema import BaseNode, Document
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
    DocumentSummaryIndex
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.base import BaseIndex

logger = logging.getLogger(__name__)


class IndexBuilder(ABC):
    """
    インデックスビルダー基底クラス
    """
    
    def __init__(
        self,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = True,
        **kwargs
    ):
        """
        IndexBuilderの初期化
        
        Args:
            storage_context: StorageContext
            show_progress: 進捗を表示するか
            **kwargs: 追加パラメータ
        """
        self.storage_context = storage_context
        self.show_progress = show_progress
        self.kwargs = kwargs
        self._index = None
    
    @abstractmethod
    def build_from_nodes(self, nodes: List[BaseNode]) -> BaseIndex:
        """
        ノードからインデックスを構築
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            構築されたインデックス
        """
        pass
    
    @abstractmethod
    def build_from_documents(self, documents: List[Document]) -> BaseIndex:
        """
        ドキュメントからインデックスを構築
        
        Args:
            documents: Documentのリスト
            
        Returns:
            構築されたインデックス
        """
        pass
    
    def get_index(self) -> Optional[BaseIndex]:
        """構築されたインデックスを取得"""
        return self._index


class VectorIndexBuilder(IndexBuilder):
    """
    ベクトルインデックスビルダー
    """
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """
        ノードからベクトルインデックスを構築
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            VectorStoreIndex
        """
        try:
            self._index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"ベクトルインデックスを構築: {len(nodes)}ノード")
            return self._index
        except Exception as e:
            logger.error(f"ベクトルインデックス構築エラー: {e}")
            raise
    
    def build_from_documents(self, documents: List[Document]) -> VectorStoreIndex:
        """
        ドキュメントからベクトルインデックスを構築
        
        Args:
            documents: Documentのリスト
            
        Returns:
            VectorStoreIndex
        """
        try:
            self._index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"ベクトルインデックスを構築: {len(documents)}ドキュメント")
            return self._index
        except Exception as e:
            logger.error(f"ベクトルインデックス構築エラー: {e}")
            raise


class SummaryIndexBuilder(IndexBuilder):
    """
    サマリーインデックスビルダー
    """
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> SummaryIndex:
        """
        ノードからサマリーインデックスを構築
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            SummaryIndex
        """
        try:
            self._index = SummaryIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"サマリーインデックスを構築: {len(nodes)}ノード")
            return self._index
        except Exception as e:
            logger.error(f"サマリーインデックス構築エラー: {e}")
            raise
    
    def build_from_documents(self, documents: List[Document]) -> SummaryIndex:
        """
        ドキュメントからサマリーインデックスを構築
        
        Args:
            documents: Documentのリスト
            
        Returns:
            SummaryIndex
        """
        try:
            self._index = SummaryIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"サマリーインデックスを構築: {len(documents)}ドキュメント")
            return self._index
        except Exception as e:
            logger.error(f"サマリーインデックス構築エラー: {e}")
            raise


class TreeIndexBuilder(IndexBuilder):
    """
    ツリーインデックスビルダー
    """
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> TreeIndex:
        """
        ノードからツリーインデックスを構築
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            TreeIndex
        """
        try:
            self._index = TreeIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"ツリーインデックスを構築: {len(nodes)}ノード")
            return self._index
        except Exception as e:
            logger.error(f"ツリーインデックス構築エラー: {e}")
            raise
    
    def build_from_documents(self, documents: List[Document]) -> TreeIndex:
        """
        ドキュメントからツリーインデックスを構築
        
        Args:
            documents: Documentのリスト
            
        Returns:
            TreeIndex
        """
        try:
            self._index = TreeIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"ツリーインデックスを構築: {len(documents)}ドキュメント")
            return self._index
        except Exception as e:
            logger.error(f"ツリーインデックス構築エラー: {e}")
            raise


class KeywordIndexBuilder(IndexBuilder):
    """
    キーワードインデックスビルダー
    """
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> KeywordTableIndex:
        """
        ノードからキーワードインデックスを構築
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            KeywordTableIndex
        """
        try:
            self._index = KeywordTableIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"キーワードインデックスを構築: {len(nodes)}ノード")
            return self._index
        except Exception as e:
            logger.error(f"キーワードインデックス構築エラー: {e}")
            raise
    
    def build_from_documents(self, documents: List[Document]) -> KeywordTableIndex:
        """
        ドキュメントからキーワードインデックスを構築
        
        Args:
            documents: Documentのリスト
            
        Returns:
            KeywordTableIndex
        """
        try:
            self._index = KeywordTableIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"キーワードインデックスを構築: {len(documents)}ドキュメント")
            return self._index
        except Exception as e:
            logger.error(f"キーワードインデックス構築エラー: {e}")
            raise


class KnowledgeGraphIndexBuilder(IndexBuilder):
    """
    ナレッジグラフインデックスビルダー
    """
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> KnowledgeGraphIndex:
        """
        ノードからナレッジグラフインデックスを構築
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            KnowledgeGraphIndex
        """
        try:
            self._index = KnowledgeGraphIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"ナレッジグラフインデックスを構築: {len(nodes)}ノード")
            return self._index
        except Exception as e:
            logger.error(f"ナレッジグラフインデックス構築エラー: {e}")
            raise
    
    def build_from_documents(self, documents: List[Document]) -> KnowledgeGraphIndex:
        """
        ドキュメントからナレッジグラフインデックスを構築
        
        Args:
            documents: Documentのリスト
            
        Returns:
            KnowledgeGraphIndex
        """
        try:
            self._index = KnowledgeGraphIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                show_progress=self.show_progress,
                **self.kwargs
            )
            logger.info(f"ナレッジグラフインデックスを構築: {len(documents)}ドキュメント")
            return self._index
        except Exception as e:
            logger.error(f"ナレッジグラフインデックス構築エラー: {e}")
            raise


class DocumentSummaryIndexBuilder(IndexBuilder):
    """
    ドキュメントサマリーインデックスビルダー
    """
    
    def __init__(
        self,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = True,
        response_synthesizer: Optional[Any] = None,
        **kwargs
    ):
        """
        DocumentSummaryIndexBuilderの初期化
        
        Args:
            storage_context: StorageContext
            show_progress: 進捗を表示するか
            response_synthesizer: ResponseSynthesizer
            **kwargs: 追加パラメータ
        """
        super().__init__(storage_context, show_progress, **kwargs)
        self.response_synthesizer = response_synthesizer
    
    def build_from_nodes(self, nodes: List[BaseNode]) -> DocumentSummaryIndex:
        """
        ノードからドキュメントサマリーインデックスを構築
        
        Args:
            nodes: BaseNodeのリスト
            
        Returns:
            DocumentSummaryIndex
        """
        try:
            # DocumentSummaryIndexはノードから直接構築できない
            # ドキュメントに変換する必要がある
            logger.warning("DocumentSummaryIndexはドキュメントから構築してください")
            raise NotImplementedError("DocumentSummaryIndexはノードから直接構築できません")
        except Exception as e:
            logger.error(f"ドキュメントサマリーインデックス構築エラー: {e}")
            raise
    
    def build_from_documents(self, documents: List[Document]) -> DocumentSummaryIndex:
        """
        ドキュメントからドキュメントサマリーインデックスを構築
        
        Args:
            documents: Documentのリスト
            
        Returns:
            DocumentSummaryIndex
        """
        try:
            kwargs = {
                "storage_context": self.storage_context,
                "show_progress": self.show_progress,
                **self.kwargs
            }
            
            if self.response_synthesizer:
                kwargs["response_synthesizer"] = self.response_synthesizer
            
            self._index = DocumentSummaryIndex.from_documents(
                documents=documents,
                **kwargs
            )
            logger.info(f"ドキュメントサマリーインデックスを構築: {len(documents)}ドキュメント")
            return self._index
        except Exception as e:
            logger.error(f"ドキュメントサマリーインデックス構築エラー: {e}")
            raise
