import os
import logging
import uuid
import re
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pathlib import Path
from datetime import datetime
from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode, TextNode, Node
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor, SummaryExtractor, DocumentContextExtractor, QuestionsAnsweredExtractor
from llama_index.extractors.entity import EntityExtractor
from services.models import ChunkingResult, EntityRelation
from configs import ProcessingConfig
from db.database_manager import db_manager

logger = logging.getLogger(__name__)

class ChunkingService:
    """
    ドキュメントチャンク化サービス
    ドキュメントを検索・処理に適したサイズに分割
    """
    
    def __init__(self, config: Optional[ProcessingConfig]):
        self.config = config or ProcessingConfig()

        
        # ノードパーサーの初期化
        if self.config.chunk_method == "sentence":
            self.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.chunk_method == "token":
            self.node_parser = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        else:  # simple
            self.node_parser = SimpleNodeParser(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

        # エクストラクターの初期化
        self.extractors = [
            TitleExtractor(),
            KeywordExtractor(),
            SummaryExtractor(),
            DocumentContextExtractor(docstore=db_manager.docstore, llm=Settings.llm),
            QuestionsAnsweredExtractor(),
        ]

        logger.info("ChunkingService initialized")
    
    def chunk_documents(self, documents: List[Document]) -> List[ChunkingResult]:
        """
        複数ドキュメントをチャンク化
        """
        try:
            start_time = datetime.now()
            logger.info(f"ドキュメントチャンク化開始: {documents[0].id_}")
            
            # ドキュメントをノードに分割
            nodes = self.node_parser.get_nodes_from_documents(documents, True)
            
            # 各ノードに元ドキュメントの情報を設定
            for node in nodes:
                if not node.metadata:
                    node.metadata = {}
                node.metadata.update({
                    "created_at" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #     'source_document_id': document.id_,
                #     'source_document_metadata': document.metadata or {},
                #     'chunk_method': self.config.chunk_method,
                #     'chunk_size': self.config.chunk_size,
                #     'chunk_overlap': self.config.chunk_overlap
                 })

            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"チャンク化完了: {len(nodes)}個のチャンクを生成 ({processing_time:.2f}秒)")
            
            return ChunkingResult(
                success=True,
                nodes=nodes,  # BaseNodeオブジェクトのリストをそのまま返す
                processing_time=processing_time,
                metadata={
                    'chunk_count': len(nodes),
                    'chunk_method': self.config.chunk_method,
                    # 'extractors_applied': [extractor.__class__.__name__ for extractor in self.extractors]
                }
            )
            
        except Exception as e:
            logger.error(f"チャンク化エラー: {e}", exc_info=True)
            return ChunkingResult(
                success=False,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0.0
            )

    def chunk_document(self, document: Document) -> ChunkingResult:
        """
        ドキュメントをチャンク化
        """
        try:
            start_time = datetime.now()
            logger.info(f"ドキュメントチャンク化開始: {document.id_}")
            
            # ドキュメントをノードに分割
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            # 各ノードに元ドキュメントの情報を設定
            # for node in nodes:
            #     if not node.metadata:
            #         node.metadata = {}
            #     node.metadata.update({
            #         'source_document_id': document.id_,
            #         'source_document_metadata': document.metadata or {},
            #         'chunk_method': self.config.chunk_method,
            #         'chunk_size': self.config.chunk_size,
            #         'chunk_overlap': self.config.chunk_overlap
            #     })
            
            # エクストラクターを適用してメタデータを充実化
            # if self.extractors:
            #     logger.info("メタデータ抽出開始")
            #     for extractor in self.extractors:
            #         try:
            #             nodes = extractor.extract(nodes)
            #         except Exception as e:
            #             logger.warning(f"エクストラクター {extractor.__class__.__name__} でエラー: {e}")
            #             continue
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"チャンク化完了: {len(nodes)}個のチャンクを生成 ({processing_time:.2f}秒)")
            
            return ChunkingResult(
                success=True,
                nodes=nodes,  # BaseNodeオブジェクトのリストをそのまま返す
                processing_time=processing_time,
                metadata={
                    'chunk_count': len(nodes),
                    'original_document_id': document.id_,
                    'chunk_method': self.config.chunk_method,
                    # 'extractors_applied': [extractor.__class__.__name__ for extractor in self.extractors]
                }
            )
            
        except Exception as e:
            logger.error(f"チャンク化エラー: {e}", exc_info=True)
            return ChunkingResult(
                success=False,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0.0
            )
