import os
import logging
import asyncio
import uuid
import re
from typing import List, Dict, Any, Optional, Union, BinaryIO
from pathlib import Path
from datetime import datetime

from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode, Node
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor, KeywordExtractor, SummaryExtractor, DocumentContextExtractor, QuestionsAnsweredExtractor
from llama_index.extractors.entity import EntityExtractor
from models import ProcessingConfig, ChunkingResult, ChunkData, Neo4jData, EntityRelation
from ..llm.ollama_connector import OllamaConnector

class DocumentChunker:
    """
    ドキュメントチャンク化サービス
    ドキュメントを検索・処理に適したサイズに分割
    """
    
    def __init__(self, 
                 config: Optional[ProcessingConfig],
                 ollama: OllamaConnector):
        self.ollama = ollama
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
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
            TitleExtractor(llm=self.ollama.llm),
            KeywordExtractor(llm=self.ollama.llm),
            SummaryExtractor(llm=self.ollama.llm),
            DocumentContextExtractor(llm=self.ollama.llm),
            QuestionsAnsweredExtractor(llm=self.ollama.llm),
        ]
        
        # エンティティ抽出が有効な場合に追加
        if self.config.extract_entities:
            self.extractors.append(EntityExtractor())

    def chunk_documents(self, documents: List[Document]) -> ChunkingResult:
        """
        ドキュメントをチャンク化し、MongoDB・Neo4j用のデータを準備
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting document chunking for {len(documents)} documents")
            
            all_nodes = []
            mongodb_chunks = []
            neo4j_data = []
            
            for doc in documents:
                # ドキュメントをノードに分割
                nodes = self.node_parser.get_nodes_from_documents([doc])
                
                # メタデータ抽出
                for extractor in self.extractors:
                    nodes = extractor.extract(nodes)
                
                # MongoDB・Neo4j用データの準備
                doc_mongodb_chunks, doc_neo4j_data = self._prepare_database_data(
                    nodes, doc
                )
                
                all_nodes.extend(nodes)
                mongodb_chunks.extend(doc_mongodb_chunks)
                neo4j_data.extend(doc_neo4j_data)
            
            # 結果の文書化
            chunked_documents = [
                Document(
                    text=node.text,
                    metadata=node.metadata,
                    id_=node.id_
                ) for node in all_nodes
            ]
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ChunkingResult(
                success=True,
                documents=chunked_documents,
                processing_time=processing_time,
                metadata={
                    "total_chunks": len(all_nodes),
                    "total_documents": len(documents),
                    "chunk_method": self.config.chunk_method,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap
                },
                mongodb_chunks=mongodb_chunks,
                neo4j_data=neo4j_data
            )
            
            self.logger.info(f"Document chunking completed: {len(all_nodes)} chunks created")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during document chunking: {str(e)}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ChunkingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    async def _prepare_database_data(self, nodes: List[BaseNode], document: Document) -> tuple[List[ChunkData], List[Neo4jData]]:
        """
        ノードからMongoDB・Neo4j用のデータを準備
        """
        mongodb_chunks = []
        neo4j_data = []
        
        doc_id = document.metadata.get("doc_id", str(uuid.uuid4()))
        
        for i, node in enumerate(nodes):
            # MongoDB用チャンクデータの準備
            chunk_id = str(uuid.uuid4())
            
            # オフセット計算（簡易版）
            original_text = document.text
            offset_start = original_text.find(node.text) if original_text else 0
            offset_end = offset_start + len(node.text) if offset_start >= 0 else len(node.text)
            
            # エンティティ抽出
            entities = self._extract_entities_from_metadata(node.metadata)
            
            # メタデータの準備
            chunk_metadata = {
                "section_title": node.metadata.get("section_title", ""),
                "page": node.metadata.get("page", 0),
                "language": document.metadata.get("language", "unknown"),
                "confidence": node.metadata.get("confidence", 1.0),
                "channel": document.metadata.get("channel", ""),
                "entities": entities,
                "keywords": node.metadata.get("keywords", []),
                "summary": node.metadata.get("summary", ""),
                "title": node.metadata.get("title", "")
            }
            
            chunk_data = ChunkData(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=node.text,
                chunk_index=i,
                offset_start=offset_start,
                offset_end=offset_end,
                metadata=chunk_metadata
            )
            
            mongodb_chunks.append(chunk_data)
            
            # Neo4j用データの準備
            if entities:
                relations = self._extract_entity_relations(node.text, entities)
                
                neo4j_chunk_data = Neo4jData(
                    entities=entities,
                    relations=relations,
                    chunk_id=chunk_id,
                    doc_id=doc_id
                )
                
                neo4j_data.append(neo4j_chunk_data)
        
        return mongodb_chunks, neo4j_data

    def _extract_entities_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        メタデータからエンティティを抽出
        """
        entities = []
        
        # EntityExtractorが抽出したエンティティを取得
        if "entities" in metadata:
            entities.extend(metadata["entities"])
        
        # キーワードもエンティティとして扱う
        if "keywords" in metadata:
            entities.extend(metadata["keywords"])
        
        # 重複除去と正規化
        entities = list(set([entity.strip() for entity in entities if entity.strip()]))
        
        return entities

    def _extract_entity_relations(self, text: str, entities: List[str]) -> List[EntityRelation]:
        """
        テキストからエンティティ間の関係を抽出（簡易版）
        """
        relations = []
        
        # 簡易的な関係抽出（同一文中に出現するエンティティを関連とする）
        sentences = re.split(r'[.!?。！？]', text)
        
        for sentence in sentences:
            sentence_entities = [entity for entity in entities if entity in sentence]
            
            # 同一文中の全エンティティペアに関係を設定
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    relation = EntityRelation(
                        source_entity=entity1,
                        target_entity=entity2,
                        relation_type="co_occurrence",
                        confidence=0.7,
                        context=sentence.strip()
                    )
                    relations.append(relation)
        
        return relations

    def get_chunk_statistics(self, result: ChunkingResult) -> Dict[str, Any]:
        """
        チャンク化結果の統計情報を取得
        """
        if not result.success:
            return {"error": result.error_message}
        
        total_entities = sum(len(data.entities) for data in result.neo4j_data)
        total_relations = sum(len(data.relations) for data in result.neo4j_data)
        
        chunk_sizes = [len(chunk.text) for chunk in result.mongodb_chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        return {
            "total_chunks": len(result.mongodb_chunks),
            "total_entities": total_entities,
            "total_relations": total_relations,
            "average_chunk_size": avg_chunk_size,
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "processing_time": result.processing_time
        }