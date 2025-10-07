import logging
import numpy as np
from typing import List, Optional
from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode, TextNode
from llm.ollama_connector import OllamaConnector
from configs import ProcessingConfig
logger = logging.getLogger(__name__)

class EmbeddingService:
    """OllamaによるDocument/Nodeのエンベディングサービス"""
    
    def __init__(self):
        pass
    
    def embed_document(self, document: Document) -> Document:
        """
        Documentにエンベディングを追加
        """
        if Settings.embed_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            # ドキュメントのテキストからエンベディングを生成
            embedding = Settings.embed_model.get_text_embedding(document.text)
            
            # Documentにエンベディングを設定
            document.embedding = embedding
            
            logger.debug(f"Document embedded successfully. Embedding dimension: {len(embedding)}")
            return document
            
        except Exception as e:
            logger.error(f"Failed to embed document: {e}")
            raise
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        複数のDocumentにエンベディングを追加
        """
        embedded_documents = []
        
        for i, doc in enumerate(documents):
            try:
                embedded_doc = self.embed_document(doc)
                embedded_documents.append(embedded_doc)
                logger.debug(f"Processed document {i+1}/{len(documents)}")
            except Exception as e:
                logger.error(f"Failed to embed document {i+1}: {e}")
                # エラーが発生したドキュメントはスキップ
                continue
        
        logger.info(f"Successfully embedded {len(embedded_documents)}/{len(documents)} documents")
        return embedded_documents
    
    def embed_node(self, node: BaseNode) -> BaseNode:
        """
        Nodeにエンベディングを追加
        """
        if Settings.embed_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            # ノードのテキストからエンベディングを生成
            text_content = node.get_content()
            embedding = Settings.embed_model.get_text_embedding(text_content)

            # Nodeにエンベディングを設定
            node.embedding = embedding
            
            logger.debug(f"Node embedded successfully. Embedding dimension: {len(embedding)}")
            return node
            
        except Exception as e:
            logger.error(f"Failed to embed node: {e}")
            raise
    
    def embed_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        複数のNodeにエンベディングを追加
        """
        embedded_nodes = []
        
        for i, node in enumerate(nodes):
            try:
                embedded_node = self.embed_node(node)
                embedded_nodes.append(embedded_node)
                logger.debug(f"Processed node {i+1}/{len(nodes)}")
            except Exception as e:
                logger.error(f"Failed to embed node {i+1}: {e}")
                # エラーが発生したノードはスキップ
                continue
        
        logger.info(f"Successfully embedded {len(embedded_nodes)}/{len(nodes)} nodes")
        return embedded_nodes
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        テキストからエンベディングを直接取得
        """
        if Settings.embed_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = Settings.embed_model.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get text embedding: {e}")
            raise
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        クエリからエンベディングを取得
        """
        if Settings.embed_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = Settings.embed_model.get_query_embedding(query)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            raise
    
