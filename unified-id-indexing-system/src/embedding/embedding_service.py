from typing import List, Optional
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llm.ollama_connector import OllamaConnector
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings for documents and nodes using Ollama."""
    
    def __init__(self, connector: OllamaConnector):
        self.connector = connector
        self.embedding_model = OllamaEmbedding(connector)

    def embed_document(self, document: Document) -> Document:
        """Adds embedding to a Document."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = self.embedding_model.get_text_embedding(document.text)
            document.embedding = embedding
            logger.debug(f"Document embedded successfully. Embedding dimension: {len(embedding)}")
            return document
            
        except Exception as e:
            logger.error(f"Failed to embed document: {e}")
            raise
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Adds embedding to a list of Documents."""
        embedded_documents = []
        
        for i, doc in enumerate(documents):
            try:
                embedded_doc = self.embed_document(doc)
                embedded_documents.append(embedded_doc)
                logger.debug(f"Processed document {i+1}/{len(documents)}")
            except Exception as e:
                logger.error(f"Failed to embed document {i+1}: {e}")
                continue
        
        logger.info(f"Successfully embedded {len(embedded_documents)}/{len(documents)} documents")
        return embedded_documents
    
    def embed_node(self, node: BaseNode) -> BaseNode:
        """Adds embedding to a Node."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            text_content = node.get_content()
            embedding = self.embedding_model.get_text_embedding(text_content)
            node.embedding = embedding
            logger.debug(f"Node embedded successfully. Embedding dimension: {len(embedding)}")
            return node
            
        except Exception as e:
            logger.error(f"Failed to embed node: {e}")
            raise
    
    def embed_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Adds embedding to a list of Nodes."""
        embedded_nodes = []
        
        for i, node in enumerate(nodes):
            try:
                embedded_node = self.embed_node(node)
                embedded_nodes.append(embedded_node)
                logger.debug(f"Processed node {i+1}/{len(nodes)}")
            except Exception as e:
                logger.error(f"Failed to embed node {i+1}: {e}")
                continue
        
        logger.info(f"Successfully embedded {len(embedded_nodes)}/{len(nodes)} nodes")
        return embedded_nodes
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Gets embedding directly from text."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = self.embedding_model.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get text embedding: {e}")
            raise
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Gets embedding from a query."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        
        try:
            embedding = self.embedding_model.get_query_embedding(query)
            return embedding
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            raise
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculates cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def check_model_availability(self) -> bool:
        """Checks the availability of the embedding model."""
        try:
            if not self.connector.check_connection():
                return False
            
            available_models = self.connector.get_available_models()
            return self.model_name in available_models
            
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Gets the dimension of the embeddings."""
        try:
            test_embedding = self.get_text_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            return None