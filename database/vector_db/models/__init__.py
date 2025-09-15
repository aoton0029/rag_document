"""
Milvus vector database models for RAGShelf.
"""
from .document_embedding import DocumentEmbedding
from .query_embedding import QueryEmbedding

__all__ = ["DocumentEmbedding", "QueryEmbedding"]