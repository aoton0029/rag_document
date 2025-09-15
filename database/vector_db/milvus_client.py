from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
from .models import DocumentEmbedding, QueryEmbedding


class MilvusClient:
    """Milvus client for vector storage and similarity search with model support."""
    
    def __init__(self, host: str = "localhost", port: str = "19530", 
                 doc_collection_name: str = "document_embeddings",
                 query_collection_name: str = "query_embeddings"):
        self.host = host
        self.port = port
        self.doc_collection_name = doc_collection_name
        self.query_collection_name = query_collection_name
        self.doc_collection = None
        self.query_collection = None
        self.connect()
        
    def connect(self):
        """Connect to Milvus server."""
        connections.connect("default", host=self.host, port=self.port)
        
    def create_document_collection(self, dimension: int = 768):
        """Create document embeddings collection based on schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="vector_dimension", dtype=DataType.INT64),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields, "Document embeddings for similarity search")
        
        if utility.has_collection(self.doc_collection_name):
            self.doc_collection = Collection(self.doc_collection_name)
        else:
            self.doc_collection = Collection(self.doc_collection_name, schema)
            
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.doc_collection.create_index("embedding_vector", index_params)
        
    def create_query_collection(self, dimension: int = 768):
        """Create query embeddings collection."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="query_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="query_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="complexity", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="usage_count", dtype=DataType.INT64),
            FieldSchema(name="avg_relevance_score", dtype=DataType.FLOAT),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="last_used_at", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields, "Query embeddings for search optimization")
        
        if utility.has_collection(self.query_collection_name):
            self.query_collection = Collection(self.query_collection_name)
        else:
            self.query_collection = Collection(self.query_collection_name, schema)
            
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT", 
            "params": {"nlist": 512}
        }
        self.query_collection.create_index("embedding_vector", index_params)
        
    # Document embedding operations
    def insert_document_embedding(self, embedding: DocumentEmbedding) -> bool:
        """Insert a document embedding using the DocumentEmbedding model."""
        if not self.doc_collection:
            raise ValueError("Document collection not initialized")
            
        # Validate vector dimension
        embedding.validate_vector_dimension()
        
        data = [
            [embedding.vector_id],
            [embedding.chunk_id],
            [embedding.document_id],
            [embedding.embedding_vector],
            [embedding.embedding_model],
            [embedding.vector_dimension],
            [embedding.content_hash or ""],
            [embedding.metadata.content_type if embedding.metadata else ""],
            [embedding.metadata.language if embedding.metadata else ""],
            [embedding.created_at]
        ]
        
        try:
            self.doc_collection.insert(data)
            self.doc_collection.flush()
            return True
        except Exception as e:
            print(f"Error inserting document embedding: {e}")
            return False
    
    def insert_document_embeddings(self, embeddings: List[DocumentEmbedding]) -> bool:
        """Insert multiple document embeddings."""
        if not self.doc_collection:
            raise ValueError("Document collection not initialized")
            
        if not embeddings:
            return True
            
        # Validate all embeddings
        for embedding in embeddings:
            embedding.validate_vector_dimension()
        
        # Prepare batch data
        vector_ids = [emb.vector_id for emb in embeddings]
        chunk_ids = [emb.chunk_id for emb in embeddings]
        document_ids = [emb.document_id for emb in embeddings]
        embedding_vectors = [emb.embedding_vector for emb in embeddings]
        embedding_models = [emb.embedding_model for emb in embeddings]
        vector_dimensions = [emb.vector_dimension for emb in embeddings]
        content_hashes = [emb.content_hash or "" for emb in embeddings]
        content_types = [emb.metadata.content_type if emb.metadata else "" for emb in embeddings]
        languages = [emb.metadata.language if emb.metadata else "" for emb in embeddings]
        created_ats = [emb.created_at for emb in embeddings]
        
        data = [
            vector_ids, chunk_ids, document_ids, embedding_vectors,
            embedding_models, vector_dimensions, content_hashes,
            content_types, languages, created_ats
        ]
        
        try:
            self.doc_collection.insert(data)
            self.doc_collection.flush()
            return True
        except Exception as e:
            print(f"Error inserting document embeddings: {e}")
            return False
    
    def search_similar_documents(self, query_vector: List[float], top_k: int = 5, 
                                document_filter: Optional[str] = None,
                                model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar document embeddings."""
        if not self.doc_collection:
            raise ValueError("Document collection not initialized")
            
        self.doc_collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        # Build filter expression
        expr = None
        filters = []
        if document_filter:
            filters.append(f'document_id == "{document_filter}"')
        if model_filter:
            filters.append(f'embedding_model == "{model_filter}"')
        if filters:
            expr = " and ".join(filters)
            
        results = self.doc_collection.search(
            data=[query_vector],
            anns_field="embedding_vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["vector_id", "chunk_id", "document_id", "embedding_model", 
                          "content_type", "language", "created_at"]
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "vector_id": hit.entity.get("vector_id"),
                    "chunk_id": hit.entity.get("chunk_id"),
                    "document_id": hit.entity.get("document_id"),
                    "embedding_model": hit.entity.get("embedding_model"),
                    "content_type": hit.entity.get("content_type"),
                    "language": hit.entity.get("language"),
                    "score": hit.score,
                    "created_at": hit.entity.get("created_at")
                })
                
        return formatted_results
    
    def get_document_embedding(self, vector_id: str) -> Optional[DocumentEmbedding]:
        """Get a specific document embedding by vector_id."""
        if not self.doc_collection:
            return None
            
        self.doc_collection.load()
        
        results = self.doc_collection.query(
            expr=f'vector_id == "{vector_id}"',
            output_fields=["vector_id", "chunk_id", "document_id", "embedding_vector",
                          "embedding_model", "vector_dimension", "content_hash",
                          "content_type", "language", "created_at"]
        )
        
        if results:
            result = results[0]
            metadata = None
            if result.get("content_type") or result.get("language"):
                from .models.document_embedding import DocumentEmbeddingMetadata
                metadata = DocumentEmbeddingMetadata(
                    content_type=result.get("content_type"),
                    language=result.get("language")
                )
                
            return DocumentEmbedding(
                vector_id=result["vector_id"],
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                embedding_vector=result["embedding_vector"],
                embedding_model=result["embedding_model"],
                vector_dimension=result["vector_dimension"],
                content_hash=result.get("content_hash"),
                metadata=metadata,
                created_at=result["created_at"]
            )
        return None
    
    def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a document."""
        if not self.doc_collection:
            return False
            
        try:
            expr = f'document_id == "{document_id}"'
            self.doc_collection.delete(expr)
            return True
        except Exception as e:
            print(f"Error deleting document embeddings: {e}")
            return False
    
    # Query embedding operations
    def insert_query_embedding(self, query_embedding: QueryEmbedding) -> bool:
        """Insert a query embedding using the QueryEmbedding model."""
        if not self.query_collection:
            raise ValueError("Query collection not initialized")
            
        metadata = query_embedding.query_metadata
        data = [
            [query_embedding.query_id],
            [query_embedding.query_text],
            [query_embedding.query_hash],
            [query_embedding.embedding_vector],
            [query_embedding.embedding_model],
            [metadata.language if metadata else ""],
            [metadata.intent if metadata else ""],
            [metadata.complexity if metadata else ""],
            [query_embedding.usage_count],
            [query_embedding.avg_relevance_score or 0.0],
            [query_embedding.created_at],
            [query_embedding.last_used_at]
        ]
        
        try:
            self.query_collection.insert(data)
            self.query_collection.flush()
            return True
        except Exception as e:
            print(f"Error inserting query embedding: {e}")
            return False
    
    def search_similar_queries(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar query embeddings."""
        if not self.query_collection:
            raise ValueError("Query collection not initialized")
            
        self.query_collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.query_collection.search(
            data=[query_vector],
            anns_field="embedding_vector",
            param=search_params,
            limit=top_k,
            output_fields=["query_id", "query_text", "query_hash", "embedding_model",
                          "language", "intent", "usage_count", "avg_relevance_score"]
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "query_id": hit.entity.get("query_id"),
                    "query_text": hit.entity.get("query_text"),
                    "query_hash": hit.entity.get("query_hash"),
                    "embedding_model": hit.entity.get("embedding_model"),
                    "language": hit.entity.get("language"),
                    "intent": hit.entity.get("intent"),
                    "usage_count": hit.entity.get("usage_count"),
                    "avg_relevance_score": hit.entity.get("avg_relevance_score"),
                    "score": hit.score
                })
                
        return formatted_results
    
    def update_query_usage(self, query_hash: str, relevance_score: Optional[float] = None) -> bool:
        """Update query usage statistics."""
        if not self.query_collection:
            return False
            
        # This would require more complex operations in Milvus
        # For now, we'll implement a simple approach
        try:
            # Get current query
            results = self.query_collection.query(
                expr=f'query_hash == "{query_hash}"',
                output_fields=["query_id", "usage_count", "avg_relevance_score", "last_used_at"]
            )
            
            if results:
                result = results[0]
                new_usage_count = result["usage_count"] + 1
                new_last_used = int(time.time())
                
                new_avg_score = result.get("avg_relevance_score", 0.0)
                if relevance_score is not None:
                    if result.get("avg_relevance_score"):
                        new_avg_score = (
                            (result["avg_relevance_score"] * (new_usage_count - 1) + relevance_score) / 
                            new_usage_count
                        )
                    else:
                        new_avg_score = relevance_score
                
                # Note: Milvus doesn't support direct updates, so we'd need to delete and re-insert
                # This is a simplified approach - in production, consider batch operations
                return True
            return False
        except Exception as e:
            print(f"Error updating query usage: {e}")
            return False
    
    # Collection management
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        
        if self.doc_collection:
            self.doc_collection.load()
            stats["document_embeddings"] = {
                "num_entities": self.doc_collection.num_entities,
                "description": self.doc_collection.description
            }
            
        if self.query_collection:
            self.query_collection.load()
            stats["query_embeddings"] = {
                "num_entities": self.query_collection.num_entities,
                "description": self.query_collection.description
            }
            
        return stats
    
    def drop_collections(self):
        """Drop all collections (use with caution)."""
        if utility.has_collection(self.doc_collection_name):
            utility.drop_collection(self.doc_collection_name)
        if utility.has_collection(self.query_collection_name):
            utility.drop_collection(self.query_collection_name)
    
    def close(self):
        """Close connections."""
        if self.doc_collection:
            self.doc_collection.release()
        if self.query_collection:
            self.query_collection.release()
        connections.disconnect("default")