from typing import List, Dict, Any
from core.unified_id import UnifiedID
from db.database_manager import db_manager

class VectorSearch:
    """Handles vector-based search operations using unified IDs."""

    def __init__(self):
        self.db_manager = db_manager

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Perform a vector search using the provided query vector.

        Args:
            query_vector: The vector representation of the search query.
            top_k: The number of top results to return.

        Returns:
            A list of documents matching the search criteria.
        """
        # Perform the vector search in the vector store (e.g., Milvus)
        results = self.db_manager.vector_store.search(query_vector, top_k)
        
        # Map results to include unified IDs and other relevant information
        documents = []
        for result in results:
            unified_id = result.get('unified_id')
            document = self.db_manager.docstore.get_document_by_id(unified_id)
            documents.append({
                "unified_id": unified_id,
                "score": result.get('score'),
                "document": document
            })
        
        return documents

    def get_similar_documents(self, document_id: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents based on the given document ID.

        Args:
            document_id: The ID of the document to find similar documents for.
            top_k: The number of similar documents to return.

        Returns:
            A list of similar documents.
        """
        # Retrieve the document's vector from the database
        document = self.db_manager.docstore.get_document_by_id(document_id)
        if not document:
            return []

        # Perform a vector search using the document's vector
        return self.search(document['embedding'], top_k)