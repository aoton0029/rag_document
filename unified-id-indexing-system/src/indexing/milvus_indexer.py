from llama_index.vector_stores.milvus import MilvusVectorStore
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class MilvusIndexer:
    def __init__(self):
        self.vector_store = MilvusVectorStore(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_collection_name
        )

    def index_document(self, unified_id: str, embedding: list, metadata: dict):
        try:
            self.vector_store.insert(
                vectors=[embedding],
                metadatas=[metadata],
                ids=[unified_id]
            )
            logger.info(f"Document indexed successfully with unified ID: {unified_id}")
        except Exception as e:
            logger.error(f"Failed to index document with unified ID {unified_id}: {e}")
            raise

    def delete_document(self, unified_id: str):
        try:
            self.vector_store.delete(ids=[unified_id])
            logger.info(f"Document deleted successfully with unified ID: {unified_id}")
        except Exception as e:
            logger.error(f"Failed to delete document with unified ID {unified_id}: {e}")
            raise

    def update_document(self, unified_id: str, embedding: list, metadata: dict):
        self.delete_document(unified_id)
        self.index_document(unified_id, embedding, metadata)

    def search(self, query_embedding: list, top_k: int = 10):
        try:
            results = self.vector_store.query(
                query_vector=query_embedding,
                top_k=top_k
            )
            logger.info(f"Search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise