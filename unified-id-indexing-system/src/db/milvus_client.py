from llama_index.vector_stores.milvus import MilvusVectorStore
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self):
        self.vector_store = None

    def initialize_connection(self):
        """Initialize connection to the Milvus vector store."""
        try:
            self.vector_store = MilvusVectorStore(
                host=settings.milvus_host,
                port=settings.milvus_port,
                collection_name=settings.milvus_collection_name,
                index_config={
                    "index_type": settings.milvus_index_type,
                    "metric_type": settings.milvus_metric_type
                }
            )
            logger.info("Milvus connection initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus connection: {e}")
            raise

    def insert_vectors(self, vectors, metadata):
        """Insert vectors into the Milvus vector store."""
        try:
            if self.vector_store is None:
                self.initialize_connection()
            self.vector_store.insert(vectors, metadata)
            logger.info("Vectors inserted successfully.")
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise

    def search_vectors(self, query_vector, top_k):
        """Search for similar vectors in the Milvus vector store."""
        try:
            if self.vector_store is None:
                self.initialize_connection()
            results = self.vector_store.search(query_vector, top_k)
            logger.info("Search completed successfully.")
            return results
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    def close_connection(self):
        """Close the connection to the Milvus vector store."""
        if self.vector_store is not None:
            self.vector_store.close()
            logger.info("Milvus connection closed.")