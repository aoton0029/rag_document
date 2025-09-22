from redis import Redis
import logging

logger = logging.getLogger(__name__)

class RedisIndexer:
    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    def index_document(self, unified_id: str, document_data: dict) -> bool:
        try:
            self.redis_client.hmset(unified_id, document_data)
            logger.info(f"Document indexed successfully with unified ID: {unified_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to index document with unified ID {unified_id}: {e}")
            return False

    def get_document(self, unified_id: str) -> dict:
        try:
            document = self.redis_client.hgetall(unified_id)
            logger.info(f"Document retrieved successfully with unified ID: {unified_id}")
            return document
        except Exception as e:
            logger.error(f"Failed to retrieve document with unified ID {unified_id}: {e}")
            return {}

    def delete_document(self, unified_id: str) -> bool:
        try:
            self.redis_client.delete(unified_id)
            logger.info(f"Document deleted successfully with unified ID: {unified_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document with unified ID {unified_id}: {e}")
            return False

    def index_documents(self, documents: list) -> None:
        for doc in documents:
            unified_id = doc.get('unified_id')
            self.index_document(unified_id, doc)