from db.database_manager import db_manager
from core.unified_id import UnifiedID
from core.correlation_id import CorrelationID
from core.global_sequence import GlobalSequence
import logging

logger = logging.getLogger(__name__)

class MongoIndexer:
    def __init__(self):
        self.db_manager = db_manager

    def index_document(self, document):
        unified_id = UnifiedID().generate()
        correlation_id = CorrelationID().generate()
        global_sequence = GlobalSequence().generate()

        try:
            # Prepare the document for indexing
            document['unified_id'] = unified_id
            document['correlation_id'] = correlation_id
            document['global_sequence'] = global_sequence

            # Index the document in MongoDB
            mongo_client = self.db_manager.mongo
            mongo_client.insert_document(document)

            logger.info(f"Document indexed successfully with unified_id: {unified_id}")

        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise

    def index_documents(self, documents):
        for document in documents:
            self.index_document(document)