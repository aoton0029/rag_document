import logging
from src.db.database_manager import db_manager
from src.indexing.index_registry import IndexRegistry
from src.core.unified_id import UnifiedID

logger = logging.getLogger(__name__)

def migrate_data():
    logger.info("Starting data migration...")

    # Initialize the index registry
    index_registry = IndexRegistry()

    # Fetch all documents from the source database
    source_documents = db_manager.mongo.get_all_documents()

    for document in source_documents:
        try:
            # Generate a unified ID for the document
            unified_id = UnifiedID.generate()

            # Migrate the document to the target database
            db_manager.milvus.insert_document(unified_id, document)

            # Update the index registry
            index_registry.update_index_status(unified_id, "migrated")

            logger.info(f"Document {document['_id']} migrated with unified ID {unified_id}")

        except Exception as e:
            logger.error(f"Failed to migrate document {document['_id']}: {e}")

    logger.info("Data migration completed.")

if __name__ == "__main__":
    migrate_data()