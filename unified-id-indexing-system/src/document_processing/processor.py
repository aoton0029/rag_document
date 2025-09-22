from typing import Optional
import logging
from .ingest_service import DocumentIngestionService
from .preprocessor_service import DocumentPreprocessor
from .chunk_service import DocumentChunker
from ..db.database_manager import db_manager
from ..core.unified_id import UnifiedID

class IntegratedDocumentProcessor:
    """
    Integrated Document Processing Service
    Orchestrates ingestion, preprocessing, and chunking of documents.
    """
    
    def __init__(self, 
                 unified_id_generator: Optional[UnifiedID] = None):
        self.unified_id_generator = unified_id_generator or UnifiedID()
        self.ingestion_service = DocumentIngestionService()
        self.preprocessor = DocumentPreprocessor()
        self.chunker = DocumentChunker()

    def process_file(self, file_path: str):
        unified_id = self.unified_id_generator.generate()
        logging.info(f"Processing file with unified ID: {unified_id}")

        ingestion_result = self.ingestion_service.ingest_from_file_path(file_path=file_path)
        if not ingestion_result.success:
            logging.error(f"Ingestion failed: {ingestion_result.error_message}")
            return None
        
        preprocessing_result = self.preprocessor.preprocess(ingestion_result.documents)
        if not preprocessing_result.success:
            logging.error(f"Preprocessing failed: {preprocessing_result.error_message}")
            return None
        
        chunking_result = self.chunker.chunk_documents(preprocessing_result.documents)
        if not chunking_result.success:
            logging.error(f"Chunking failed: {chunking_result.error_message}")
            return None
        
        # Save results to the database using db_manager
        db_manager.save_documents(unified_id, ingestion_result.documents, preprocessing_result.documents, chunking_result.documents)

        return {
            "unified_id": unified_id,
            "ingestion": ingestion_result,
            "preprocessing": preprocessing_result,
            "chunking": chunking_result
        }