from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ..repositories.document_metadata_repository import DocumentMetadataRepository
from ..models.document_metadata import DocumentMetadata

class DocumentService:
    """Service layer for document management operations."""
    
    def __init__(self, db_type: Optional[str] = None):
        self.db_type = db_type
        self.document_repo = DocumentMetadataRepository(db_type)
        self.logger = logging.getLogger(__name__)
    
    # Document management
    def register_document(self, filename: str, file_path: str, 
                         file_size: int, file_type: str,
                         content_hash: str = "",
                         user_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[DocumentMetadata]:
        """Register a new document in the system."""
        try:
            # Check if document already exists by hash
            if content_hash:
                existing = self.document_repo.get_by_content_hash(content_hash, self.db_type)
                if existing:
                    self.logger.info(f"Document already exists with hash: {content_hash}")
                    return existing
            
            # Create document metadata
            document = self.document_repo.create_document_metadata(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                content_hash=content_hash,
                user_id=user_id,
                metadata_json=metadata,
                db_type=self.db_type
            )
            
            if document:
                self.logger.info(f"Document registered: {document.id}")
            
            return document
        except Exception as e:
            self.logger.error(f"Error registering document: {e}")
            return None
    
    def get_document_by_id(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document by ID."""
        return self.document_repo.get_by_id(document_id, self.db_type)
    
    def get_user_documents(self, user_id: str, 
                          status_filter: Optional[str] = None,
                          file_type_filter: Optional[str] = None,
                          limit: int = 50, offset: int = 0) -> List[DocumentMetadata]:
        """Get documents for a user."""
        try:
            return self.document_repo.get_user_documents(
                user_id=user_id,
                status_filter=status_filter,
                file_type_filter=file_type_filter,
                limit=limit,
                offset=offset,
                db_type=self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting user documents: {e}")
            return []
    
    def update_processing_status(self, document_id: str, status: str) -> bool:
        """Update document processing status."""
        try:
            valid_statuses = ["pending", "processing", "completed", "failed"]
            if status not in valid_statuses:
                self.logger.error(f"Invalid status: {status}")
                return False
            
            success = self.document_repo.update_processing_status(
                document_id, status, self.db_type
            )
            
            if success:
                self.logger.info(f"Document {document_id} status updated to {status}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error updating document status: {e}")
            return False
    
    def update_chunk_count(self, document_id: str, chunk_count: int) -> bool:
        """Update document chunk count after processing."""
        try:
            return self.document_repo.update_chunk_count(
                document_id, chunk_count, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error updating chunk count: {e}")
            return False
    
    def update_document_metadata(self, document_id: str, 
                               metadata: Dict[str, Any]) -> bool:
        """Update document metadata."""
        try:
            return self.document_repo.update_metadata(
                document_id, metadata, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error updating document metadata: {e}")
            return False
    
    def search_documents(self, search_term: str, user_id: Optional[str] = None,
                        limit: int = 50) -> List[DocumentMetadata]:
        """Search documents by filename or metadata."""
        try:
            return self.document_repo.search_documents(
                search_term=search_term,
                user_id=user_id,
                limit=limit,
                db_type=self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def get_documents_by_status(self, status: str, limit: int = 100) -> List[DocumentMetadata]:
        """Get documents by processing status."""
        try:
            return self.document_repo.get_documents_by_status(
                status, limit, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting documents by status: {e}")
            return []
    
    def get_documents_by_type(self, file_type: str, limit: int = 100) -> List[DocumentMetadata]:
        """Get documents by file type."""
        try:
            return self.document_repo.get_documents_by_type(
                file_type, limit, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting documents by type: {e}")
            return []
    
    def get_recent_documents(self, days: int = 7, limit: int = 50) -> List[DocumentMetadata]:
        """Get recently uploaded documents."""
        try:
            return self.document_repo.get_recent_documents(
                days, limit, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting recent documents: {e}")
            return []
    
    def get_large_documents(self, min_size_mb: float = 10, 
                           limit: int = 50) -> List[DocumentMetadata]:
        """Get large documents above size threshold."""
        try:
            return self.document_repo.get_large_documents(
                min_size_mb, limit, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting large documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document metadata."""
        try:
            success = self.document_repo.delete(document_id, self.db_type)
            if success:
                self.logger.info(f"Document deleted: {document_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False
    
    # Statistics and analytics
    def get_document_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get document statistics."""
        try:
            return self.document_repo.get_document_statistics(user_id, self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting document statistics: {e}")
            return {}
    
    def get_processing_queue_status(self) -> Dict[str, Any]:
        """Get processing queue status."""
        try:
            pending = len(self.get_documents_by_status("pending"))
            processing = len(self.get_documents_by_status("processing"))
            failed = len(self.get_documents_by_status("failed"))
            
            return {
                "pending_count": pending,
                "processing_count": processing,
                "failed_count": failed,
                "total_in_queue": pending + processing + failed
            }
        except Exception as e:
            self.logger.error(f"Error getting queue status: {e}")
            return {}
    
    # Maintenance operations
    def cleanup_failed_documents(self, days_old: int = 7) -> int:
        """Clean up old failed documents."""
        try:
            count = self.document_repo.cleanup_failed_documents(days_old, self.db_type)
            if count > 0:
                self.logger.info(f"Cleaned up {count} failed documents")
            return count
        except Exception as e:
            self.logger.error(f"Error cleaning up failed documents: {e}")
            return 0
    
    def retry_failed_documents(self, limit: int = 10) -> List[DocumentMetadata]:
        """Get failed documents to retry processing."""
        try:
            failed_docs = self.get_documents_by_status("failed", limit)
            
            # Reset status to pending for retry
            for doc in failed_docs:
                self.update_processing_status(doc.id, "pending")
            
            return failed_docs
        except Exception as e:
            self.logger.error(f"Error retrying failed documents: {e}")
            return []
    
    # Batch operations
    def bulk_update_status(self, document_ids: List[str], status: str) -> int:
        """Bulk update document status."""
        try:
            count = 0
            for doc_id in document_ids:
                if self.update_processing_status(doc_id, status):
                    count += 1
            return count
        except Exception as e:
            self.logger.error(f"Error bulk updating status: {e}")
            return 0
    
    def get_documents_needing_processing(self, limit: int = 50) -> List[DocumentMetadata]:
        """Get documents that need processing."""
        try:
            # Get pending documents
            pending = self.get_documents_by_status("pending", limit)
            
            # If not enough pending, also get failed documents for retry
            if len(pending) < limit:
                remaining = limit - len(pending)
                failed = self.get_documents_by_status("failed", remaining)
                pending.extend(failed)
            
            return pending
        except Exception as e:
            self.logger.error(f"Error getting documents needing processing: {e}")
            return []