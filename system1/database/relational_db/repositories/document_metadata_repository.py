from typing import Optional, List, Dict, Any
from sqlalchemy import text, func, and_, desc
from datetime import datetime, timedelta
import uuid

from ..models.document_metadata import DocumentMetadata
from .base_repository import BaseRepository

class DocumentMetadataRepository(BaseRepository[DocumentMetadata]):
    """Repository for DocumentMetadata model with document management operations."""
    
    def __init__(self, db_type: Optional[str] = None):
        super().__init__(DocumentMetadata, db_type)
    
    # Document management methods
    def create_document_metadata(self, filename: str, file_path: str, 
                                file_size: int, file_type: str,
                                content_hash: str = "",
                                user_id: Optional[str] = None,
                                chunk_count: int = 0,
                                metadata_json: Optional[Dict[str, Any]] = None,
                                db_type: Optional[str] = None) -> Optional[DocumentMetadata]:
        """Create document metadata record."""
        doc_data = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "file_path": file_path,
            "file_size": file_size,
            "file_type": file_type,
            "content_hash": content_hash,
            "user_id": user_id,
            "chunk_count": chunk_count,
            "processing_status": "pending",
            "metadata_json": metadata_json or {}
        }
        
        return self.create(doc_data, db_type)
    
    def get_by_filename(self, filename: str, user_id: Optional[str] = None,
                       db_type: Optional[str] = None) -> Optional[DocumentMetadata]:
        """Get document by filename."""
        filters = {"filename": filename}
        if user_id:
            filters["user_id"] = user_id
            
        result = self.find_by_filters(filters, limit=1, db_type=db_type)
        return result[0] if result else None
    
    def get_by_content_hash(self, content_hash: str, 
                           db_type: Optional[str] = None) -> Optional[DocumentMetadata]:
        """Get document by content hash."""
        result = self.find_by_filters({"content_hash": content_hash}, limit=1, db_type=db_type)
        return result[0] if result else None
    
    def get_user_documents(self, user_id: str, 
                          status_filter: Optional[str] = None,
                          file_type_filter: Optional[str] = None,
                          limit: int = 50, offset: int = 0,
                          db_type: Optional[str] = None) -> List[DocumentMetadata]:
        """Get documents for user with optional filters."""
        filters = {"user_id": user_id}
        
        if status_filter:
            filters["processing_status"] = status_filter
        
        if file_type_filter:
            filters["file_type"] = file_type_filter
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            offset=offset,
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    def get_documents_by_status(self, status: str, limit: int = 100,
                               db_type: Optional[str] = None) -> List[DocumentMetadata]:
        """Get documents by processing status."""
        return self.find_by_filters(
            {"processing_status": status}, 
            limit=limit, 
            order_by="created_at",
            db_type=db_type
        )
    
    def get_documents_by_type(self, file_type: str, limit: int = 100,
                             db_type: Optional[str] = None) -> List[DocumentMetadata]:
        """Get documents by file type."""
        return self.find_by_filters(
            {"file_type": file_type}, 
            limit=limit, 
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    def update_processing_status(self, document_id: str, status: str,
                                db_type: Optional[str] = None) -> bool:
        """Update document processing status."""
        result = self.update(document_id, {"processing_status": status}, db_type)
        return result is not None
    
    def update_chunk_count(self, document_id: str, chunk_count: int,
                          db_type: Optional[str] = None) -> bool:
        """Update document chunk count."""
        result = self.update(document_id, {"chunk_count": chunk_count}, db_type)
        return result is not None
    
    def update_metadata(self, document_id: str, metadata_json: Dict[str, Any],
                       db_type: Optional[str] = None) -> bool:
        """Update document metadata JSON."""
        result = self.update(document_id, {"metadata_json": metadata_json}, db_type)
        return result is not None
    
    def search_documents(self, search_term: str, user_id: Optional[str] = None,
                        limit: int = 50, db_type: Optional[str] = None) -> List[DocumentMetadata]:
        """Search documents by filename or metadata."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                query = session.query(DocumentMetadata)
                
                # Add filename search
                query = query.filter(
                    DocumentMetadata.filename.ilike(f"%{search_term}%")
                )
                
                # Filter by user if specified
                if user_id:
                    query = query.filter(DocumentMetadata.user_id == user_id)
                
                return query.order_by(desc(DocumentMetadata.created_at)).limit(limit).all()
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def get_recent_documents(self, days: int = 7, limit: int = 50,
                           db_type: Optional[str] = None) -> List[DocumentMetadata]:
        """Get recently created documents."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filters = {"created_at": {"gte": cutoff_date}}
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    def get_large_documents(self, min_size_mb: float = 10, limit: int = 50,
                           db_type: Optional[str] = None) -> List[DocumentMetadata]:
        """Get large documents above size threshold."""
        min_size_bytes = int(min_size_mb * 1024 * 1024)
        filters = {"file_size": {"gte": min_size_bytes}}
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            order_by="file_size", 
            order_desc=True,
            db_type=db_type
        )
    
    def get_document_statistics(self, user_id: Optional[str] = None,
                               db_type: Optional[str] = None) -> Dict[str, Any]:
        """Get document statistics."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                query = session.query(DocumentMetadata)
                
                if user_id:
                    query = query.filter(DocumentMetadata.user_id == user_id)
                
                total_documents = query.count()
                total_size = query.with_entities(func.sum(DocumentMetadata.file_size)).scalar() or 0
                total_chunks = query.with_entities(func.sum(DocumentMetadata.chunk_count)).scalar() or 0
                
                # Status distribution
                status_stats = session.query(
                    DocumentMetadata.processing_status,
                    func.count(DocumentMetadata.id)
                ).group_by(DocumentMetadata.processing_status)
                
                if user_id:
                    status_stats = status_stats.filter(DocumentMetadata.user_id == user_id)
                
                status_distribution = dict(status_stats.all())
                
                # File type distribution
                type_stats = session.query(
                    DocumentMetadata.file_type,
                    func.count(DocumentMetadata.id)
                ).group_by(DocumentMetadata.file_type)
                
                if user_id:
                    type_stats = type_stats.filter(DocumentMetadata.user_id == user_id)
                
                type_distribution = dict(type_stats.all())
                
                # Documents created in last 30 days
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                recent_query = query.filter(DocumentMetadata.created_at >= thirty_days_ago)
                recent_documents = recent_query.count()
                
                return {
                    "total_documents": total_documents,
                    "total_size_bytes": total_size,
                    "total_size_mb": total_size / (1024 * 1024) if total_size else 0,
                    "total_chunks": total_chunks,
                    "recent_documents_30_days": recent_documents,
                    "average_size_mb": (total_size / (1024 * 1024)) / total_documents if total_documents > 0 else 0,
                    "average_chunks_per_document": total_chunks / total_documents if total_documents > 0 else 0,
                    "status_distribution": status_distribution,
                    "file_type_distribution": type_distribution
                }
        except Exception as e:
            self.logger.error(f"Error getting document statistics: {e}")
            return {}
    
    def cleanup_failed_documents(self, days_old: int = 7, 
                                db_type: Optional[str] = None) -> int:
        """Clean up old failed documents."""
        try:
            from ..database import get_db
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            with get_db(db_type or self.db_type) as session:
                deleted = session.query(DocumentMetadata).filter(
                    and_(
                        DocumentMetadata.processing_status == "failed",
                        DocumentMetadata.created_at < cutoff_date
                    )
                ).delete(synchronize_session=False)
                session.commit()
                return deleted
        except Exception as e:
            self.logger.error(f"Error cleaning up failed documents: {e}")
            return 0