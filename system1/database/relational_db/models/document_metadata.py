"""
Document metadata model for SQLAlchemy.
"""
from sqlalchemy import Column, String, DateTime, Text, BigInteger, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DocumentMetadata(Base):
    """Document metadata model for relational database."""
    
    __tablename__ = "document_metadata"
    
    # Primary key
    id = Column(String(36), primary_key=True, index=True, doc="Document UUID")
    
    # Foreign key to users table
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Document owner"
    )
    
    # File information
    filename = Column(String(255), nullable=False, doc="Original filename")
    content_type = Column(String(100), nullable=False, doc="MIME content type")
    file_size = Column(BigInteger, nullable=False, doc="File size in bytes")
    file_hash = Column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        doc="SHA-256 file hash"
    )
    storage_path = Column(String(500), nullable=False, doc="Storage location path")
    
    # Processing information
    processing_status = Column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
        doc="Processing status (pending, processing, completed, failed, indexed)"
    )
    error_message = Column(Text, nullable=True, doc="Error message if processing failed")
    
    # Metadata
    metadata = Column(Text, nullable=True, doc="JSON document metadata")
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        doc="Upload timestamp"
    )
    processed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Processing completion time"
    )

    # Relationships
    user = relationship("User", back_populates="documents")

    def __repr__(self):
        return f"<DocumentMetadata(id='{self.id}', filename='{self.filename}', status='{self.processing_status}')>"