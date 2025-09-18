"""
Session model for SQLAlchemy.
"""
from sqlalchemy import Column, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Session(Base):
    """Session model for relational database."""
    
    __tablename__ = "sessions"
    
    # Primary key
    id = Column(String(36), primary_key=True, index=True, doc="Session UUID")
    
    # Foreign key to users table
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="User identifier"
    )
    
    # Session information
    token_hash = Column(String(255), nullable=False, doc="Hashed session token")
    ip_address = Column(String(45), nullable=True, doc="IP address (IPv4/IPv6)")
    user_agent = Column(Text, nullable=True, doc="User agent string")
    
    # Session status and metadata
    is_active = Column(Boolean, nullable=False, default=True, doc="Session active status")
    device_info = Column(Text, nullable=True, doc="JSON device information")
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        doc="Session creation timestamp"
    )
    last_accessed = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        index=True,
        doc="Last access timestamp"
    )
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Session expiration timestamp"
    )

    # Relationships
    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<Session(id='{self.id}', user_id='{self.user_id}', is_active={self.is_active})>"