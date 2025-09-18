"""
Query log model for SQLAlchemy.
"""
from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class QueryLog(Base):
    """Query log model for relational database."""
    
    __tablename__ = "query_logs"
    
    # Primary key
    id = Column(String(36), primary_key=True, index=True, doc="Query log UUID")
    
    # Foreign keys
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="User who made the query"
    )
    session_id = Column(
        String(36),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Session identifier"
    )
    
    # Query information
    query_text = Column(Text, nullable=False, doc="Original query text")
    normalized_query = Column(Text, nullable=True, doc="Normalized query text")
    query_type = Column(
        String(20),
        nullable=False,
        index=True,
        doc="Type of query (search, chat, analysis, comparison)"
    )
    
    # Result information
    result_count = Column(Integer, nullable=False, default=0, doc="Number of results returned")
    response_time_ms = Column(
        Integer,
        nullable=False,
        index=True,
        doc="Response time in milliseconds"
    )
    success = Column(Boolean, nullable=False, default=True, doc="Query success status")
    error_message = Column(Text, nullable=True, doc="Error message if query failed")
    
    # Timestamp
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        doc="Query timestamp"
    )

    # Relationships
    user = relationship("User", back_populates="query_logs")
    session = relationship("Session", back_populates="query_logs")

    def __repr__(self):
        return f"<QueryLog(id='{self.id}', query_type='{self.query_type}', success={self.success})>"