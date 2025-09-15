"""
User model for SQLAlchemy.
"""
from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    """User model for relational database."""
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(String(36), primary_key=True, index=True, doc="UUID user identifier")
    
    # Basic user information
    username = Column(String(50), unique=True, nullable=False, index=True, doc="Unique username")
    email = Column(String(255), unique=True, nullable=False, index=True, doc="User email address")
    password_hash = Column(String(255), nullable=False, doc="Hashed password")
    salt = Column(String(32), nullable=False, doc="Password salt")
    
    # Optional user details
    first_name = Column(String(100), nullable=True, doc="User first name")
    last_name = Column(String(100), nullable=True, doc="User last name")
    
    # User role and status
    role = Column(
        String(20), 
        nullable=False, 
        default="user", 
        index=True,
        doc="User role (admin, user, guest)"
    )
    is_active = Column(Boolean, nullable=False, default=True, doc="Account active status")
    
    # User preferences and metadata
    preferences = Column(Text, nullable=True, doc="JSON user preferences")
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(),
        index=True,
        doc="Account creation timestamp"
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        doc="Last update timestamp"
    )
    last_login = Column(DateTime(timezone=True), nullable=True, doc="Last login timestamp")

    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}', email='{self.email}')>"