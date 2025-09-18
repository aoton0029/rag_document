"""
Database configuration and management for RAGShelf relational database.
"""
import os
from typing import Dict, Optional, Any, List, Union, Type
from contextlib import contextmanager
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Import models
from .models import User, Session as UserSession, DocumentMetadata, QueryLog

# Load environment variables
load_dotenv()

# Database type definitions
class DBType:
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"

# Database connection URL generation
def get_connection_url(db_type: str) -> str:
    """Generate connection URL based on database type."""
    if db_type == DBType.SQLITE:
        return os.getenv("SQLITE_DATABASE_URL", "sqlite:///./ragshelf.db")
    
    elif db_type == DBType.SQLSERVER:
        host = os.getenv("SQLSERVER_HOST", "localhost")
        port = os.getenv("SQLSERVER_PORT", "1433")
        user = os.getenv("SQLSERVER_USER", "sa")
        password = os.getenv("SQLSERVER_PASSWORD", "")
        database = os.getenv("SQLSERVER_DATABASE", "ragshelf")
        return f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

# Create declarative base from the first model (they should all use the same base)
Base = declarative_base()

# Update all models to use the same Base
User.__table__.metadata = Base.metadata
UserSession.__table__.metadata = Base.metadata  
DocumentMetadata.__table__.metadata = Base.metadata
QueryLog.__table__.metadata = Base.metadata

# Engine and session factory management
class DatabaseManager:
    """Database manager for handling multiple database connections."""
    
    def __init__(self):
        self.engines: Dict[str, Engine] = {}
        self.session_factories: Dict[str, scoped_session] = {}
        self.default_db = os.getenv("DEFAULT_DB", DBType.SQLITE)
        self.Base = Base

    def init_db(self, db_type: Optional[str] = None) -> None:
        """Initialize engine and session factory for specified database."""
        db_type = db_type or self.default_db
        
        if db_type not in self.engines:
            connection_url = get_connection_url(db_type)
            
            # Database-specific engine settings
            kwargs = {"pool_pre_ping": True}
            if db_type == DBType.SQLITE:
                kwargs["connect_args"] = {"check_same_thread": False}
            
            engine = create_engine(connection_url, **kwargs)
            self.engines[db_type] = engine
            
            session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.session_factories[db_type] = scoped_session(session_factory)

    def get_engine(self, db_type: Optional[str] = None) -> Engine:
        """Get engine for specified database."""
        db_type = db_type or self.default_db
        
        if db_type not in self.engines:
            self.init_db(db_type)
        
        return self.engines[db_type]

    def get_session(self, db_type: Optional[str] = None) -> scoped_session:
        """Get session for specified database."""
        db_type = db_type or self.default_db
        
        if db_type not in self.session_factories:
            self.init_db(db_type)
        
        return self.session_factories[db_type]

    def create_tables(self, db_type: Optional[str] = None) -> None:
        """Create tables in specified database."""
        db_type = db_type or self.default_db
        engine = self.get_engine(db_type)
        self.Base.metadata.create_all(bind=engine)

    def drop_tables(self, db_type: Optional[str] = None) -> None:
        """Drop all tables (use with caution)."""
        db_type = db_type or self.default_db
        engine = self.get_engine(db_type)
        self.Base.metadata.drop_all(bind=engine)

    def execute_raw_sql(self, sql: str, params: Dict = None, is_select: bool = True,
                        db_type: Optional[str] = None) -> Union[List[Dict], int]:
        """Execute raw SQL query helper function."""
        db_type = db_type or self.default_db
        engine = self.get_engine(db_type)
        
        with engine.connect() as conn:
            if isinstance(sql, str):
                sql = text(sql)
                
            result = conn.execute(sql, params or {})
            
            if is_select:
                return [dict(row) for row in result]
            else:
                conn.commit()
                return result.rowcount

    # Model-specific operations
    def create_user(self, user: User, db_type: Optional[str] = None) -> bool:
        """Create a new user."""
        try:
            with get_db(db_type) as session:
                session.add(user)
                session.commit()
                return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    def get_user_by_id(self, user_id: str, db_type: Optional[str] = None) -> Optional[User]:
        """Get user by ID."""
        try:
            with get_db(db_type) as session:
                return session.query(User).filter(User.id == user_id).first()
        except Exception as e:
            print(f"Error getting user: {e}")
            return None

    def get_user_by_username(self, username: str, db_type: Optional[str] = None) -> Optional[User]:
        """Get user by username."""
        try:
            with get_db(db_type) as session:
                return session.query(User).filter(User.username == username).first()
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None

    def get_user_by_email(self, email: str, db_type: Optional[str] = None) -> Optional[User]:
        """Get user by email."""
        try:
            with get_db(db_type) as session:
                return session.query(User).filter(User.email == email).first()
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None

    def create_document_metadata(self, doc_metadata: DocumentMetadata, 
                               db_type: Optional[str] = None) -> bool:
        """Create document metadata."""
        try:
            with get_db(db_type) as session:
                session.add(doc_metadata)
                session.commit()
                return True
        except Exception as e:
            print(f"Error creating document metadata: {e}")
            return False

    def get_document_metadata(self, document_id: str, 
                            db_type: Optional[str] = None) -> Optional[DocumentMetadata]:
        """Get document metadata by ID."""
        try:
            with get_db(db_type) as session:
                return session.query(DocumentMetadata).filter(
                    DocumentMetadata.id == document_id
                ).first()
        except Exception as e:
            print(f"Error getting document metadata: {e}")
            return None

    def get_documents_by_user(self, user_id: str, limit: int = 20,
                            db_type: Optional[str] = None) -> List[DocumentMetadata]:
        """Get documents by user ID."""
        try:
            with get_db(db_type) as session:
                return session.query(DocumentMetadata).filter(
                    DocumentMetadata.user_id == user_id
                ).limit(limit).all()
        except Exception as e:
            print(f"Error getting documents by user: {e}")
            return []

    def log_query(self, query_log: QueryLog, db_type: Optional[str] = None) -> bool:
        """Log a query."""
        try:
            with get_db(db_type) as session:
                session.add(query_log)
                session.commit()
                return True
        except Exception as e:
            print(f"Error logging query: {e}")
            return False

    def get_query_logs_by_user(self, user_id: str, limit: int = 50,
                             db_type: Optional[str] = None) -> List[QueryLog]:
        """Get query logs by user ID."""
        try:
            with get_db(db_type) as session:
                return session.query(QueryLog).filter(
                    QueryLog.user_id == user_id
                ).order_by(QueryLog.created_at.desc()).limit(limit).all()
        except Exception as e:
            print(f"Error getting query logs: {e}")
            return []

    def get_database_stats(self, db_type: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with get_db(db_type) as session:
                stats = {}
                
                # User statistics
                stats["users"] = {
                    "total": session.query(User).count(),
                    "active": session.query(User).filter(User.is_active == True).count()
                }
                
                # Document statistics
                stats["documents"] = {
                    "total": session.query(DocumentMetadata).count()
                }
                
                # Query statistics
                stats["queries"] = {
                    "total": session.query(QueryLog).count(),
                    "successful": session.query(QueryLog).filter(QueryLog.success == True).count()
                }
                
                return stats
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

# Singleton instance
db_manager = DatabaseManager()

# Database session context manager
@contextmanager
def get_db(db_type: Optional[str] = None) -> Session:
    """Context manager for getting database session."""
    db_type = db_type or db_manager.default_db
    session = db_manager.get_session(db_type)
    
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.remove()

# Convenience functions
def init_database(db_type: Optional[str] = None):
    """Initialize database tables."""
    db_manager.create_tables(db_type)

def get_db_session(db_type: Optional[str] = None):
    """Get database session (for dependency injection)."""
    return db_manager.get_session(db_type)