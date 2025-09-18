from typing import Generic, TypeVar, Type, List, Optional, Any, Dict, Union
from sqlalchemy import text, desc, asc, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import logging

from ..database import get_db
from ..sql_loader import sql_loader

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository class providing common CRUD operations for all models."""
    
    def __init__(self, model: Type[T], db_type: Optional[str] = None):
        self.model = model
        self.db_type = db_type
        self.logger = logging.getLogger(f"{__name__}.{model.__name__}Repository")
    
    # Core CRUD Operations
    def get_by_id(self, id: Any, db_type: Optional[str] = None) -> Optional[T]:
        """Get entity by ID."""
        try:
            with get_db(db_type or self.db_type) as session:
                return session.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting {self.model.__name__} by id {id}: {e}")
            return None
    
    def get_all(self, limit: Optional[int] = None, offset: int = 0, 
                db_type: Optional[str] = None) -> List[T]:
        """Get all entities with optional pagination."""
        try:
            with get_db(db_type or self.db_type) as session:
                query = session.query(self.model)
                if offset > 0:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)
                return query.all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting all {self.model.__name__}: {e}")
            return []
    
    def create(self, obj_in: Dict[str, Any], db_type: Optional[str] = None) -> Optional[T]:
        """Create new entity."""
        try:
            with get_db(db_type or self.db_type) as session:
                # Add created_at if model has it
                if hasattr(self.model, 'created_at') and 'created_at' not in obj_in:
                    obj_in['created_at'] = datetime.utcnow()
                
                # Add updated_at if model has it
                if hasattr(self.model, 'updated_at') and 'updated_at' not in obj_in:
                    obj_in['updated_at'] = datetime.utcnow()
                
                obj = self.model(**obj_in)
                session.add(obj)
                session.commit()
                session.refresh(obj)
                return obj
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating {self.model.__name__}: {e}")
            return None
    
    def update(self, id: Any, obj_in: Dict[str, Any], 
               db_type: Optional[str] = None) -> Optional[T]:
        """Update entity by ID."""
        try:
            with get_db(db_type or self.db_type) as session:
                obj = session.query(self.model).filter(self.model.id == id).first()
                if obj:
                    # Update updated_at if model has it
                    if hasattr(self.model, 'updated_at'):
                        obj_in['updated_at'] = datetime.utcnow()
                    
                    for key, value in obj_in.items():
                        if hasattr(obj, key):
                            setattr(obj, key, value)
                    session.commit()
                    session.refresh(obj)
                return obj
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating {self.model.__name__} {id}: {e}")
            return None
    
    def delete(self, id: Any, db_type: Optional[str] = None) -> bool:
        """Delete entity by ID."""
        try:
            with get_db(db_type or self.db_type) as session:
                obj = session.query(self.model).filter(self.model.id == id).first()
                if obj:
                    session.delete(obj)
                    session.commit()
                    return True
                return False
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting {self.model.__name__} {id}: {e}")
            return False
    
    def exists(self, id: Any, db_type: Optional[str] = None) -> bool:
        """Check if entity exists by ID."""
        try:
            with get_db(db_type or self.db_type) as session:
                return session.query(self.model.id).filter(self.model.id == id).first() is not None
        except SQLAlchemyError as e:
            self.logger.error(f"Error checking existence of {self.model.__name__} {id}: {e}")
            return False
    
    def count(self, filters: Optional[Dict[str, Any]] = None, 
              db_type: Optional[str] = None) -> int:
        """Count entities with optional filters."""
        try:
            with get_db(db_type or self.db_type) as session:
                query = session.query(self.model)
                if filters:
                    query = self._apply_filters(query, filters)
                return query.count()
        except SQLAlchemyError as e:
            self.logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0
    
    def find_by_filters(self, filters: Dict[str, Any], 
                       limit: Optional[int] = None, 
                       offset: int = 0,
                       order_by: Optional[str] = None,
                       order_desc: bool = False,
                       db_type: Optional[str] = None) -> List[T]:
        """Find entities by filters with optional ordering and pagination."""
        try:
            with get_db(db_type or self.db_type) as session:
                query = session.query(self.model)
                query = self._apply_filters(query, filters)
                
                # Apply ordering
                if order_by and hasattr(self.model, order_by):
                    column = getattr(self.model, order_by)
                    query = query.order_by(desc(column) if order_desc else asc(column))
                
                # Apply pagination
                if offset > 0:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)
                
                return query.all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error finding {self.model.__name__} with filters: {e}")
            return []
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to query."""
        for key, value in filters.items():
            if hasattr(self.model, key):
                column = getattr(self.model, key)
                if isinstance(value, list):
                    query = query.filter(column.in_(value))
                elif isinstance(value, dict):
                    # Handle range filters like {'gte': 10, 'lte': 20}
                    if 'gte' in value:
                        query = query.filter(column >= value['gte'])
                    if 'lte' in value:
                        query = query.filter(column <= value['lte'])
                    if 'gt' in value:
                        query = query.filter(column > value['gt'])
                    if 'lt' in value:
                        query = query.filter(column < value['lt'])
                    if 'like' in value:
                        query = query.filter(column.like(f"%{value['like']}%"))
                else:
                    query = query.filter(column == value)
        return query
    
    # Raw SQL execution methods
    def execute_named_query(self, query_name: str, params: Dict[str, Any] = None,
                           db_type: Optional[str] = None) -> List[Dict]:
        """Execute named SQL query from loader."""
        try:
            query = sql_loader.get_query(query_name, db_type or self.db_type)
            return self.execute_raw_query(query, params, db_type)
        except Exception as e:
            self.logger.error(f"Error executing named query {query_name}: {e}")
            return []
    
    def execute_raw_query(self, query: str, params: Dict[str, Any] = None,
                         db_type: Optional[str] = None) -> List[Dict]:
        """Execute raw SQL query."""
        try:
            with get_db(db_type or self.db_type) as session:
                result = session.execute(text(query), params or {})
                return [dict(row) for row in result]
        except SQLAlchemyError as e:
            self.logger.error(f"Error executing raw query: {e}")
            return []
        return False
    
    # 直接SQLを実行するメソッド
    def execute_sql(self, sql: str, params: Dict = None, is_select: bool = True) -> Union[List[Dict], int]:
        """セッション内で直接SQLを実行"""
        if isinstance(sql, str):
            sql = text(sql)
        
        if is_select:
            result = self.db_session.execute(sql, params or {})
            return [dict(row) for row in result]
        else:
            result = self.db_session.execute(sql, params or {})
            self.db_session.commit()
            return result.rowcount
    
    def execute_named_query(self, query_name: str, params: Dict = None, is_select: bool = True) -> Union[List[Dict], int]:
        """事前に定義された名前付きクエリを実行"""
        sql = sql_loader.get_query(query_name, self.db_type)
        if not sql:
            raise ValueError(f"Named query '{query_name}' not found for DB type '{self.db_type}'")
        
        return self.execute_sql(sql, params, is_select)