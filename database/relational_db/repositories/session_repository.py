from typing import Optional, List, Dict, Any
from sqlalchemy import text, func, and_
from datetime import datetime, timedelta
import uuid

from ..models.session import Session
from .base_repository import BaseRepository

class SessionRepository(BaseRepository[Session]):
    """Repository for Session model with session management operations."""
    
    def __init__(self, db_type: Optional[str] = None):
        super().__init__(Session, db_type)
    
    # Session management methods
    def create_session(self, user_id: str, 
                      session_data: Optional[Dict[str, Any]] = None,
                      expires_in_hours: int = 24,
                      db_type: Optional[str] = None) -> Optional[Session]:
        """Create new user session."""
        session_data = session_data or {}
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        session_info = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "session_data": session_data,
            "expires_at": expires_at,
            "is_active": True
        }
        
        return self.create(session_info, db_type)
    
    def get_active_session(self, session_id: str, 
                          db_type: Optional[str] = None) -> Optional[Session]:
        """Get active session by ID."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                return session.query(Session).filter(
                    and_(
                        Session.id == session_id,
                        Session.is_active == True,
                        Session.expires_at > datetime.utcnow()
                    )
                ).first()
        except Exception as e:
            self.logger.error(f"Error getting active session {session_id}: {e}")
            return None
    
    def get_user_sessions(self, user_id: str, active_only: bool = True,
                         limit: int = 50, db_type: Optional[str] = None) -> List[Session]:
        """Get sessions for user."""
        filters = {"user_id": user_id}
        if active_only:
            filters["is_active"] = True
            filters["expires_at"] = {"gt": datetime.utcnow()}
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    def update_session_data(self, session_id: str, 
                           session_data: Dict[str, Any],
                           db_type: Optional[str] = None) -> bool:
        """Update session data."""
        result = self.update(session_id, {"session_data": session_data}, db_type)
        return result is not None
    
    def extend_session(self, session_id: str, hours: int = 24,
                      db_type: Optional[str] = None) -> bool:
        """Extend session expiry time."""
        new_expires_at = datetime.utcnow() + timedelta(hours=hours)
        result = self.update(session_id, {"expires_at": new_expires_at}, db_type)
        return result is not None
    
    def invalidate_session(self, session_id: str, 
                          db_type: Optional[str] = None) -> bool:
        """Invalidate (deactivate) session."""
        result = self.update(session_id, {"is_active": False}, db_type)
        return result is not None
    
    def invalidate_user_sessions(self, user_id: str, 
                                db_type: Optional[str] = None) -> int:
        """Invalidate all sessions for user."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                updated = session.query(Session).filter(
                    Session.user_id == user_id
                ).update(
                    {"is_active": False, "updated_at": datetime.utcnow()},
                    synchronize_session=False
                )
                session.commit()
                return updated
        except Exception as e:
            self.logger.error(f"Error invalidating user sessions for {user_id}: {e}")
            return 0
    
    def cleanup_expired_sessions(self, db_type: Optional[str] = None) -> int:
        """Remove expired sessions."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                deleted = session.query(Session).filter(
                    Session.expires_at < datetime.utcnow()
                ).delete(synchronize_session=False)
                session.commit()
                return deleted
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def get_session_statistics(self, db_type: Optional[str] = None) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                total_sessions = session.query(Session).count()
                active_sessions = session.query(Session).filter(
                    and_(
                        Session.is_active == True,
                        Session.expires_at > datetime.utcnow()
                    )
                ).count()
                
                # Sessions created today
                today = datetime.utcnow().date()
                sessions_today = session.query(Session).filter(
                    func.date(Session.created_at) == today
                ).count()
                
                # Average session duration
                avg_duration = session.query(
                    func.avg(
                        func.extract('epoch', Session.expires_at) - 
                        func.extract('epoch', Session.created_at)
                    )
                ).scalar()
                
                return {
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "expired_sessions": total_sessions - active_sessions,
                    "sessions_created_today": sessions_today,
                    "average_duration_hours": (avg_duration / 3600) if avg_duration else 0
                }
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            return {}