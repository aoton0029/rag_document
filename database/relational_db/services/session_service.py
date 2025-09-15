from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import secrets
import logging

from ..repositories.session_repository import SessionRepository
from ..repositories.user_repository import UserRepository
from ..models.session import Session

class SessionService:
    """Service layer for session management and authentication."""
    
    def __init__(self, db_type: Optional[str] = None):
        self.db_type = db_type
        self.session_repo = SessionRepository(db_type)
        self.user_repo = UserRepository(db_type)
        self.logger = logging.getLogger(__name__)
    
    # Session lifecycle
    def create_session(self, user_id: str, 
                      session_type: str = "web",
                      expires_hours: int = 24,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[Session]:
        """Create a new user session."""
        try:
            # Verify user exists
            user = self.user_repo.get_by_id(user_id, self.db_type)
            if not user:
                self.logger.warning(f"Cannot create session for non-existent user: {user_id}")
                return None
            
            # Generate secure token
            token = self._generate_session_token()
            
            # Create session
            session = self.session_repo.create_session(
                user_id=user_id,
                session_token=token,
                session_type=session_type,
                expires_hours=expires_hours,
                metadata_json=metadata,
                db_type=self.db_type
            )
            
            if session:
                self.logger.info(f"Session created for user {user_id}: {session.id}")
            
            return session
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[Session]:
        """Validate a session token and return session if valid."""
        try:
            session = self.session_repo.get_by_token(session_token, self.db_type)
            
            if not session:
                return None
            
            # Check if session is expired
            if session.expires_at and session.expires_at < datetime.utcnow():
                self.logger.debug(f"Session expired: {session.id}")
                # Optionally clean up expired session
                self.invalidate_session(session_token)
                return None
            
            # Update last accessed time
            self.session_repo.update_last_accessed(session.id, self.db_type)
            
            return session
        except Exception as e:
            self.logger.error(f"Error validating session: {e}")
            return None
    
    def refresh_session(self, session_token: str, 
                       expires_hours: int = 24) -> Optional[Session]:
        """Refresh a session's expiration time."""
        try:
            session = self.validate_session(session_token)
            if not session:
                return None
            
            new_expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
            
            # Update expiration
            filters = {"id": session.id}
            updates = {"expires_at": new_expires_at}
            
            updated = self.session_repo.update(filters, updates, self.db_type)
            if updated > 0:
                # Refetch updated session
                session = self.session_repo.get_by_id(session.id, self.db_type)
                self.logger.debug(f"Session refreshed: {session.id}")
            
            return session
        except Exception as e:
            self.logger.error(f"Error refreshing session: {e}")
            return None
    
    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session token."""
        try:
            session = self.session_repo.get_by_token(session_token, self.db_type)
            if not session:
                return True  # Already invalid
            
            # Soft delete by setting inactive
            filters = {"id": session.id}
            updates = {"is_active": False}
            
            updated = self.session_repo.update(filters, updates, self.db_type)
            
            if updated > 0:
                self.logger.info(f"Session invalidated: {session.id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error invalidating session: {e}")
            return False
    
    def invalidate_user_sessions(self, user_id: str, 
                                exclude_session_id: Optional[str] = None) -> int:
        """Invalidate all sessions for a user, optionally excluding one."""
        try:
            user_sessions = self.session_repo.get_user_active_sessions(
                user_id, self.db_type
            )
            
            invalidated = 0
            for session in user_sessions:
                if exclude_session_id and session.id == exclude_session_id:
                    continue
                
                if self.invalidate_session(session.session_token):
                    invalidated += 1
            
            self.logger.info(f"Invalidated {invalidated} sessions for user {user_id}")
            return invalidated
        except Exception as e:
            self.logger.error(f"Error invalidating user sessions: {e}")
            return 0
    
    # Session management
    def get_user_sessions(self, user_id: str, 
                         active_only: bool = True,
                         limit: int = 50) -> List[Session]:
        """Get sessions for a user."""
        try:
            if active_only:
                return self.session_repo.get_user_active_sessions(user_id, self.db_type)
            else:
                # Get all sessions with limit
                filters = {"user_id": user_id}
                return self.session_repo.get_all(
                    filters=filters, 
                    limit=limit,
                    order_by="created_at DESC",
                    db_type=self.db_type
                )
        except Exception as e:
            self.logger.error(f"Error getting user sessions: {e}")
            return []
    
    def get_session_details(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information."""
        try:
            session = self.session_repo.get_by_token(session_token, self.db_type)
            if not session:
                return None
            
            # Get user details
            user = self.user_repo.get_by_id(session.user_id, self.db_type)
            
            return {
                "session_id": session.id,
                "user_id": session.user_id,
                "username": user.username if user else None,
                "session_type": session.session_type,
                "created_at": session.created_at,
                "expires_at": session.expires_at,
                "last_accessed_at": session.last_accessed_at,
                "is_active": session.is_active,
                "metadata": session.metadata_json,
                "is_expired": session.expires_at < datetime.utcnow() if session.expires_at else False
            }
        except Exception as e:
            self.logger.error(f"Error getting session details: {e}")
            return None
    
    # Authentication flow
    def authenticate_user(self, username_or_email: str, password: str,
                         session_type: str = "web",
                         expires_hours: int = 24) -> Optional[Dict[str, Any]]:
        """Authenticate user and create session."""
        try:
            # Authenticate user
            user = self.user_repo.authenticate_user(
                username_or_email, password, self.db_type
            )
            
            if not user:
                self.logger.warning(f"Authentication failed for: {username_or_email}")
                return None
            
            # Create session
            session = self.create_session(
                user_id=user.id,
                session_type=session_type,
                expires_hours=expires_hours
            )
            
            if not session:
                self.logger.error(f"Failed to create session for user: {user.id}")
                return None
            
            return {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                    "is_active": user.is_active
                },
                "session": {
                    "token": session.session_token,
                    "expires_at": session.expires_at,
                    "session_type": session.session_type
                }
            }
        except Exception as e:
            self.logger.error(f"Error during authentication: {e}")
            return None
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session."""
        try:
            return self.invalidate_session(session_token)
        except Exception as e:
            self.logger.error(f"Error during logout: {e}")
            return False
    
    def logout_all_devices(self, session_token: str) -> int:
        """Logout user from all devices."""
        try:
            session = self.validate_session(session_token)
            if not session:
                return 0
            
            return self.invalidate_user_sessions(session.user_id)
        except Exception as e:
            self.logger.error(f"Error during logout all: {e}")
            return 0
    
    # Session analytics
    def get_session_statistics(self, user_id: Optional[str] = None,
                              hours: int = 24) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            return self.session_repo.get_session_statistics(
                user_id, hours, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            return {}
    
    def get_active_users_count(self) -> int:
        """Get count of users with active sessions."""
        try:
            return self.session_repo.get_active_users_count(self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting active users count: {e}")
            return 0
    
    def get_concurrent_sessions(self) -> List[Dict[str, Any]]:
        """Get information about concurrent sessions."""
        try:
            # Get all active sessions
            filters = {"is_active": True}
            active_sessions = self.session_repo.get_all(
                filters=filters,
                order_by="last_accessed_at DESC",
                db_type=self.db_type
            )
            
            # Group by user
            user_sessions = {}
            for session in active_sessions:
                if session.user_id not in user_sessions:
                    user_sessions[session.user_id] = []
                user_sessions[session.user_id].append(session)
            
            # Find users with multiple active sessions
            concurrent = []
            for user_id, sessions in user_sessions.items():
                if len(sessions) > 1:
                    user = self.user_repo.get_by_id(user_id, self.db_type)
                    concurrent.append({
                        "user_id": user_id,
                        "username": user.username if user else None,
                        "session_count": len(sessions),
                        "session_types": list(set(s.session_type for s in sessions)),
                        "latest_access": max(s.last_accessed_at for s in sessions if s.last_accessed_at)
                    })
            
            return sorted(concurrent, key=lambda x: x["session_count"], reverse=True)
        except Exception as e:
            self.logger.error(f"Error getting concurrent sessions: {e}")
            return []
    
    # Maintenance operations
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            count = self.session_repo.cleanup_expired_sessions(self.db_type)
            if count > 0:
                self.logger.info(f"Cleaned up {count} expired sessions")
            return count
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old inactive sessions."""
        try:
            count = self.session_repo.cleanup_old_sessions(days_old, self.db_type)
            if count > 0:
                self.logger.info(f"Cleaned up {count} old sessions")
            return count
        except Exception as e:
            self.logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def session_health_check(self) -> Dict[str, Any]:
        """Perform session system health check."""
        try:
            stats = self.get_session_statistics(hours=24)
            active_count = self.get_active_users_count()
            concurrent = len(self.get_concurrent_sessions())
            
            # Check for anomalies
            warnings = []
            if stats.get("total_sessions", 0) > 1000:
                warnings.append("High session creation rate")
            if concurrent > 50:
                warnings.append("High concurrent session count")
            
            return {
                "status": "healthy" if not warnings else "warning",
                "active_sessions": stats.get("active_sessions", 0),
                "active_users": active_count,
                "concurrent_users": concurrent,
                "session_creation_rate": stats.get("total_sessions", 0),
                "warnings": warnings,
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error in session health check: {e}")
            return {"status": "error", "error": str(e)}
    
    # Utility methods
    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        return secrets.token_urlsafe(32)
    
    def is_session_valid(self, session_token: str) -> bool:
        """Check if session token is valid without updating access time."""
        try:
            session = self.session_repo.get_by_token(session_token, self.db_type)
            
            if not session or not session.is_active:
                return False
            
            # Check expiration
            if session.expires_at and session.expires_at < datetime.utcnow():
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking session validity: {e}")
            return False
    
    def extend_session(self, session_token: str, 
                      additional_hours: int = 24) -> Optional[Session]:
        """Extend session expiration by additional hours."""
        try:
            session = self.validate_session(session_token)
            if not session:
                return None
            
            if session.expires_at:
                new_expires_at = session.expires_at + timedelta(hours=additional_hours)
            else:
                new_expires_at = datetime.utcnow() + timedelta(hours=additional_hours)
            
            # Update expiration
            filters = {"id": session.id}
            updates = {"expires_at": new_expires_at}
            
            updated = self.session_repo.update(filters, updates, self.db_type)
            if updated > 0:
                session = self.session_repo.get_by_id(session.id, self.db_type)
                self.logger.debug(f"Session extended: {session.id}")
            
            return session
        except Exception as e:
            self.logger.error(f"Error extending session: {e}")
            return None