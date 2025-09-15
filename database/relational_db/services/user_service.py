from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ..repositories.user_repository import UserRepository
from ..repositories.session_repository import SessionRepository
from ..models.user import User
from ..models.session import Session

class UserService:
    """Service layer for user management operations."""
    
    def __init__(self, db_type: Optional[str] = None):
        self.db_type = db_type
        self.user_repo = UserRepository(db_type)
        self.session_repo = SessionRepository(db_type)
        self.logger = logging.getLogger(__name__)
    
    # User management
    def create_user(self, username: str, email: str, password: str,
                   first_name: Optional[str] = None,
                   last_name: Optional[str] = None,
                   role: str = "user") -> Optional[User]:
        """Create new user account."""
        try:
            # Validate input
            if not username or not email or not password:
                self.logger.error("Missing required fields for user creation")
                return None
            
            if len(password) < 8:
                self.logger.error("Password too short")
                return None
            
            # Create user
            user = self.user_repo.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                role=role,
                db_type=self.db_type
            )
            
            if user:
                self.logger.info(f"User created successfully: {user.id}")
            
            return user
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return None
    
    def authenticate_user(self, identifier: str, password: str) -> Optional[User]:
        """Authenticate user by email/username and password."""
        try:
            # Get user by email or username
            user = self.user_repo.get_by_email_or_username(identifier, self.db_type)
            
            if not user:
                self.logger.warning(f"User not found: {identifier}")
                return None
            
            if not user.is_active:
                self.logger.warning(f"User account inactive: {identifier}")
                return None
            
            # Verify password
            if self.user_repo.verify_password(user, password):
                self.logger.info(f"User authenticated: {user.id}")
                return user
            else:
                self.logger.warning(f"Invalid password for user: {identifier}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error authenticating user: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.user_repo.get_by_id(user_id, self.db_type)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = self.user_repo.get_by_email(email, self.db_type)
        return result[0] if result else None
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile information."""
        try:
            # Remove sensitive fields that shouldn't be updated this way
            safe_updates = {k: v for k, v in updates.items() 
                          if k not in ['id', 'password_hash', 'salt', 'created_at']}
            
            result = self.user_repo.update(user_id, safe_updates, self.db_type)
            if result:
                self.logger.info(f"User profile updated: {user_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            return False
    
    def change_password(self, user_id: str, current_password: str, 
                       new_password: str) -> bool:
        """Change user password."""
        try:
            # Get current user
            user = self.get_user_by_id(user_id)
            if not user:
                return False
            
            # Verify current password
            if not self.user_repo.verify_password(user, current_password):
                self.logger.warning(f"Invalid current password for user: {user_id}")
                return False
            
            # Validate new password
            if len(new_password) < 8:
                self.logger.error("New password too short")
                return False
            
            # Update password
            success = self.user_repo.update_password(user_id, new_password, self.db_type)
            if success:
                # Invalidate all user sessions for security
                self.session_repo.invalidate_user_sessions(user_id, self.db_type)
                self.logger.info(f"Password changed for user: {user_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error changing password: {e}")
            return False
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        try:
            success = self.user_repo.deactivate_user(user_id, self.db_type)
            if success:
                # Invalidate all user sessions
                self.session_repo.invalidate_user_sessions(user_id, self.db_type)
                self.logger.info(f"User deactivated: {user_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error deactivating user: {e}")
            return False
    
    def activate_user(self, user_id: str) -> bool:
        """Activate user account."""
        try:
            success = self.user_repo.activate_user(user_id, self.db_type)
            if success:
                self.logger.info(f"User activated: {user_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error activating user: {e}")
            return False
    
    # User preferences
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            return self.user_repo.update_preferences(user_id, preferences, self.db_type)
        except Exception as e:
            self.logger.error(f"Error updating preferences: {e}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        try:
            return self.user_repo.get_preferences(user_id, self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting preferences: {e}")
            return {}
    
    # Session management
    def create_user_session(self, user_id: str, 
                           session_data: Optional[Dict[str, Any]] = None,
                           expires_in_hours: int = 24) -> Optional[Session]:
        """Create new user session."""
        try:
            # Verify user exists and is active
            user = self.get_user_by_id(user_id)
            if not user or not user.is_active:
                return None
            
            session = self.session_repo.create_session(
                user_id=user_id,
                session_data=session_data,
                expires_in_hours=expires_in_hours,
                db_type=self.db_type
            )
            
            if session:
                self.logger.info(f"Session created for user: {user_id}")
            
            return session
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return None
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user."""
        try:
            session = self.session_repo.get_active_session(session_id, self.db_type)
            if not session:
                return None
            
            user = self.get_user_by_id(session.user_id)
            if not user or not user.is_active:
                # Invalidate session if user is inactive
                self.session_repo.invalidate_session(session_id, self.db_type)
                return None
            
            return user
        except Exception as e:
            self.logger.error(f"Error validating session: {e}")
            return None
    
    def logout_user(self, session_id: str) -> bool:
        """Logout user by invalidating session."""
        try:
            return self.session_repo.invalidate_session(session_id, self.db_type)
        except Exception as e:
            self.logger.error(f"Error logging out user: {e}")
            return False
    
    # User search and management
    def search_users(self, search_term: str, active_only: bool = True,
                    limit: int = 50) -> List[User]:
        """Search users."""
        try:
            return self.user_repo.search_users(search_term, active_only, limit, self.db_type)
        except Exception as e:
            self.logger.error(f"Error searching users: {e}")
            return []
    
    def get_users_by_role(self, role: str, active_only: bool = True) -> List[User]:
        """Get users by role."""
        try:
            return self.user_repo.get_users_by_role(role, active_only, self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting users by role: {e}")
            return []
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics."""
        try:
            return self.user_repo.get_user_statistics(self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting user statistics: {e}")
            return {}
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            return self.session_repo.get_session_statistics(self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            return {}
    
    # Admin operations
    def bulk_activate_users(self, user_ids: List[str]) -> int:
        """Bulk activate users (admin operation)."""
        try:
            return self.user_repo.bulk_activate_users(user_ids, self.db_type)
        except Exception as e:
            self.logger.error(f"Error bulk activating users: {e}")
            return 0
    
    def bulk_deactivate_users(self, user_ids: List[str]) -> int:
        """Bulk deactivate users (admin operation)."""
        try:
            return self.user_repo.bulk_deactivate_users(user_ids, self.db_type)
        except Exception as e:
            self.logger.error(f"Error bulk deactivating users: {e}")
            return 0
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            return self.session_repo.cleanup_expired_sessions(self.db_type)
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {e}")
            return 0