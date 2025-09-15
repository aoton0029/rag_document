from typing import Optional, List, Dict, Any
from sqlalchemy import text, func, or_
from datetime import datetime, timedelta
import hashlib
import uuid
import secrets

from ..models.user import User
from .base_repository import BaseRepository

class UserRepository(BaseRepository[User]):
    """Repository for User model with specific user management operations."""
    
    def __init__(self, db_type: Optional[str] = None):
        super().__init__(User, db_type)
    
    # Authentication and user lookup methods
    def get_by_email(self, email: str, db_type: Optional[str] = None) -> Optional[User]:
        """Get user by email address."""
        return self.find_by_filters({"email": email}, limit=1, db_type=db_type)
    
    def get_by_username(self, username: str, db_type: Optional[str] = None) -> Optional[User]:
        """Get user by username."""
        result = self.find_by_filters({"username": username}, limit=1, db_type=db_type)
        return result[0] if result else None
    
    def get_by_email_or_username(self, identifier: str, 
                                db_type: Optional[str] = None) -> Optional[User]:
        """Get user by email or username."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                return session.query(User).filter(
                    or_(User.email == identifier, User.username == identifier)
                ).first()
        except Exception as e:
            self.logger.error(f"Error getting user by identifier {identifier}: {e}")
            return None
    
    def verify_password(self, user: User, password: str) -> bool:
        """Verify user password."""
        if not user or not user.password_hash or not user.salt:
            return False
        
        # Hash the provided password with user's salt
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            user.salt.encode('utf-8'),
            100000
        ).hex()
        
        return password_hash == user.password_hash
    
    def create_user(self, username: str, email: str, password: str,
                   first_name: Optional[str] = None,
                   last_name: Optional[str] = None,
                   role: str = "user",
                   db_type: Optional[str] = None) -> Optional[User]:
        """Create new user with hashed password."""
        # Check if user already exists
        if self.get_by_email(email, db_type) or self.get_by_username(username, db_type):
            self.logger.warning(f"User with email {email} or username {username} already exists")
            return None
        
        # Generate salt and hash password
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        user_data = {
            "id": str(uuid.uuid4()),
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "salt": salt,
            "first_name": first_name,
            "last_name": last_name,
            "role": role,
            "is_active": True,
            "preferences": None
        }
        
        return self.create(user_data, db_type)
    
    def update_password(self, user_id: str, new_password: str,
                       db_type: Optional[str] = None) -> bool:
        """Update user password."""
        # Generate new salt and hash
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            new_password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        result = self.update(user_id, {
            "password_hash": password_hash,
            "salt": salt
        }, db_type)
        
        return result is not None
    
    def activate_user(self, user_id: str, db_type: Optional[str] = None) -> bool:
        """Activate user account."""
        result = self.update(user_id, {"is_active": True}, db_type)
        return result is not None
    
    def deactivate_user(self, user_id: str, db_type: Optional[str] = None) -> bool:
        """Deactivate user account."""
        result = self.update(user_id, {"is_active": False}, db_type)
        return result is not None
    
    def update_preferences(self, user_id: str, preferences: Dict[str, Any],
                          db_type: Optional[str] = None) -> bool:
        """Update user preferences."""
        import json
        result = self.update(user_id, {
            "preferences": json.dumps(preferences)
        }, db_type)
        return result is not None
    
    def get_preferences(self, user_id: str, db_type: Optional[str] = None) -> Dict[str, Any]:
        """Get user preferences."""
        user = self.get_by_id(user_id, db_type)
        if user and user.preferences:
            import json
            try:
                return json.loads(user.preferences)
            except json.JSONDecodeError:
                return {}
        return {}
    
    # User search and filtering
    def search_users(self, search_term: str, active_only: bool = True,
                    limit: int = 50, db_type: Optional[str] = None) -> List[User]:
        """Search users by username, email, first_name, or last_name."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                query = session.query(User)
                
                # Add search filters
                search_filter = or_(
                    User.username.ilike(f"%{search_term}%"),
                    User.email.ilike(f"%{search_term}%"),
                    User.first_name.ilike(f"%{search_term}%"),
                    User.last_name.ilike(f"%{search_term}%")
                )
                query = query.filter(search_filter)
                
                # Filter by active status
                if active_only:
                    query = query.filter(User.is_active == True)
                
                return query.limit(limit).all()
        except Exception as e:
            self.logger.error(f"Error searching users: {e}")
            return []
    
    def get_users_by_role(self, role: str, active_only: bool = True,
                         db_type: Optional[str] = None) -> List[User]:
        """Get users by role."""
        filters = {"role": role}
        if active_only:
            filters["is_active"] = True
        
        return self.find_by_filters(filters, db_type=db_type)
    
    def get_recent_users(self, days: int = 7, limit: int = 50,
                        db_type: Optional[str] = None) -> List[User]:
        """Get recently created users."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        filters = {"created_at": {"gte": cutoff_date}}
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    # User statistics
    def get_user_statistics(self, db_type: Optional[str] = None) -> Dict[str, Any]:
        """Get user statistics."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                total_users = session.query(User).count()
                active_users = session.query(User).filter(User.is_active == True).count()
                admin_users = session.query(User).filter(User.role == "admin").count()
                
                # Users created in last 30 days
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                recent_users = session.query(User).filter(
                    User.created_at >= thirty_days_ago
                ).count()
                
                # Role distribution
                role_stats = session.query(
                    User.role, func.count(User.id)
                ).group_by(User.role).all()
                
                return {
                    "total_users": total_users,
                    "active_users": active_users,
                    "inactive_users": total_users - active_users,
                    "admin_users": admin_users,
                    "recent_users_30_days": recent_users,
                    "role_distribution": dict(role_stats),
                    "activity_rate": active_users / total_users if total_users > 0 else 0
                }
        except Exception as e:
            self.logger.error(f"Error getting user statistics: {e}")
            return {}
    
    # Bulk operations
    def bulk_activate_users(self, user_ids: List[str], 
                           db_type: Optional[str] = None) -> int:
        """Bulk activate users."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                updated = session.query(User).filter(
                    User.id.in_(user_ids)
                ).update(
                    {"is_active": True, "updated_at": datetime.utcnow()},
                    synchronize_session=False
                )
                session.commit()
                return updated
        except Exception as e:
            self.logger.error(f"Error bulk activating users: {e}")
            return 0
    
    def bulk_deactivate_users(self, user_ids: List[str], 
                             db_type: Optional[str] = None) -> int:
        """Bulk deactivate users."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                updated = session.query(User).filter(
                    User.id.in_(user_ids)
                ).update(
                    {"is_active": False, "updated_at": datetime.utcnow()},
                    synchronize_session=False
                )
                session.commit()
                return updated
        except Exception as e:
            self.logger.error(f"Error bulk deactivating users: {e}")
            return 0