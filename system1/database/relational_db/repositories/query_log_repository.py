from typing import Optional, List, Dict, Any
from sqlalchemy import text, func, and_, desc
from datetime import datetime, timedelta
import uuid

from ..models.query_log import QueryLog
from .base_repository import BaseRepository

class QueryLogRepository(BaseRepository[QueryLog]):
    """Repository for QueryLog model with query analytics operations."""
    
    def __init__(self, db_type: Optional[str] = None):
        super().__init__(QueryLog, db_type)
    
    # Query logging methods
    def log_query(self, user_id: Optional[str], query_text: str, response_text: str,
                 conversation_id: Optional[str] = None,
                 turn_id: Optional[str] = None,
                 search_results_count: int = 0,
                 retrieval_time: float = 0.0,
                 generation_time: float = 0.0,
                 total_time: float = 0.0,
                 success: bool = True,
                 metadata_json: Optional[Dict[str, Any]] = None,
                 db_type: Optional[str] = None) -> Optional[QueryLog]:
        """Log a query with performance metrics."""
        log_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "query_text": query_text,
            "response_text": response_text,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "search_results_count": search_results_count,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "success": success,
            "metadata_json": metadata_json or {}
        }
        
        return self.create(log_data, db_type)
    
    def get_user_query_history(self, user_id: str, limit: int = 50, offset: int = 0,
                              successful_only: bool = False,
                              db_type: Optional[str] = None) -> List[QueryLog]:
        """Get query history for user."""
        filters = {"user_id": user_id}
        if successful_only:
            filters["success"] = True
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            offset=offset,
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    def get_conversation_queries(self, conversation_id: str,
                               db_type: Optional[str] = None) -> List[QueryLog]:
        """Get all queries in a conversation."""
        return self.find_by_filters(
            {"conversation_id": conversation_id}, 
            order_by="created_at",
            db_type=db_type
        )
    
    def get_recent_queries(self, hours: int = 24, limit: int = 100,
                          successful_only: bool = False,
                          db_type: Optional[str] = None) -> List[QueryLog]:
        """Get recent queries."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        filters = {"created_at": {"gte": cutoff_time}}
        
        if successful_only:
            filters["success"] = True
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    def get_slow_queries(self, min_time_seconds: float = 5.0, limit: int = 50,
                        db_type: Optional[str] = None) -> List[QueryLog]:
        """Get slow queries above time threshold."""
        filters = {"total_time": {"gte": min_time_seconds}}
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            order_by="total_time", 
            order_desc=True,
            db_type=db_type
        )
    
    def get_failed_queries(self, hours: int = 24, limit: int = 50,
                          db_type: Optional[str] = None) -> List[QueryLog]:
        """Get failed queries in time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        filters = {
            "success": False,
            "created_at": {"gte": cutoff_time}
        }
        
        return self.find_by_filters(
            filters, 
            limit=limit, 
            order_by="created_at", 
            order_desc=True,
            db_type=db_type
        )
    
    def search_queries(self, search_term: str, user_id: Optional[str] = None,
                      limit: int = 50, db_type: Optional[str] = None) -> List[QueryLog]:
        """Search queries by text content."""
        try:
            from ..database import get_db
            with get_db(db_type or self.db_type) as session:
                query = session.query(QueryLog)
                
                # Search in query text
                query = query.filter(
                    QueryLog.query_text.ilike(f"%{search_term}%")
                )
                
                # Filter by user if specified
                if user_id:
                    query = query.filter(QueryLog.user_id == user_id)
                
                return query.order_by(desc(QueryLog.created_at)).limit(limit).all()
        except Exception as e:
            self.logger.error(f"Error searching queries: {e}")
            return []
    
    # Analytics methods
    def get_query_statistics(self, user_id: Optional[str] = None,
                            hours: int = 24,
                            db_type: Optional[str] = None) -> Dict[str, Any]:
        """Get query analytics for time period."""
        try:
            from ..database import get_db
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with get_db(db_type or self.db_type) as session:
                query = session.query(QueryLog).filter(
                    QueryLog.created_at >= cutoff_time
                )
                
                if user_id:
                    query = query.filter(QueryLog.user_id == user_id)
                
                total_queries = query.count()
                successful_queries = query.filter(QueryLog.success == True).count()
                failed_queries = total_queries - successful_queries
                
                # Performance metrics
                avg_total_time = query.with_entities(
                    func.avg(QueryLog.total_time)
                ).scalar() or 0
                
                avg_retrieval_time = query.with_entities(
                    func.avg(QueryLog.retrieval_time)
                ).scalar() or 0
                
                avg_generation_time = query.with_entities(
                    func.avg(QueryLog.generation_time)
                ).scalar() or 0
                
                avg_results_count = query.with_entities(
                    func.avg(QueryLog.search_results_count)
                ).scalar() or 0
                
                # Query length statistics
                avg_query_length = query.with_entities(
                    func.avg(func.length(QueryLog.query_text))
                ).scalar() or 0
                
                return {
                    "time_period_hours": hours,
                    "total_queries": total_queries,
                    "successful_queries": successful_queries,
                    "failed_queries": failed_queries,
                    "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                    "average_total_time": float(avg_total_time),
                    "average_retrieval_time": float(avg_retrieval_time),
                    "average_generation_time": float(avg_generation_time),
                    "average_results_count": float(avg_results_count),
                    "average_query_length": float(avg_query_length)
                }
        except Exception as e:
            self.logger.error(f"Error getting query statistics: {e}")
            return {}
    
    def get_user_activity_stats(self, days: int = 30, 
                               db_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user activity statistics."""
        try:
            from ..database import get_db
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            with get_db(db_type or self.db_type) as session:
                user_stats = session.query(
                    QueryLog.user_id,
                    func.count(QueryLog.id).label('total_queries'),
                    func.sum(func.case(
                        (QueryLog.success == True, 1),
                        else_=0
                    )).label('successful_queries'),
                    func.avg(QueryLog.total_time).label('avg_response_time'),
                    func.max(QueryLog.created_at).label('last_activity')
                ).filter(
                    QueryLog.created_at >= cutoff_time
                ).group_by(QueryLog.user_id).all()
                
                return [{
                    "user_id": stat.user_id,
                    "total_queries": stat.total_queries,
                    "successful_queries": stat.successful_queries,
                    "success_rate": stat.successful_queries / stat.total_queries if stat.total_queries > 0 else 0,
                    "average_response_time": float(stat.avg_response_time or 0),
                    "last_activity": stat.last_activity.isoformat() if stat.last_activity else None
                } for stat in user_stats]
        except Exception as e:
            self.logger.error(f"Error getting user activity stats: {e}")
            return []
    
    def get_popular_queries(self, days: int = 7, limit: int = 20,
                           db_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get most popular query patterns."""
        try:
            from ..database import get_db
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            with get_db(db_type or self.db_type) as session:
                # Group by normalized query text (first 100 chars)
                popular_queries = session.query(
                    func.substr(QueryLog.query_text, 1, 100).label('query_pattern'),
                    func.count(QueryLog.id).label('frequency'),
                    func.avg(QueryLog.total_time).label('avg_time'),
                    func.avg(QueryLog.search_results_count).label('avg_results')
                ).filter(
                    QueryLog.created_at >= cutoff_time,
                    QueryLog.success == True
                ).group_by(
                    func.substr(QueryLog.query_text, 1, 100)
                ).order_by(
                    desc(func.count(QueryLog.id))
                ).limit(limit).all()
                
                return [{
                    "query_pattern": query.query_pattern,
                    "frequency": query.frequency,
                    "average_response_time": float(query.avg_time or 0),
                    "average_results_count": float(query.avg_results or 0)
                } for query in popular_queries]
        except Exception as e:
            self.logger.error(f"Error getting popular queries: {e}")
            return []
    
    def cleanup_old_logs(self, days_old: int = 90, 
                        db_type: Optional[str] = None) -> int:
        """Clean up old query logs."""
        try:
            from ..database import get_db
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            with get_db(db_type or self.db_type) as session:
                deleted = session.query(QueryLog).filter(
                    QueryLog.created_at < cutoff_date
                ).delete(synchronize_session=False)
                session.commit()
                return deleted
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {e}")
            return 0