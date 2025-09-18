from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from ..repositories.query_log_repository import QueryLogRepository
from ..models.query_log import QueryLog

class AnalyticsService:
    """Service layer for query analytics and insights."""
    
    def __init__(self, db_type: Optional[str] = None):
        self.db_type = db_type
        self.query_repo = QueryLogRepository(db_type)
        self.logger = logging.getLogger(__name__)
    
    # Query logging
    def log_user_query(self, user_id: Optional[str], query_text: str, 
                      response_text: str,
                      conversation_id: Optional[str] = None,
                      turn_id: Optional[str] = None,
                      search_results_count: int = 0,
                      retrieval_time: float = 0.0,
                      generation_time: float = 0.0,
                      total_time: float = 0.0,
                      success: bool = True,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[QueryLog]:
        """Log a user query with performance metrics."""
        try:
            query_log = self.query_repo.log_query(
                user_id=user_id,
                query_text=query_text,
                response_text=response_text,
                conversation_id=conversation_id,
                turn_id=turn_id,
                search_results_count=search_results_count,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                success=success,
                metadata_json=metadata,
                db_type=self.db_type
            )
            
            if query_log:
                self.logger.debug(f"Query logged: {query_log.id}")
            
            return query_log
        except Exception as e:
            self.logger.error(f"Error logging query: {e}")
            return None
    
    # Query analytics
    def get_user_query_history(self, user_id: str, limit: int = 50, 
                              offset: int = 0,
                              successful_only: bool = False) -> List[QueryLog]:
        """Get query history for a user."""
        try:
            return self.query_repo.get_user_query_history(
                user_id=user_id,
                limit=limit,
                offset=offset,
                successful_only=successful_only,
                db_type=self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting user query history: {e}")
            return []
    
    def get_conversation_queries(self, conversation_id: str) -> List[QueryLog]:
        """Get all queries in a conversation."""
        try:
            return self.query_repo.get_conversation_queries(
                conversation_id, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting conversation queries: {e}")
            return []
    
    def search_queries(self, search_term: str, user_id: Optional[str] = None,
                      limit: int = 50) -> List[QueryLog]:
        """Search queries by content."""
        try:
            return self.query_repo.search_queries(
                search_term=search_term,
                user_id=user_id,
                limit=limit,
                db_type=self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error searching queries: {e}")
            return []
    
    # Performance analytics
    def get_slow_queries(self, min_time_seconds: float = 5.0, 
                        limit: int = 50) -> List[QueryLog]:
        """Get slow queries above time threshold."""
        try:
            return self.query_repo.get_slow_queries(
                min_time_seconds, limit, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting slow queries: {e}")
            return []
    
    def get_failed_queries(self, hours: int = 24, limit: int = 50) -> List[QueryLog]:
        """Get failed queries in time period."""
        try:
            return self.query_repo.get_failed_queries(
                hours, limit, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting failed queries: {e}")
            return []
    
    def get_recent_queries(self, hours: int = 24, limit: int = 100,
                          successful_only: bool = False) -> List[QueryLog]:
        """Get recent queries."""
        try:
            return self.query_repo.get_recent_queries(
                hours, limit, successful_only, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting recent queries: {e}")
            return []
    
    # Statistics and insights
    def get_query_statistics(self, user_id: Optional[str] = None,
                            hours: int = 24) -> Dict[str, Any]:
        """Get query statistics for time period."""
        try:
            return self.query_repo.get_query_statistics(
                user_id, hours, self.db_type
            )
        except Exception as e:
            self.logger.error(f"Error getting query statistics: {e}")
            return {}
    
    def get_user_activity_stats(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get user activity statistics."""
        try:
            return self.query_repo.get_user_activity_stats(days, self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting user activity stats: {e}")
            return []
    
    def get_popular_queries(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most popular query patterns."""
        try:
            return self.query_repo.get_popular_queries(days, limit, self.db_type)
        except Exception as e:
            self.logger.error(f"Error getting popular queries: {e}")
            return []
    
    def get_system_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive system performance metrics."""
        try:
            # Get basic statistics
            stats = self.get_query_statistics(hours=hours)
            
            # Get slow queries count
            slow_queries = len(self.get_slow_queries(limit=1000))
            
            # Get failed queries count
            failed_queries = len(self.get_failed_queries(hours=hours, limit=1000))
            
            # Calculate performance score (0-100)
            total_queries = stats.get("total_queries", 0)
            success_rate = stats.get("success_rate", 0)
            avg_time = stats.get("average_total_time", 0)
            
            # Performance score based on success rate and response time
            performance_score = 0
            if total_queries > 0:
                time_score = max(0, 100 - (avg_time * 10))  # Penalty for slow responses
                performance_score = (success_rate * 0.7 + time_score * 0.3) * 100
            
            return {
                **stats,
                "slow_queries_count": slow_queries,
                "failed_queries_count": failed_queries,
                "performance_score": round(performance_score, 2),
                "health_status": self._get_health_status(performance_score, success_rate)
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_health_status(self, performance_score: float, success_rate: float) -> str:
        """Determine system health status."""
        if performance_score >= 90 and success_rate >= 0.95:
            return "excellent"
        elif performance_score >= 80 and success_rate >= 0.90:
            return "good"
        elif performance_score >= 70 and success_rate >= 0.80:
            return "fair"
        elif performance_score >= 60 and success_rate >= 0.70:
            return "poor"
        else:
            return "critical"
    
    def get_user_engagement_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get user engagement metrics."""
        try:
            user_stats = self.get_user_activity_stats(days)
            
            if not user_stats:
                return {}
            
            total_users = len(user_stats)
            active_users = len([u for u in user_stats if u["total_queries"] > 0])
            
            total_queries = sum(u["total_queries"] for u in user_stats)
            avg_queries_per_user = total_queries / total_users if total_users > 0 else 0
            
            # Power users (>10 queries)
            power_users = len([u for u in user_stats if u["total_queries"] > 10])
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "inactive_users": total_users - active_users,
                "engagement_rate": active_users / total_users if total_users > 0 else 0,
                "total_queries": total_queries,
                "average_queries_per_user": avg_queries_per_user,
                "power_users": power_users,
                "power_user_rate": power_users / total_users if total_users > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting engagement metrics: {e}")
            return {}
    
    def get_query_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get query trends and patterns."""
        try:
            # Get popular queries
            popular = self.get_popular_queries(days, 10)
            
            # Get recent activity
            recent_stats = self.get_query_statistics(hours=24)
            week_stats = self.get_query_statistics(hours=24*7)
            
            # Calculate trends
            daily_queries = recent_stats.get("total_queries", 0)
            weekly_queries = week_stats.get("total_queries", 0)
            
            query_trend = "stable"
            if weekly_queries > 0:
                daily_avg = weekly_queries / 7
                if daily_queries > daily_avg * 1.2:
                    query_trend = "increasing"
                elif daily_queries < daily_avg * 0.8:
                    query_trend = "decreasing"
            
            return {
                "popular_queries": popular,
                "daily_queries": daily_queries,
                "weekly_queries": weekly_queries,
                "query_trend": query_trend,
                "average_daily_queries": weekly_queries / 7 if weekly_queries > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting query trends: {e}")
            return {}
    
    # Maintenance operations
    def cleanup_old_logs(self, days_old: int = 90) -> int:
        """Clean up old query logs."""
        try:
            count = self.query_repo.cleanup_old_logs(days_old, self.db_type)
            if count > 0:
                self.logger.info(f"Cleaned up {count} old query logs")
            return count
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {e}")
            return 0
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily analytics report."""
        try:
            return {
                "date": datetime.utcnow().isoformat(),
                "query_statistics": self.get_query_statistics(hours=24),
                "performance_metrics": self.get_system_performance_metrics(hours=24),
                "user_engagement": self.get_user_engagement_metrics(days=1),
                "slow_queries": len(self.get_slow_queries(limit=1000)),
                "failed_queries": len(self.get_failed_queries(hours=24, limit=1000)),
                "popular_queries": self.get_popular_queries(days=1, limit=5)
            }
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
            return {}
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly analytics report."""
        try:
            return {
                "week_ending": datetime.utcnow().isoformat(),
                "query_statistics": self.get_query_statistics(hours=24*7),
                "performance_metrics": self.get_system_performance_metrics(hours=24*7),
                "user_engagement": self.get_user_engagement_metrics(days=7),
                "query_trends": self.get_query_trends(days=7),
                "top_users": self.get_user_activity_stats(days=7)[:10],
                "popular_queries": self.get_popular_queries(days=7, limit=10)
            }
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {e}")
            return {}