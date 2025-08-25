"""
Audit Repository Implementation

Provides audit logging and security tracking with:
- Comprehensive action logging
- User activity tracking
- Security event monitoring
- Compliance reporting
"""

import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, func, desc
from sqlalchemy.sql import Select

from .base import BaseRepository, RepositoryError
from ..models import AuditLog, User
from ...core.logger import get_logger

logger = get_logger(__name__)


class AuditRepository(BaseRepository[AuditLog]):
    """Audit log repository for security tracking and compliance."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize audit repository.
        
        Args:
            session: AsyncSession instance
        """
        super().__init__(session, AuditLog)
    
    def _add_relationship_loading(self, query: Select) -> Select:
        """
        Add eager loading for audit relationships.
        
        Args:
            query: SQLAlchemy query
            
        Returns:
            Query with relationship loading options
        """
        return query.options(
            selectinload(AuditLog.user)
        )
    
    async def log_action(
        self,
        user_id: Optional[uuid.UUID] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        result: str = "success",
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AuditLog]:
        """
        Log audit event with comprehensive details.
        
        Args:
            user_id: ID of user performing action
            action: Action being performed
            resource_type: Type of resource being acted upon
            resource_id: ID of the resource
            ip_address: Client IP address
            user_agent: Client user agent
            old_values: Previous values (for updates)
            new_values: New values (for creates/updates)
            result: Result of the action (success/failure)
            error_message: Error message if action failed
            metadata: Additional metadata
            
        Returns:
            Created AuditLog instance or None on failure
            
        Raises:
            RepositoryError: If logging fails
        """
        try:
            # Convert dictionaries to JSON strings
            old_values_json = json.dumps(old_values) if old_values else None
            new_values_json = json.dumps(new_values) if new_values else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Create audit log entry
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                old_values=old_values_json,
                new_values=new_values_json,
                result=result,
                error_message=error_message
            )
            
            self.session.add(audit_log)
            await self.session.flush()
            await self.session.refresh(audit_log)
            
            # Log at appropriate level based on result
            if result == "success":
                self.logger.debug(
                    f"Audit logged: {action}",
                    extra={
                        "audit_id": str(audit_log.id),
                        "user_id": str(user_id) if user_id else None,
                        "action": action,
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "ip_address": ip_address
                    }
                )
            else:
                self.logger.warning(
                    f"Failed action logged: {action}",
                    extra={
                        "audit_id": str(audit_log.id),
                        "user_id": str(user_id) if user_id else None,
                        "action": action,
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "error_message": error_message,
                        "ip_address": ip_address
                    }
                )
            
            return audit_log
            
        except Exception as e:
            # Don't rollback session for audit failures - log and continue
            self.logger.error(
                f"Failed to log audit event: {e}",
                extra={
                    "action": action,
                    "user_id": str(user_id) if user_id else None,
                    "resource_type": resource_type,
                    "error": str(e)
                }
            )
            # Don't raise exception for audit failures to avoid disrupting main operations
            return None
    
    async def get_user_activity(
        self,
        user_id: uuid.UUID,
        days: int = 30,
        action_filter: Optional[str] = None,
        resource_type_filter: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """
        Get user activity for specified period.
        
        Args:
            user_id: User ID
            days: Number of days to look back
            action_filter: Filter by specific action
            resource_type_filter: Filter by resource type
            limit: Maximum number of records
            
        Returns:
            List of AuditLog instances for user
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            filters = {
                'user_id': user_id,
                'created_at__gte': start_date
            }
            
            if action_filter:
                filters['action'] = action_filter
            if resource_type_filter:
                filters['resource_type'] = resource_type_filter
            
            activity = await self.get_all(
                filters=filters,
                limit=limit,
                order_by='created_at',
                order_desc=True,
                load_relationships=True
            )
            
            self.logger.debug(
                f"Retrieved {len(activity)} activity records for user {user_id}",
                extra={
                    "user_id": str(user_id),
                    "days": days,
                    "action_filter": action_filter,
                    "resource_type_filter": resource_type_filter,
                    "record_count": len(activity)
                }
            )
            
            return activity
            
        except Exception as e:
            self.logger.error(f"Error getting user activity for {user_id}: {e}")
            raise RepositoryError(
                "Failed to retrieve user activity",
                "GET_USER_ACTIVITY_ERROR",
                {
                    "user_id": str(user_id),
                    "days": days,
                    "action_filter": action_filter,
                    "error": str(e)
                }
            )
    
    async def get_security_events(
        self,
        days: int = 7,
        event_types: Optional[List[str]] = None,
        severity: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """
        Get security-related audit events.
        
        Args:
            days: Number of days to look back
            event_types: List of event types to filter by
            severity: Filter by result (success/failure)
            limit: Maximum number of records
            
        Returns:
            List of security-related AuditLog instances
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Security-related actions
            security_actions = event_types or [
                'login', 'logout', 'login_failed', 'password_change',
                'password_reset', 'account_locked', 'role_assigned',
                'role_revoked', 'permission_granted', 'permission_denied'
            ]
            
            filters = {
                'created_at__gte': start_date,
                'action__in': security_actions
            }
            
            if severity:
                filters['result'] = severity
            
            events = await self.get_all(
                filters=filters,
                limit=limit,
                order_by='created_at',
                order_desc=True,
                load_relationships=True
            )
            
            self.logger.debug(
                f"Retrieved {len(events)} security events",
                extra={
                    "days": days,
                    "event_types": event_types,
                    "severity": severity,
                    "event_count": len(events)
                }
            )
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting security events: {e}")
            raise RepositoryError(
                "Failed to retrieve security events",
                "GET_SECURITY_EVENTS_ERROR",
                {
                    "days": days,
                    "event_types": event_types,
                    "severity": severity,
                    "error": str(e)
                }
            )
    
    async def get_failed_actions(
        self,
        days: int = 1,
        action_filter: Optional[str] = None,
        user_id: Optional[uuid.UUID] = None,
        ip_address: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """
        Get failed actions for security monitoring.
        
        Args:
            days: Number of days to look back
            action_filter: Filter by specific action
            user_id: Filter by user ID
            ip_address: Filter by IP address
            limit: Maximum number of records
            
        Returns:
            List of failed action AuditLog instances
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            filters = {
                'created_at__gte': start_date,
                'result': 'failure'
            }
            
            if action_filter:
                filters['action'] = action_filter
            if user_id:
                filters['user_id'] = user_id
            if ip_address:
                filters['ip_address'] = ip_address
            
            failed_actions = await self.get_all(
                filters=filters,
                limit=limit,
                order_by='created_at',
                order_desc=True,
                load_relationships=True
            )
            
            self.logger.debug(
                f"Retrieved {len(failed_actions)} failed actions",
                extra={
                    "days": days,
                    "action_filter": action_filter,
                    "user_id": str(user_id) if user_id else None,
                    "ip_address": ip_address,
                    "failed_count": len(failed_actions)
                }
            )
            
            return failed_actions
            
        except Exception as e:
            self.logger.error(f"Error getting failed actions: {e}")
            raise RepositoryError(
                "Failed to retrieve failed actions",
                "GET_FAILED_ACTIONS_ERROR",
                {
                    "days": days,
                    "action_filter": action_filter,
                    "user_id": str(user_id) if user_id else None,
                    "ip_address": ip_address,
                    "error": str(e)
                }
            )
    
    async def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
        days: int = 30,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get audit history for a specific resource.
        
        Args:
            resource_type: Type of resource
            resource_id: ID of the resource
            days: Number of days to look back
            limit: Maximum number of records
            
        Returns:
            List of AuditLog instances for the resource
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            filters = {
                'resource_type': resource_type,
                'resource_id': resource_id,
                'created_at__gte': start_date
            }
            
            history = await self.get_all(
                filters=filters,
                limit=limit,
                order_by='created_at',
                order_desc=True,
                load_relationships=True
            )
            
            self.logger.debug(
                f"Retrieved {len(history)} history records for {resource_type}:{resource_id}",
                extra={
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "days": days,
                    "record_count": len(history)
                }
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting resource history: {e}")
            raise RepositoryError(
                "Failed to retrieve resource history",
                "GET_RESOURCE_HISTORY_ERROR",
                {
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "days": days,
                    "error": str(e)
                }
            )
    
    async def get_audit_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get audit statistics for monitoring and reporting.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with audit statistics
            
        Raises:
            RepositoryError: If operation fails
        """
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get all audit records for the period
            records = await self.get_all(
                filters={'created_at__gte': start_date},
                limit=100000  # Large limit for statistics
            )
            
            # Calculate statistics
            total_actions = len(records)
            
            action_counts = {}
            resource_type_counts = {}
            result_counts = {}
            daily_counts = {}
            user_activity = {}
            
            for record in records:
                # Action counts
                action = record.action or 'unknown'
                action_counts[action] = action_counts.get(action, 0) + 1
                
                # Resource type counts
                resource_type = record.resource_type or 'unknown'
                resource_type_counts[resource_type] = resource_type_counts.get(resource_type, 0) + 1
                
                # Result counts
                result = record.result or 'unknown'
                result_counts[result] = result_counts.get(result, 0) + 1
                
                # Daily activity
                day = record.created_at.date().isoformat()
                daily_counts[day] = daily_counts.get(day, 0) + 1
                
                # User activity
                if record.user_id:
                    user_id = str(record.user_id)
                    user_activity[user_id] = user_activity.get(user_id, 0) + 1
            
            # Calculate success rate
            successful_actions = result_counts.get('success', 0)
            success_rate = (successful_actions / total_actions * 100) if total_actions > 0 else 0
            
            # Get top users by activity
            top_users = sorted(
                user_activity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Get most frequent actions
            top_actions = sorted(
                action_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            statistics = {
                'period_days': days,
                'total_actions': total_actions,
                'success_rate': round(success_rate, 2),
                'unique_users': len(user_activity),
                'action_counts': action_counts,
                'resource_type_counts': resource_type_counts,
                'result_counts': result_counts,
                'daily_activity': daily_counts,
                'top_users': [{'user_id': uid, 'action_count': count} for uid, count in top_users],
                'top_actions': [{'action': action, 'count': count} for action, count in top_actions],
                'generated_at': datetime.now(timezone.utc),
                'start_date': start_date,
                'end_date': datetime.now(timezone.utc)
            }
            
            self.logger.debug(
                f"Generated audit statistics for {days} days",
                extra={
                    "days": days,
                    "total_actions": total_actions,
                    "success_rate": success_rate,
                    "unique_users": len(user_activity)
                }
            )
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error getting audit statistics: {e}")
            raise RepositoryError(
                "Failed to get audit statistics",
                "GET_AUDIT_STATISTICS_ERROR",
                {"days": days, "error": str(e)}
            )
    
    async def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """
        Clean up old audit records to manage storage.
        
        Args:
            days_to_keep: Number of days of records to keep
            
        Returns:
            Number of records deleted
            
        Raises:
            RepositoryError: If cleanup fails
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            # Count records to be deleted
            count = await self.count({'created_at__lt': cutoff_date})
            
            if count == 0:
                self.logger.info("No old audit records to clean up")
                return 0
            
            # Delete old records
            deleted_count = await self.bulk_delete(
                await self.session.scalars(
                    select(AuditLog.id).where(AuditLog.created_at < cutoff_date)
                ).all()
            )
            
            self.logger.info(
                f"Cleaned up {deleted_count} old audit records",
                extra={
                    "days_to_keep": days_to_keep,
                    "cutoff_date": cutoff_date,
                    "deleted_count": deleted_count
                }
            )
            
            return deleted_count
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error cleaning up old audit records: {e}")
            raise RepositoryError(
                "Failed to cleanup old audit records",
                "CLEANUP_AUDIT_ERROR",
                {"days_to_keep": days_to_keep, "error": str(e)}
            )
