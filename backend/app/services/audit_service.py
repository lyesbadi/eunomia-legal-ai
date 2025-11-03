"""
EUNOMIA Legal AI Platform - Audit Service
Business logic for GDPR audit logging and compliance
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from app.models.audit_log import AuditLog, ActionType, ResourceType
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# AUDIT SERVICE
# ============================================================================
class AuditService:
    """
    Service for GDPR audit logging operations.
    
    Handles:
    - Creating audit logs
    - Querying audit logs
    - Cleanup old logs
    - GDPR compliance reporting
    """
    
    @staticmethod
    async def log(
        db: AsyncSession,
        user_id: Optional[int],
        action: ActionType,
        resource_type: ResourceType,
        resource_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Create audit log entry.
        
        Args:
            db: Database session
            user_id: User ID (None for anonymous actions)
            action: Action type
            resource_type: Resource type affected
            resource_id: Resource ID affected
            ip_address: Client IP address
            user_agent: Client user agent
            details: Additional details (JSON)
            success: Whether action succeeded
            error_message: Error message if failed
            
        Returns:
            Created audit log entry
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            success=success,
            error_message=error_message
        )
        
        db.add(audit_log)
        await db.commit()
        await db.refresh(audit_log)
        
        return audit_log
    
    @staticmethod
    async def get_user_logs(
        db: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        action: Optional[ActionType] = None,
        resource_type: Optional[ResourceType] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> tuple[List[AuditLog], int]:
        """
        Get audit logs for specific user.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records
            action: Filter by action type
            resource_type: Filter by resource type
            date_from: Filter by date (from)
            date_to: Filter by date (to)
            
        Returns:
            Tuple of (logs list, total count)
        """
        query = select(AuditLog).where(AuditLog.user_id == user_id)
        
        # Apply filters
        conditions = []
        
        if action:
            conditions.append(AuditLog.action == action)
        
        if resource_type:
            conditions.append(AuditLog.resource_type == resource_type)
        
        if date_from:
            conditions.append(AuditLog.timestamp >= date_from)
        
        if date_to:
            conditions.append(AuditLog.timestamp <= date_to)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        query = query.order_by(desc(AuditLog.timestamp))
        
        # Execute
        result = await db.execute(query)
        logs = result.scalars().all()
        
        return list(logs), total
    
    @staticmethod
    async def get_resource_logs(
        db: AsyncSession,
        resource_type: ResourceType,
        resource_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> tuple[List[AuditLog], int]:
        """
        Get audit logs for specific resource.
        
        Args:
            db: Database session
            resource_type: Resource type
            resource_id: Resource ID
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            Tuple of (logs list, total count)
        """
        query = select(AuditLog).where(
            and_(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id
            )
        )
        
        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        query = query.order_by(desc(AuditLog.timestamp))
        
        # Execute
        result = await db.execute(query)
        logs = result.scalars().all()
        
        return list(logs), total
    
    @staticmethod
    async def cleanup_old_logs(
        db: AsyncSession,
        days: int = 365
    ) -> int:
        """
        Delete audit logs older than specified days (GDPR retention).
        
        Args:
            db: Database session
            days: Number of days to retain (default: 365)
            
        Returns:
            Number of logs deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get logs to delete
        result = await db.execute(
            select(AuditLog).where(AuditLog.timestamp < cutoff_date)
        )
        logs_to_delete = result.scalars().all()
        
        count = len(logs_to_delete)
        
        # Delete logs
        for log in logs_to_delete:
            await db.delete(log)
        
        await db.commit()
        
        logger.info(f"Cleaned up {count} audit logs older than {days} days")
        return count
    
    @staticmethod
    async def get_failed_logins(
        db: AsyncSession,
        user_id: int,
        hours: int = 24
    ) -> int:
        """
        Get count of failed login attempts in last N hours.
        
        Args:
            db: Database session
            user_id: User ID
            hours: Number of hours to look back
            
        Returns:
            Count of failed login attempts
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        result = await db.execute(
            select(func.count(AuditLog.id))
            .where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.action == ActionType.LOGIN_FAILED,
                    AuditLog.timestamp >= cutoff_time
                )
            )
        )
        
        return result.scalar() or 0
    
    @staticmethod
    async def get_statistics(
        db: AsyncSession,
        user_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get audit log statistics.
        
        Args:
            db: Database session
            user_id: Optional user ID (None for all users)
            days: Number of days to analyze
            
        Returns:
            Dictionary with statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(AuditLog).where(AuditLog.timestamp >= cutoff_date)
        
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        
        # Total logs
        total_result = await db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = total_result.scalar() or 0
        
        # By action type
        by_action = {}
        for action in ActionType:
            count_result = await db.execute(
                select(func.count(AuditLog.id))
                .where(
                    and_(
                        AuditLog.timestamp >= cutoff_date,
                        AuditLog.action == action,
                        AuditLog.user_id == user_id if user_id else True
                    )
                )
            )
            count = count_result.scalar() or 0
            if count > 0:
                by_action[action.value] = count
        
        # By resource type
        by_resource = {}
        for resource in ResourceType:
            count_result = await db.execute(
                select(func.count(AuditLog.id))
                .where(
                    and_(
                        AuditLog.timestamp >= cutoff_date,
                        AuditLog.resource_type == resource,
                        AuditLog.user_id == user_id if user_id else True
                    )
                )
            )
            count = count_result.scalar() or 0
            if count > 0:
                by_resource[resource.value] = count
        
        # Failed actions
        failed_result = await db.execute(
            select(func.count(AuditLog.id))
            .where(
                and_(
                    AuditLog.timestamp >= cutoff_date,
                    AuditLog.success == False,
                    AuditLog.user_id == user_id if user_id else True
                )
            )
        )
        failed_count = failed_result.scalar() or 0
        
        return {
            "total_logs": total,
            "by_action": by_action,
            "by_resource": by_resource,
            "failed_actions": failed_count,
            "period_days": days
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
async def log_action(
    db: AsyncSession,
    user_id: Optional[int],
    action: ActionType,
    resource_type: ResourceType,
    **kwargs
) -> AuditLog:
    """Convenience function to log action."""
    return await AuditService.log(
        db, user_id, action, resource_type, **kwargs
    )


async def get_user_audit_logs(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 100
) -> tuple[List[AuditLog], int]:
    """Get user audit logs."""
    return await AuditService.get_user_logs(db, user_id, skip, limit)