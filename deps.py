"""
EUNOMIA Legal AI Platform - API Dependencies (FIXED)
FastAPI dependencies for authentication, database, RBAC, and utilities
"""
from typing import Optional, AsyncGenerator, Callable
from datetime import datetime
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from redis.asyncio import Redis
import logging
import time
from collections import defaultdict

from app.core.database import get_db
from app.core.security import decode_token
from app.core.config import settings
from app.models.user import User, UserRole
from app.models.audit_log import AuditLog, ActionType, ResourceType
from app.schemas.auth import TokenData


# ============================================================================
# LOGGER
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# SECURITY SCHEME
# ============================================================================
security = HTTPBearer(
    scheme_name="Bearer",
    description="JWT Bearer token authentication",
    auto_error=False
)


# ============================================================================
# RATE LIMITING (IN-MEMORY - FOR PRODUCTION USE REDIS)
# ============================================================================
# Simple in-memory rate limiter
_rate_limit_storage = defaultdict(list)
_rate_limit_lock = None

async def rate_limit_strict(request: Request):
    """
    Strict rate limiting dependency for sensitive endpoints.
    
    Limits:
    - 5 requests per minute per IP for registration/login
    
    Args:
        request: FastAPI request
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old requests (older than 60 seconds)
    _rate_limit_storage[client_ip] = [
        t for t in _rate_limit_storage[client_ip] 
        if current_time - t < 60
    ]
    
    # Check limit (5 requests per minute)
    if len(_rate_limit_storage[client_ip]) >= 5:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later."
        )
    
    # Add current request
    _rate_limit_storage[client_ip].append(current_time)


async def rate_limit_standard(request: Request):
    """
    Standard rate limiting dependency.
    
    Limits:
    - 100 requests per minute per IP
    
    Args:
        request: FastAPI request
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old requests
    _rate_limit_storage[client_ip] = [
        t for t in _rate_limit_storage[client_ip] 
        if current_time - t < 60
    ]
    
    # Check limit (100 requests per minute)
    if len(_rate_limit_storage[client_ip]) >= 100:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later."
        )
    
    _rate_limit_storage[client_ip].append(current_time)


# ============================================================================
# REDIS DEPENDENCY
# ============================================================================
async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    Dependency to get Redis connection.
    
    Yields:
        Redis: Redis async client
    """
    redis_client = Redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    try:
        yield redis_client
    finally:
        await redis_client.close()


# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================
async def get_token_from_header(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Extract JWT token from Authorization header.
    
    Args:
        credentials: HTTP bearer credentials
        
    Returns:
        JWT token string
        
    Raises:
        HTTPException: If no token provided
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please provide a valid token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return credentials.credentials


async def get_current_user(
    token: str = Depends(get_token_from_header),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current user from JWT token.
    
    Args:
        token: JWT token from header
        db: Database session
        
    Returns:
        User: Current user object
        
    Raises:
        HTTPException: If token invalid or user not found
    """
    # Decode token
    token_data = decode_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == token_data.user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user only if account is active.
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        User: Active user object
        
    Raises:
        HTTPException: If user account is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact support."
        )
    
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current user only if email is verified.
    
    Args:
        current_user: User from get_current_active_user dependency
        
    Returns:
        User: Verified user object
        
    Raises:
        HTTPException: If email is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required. Please verify your email address."
        )
    
    return current_user


# ============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# ============================================================================
class RoleChecker:
    """
    Dependency factory for role-based access control.
    """
    
    def __init__(self, allowed_roles: list[UserRole]):
        """
        Initialize role checker.
        
        Args:
            allowed_roles: List of allowed user roles
        """
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_active_user)) -> User:
        """
        Check if user has required role.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User: User with required role
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {[r.value for r in self.allowed_roles]}"
            )
        
        return current_user


# Convenience role checkers
require_admin = RoleChecker([UserRole.ADMIN])
require_user_or_admin = RoleChecker([UserRole.USER, UserRole.ADMIN])


# ============================================================================
# AUDIT LOGGING
# ============================================================================
class AuditLogger:
    """
    Audit logger for tracking user actions.
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize audit logger.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def log(
        self,
        user_id: int,
        action: ActionType,
        resource_type: ResourceType,
        resource_id: Optional[int] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """
        Log an audit event.
        
        Args:
            user_id: User who performed the action
            action: Type of action performed
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional details (JSON)
            ip_address: Client IP address
            user_agent: Client user agent
        """
        try:
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.utcnow()
            )
            
            self.db.add(audit_log)
            await self.db.commit()
            
            logger.info(
                f"Audit log created: user_id={user_id}, "
                f"action={action.value}, resource={resource_type.value}:{resource_id}"
            )
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            await self.db.rollback()


async def get_audit_logger(
    db: AsyncSession = Depends(get_db)
) -> AuditLogger:
    """
    Get audit logger instance.
    
    Args:
        db: Database session
        
    Returns:
        AuditLogger: Audit logger instance
    """
    return AuditLogger(db)


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    "get_db",
    "get_redis",
    "get_token_from_header",
    "get_current_user",
    "get_current_active_user",
    "get_current_verified_user",
    "RoleChecker",
    "require_admin",
    "require_user_or_admin",
    "AuditLogger",
    "get_audit_logger",
    "rate_limit_strict",
    "rate_limit_standard",
]
