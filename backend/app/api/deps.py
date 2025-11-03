"""
EUNOMIA Legal AI Platform - API Dependencies
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

from app.core.database import get_async_session
from app.core.security import decode_access_token
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
# DATABASE DEPENDENCY
# ============================================================================
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get async database session.
    
    Yields:
        AsyncSession: SQLAlchemy async session
        
    Example:
```python
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
```
    
    Note:
        Session is automatically committed and closed after request.
    """
    async for session in get_async_session():
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# REDIS DEPENDENCY
# ============================================================================
async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    Dependency to get Redis connection.
    
    Yields:
        Redis: Redis async client
        
    Example:
```python
        @router.get("/cache/{key}")
        async def get_cache(key: str, redis: Redis = Depends(get_redis)):
            value = await redis.get(key)
            return {"value": value}
```
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
        credentials: HTTP Bearer credentials
        
    Returns:
        JWT token string
        
    Raises:
        HTTPException: If token is missing or invalid format
        
    Example:
```python
        @router.get("/protected")
        async def protected(token: str = Depends(get_token_from_header)):
            return {"token": token[:10] + "..."}
```
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials


async def get_current_user(
    token: str = Depends(get_token_from_header),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        token: JWT access token
        db: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        HTTPException: If token is invalid or user not found
        
    Example:
```python
        @router.get("/me")
        async def get_me(user: User = Depends(get_current_user)):
            return {"email": user.email, "role": user.role}
```
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        token_data = decode_access_token(token)
        if token_data is None:
            raise credentials_exception
        
        user_id = token_data.get("user_id")
        if user_id is None:
            raise credentials_exception
            
    except Exception as e:
        logger.warning(f"Token decode error: {e}")
        raise credentials_exception
    
    # Fetch user from database
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            raise credentials_exception
        
        # Update last activity timestamp
        user.last_activity = datetime.utcnow()
        await db.commit()
        
        return user
        
    except Exception as e:
        logger.error(f"Database error fetching user {user_id}: {e}")
        raise credentials_exception


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
        
    Example:
```python
        @router.post("/documents/upload")
        async def upload(user: User = Depends(get_current_active_user)):
            # Only active users can upload
            return {"user_id": user.id}
```
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
        
    Example:
```python
        @router.post("/documents/upload")
        async def upload(user: User = Depends(get_current_verified_user)):
            # Only verified users can upload
            return {"user_id": user.id}
```
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
    
    Checks if current user has required role(s).
    """
    
    def __init__(self, allowed_roles: list[UserRole]):
        """
        Initialize role checker.
        
        Args:
            allowed_roles: List of allowed user roles
            
        Example:
```python
            # Only admins can access
            require_admin = RoleChecker([UserRole.ADMIN])
            
            @router.delete("/users/{user_id}")
            async def delete_user(
                user_id: int,
                current_user: User = Depends(require_admin)
            ):
                # Only admins reach here
                pass
```
        """
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_active_user)) -> User:
        """
        Check if user has required role.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User: User object if authorized
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        if current_user.role not in self.allowed_roles:
            logger.warning(
                f"Access denied: User {current_user.id} ({current_user.role}) "
                f"attempted to access endpoint requiring {self.allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {', '.join([r.value for r in self.allowed_roles])}"
            )
        
        return current_user


# Convenience role checker instances
require_admin = RoleChecker([UserRole.ADMIN])
require_manager = RoleChecker([UserRole.ADMIN, UserRole.MANAGER])
require_user = RoleChecker([UserRole.ADMIN, UserRole.MANAGER, UserRole.USER])


# ============================================================================
# RESOURCE OWNERSHIP VERIFICATION
# ============================================================================
async def verify_document_ownership(
    document_id: int,
    current_user: User,
    db: AsyncSession
) -> bool:
    """
    Verify that current user owns the document.
    
    Args:
        document_id: Document ID to check
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        bool: True if user owns document or is admin
        
    Raises:
        HTTPException: If document not found or access denied
        
    Example:
```python
        @router.get("/documents/{document_id}")
        async def get_document(
            document_id: int,
            current_user: User = Depends(get_current_user),
            db: AsyncSession = Depends(get_db)
        ):
            await verify_document_ownership(document_id, current_user, db)
            # User is authorized to access document
            return {"document_id": document_id}
```
    """
    from app.models.document import Document
    
    # Admins can access all documents
    if current_user.role == UserRole.ADMIN:
        return True
    
    # Check if document exists and belongs to user
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    if document.user_id != current_user.id:
        logger.warning(
            f"Access denied: User {current_user.id} attempted to access "
            f"document {document_id} owned by user {document.user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this document"
        )
    
    return True


# ============================================================================
# AUDIT LOGGING DEPENDENCY
# ============================================================================
class AuditLogger:
    """
    Helper class for creating audit logs.
    """
    
    def __init__(self, db: AsyncSession, request: Request):
        """
        Initialize audit logger.
        
        Args:
            db: Database session
            request: FastAPI request object
        """
        self.db = db
        self.request = request
    
    async def log(
        self,
        user_id: Optional[int],
        action: ActionType,
        resource_type: ResourceType,
        resource_id: Optional[int] = None,
        details: Optional[dict] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Create audit log entry.
        
        Args:
            user_id: User ID performing action (None for anonymous)
            action: Type of action performed
            resource_type: Type of resource affected
            resource_id: ID of affected resource
            details: Additional details (JSON)
            success: Whether action succeeded
            error_message: Error message if failed
            
        Returns:
            AuditLog: Created audit log entry
            
        Example:
```python
            @router.post("/documents/upload")
            async def upload(
                audit: AuditLogger = Depends(get_audit_logger),
                current_user: User = Depends(get_current_user)
            ):
                # ... upload logic ...
                await audit.log(
                    user_id=current_user.id,
                    action=ActionType.DOCUMENT_UPLOAD,
                    resource_type=ResourceType.DOCUMENT,
                    resource_id=document.id,
                    details={"filename": file.filename}
                )
```
        """
        # Extract IP and user agent
        ip_address = self.request.client.host if self.request.client else None
        user_agent = self.request.headers.get("user-agent")
        
        # Create audit log entry
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
        
        self.db.add(audit_log)
        await self.db.commit()
        await self.db.refresh(audit_log)
        
        logger.info(
            f"Audit log created: {action.value} on {resource_type.value} "
            f"by user {user_id} (success={success})"
        )
        
        return audit_log


async def get_audit_logger(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> AuditLogger:
    """
    Dependency to get audit logger instance.
    
    Args:
        request: FastAPI request object
        db: Database session
        
    Returns:
        AuditLogger: Audit logger instance
        
    Example:
```python
        @router.post("/login")
        async def login(
            audit: AuditLogger = Depends(get_audit_logger)
        ):
            try:
                # ... login logic ...
                await audit.log(
                    user_id=user.id,
                    action=ActionType.LOGIN_SUCCESS,
                    resource_type=ResourceType.USER
                )
            except Exception as e:
                await audit.log(
                    user_id=None,
                    action=ActionType.LOGIN_FAILED,
                    resource_type=ResourceType.USER,
                    success=False,
                    error_message=str(e)
                )
```
    """
    return AuditLogger(db, request)


# ============================================================================
# PAGINATION DEPENDENCY
# ============================================================================
class PaginationParams:
    """
    Pagination parameters for list endpoints.
    """
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 20
    ):
        """
        Initialize pagination parameters.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page (1-100)
            
        Raises:
            HTTPException: If parameters are invalid
        """
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Page must be >= 1"
            )
        
        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Page size must be between 1 and 100"
            )
        
        self.page = page
        self.page_size = page_size
    
    @property
    def offset(self) -> int:
        """Calculate offset for SQL query."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit for SQL query."""
        return self.page_size


async def get_pagination(
    page: int = 1,
    page_size: int = 20
) -> PaginationParams:
    """
    Dependency to get pagination parameters.
    
    Args:
        page: Page number (1-indexed)
        page_size: Items per page (1-100)
        
    Returns:
        PaginationParams: Validated pagination parameters
        
    Example:
```python
        @router.get("/documents")
        async def list_documents(
            pagination: PaginationParams = Depends(get_pagination),
            db: AsyncSession = Depends(get_db)
        ):
            result = await db.execute(
                select(Document)
                .offset(pagination.offset)
                .limit(pagination.limit)
            )
            return result.scalars().all()
```
    """
    return PaginationParams(page=page, page_size=page_size)


# ============================================================================
# RATE LIMITING DEPENDENCY
# ============================================================================
class RateLimiter:
    """
    Simple rate limiter using Redis.
    """
    
    def __init__(
        self,
        requests: int = 60,
        window: int = 60,
        key_prefix: str = "rate_limit"
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests: Number of requests allowed
            window: Time window in seconds
            key_prefix: Redis key prefix
        """
        self.requests = requests
        self.window = window
        self.key_prefix = key_prefix
    
    async def __call__(
        self,
        request: Request,
        redis: Redis = Depends(get_redis)
    ) -> None:
        """
        Check rate limit for current request.
        
        Args:
            request: FastAPI request object
            redis: Redis connection
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        # Get client identifier (IP or user ID if authenticated)
        client_id = request.client.host if request.client else "unknown"
        
        # Try to get user ID from token
        try:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                token_data = decode_access_token(token)
                if token_data:
                    client_id = f"user_{token_data.get('user_id')}"
        except Exception:
            pass  # Use IP if token decode fails
        
        # Create Redis key
        key = f"{self.key_prefix}:{client_id}"
        
        # Increment counter
        current = await redis.incr(key)
        
        # Set expiration on first request
        if current == 1:
            await redis.expire(key, self.window)
        
        # Check if limit exceeded
        if current > self.requests:
            ttl = await redis.ttl(key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {ttl} seconds.",
                headers={"Retry-After": str(ttl)}
            )


# Rate limiter instances
rate_limit_standard = RateLimiter(requests=60, window=60)  # 60 req/min
rate_limit_strict = RateLimiter(requests=10, window=60)    # 10 req/min (auth endpoints)


# ============================================================================
# QUERY FILTERS DEPENDENCY
# ============================================================================
class DocumentFilters:
    """
    Common filters for document list endpoints.
    """
    
    def __init__(
        self,
        status: Optional[str] = None,
        document_type: Optional[str] = None,
        search: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ):
        """
        Initialize document filters.
        
        Args:
            status: Filter by document status
            document_type: Filter by document type
            search: Search in filename/title
            date_from: Filter by upload date (from)
            date_to: Filter by upload date (to)
        """
        self.status = status
        self.document_type = document_type
        self.search = search
        self.date_from = date_from
        self.date_to = date_to


async def get_document_filters(
    status: Optional[str] = None,
    document_type: Optional[str] = None,
    search: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None
) -> DocumentFilters:
    """
    Dependency to get document filters.
    
    Returns:
        DocumentFilters: Filter parameters
        
    Example:
```python
        @router.get("/documents")
        async def list_documents(
            filters: DocumentFilters = Depends(get_document_filters),
            db: AsyncSession = Depends(get_db)
        ):
            query = select(Document)
            
            if filters.status:
                query = query.where(Document.status == filters.status)
            
            if filters.search:
                query = query.where(
                    Document.filename.ilike(f"%{filters.search}%")
                )
            
            result = await db.execute(query)
            return result.scalars().all()
```
    """
    return DocumentFilters(
        status=status,
        document_type=document_type,
        search=search,
        date_from=date_from,
        date_to=date_to
    )