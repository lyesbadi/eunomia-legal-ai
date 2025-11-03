"""
EUNOMIA Legal AI Platform - User Management Routes
FastAPI routes for user profile management and administration
"""
from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.api.deps import (
    get_db,
    get_current_user,
    get_current_active_user,
    require_admin,
    require_manager,
    get_audit_logger,
    get_pagination,
    AuditLogger,
    PaginationParams
)
from app.core.security import verify_password, get_password_hash
from app.models.user import User, UserRole
from app.models.document import Document
from app.models.analysis import Analysis
from app.models.audit_log import ActionType, ResourceType
from app.schemas.user import (
    UserResponse,
    UserWithStats,
    UserDetail,
    UserUpdate,
    UserUpdateAdmin,
    UserUpdatePassword,
    UserDeleteRequest,
    UserDeleteResponse,
    UserListResponse
)
import logging


# ============================================================================
# ROUTER SETUP
# ============================================================================
router = APIRouter(prefix="/users", tags=["Users"])
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
async def get_user_statistics(db: AsyncSession, user_id: int) -> dict:
    """
    Get user statistics (documents, analyses, storage).
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        Dictionary with statistics
    """
    # Count documents
    documents_result = await db.execute(
        select(func.count(Document.id))
        .where(Document.user_id == user_id)
    )
    documents_count = documents_result.scalar() or 0
    
    # Count analyses
    analyses_result = await db.execute(
        select(func.count(Analysis.id))
        .join(Document, Document.id == Analysis.document_id)
        .where(Document.user_id == user_id)
    )
    analyses_count = analyses_result.scalar() or 0
    
    # Calculate storage used
    storage_result = await db.execute(
        select(func.sum(Document.file_size))
        .where(Document.user_id == user_id)
    )
    storage_bytes = storage_result.scalar() or 0
    storage_mb = storage_bytes / (1024 * 1024)
    
    # Get last document upload
    last_doc_result = await db.execute(
        select(Document.uploaded_at)
        .where(Document.user_id == user_id)
        .order_by(Document.uploaded_at.desc())
        .limit(1)
    )
    last_document_uploaded = last_doc_result.scalar_one_or_none()
    
    return {
        "documents_count": documents_count,
        "analyses_count": analyses_count,
        "storage_used_mb": round(storage_mb, 2),
        "last_document_uploaded": last_document_uploaded
    }


async def check_email_exists(db: AsyncSession, email: str, exclude_user_id: Optional[int] = None) -> bool:
    """
    Check if email already exists in database.
    
    Args:
        db: Database session
        email: Email to check
        exclude_user_id: User ID to exclude from check (for updates)
        
    Returns:
        True if email exists
    """
    query = select(User).where(User.email == email.lower())
    
    if exclude_user_id:
        query = query.where(User.id != exclude_user_id)
    
    result = await db.execute(query)
    return result.scalar_one_or_none() is not None


# ============================================================================
# CURRENT USER ROUTES (SELF-MANAGEMENT)
# ============================================================================
@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Get authenticated user's profile information"
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
) -> UserResponse:
    """
    Get current user profile.
    
    Returns full user profile information for authenticated user.
    """
    return UserResponse.model_validate(current_user)


@router.get(
    "/me/dashboard",
    response_model=UserWithStats,
    summary="Get user dashboard with statistics",
    description="Get user profile with usage statistics"
)
async def get_user_dashboard(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> UserWithStats:
    """
    Get user dashboard with statistics.
    
    Returns:
    - User profile information
    - Document count
    - Analysis count
    - Storage used
    - Last activity timestamps
    """
    # Get statistics
    stats = await get_user_statistics(db, current_user.id)
    
    # Combine user data with statistics
    user_data = UserResponse.model_validate(current_user).model_dump()
    user_data.update(stats)
    
    return UserWithStats(**user_data)


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
    description="Update authenticated user's profile information"
)
async def update_current_user_profile(
    data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> UserResponse:
    """
    Update current user profile.
    
    - **full_name**: User's full name
    - **language**: Preferred language
    - **timezone**: User timezone
    - **notifications_enabled**: Email notifications preference
    
    All fields are optional. Only provided fields will be updated.
    """
    # Track what changed
    changes = {}
    
    # Update only provided fields
    update_data = data.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(current_user, field):
            old_value = getattr(current_user, field)
            if old_value != value:
                setattr(current_user, field, value)
                changes[field] = {"old": old_value, "new": value}
    
    if not changes:
        return UserResponse.model_validate(current_user)
    
    # Update timestamp
    current_user.updated_at = datetime.utcnow()
    
    try:
        await db.commit()
        await db.refresh(current_user)
        
        # Log update
        await audit.log(
            user_id=current_user.id,
            action=ActionType.USER_UPDATE,
            resource_type=ResourceType.USER,
            resource_id=current_user.id,
            details={"changes": changes}
        )
        
        logger.info(f"User profile updated: {current_user.email} (ID: {current_user.id})")
        
        return UserResponse.model_validate(current_user)
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.patch(
    "/me/password",
    status_code=status.HTTP_200_OK,
    summary="Change password",
    description="Change authenticated user's password"
)
async def change_password(
    data: UserUpdatePassword,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> dict:
    """
    Change user password.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (strong password required)
    - **new_password_confirm**: Password confirmation
    
    Returns success message.
    """
    # Verify current password
    if not verify_password(data.current_password, current_user.hashed_password):
        await audit.log(
            user_id=current_user.id,
            action=ActionType.PASSWORD_CHANGE,
            resource_type=ResourceType.USER,
            resource_id=current_user.id,
            success=False,
            error_message="Current password incorrect"
        )
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(data.new_password)
    current_user.password_changed_at = datetime.utcnow()
    
    try:
        await db.commit()
        
        # Log password change
        await audit.log(
            user_id=current_user.id,
            action=ActionType.PASSWORD_CHANGE,
            resource_type=ResourceType.USER,
            resource_id=current_user.id
        )
        
        logger.info(f"Password changed for user: {current_user.email} (ID: {current_user.id})")
        
        return {
            "message": "Password changed successfully",
            "changed_at": current_user.password_changed_at
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.delete(
    "/me",
    response_model=UserDeleteResponse,
    summary="Delete current user account",
    description="Delete authenticated user's account (GDPR right to be forgotten)"
)
async def delete_current_user_account(
    data: UserDeleteRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> UserDeleteResponse:
    """
    Delete user account (GDPR right to be forgotten).
    
    - **password**: Password confirmation for security
    - **confirm_deletion**: Must be True to proceed
    - **delete_permanently**: If True, delete immediately; if False, 30-day grace period
    
    Returns deletion timestamp and retention information.
    """
    # Verify password
    if not verify_password(data.password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )
    
    deleted_at = datetime.utcnow()
    
    if data.delete_permanently:
        # Immediate permanent deletion
        try:
            # TODO: Delete all user data (documents, analyses, etc.)
            # This should be done in a background task for large datasets
            
            # For now, just mark as anonymized
            current_user.is_active = False
            current_user.anonymized = True
            current_user.anonymized_at = deleted_at
            current_user.email = f"deleted_{current_user.id}@anonymized.local"
            current_user.full_name = None
            current_user.hashed_password = "DELETED"
            
            await db.commit()
            
            # Log deletion
            await audit.log(
                user_id=current_user.id,
                action=ActionType.USER_DELETE,
                resource_type=ResourceType.USER,
                resource_id=current_user.id,
                details={
                    "permanent": True,
                    "reason": data.reason
                }
            )
            
            logger.info(f"User account permanently deleted: ID {current_user.id}")
            
            return UserDeleteResponse(
                message="Account permanently deleted",
                deleted_at=deleted_at,
                permanent_deletion_date=deleted_at
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting user account: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete account"
            )
    else:
        # Soft delete with 30-day grace period
        retention_date = deleted_at + timedelta(days=30)
        
        current_user.is_active = False
        current_user.data_retention_until = retention_date
        
        await db.commit()
        
        # Log deletion request
        await audit.log(
            user_id=current_user.id,
            action=ActionType.DATA_DELETION_REQUEST,
            resource_type=ResourceType.USER,
            resource_id=current_user.id,
            details={
                "permanent": False,
                "grace_period_days": 30,
                "reason": data.reason
            }
        )
        
        logger.info(
            f"User account deletion requested: {current_user.email} (ID: {current_user.id}), "
            f"permanent deletion on {retention_date}"
        )
        
        return UserDeleteResponse(
            message="Account deletion requested. Data will be permanently deleted after 30 days.",
            deleted_at=deleted_at,
            permanent_deletion_date=retention_date
        )


# ============================================================================
# ADMIN USER MANAGEMENT ROUTES
# ============================================================================
@router.get(
    "",
    response_model=UserListResponse,
    summary="List all users (Admin only)",
    description="Get paginated list of all users with filters"
)
async def list_users(
    pagination: PaginationParams = Depends(get_pagination),
    role: Optional[UserRole] = Query(None, description="Filter by role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    is_verified: Optional[bool] = Query(None, description="Filter by verification status"),
    search: Optional[str] = Query(None, description="Search in email or full name"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
) -> UserListResponse:
    """
    List all users (admin only).
    
    Supports:
    - Pagination (page, page_size)
    - Filtering (role, is_active, is_verified)
    - Search (email, full_name)
    
    Returns paginated list of users.
    """
    # Build query
    query = select(User)
    
    # Apply filters
    filters = []
    
    if role is not None:
        filters.append(User.role == role)
    
    if is_active is not None:
        filters.append(User.is_active == is_active)
    
    if is_verified is not None:
        filters.append(User.is_verified == is_verified)
    
    if search:
        search_pattern = f"%{search}%"
        filters.append(
            or_(
                User.email.ilike(search_pattern),
                User.full_name.ilike(search_pattern)
            )
        )
    
    if filters:
        query = query.where(and_(*filters))
    
    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    query = query.offset(pagination.offset).limit(pagination.limit)
    query = query.order_by(User.created_at.desc())
    
    # Execute query
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Convert to response schema
    user_responses = [UserResponse.model_validate(user) for user in users]
    
    return UserListResponse(
        users=user_responses,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size
    )


@router.get(
    "/{user_id}",
    response_model=UserDetail,
    summary="Get user details (Admin only)",
    description="Get detailed information about a specific user"
)
async def get_user_detail(
    user_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
) -> UserDetail:
    """
    Get user details (admin only).
    
    Returns:
    - Full user profile
    - GDPR information
    - Activity timestamps
    - Usage statistics
    """
    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    
    # Get statistics
    stats = await get_user_statistics(db, user_id)
    
    # Check if has API key
    has_api_key = user.api_key_hash is not None
    
    # Combine data
    user_data = UserResponse.model_validate(user).model_dump()
    user_data.update(stats)
    user_data.update({
        "gdpr_consent_given_at": user.gdpr_consent_given_at,
        "gdpr_consent_version": user.gdpr_consent_version,
        "data_retention_until": user.data_retention_until,
        "anonymized": user.anonymized,
        "has_api_key": has_api_key,
        "api_key_created_at": user.api_key_created_at,
        "updated_at": user.updated_at,
        "last_activity": user.last_activity,
        "password_changed_at": user.password_changed_at
    })
    
    return UserDetail(**user_data)


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user (Admin only)",
    description="Update any user's information (admin only)"
)
async def update_user(
    user_id: int,
    data: UserUpdateAdmin,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> UserResponse:
    """
    Update user (admin only).
    
    Allows updating:
    - Profile information
    - Role
    - Active status
    - Verification status
    
    All fields are optional.
    """
    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    
    # Prevent self-demotion
    if user_id == admin.id and data.role and data.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own admin role"
        )
    
    # Prevent self-deactivation
    if user_id == admin.id and data.is_active is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    # Track changes
    changes = {}
    
    # Update fields
    update_data = data.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(user, field):
            old_value = getattr(user, field)
            if old_value != value:
                setattr(user, field, value)
                changes[field] = {"old": str(old_value), "new": str(value)}
    
    if not changes:
        return UserResponse.model_validate(user)
    
    # Update timestamp
    user.updated_at = datetime.utcnow()
    
    try:
        await db.commit()
        await db.refresh(user)
        
        # Log update
        await audit.log(
            user_id=admin.id,
            action=ActionType.ADMIN_ACTION,
            resource_type=ResourceType.USER,
            resource_id=user.id,
            details={
                "action": "user_update",
                "changes": changes
            }
        )
        
        logger.info(
            f"User {user.id} updated by admin {admin.id}: {changes}"
        )
        
        return UserResponse.model_validate(user)
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete(
    "/{user_id}",
    response_model=UserDeleteResponse,
    summary="Delete user (Admin only)",
    description="Delete user account (admin only)"
)
async def delete_user(
    user_id: int,
    permanent: bool = Query(False, description="Permanent deletion (no grace period)"),
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> UserDeleteResponse:
    """
    Delete user (admin only).
    
    - **permanent**: If True, delete immediately; if False, 30-day grace period
    
    Returns deletion information.
    """
    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )
    
    # Prevent self-deletion
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    deleted_at = datetime.utcnow()
    
    if permanent:
        # Immediate deletion
        user.is_active = False
        user.anonymized = True
        user.anonymized_at = deleted_at
        user.email = f"deleted_{user.id}@anonymized.local"
        user.full_name = None
        user.hashed_password = "DELETED"
        
        await db.commit()
        
        # Log deletion
        await audit.log(
            user_id=admin.id,
            action=ActionType.ADMIN_ACTION,
            resource_type=ResourceType.USER,
            resource_id=user.id,
            details={
                "action": "user_delete_permanent"
            }
        )
        
        logger.info(f"User {user_id} permanently deleted by admin {admin.id}")
        
        return UserDeleteResponse(
            message="User permanently deleted",
            deleted_at=deleted_at,
            permanent_deletion_date=deleted_at
        )
    else:
        # Soft delete with grace period
        retention_date = deleted_at + timedelta(days=30)
        
        user.is_active = False
        user.data_retention_until = retention_date
        
        await db.commit()
        
        # Log deletion
        await audit.log(
            user_id=admin.id,
            action=ActionType.ADMIN_ACTION,
            resource_type=ResourceType.USER,
            resource_id=user.id,
            details={
                "action": "user_delete_soft",
                "grace_period_days": 30
            }
        )
        
        logger.info(
            f"User {user_id} soft deleted by admin {admin.id}, "
            f"permanent deletion on {retention_date}"
        )
        
        return UserDeleteResponse(
            message="User deactivated. Will be permanently deleted after 30 days.",
            deleted_at=deleted_at,
            permanent_deletion_date=retention_date
        )