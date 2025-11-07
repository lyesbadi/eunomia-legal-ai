"""
EUNOMIA Legal AI Platform - Authentication Routes
FastAPI routes for user authentication, registration, and token management
"""
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.api.deps import (
    get_db,
    get_current_user,
    get_current_active_user,
    get_audit_logger,
    AuditLogger,
    rate_limit_strict
)
from app.core.security import (
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_secure_token,
    generate_api_key
)
from app.core.config import settings
from app.models.user import User, UserRole
from app.models.audit_log import ActionType, ResourceType
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
    PasswordResetRequest,
    PasswordResetConfirm,
    PasswordResetResponse,
    EmailVerificationRequest,
    EmailVerificationResponse,
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyRevokeResponse
)
from app.schemas.user import UserResponse
import logging


# ============================================================================
# ROUTER SETUP
# ============================================================================
router = APIRouter(tags=["Authentication"])
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """
    Get user by email address.
    
    Args:
        db: Database session
        email: User email
        
    Returns:
        User or None if not found
    """
    result = await db.execute(
        select(User).where(User.email == email.lower())
    )
    return result.scalar_one_or_none()


async def send_verification_email(email: str, token: str) -> None:
    """
    Send email verification email (placeholder for actual email service).
    
    Args:
        email: User email
        token: Verification token
        
    Note:
        In production, integrate with email service (SendGrid, Mailgun, etc.)
    """
    verification_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    
    # TODO: Integrate with email service
    logger.info(f"Email verification URL for {email}: {verification_url}")
    
    # For now, just log (in production, send actual email)
    # await email_service.send_template(
    #     to=email,
    #     template="email_verification",
    #     context={"verification_url": verification_url}
    # )


async def send_password_reset_email(email: str, token: str) -> None:
    """
    Send password reset email (placeholder).
    
    Args:
        email: User email
        token: Reset token
    """
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    
    logger.info(f"Password reset URL for {email}: {reset_url}")
    
    # TODO: Integrate with email service


# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================
@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit_strict)],
    summary="Register new user account",
    description="Create a new user account with email and password. Email verification required."
)
async def register(
    data: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> RegisterResponse:
    """
    Register new user account.
    
    - **email**: Valid email address (unique)
    - **password**: Strong password (min 8 chars, 1 uppercase, 1 lowercase, 1 digit)
    - **full_name**: User's full name
    - **gdpr_consent**: Must be True to create account
    - **language**: Preferred language (default: fr)
    
    Returns user information and sends verification email.
    """
    # Check if email already exists
    existing_user = await get_user_by_email(db, data.email)
    if existing_user:
        await audit.log(
            user_id=None,
            action=ActionType.USER_REGISTER,
            resource_type=ResourceType.USER,
            success=False,
            error_message="Email already registered"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address already registered"
        )
    
    # Create new user
    try:
        # Hash password
        hashed_password = hash_password(data.password)
        
        # Generate verification token
        verification_token = generate_secure_token()
        
        # Create user object
        new_user = User(
            email=data.email.lower(),
            hashed_password=hashed_password,
            full_name=data.full_name,
            role=UserRole.USER,  # Default role
            language=data.language,
            is_active=True,
            is_verified=False,
            verification_token=verification_token,
            gdpr_consent_given_at=datetime.utcnow(),
            gdpr_consent_version=settings.TERMS_VERSION,
            data_retention_until=datetime.utcnow() + timedelta(days=3*365)  # 3 years
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Log successful registration
        await audit.log(
            user_id=new_user.id,
            action=ActionType.USER_REGISTER,
            resource_type=ResourceType.USER,
            resource_id=new_user.id,
            details={
                "email": new_user.email,
                "role": new_user.role.value
            }
        )
        
        # Send verification email in background
        background_tasks.add_task(
            send_verification_email,
            new_user.email,
            verification_token
        )
        
        logger.info(f"New user registered: {new_user.email} (ID: {new_user.id})")
        
        return RegisterResponse(
            message="Account created successfully. Please check your email to verify your account.",
            user={
                "id": new_user.id,
                "email": new_user.email,
                "full_name": new_user.full_name,
                "role": new_user.role.value,
                "is_active": new_user.is_active,
                "is_verified": new_user.is_verified
            },
            email_verification_required=True
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Registration error: {e}")
        
        await audit.log(
            user_id=None,
            action=ActionType.USER_REGISTER,
            resource_type=ResourceType.USER,
            success=False,
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create account. Please try again."
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    dependencies=[Depends(rate_limit_strict)],
    summary="Login with email and password",
    description="Authenticate user and receive access + refresh tokens"
)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> LoginResponse:
    """
    Authenticate user with email and password.
    
    Lockout policy:
    - Max 5 failed attempts
    - 15 minutes lockout duration
    - Auto-reset on successful login
    
    - **email**: User email address
    - **password**: User password
    
    Returns JWT access token and refresh token.
    """
    # Get user by email
    user = await get_user_by_email(db, data.email)
    
    if not user:
        await audit.log(
            user_id=None,
            action=ActionType.LOGIN_FAILED,
            resource_type=ResourceType.USER,
            success=False,
            error_message="Invalid credentials"
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # ✅ ÉTAPE 1 : Vérifier si le compte est verrouillé
    if user.locked_until and user.locked_until > datetime.utcnow():
        remaining = (user.locked_until - datetime.utcnow()).seconds // 60
        
        await audit.log(
            user_id=user.id,
            action=ActionType.LOGIN_FAILED,
            resource_type=ResourceType.USER,
            success=False,
            error_message=f"Account locked ({remaining} min remaining)"
        )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account locked due to too many failed attempts. Try again in {remaining} minutes."
        )
    
    # Check if account is active
    if not user.is_active:
        await audit.log(
            user_id=user.id,
            action=ActionType.LOGIN_FAILED,
            resource_type=ResourceType.USER,
            success=False,
            error_message="Account inactive"
        )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact support."
        )
    
    # Verify password
    if not verify_password(data.password, user.hashed_password):
        # ✅ ÉTAPE 2 : Incrémenter le compteur d'échecs
        user.failed_login_attempts += 1
        user.last_failed_login = datetime.utcnow()
        
        # ✅ ÉTAPE 3 : Appliquer le verrouillage au seuil
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.utcnow() + timedelta(minutes=15)
            await db.commit()
            
            await audit.log(
                user_id=user.id,
                action=ActionType.LOGIN_FAILED,
                resource_type=ResourceType.USER,
                success=False,
                error_message="Account locked after 5 failed attempts"
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account locked due to too many failed attempts. Try again in 15 minutes."
            )
        
        await db.commit()
        
        await audit.log(
            user_id=user.id,
            action=ActionType.LOGIN_FAILED,
            resource_type=ResourceType.USER,
            success=False,
            error_message="Invalid credentials"
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # ✅ ÉTAPE 4 : Réinitialiser les compteurs au succès
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.utcnow()
    user.last_activity = datetime.utcnow()
    
    # Create tokens
    access_token = create_access_token(
        data={
            "user_id": user.id,
            "email": user.email,
            "role": user.role.value
        }
    )
    
    refresh_token = create_refresh_token(
        data={
            "user_id": user.id,
            "email": user.email
        }
    )
    
    await db.commit()
    
    # Log successful login
    await audit.log(
        user_id=user.id,
        action=ActionType.LOGIN_SUCCESS,
        resource_type=ResourceType.USER,
        resource_id=user.id
    )
    
    logger.info(f"User logged in: {user.email} (ID: {user.id})")
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "is_verified": user.is_verified
        }
    )

@router.post(
    "/refresh",
    response_model=TokenRefreshResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token"
)
async def refresh_token(
    data: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> TokenRefreshResponse:
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token.
    """
    # Decode refresh token
    try:
            token_data = decode_token(data.refresh_token)
            if token_data is None or token_data.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    user_id = token_data.get("user_id")
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Create new access token
    new_access_token = create_access_token(
        data={
            "user_id": user.id,
            "email": user.email,
            "role": user.role.value
        }
    )
    
    # Update last activity
    user.last_activity = datetime.utcnow()
    await db.commit()
    
    logger.info(f"Token refreshed for user: {user.email} (ID: {user.id})")
    
    return TokenRefreshResponse(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout user",
    description="Logout current user (client-side token deletion)"
)
async def logout(
    current_user: User = Depends(get_current_user),
    audit: AuditLogger = Depends(get_audit_logger)
) -> None:
    """
    Logout user.
    
    Note: This is mainly a client-side operation (delete tokens).
    Server-side, we just log the action for audit trail.
    """
    await audit.log(
        user_id=current_user.id,
        action=ActionType.LOGOUT,
        resource_type=ResourceType.USER,
        resource_id=current_user.id
    )
    
    logger.info(f"User logged out: {current_user.email} (ID: {current_user.id})")


# ============================================================================
# EMAIL VERIFICATION
# ============================================================================
@router.post(
    "/verify-email",
    response_model=EmailVerificationResponse,
    summary="Verify email address",
    description="Verify user email using token from verification email"
)
async def verify_email(
    data: EmailVerificationRequest,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> EmailVerificationResponse:
    """
    Verify user email address.
    
    - **token**: Email verification token from registration email
    
    Returns updated user information.
    """
    # Find user with verification token
    result = await db.execute(
        select(User).where(User.verification_token == data.token)
    )
    user = result.scalar_one_or_none()
    
    if not user:

        await audit.log(
            user_id=None,
            action=ActionType.PASSWORD_RESET_COMPLETE,
            resource_type=ResourceType.USER,
            success=False,
            error_message="Invalid reset token"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Check if already verified
    if user.is_verified:
        return EmailVerificationResponse(
            message="Email already verified",
            user={
                "id": user.id,
                "email": user.email,
                "is_verified": True
            }
        )
    
    # Verify email
    user.is_verified = True
    user.verification_token = None  # Clear token after use
    await db.commit()
    
    # Log verification
    await audit.log(
        user_id=user.id,
        action=ActionType.USER_UPDATE,
        resource_type=ResourceType.USER,
        resource_id=user.id,
        details={"action": "email_verified"}
    )
    
    logger.info(f"Email verified for user: {user.email} (ID: {user.id})")
    
    return EmailVerificationResponse(
        message="Email verified successfully",
        user={
            "id": user.id,
            "email": user.email,
            "is_verified": True
        }
    )


# ============================================================================
# PASSWORD RESET
# ============================================================================
@router.post(
    "/password-reset",
    response_model=PasswordResetResponse,
    dependencies=[Depends(rate_limit_strict)],
    summary="Request password reset",
    description="Send password reset email to user"
)
async def request_password_reset(
    data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> PasswordResetResponse:
    """
    Request password reset email.
    
    - **email**: User email address
    
    Always returns success (even if email not found) for security.
    """
    # Get user by email
    user = await get_user_by_email(db, data.email)
    
    if user:
        # Generate reset token
        reset_token = generate_secure_token()
        
        # Store token in user (or separate password_reset table)
        user.password_reset_token = reset_token
        user.password_reset_expires_at = datetime.utcnow() + timedelta(hours=1) # Token expires in 1 hour
        await db.commit()
        
        # Send reset email in background
        background_tasks.add_task(
            send_password_reset_email,
            user.email,
            reset_token
        )
        
        # Log password reset request
        await audit.log(
            user_id=user.id,
            action=ActionType.PASSWORD_RESET_REQUEST,
            resource_type=ResourceType.USER,
            resource_id=user.id
        )
        
        logger.info(f"Password reset requested for: {user.email}")
    else:
        # Don't reveal if email exists or not (security)
        logger.warning(f"Password reset requested for non-existent email: {data.email}")
    
    # Always return success message
    return PasswordResetResponse(
        message="If the email exists, a password reset link has been sent."
    )


@router.post(
    "/password-reset/confirm",
    response_model=PasswordResetResponse,
    dependencies=[Depends(rate_limit_strict)],
    summary="Confirm password reset",
    description="Reset password using token from email"
)
async def confirm_password_reset(
    data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> PasswordResetResponse:
    """
    Reset password with token.
    
    - **token**: Password reset token from email
    - **new_password**: New password
    - **new_password_confirm**: Password confirmation
    
    Returns success message.
    """
    # Find user with reset token
    result = await db.execute(
        select(User).where(User.password_reset_token == data.token)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    if user.password_reset_expires_at and user.password_reset_expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired. Please request a new one."
        )
    
    # Hash new password
    user.hashed_password = hash_password(data.new_password)
    user.password_changed_at = datetime.utcnow()
    user.password_reset_token = None
    user.password_reset_expires_at = None
    await db.commit()
    
    # Log password reset
    await audit.log(
        user_id=user.id,
        action=ActionType.PASSWORD_RESET_COMPLETE,
        resource_type=ResourceType.USER,
        resource_id=user.id
    )
    
    logger.info(f"Password reset completed for: {user.email} (ID: {user.id})")
    
    return PasswordResetResponse(
        message="Password reset successfully. You can now login with your new password."
    )


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================
@router.post(
    "/api-key",
    response_model=APIKeyCreateResponse,
    summary="Create API key",
    description="Generate API key for programmatic access"
)
async def create_api_key(
    data: APIKeyCreateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> APIKeyCreateResponse:
    """
    Create API key for programmatic access.
    
    - **name**: API key name/description
    
    WARNING: The API key is only shown once. Store it securely.
    """
    # Check if user already has an API key
    if current_user.api_key_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have an active API key. Revoke it first to create a new one."
        )
    
    # Generate API key
    api_key = generate_api_key(prefix=f"user_{current_user.id}")
    api_key_hash = hash_password(api_key)
    
    # Store hashed version
    current_user.api_key_hash = api_key_hash
    current_user.api_key_created_at = datetime.utcnow()
    await db.commit()
    
    # Log API key creation
    await audit.log(
        user_id=current_user.id,
        action=ActionType.API_KEY_CREATED,
        resource_type=ResourceType.USER,
        resource_id=current_user.id,
        details={"name": data.name}
    )
    
    logger.info(f"API key created for user: {current_user.email} (ID: {current_user.id})")
    
    return APIKeyCreateResponse(
        api_key=api_key,
        name=data.name,
        created_at=current_user.api_key_created_at,
        message="API key created successfully. Store it securely - it won't be shown again."
    )


@router.delete(
    "/api-key",
    response_model=APIKeyRevokeResponse,
    summary="Revoke API key",
    description="Revoke current API key"
)
async def revoke_api_key(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> APIKeyRevokeResponse:
    """
    Revoke API key.
    
    The API key will no longer be valid for authentication.
    """
    if not current_user.api_key_hash:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active API key found"
        )
    
    # Revoke API key
    current_user.api_key_hash = None
    current_user.api_key_created_at = None
    await db.commit()
    
    # Log API key revocation
    await audit.log(
        user_id=current_user.id,
        action=ActionType.API_KEY_REVOKED,
        resource_type=ResourceType.USER,
        resource_id=current_user.id
    )
    
    logger.info(f"API key revoked for user: {current_user.email} (ID: {current_user.id})")
    
    return APIKeyRevokeResponse(
        message="API key revoked successfully",
        revoked_at=datetime.utcnow()
    )
# ============================================================================
# CURRENT USER PROFILE
# ============================================================================
@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Get authenticated user's profile information (alias for /users/me)"
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
) -> UserResponse:
    """
    Get current user profile.
    
    This is an alias for GET /api/v1/users/me for convenience.
    Returns full user profile information for authenticated user.
    
    **Authentication required:** JWT Bearer token
    
    Returns:
    - User ID, email, full name
    - Role and verification status
    - Language preferences
    - Activity timestamps
    """
    return UserResponse.model_validate(current_user)