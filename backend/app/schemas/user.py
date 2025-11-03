"""
EUNOMIA Legal AI Platform - User Schemas
Pydantic schemas for user management and profiles
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict
from enum import Enum

from app.models.user import UserRole


# ============================================================================
# USER BASE SCHEMAS
# ============================================================================
class UserBase(BaseModel):
    """
    Base user schema with common fields.
    
    Used as parent class for other user schemas.
    """
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(
        None,
        min_length=2,
        max_length=255,
        description="User's full name"
    )
    role: UserRole = Field(default=UserRole.USER, description="User role")
    language: str = Field(
        default="fr",
        max_length=10,
        description="Preferred language (ISO 639-1 code)"
    )
    timezone: Optional[str] = Field(
        default="Europe/Paris",
        max_length=50,
        description="User timezone (IANA timezone)"
    )
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """
        Validate language code against supported languages.
        
        Args:
            v: Language code (fr, en, de, etc.)
            
        Returns:
            Validated language code
            
        Raises:
            ValueError: If language not supported
        """
        supported_languages = ["fr", "en", "de", "es", "it", "nl"]
        if v.lower() not in supported_languages:
            raise ValueError(
                f"Language '{v}' not supported. "
                f"Supported languages: {', '.join(supported_languages)}"
            )
        return v.lower()
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate timezone against IANA timezone database.
        
        Args:
            v: Timezone string (e.g., "Europe/Paris")
            
        Returns:
            Validated timezone
            
        Raises:
            ValueError: If timezone invalid
        """
        if v is None:
            return "Europe/Paris"
        
        # Common valid timezones
        valid_timezones = [
            "Europe/Paris", "Europe/London", "Europe/Berlin", "Europe/Brussels",
            "Europe/Amsterdam", "Europe/Madrid", "Europe/Rome", "Europe/Zurich",
            "America/New_York", "America/Los_Angeles", "America/Chicago",
            "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney",
            "UTC"
        ]
        
        if v not in valid_timezones:
            raise ValueError(
                f"Timezone '{v}' not recognized. "
                f"Use standard IANA timezone (e.g., 'Europe/Paris')"
            )
        
        return v


# ============================================================================
# USER CREATE SCHEMAS
# ============================================================================
class UserCreate(BaseModel):
    """
    Schema for creating a new user (admin only).
    
    Differs from RegisterRequest (public) by allowing role assignment.
    """
    email: EmailStr = Field(
        ...,
        description="User email address (must be unique)"
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Strong password"
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="User's full name"
    )
    role: UserRole = Field(
        default=UserRole.USER,
        description="User role (admin can assign any role)"
    )
    language: str = Field(
        default="fr",
        description="Preferred language"
    )
    timezone: str = Field(
        default="Europe/Paris",
        description="User timezone"
    )
    is_active: bool = Field(
        default=True,
        description="Account active status"
    )
    is_verified: bool = Field(
        default=False,
        description="Email verification status (admin can bypass)"
    )
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password strength (same rules as auth.py)"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least 1 uppercase letter")
        
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least 1 lowercase letter")
        
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least 1 digit")
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "jane.smith@example.com",
                "password": "SecurePass123!",
                "full_name": "Jane Smith",
                "role": "user",
                "language": "fr",
                "timezone": "Europe/Paris",
                "is_active": True,
                "is_verified": False
            }
        }
    )


# ============================================================================
# USER UPDATE SCHEMAS
# ============================================================================
class UserUpdate(BaseModel):
    """
    Schema for updating user profile (by user themselves).
    
    All fields are optional. Only provided fields will be updated.
    """
    full_name: Optional[str] = Field(
        None,
        min_length=2,
        max_length=255,
        description="User's full name"
    )
    language: Optional[str] = Field(
        None,
        max_length=10,
        description="Preferred language"
    )
    timezone: Optional[str] = Field(
        None,
        max_length=50,
        description="User timezone"
    )
    notifications_enabled: Optional[bool] = Field(
        None,
        description="Enable email notifications"
    )
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language if provided"""
        if v is None:
            return v
        
        supported_languages = ["fr", "en", "de", "es", "it", "nl"]
        if v.lower() not in supported_languages:
            raise ValueError(
                f"Language '{v}' not supported. "
                f"Supported languages: {', '.join(supported_languages)}"
            )
        return v.lower()
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> Optional[str]:
        """Validate timezone if provided"""
        if v is None:
            return v
        
        valid_timezones = [
            "Europe/Paris", "Europe/London", "Europe/Berlin", "Europe/Brussels",
            "Europe/Amsterdam", "Europe/Madrid", "Europe/Rome", "Europe/Zurich",
            "America/New_York", "America/Los_Angeles", "America/Chicago",
            "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney",
            "UTC"
        ]
        
        if v not in valid_timezones:
            raise ValueError(
                f"Timezone '{v}' not recognized. "
                f"Use standard IANA timezone (e.g., 'Europe/Paris')"
            )
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Jane Doe Smith",
                "language": "en",
                "timezone": "Europe/London",
                "notifications_enabled": True
            }
        }
    )


class UserUpdateAdmin(BaseModel):
    """
    Schema for updating user by admin.
    
    Allows modification of restricted fields (role, active status, etc.).
    """
    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    role: Optional[UserRole] = Field(None, description="User role")
    is_active: Optional[bool] = Field(None, description="Account active status")
    is_verified: Optional[bool] = Field(None, description="Email verification status")
    language: Optional[str] = Field(None, max_length=10)
    timezone: Optional[str] = Field(None, max_length=50)
    notifications_enabled: Optional[bool] = Field(None)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "manager",
                "is_active": True,
                "is_verified": True
            }
        }
    )


class UserUpdatePassword(BaseModel):
    """
    Schema for password change (authenticated user).
    """
    current_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Current password for verification"
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New strong password"
    )
    new_password_confirm: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password confirmation"
    )
    
    @field_validator('new_password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate new password strength"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least 1 uppercase letter")
        
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least 1 lowercase letter")
        
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least 1 digit")
        
        return v
    
    @field_validator('new_password_confirm')
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Validate that passwords match"""
        if 'new_password' in info.data and v != info.data['new_password']:
            raise ValueError("Passwords do not match")
        return v
    
    @field_validator('new_password')
    @classmethod
    def password_not_same_as_current(cls, v: str, info) -> str:
        """Ensure new password is different from current"""
        if 'current_password' in info.data and v == info.data['current_password']:
            raise ValueError("New password must be different from current password")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_password": "OldSecurePass123!",
                "new_password": "NewSecurePass456!",
                "new_password_confirm": "NewSecurePass456!"
            }
        }
    )


# ============================================================================
# USER RESPONSE SCHEMAS
# ============================================================================
class UserResponse(BaseModel):
    """
    Standard user response schema for API endpoints.
    
    Excludes sensitive fields (hashed_password, verification_token, etc.).
    """
    id: int = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="Account active status")
    is_verified: bool = Field(..., description="Email verification status")
    language: str = Field(..., description="Preferred language")
    timezone: Optional[str] = Field(None, description="User timezone")
    notifications_enabled: bool = Field(..., description="Notifications enabled")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "language": "fr",
                "timezone": "Europe/Paris",
                "notifications_enabled": True,
                "created_at": "2025-01-15T10:30:00Z",
                "last_login": "2025-11-01T08:45:00Z"
            }
        }
    )


class UserPublic(BaseModel):
    """
    Public user information (for sharing, collaboration features).
    
    Minimal information for privacy protection.
    """
    id: int = Field(..., description="User ID")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: UserRole = Field(..., description="User role")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "full_name": "John Doe",
                "role": "user"
            }
        }
    )


class UserWithStats(BaseModel):
    """
    User response with usage statistics (for dashboard).
    """
    id: int = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="Account active status")
    is_verified: bool = Field(..., description="Email verification status")
    language: str = Field(..., description="Preferred language")
    timezone: Optional[str] = Field(None, description="User timezone")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    # Statistics
    documents_count: int = Field(default=0, description="Total documents uploaded")
    analyses_count: int = Field(default=0, description="Total analyses completed")
    storage_used_mb: float = Field(default=0.0, description="Storage used in MB")
    last_document_uploaded: Optional[datetime] = Field(
        None,
        description="Last document upload timestamp"
    )
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "language": "fr",
                "timezone": "Europe/Paris",
                "created_at": "2025-01-15T10:30:00Z",
                "last_login": "2025-11-01T08:45:00Z",
                "documents_count": 15,
                "analyses_count": 15,
                "storage_used_mb": 47.3,
                "last_document_uploaded": "2025-11-01T07:30:00Z"
            }
        }
    )


class UserDetail(BaseModel):
    """
    Detailed user information (admin view or user profile).
    
    Includes additional metadata and GDPR information.
    """
    id: int = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="Account active status")
    is_verified: bool = Field(..., description="Email verification status")
    language: str = Field(..., description="Preferred language")
    timezone: Optional[str] = Field(None, description="User timezone")
    notifications_enabled: bool = Field(..., description="Notifications enabled")
    
    # Timestamps
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    password_changed_at: Optional[datetime] = Field(None, description="Last password change")
    
    # GDPR
    gdpr_consent_given_at: Optional[datetime] = Field(None, description="GDPR consent timestamp")
    gdpr_consent_version: Optional[str] = Field(None, description="Terms version accepted")
    data_retention_until: Optional[datetime] = Field(None, description="Data retention date")
    anonymized: bool = Field(default=False, description="Account anonymized")
    
    # API Access
    has_api_key: bool = Field(default=False, description="Has active API key")
    api_key_created_at: Optional[datetime] = Field(None, description="API key creation date")
    
    # Statistics
    documents_count: int = Field(default=0, description="Total documents uploaded")
    analyses_count: int = Field(default=0, description="Total analyses completed")
    storage_used_mb: float = Field(default=0.0, description="Storage used in MB")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "language": "fr",
                "timezone": "Europe/Paris",
                "notifications_enabled": True,
                "created_at": "2025-01-15T10:30:00Z",
                "updated_at": "2025-10-28T14:20:00Z",
                "last_login": "2025-11-01T08:45:00Z",
                "last_activity": "2025-11-01T09:15:00Z",
                "password_changed_at": "2025-08-10T16:30:00Z",
                "gdpr_consent_given_at": "2025-01-15T10:30:00Z",
                "gdpr_consent_version": "1.0",
                "data_retention_until": "2028-01-15T10:30:00Z",
                "anonymized": False,
                "has_api_key": True,
                "api_key_created_at": "2025-02-01T09:00:00Z",
                "documents_count": 15,
                "analyses_count": 15,
                "storage_used_mb": 47.3
            }
        }
    )


# ============================================================================
# USER LIST SCHEMAS
# ============================================================================
class UserListResponse(BaseModel):
    """
    Paginated user list response (admin endpoint).
    """
    users: List[UserResponse] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "users": [
                    {
                        "id": 1,
                        "email": "john.doe@example.com",
                        "full_name": "John Doe",
                        "role": "user",
                        "is_active": True,
                        "is_verified": True,
                        "language": "fr",
                        "timezone": "Europe/Paris",
                        "notifications_enabled": True,
                        "created_at": "2025-01-15T10:30:00Z",
                        "last_login": "2025-11-01T08:45:00Z"
                    }
                ],
                "total": 150,
                "page": 1,
                "page_size": 20,
                "total_pages": 8
            }
        }
    )


# ============================================================================
# USER DELETION SCHEMAS (GDPR)
# ============================================================================
class UserDeleteRequest(BaseModel):
    """
    User deletion request (GDPR right to be forgotten).
    """
    password: str = Field(
        ...,
        min_length=8,
        description="Password confirmation for security"
    )
    confirm_deletion: bool = Field(
        ...,
        description="Explicit confirmation (must be True)"
    )
    reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional reason for deletion"
    )
    
    @field_validator('confirm_deletion')
    @classmethod
    def validate_confirmation(cls, v: bool) -> bool:
        """Ensure explicit confirmation"""
        if not v:
            raise ValueError("Deletion must be explicitly confirmed")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "password": "MySecurePass123!",
                "confirm_deletion": True,
                "reason": "No longer need the service"
            }
        }
    )


class UserDeleteResponse(BaseModel):
    """
    User deletion response.
    """
    message: str = Field(..., description="Success message")
    deleted_at: datetime = Field(..., description="Deletion timestamp")
    data_retention_until: datetime = Field(
        ...,
        description="Date when data will be permanently deleted (GDPR 30-day grace)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Account deletion requested. Data will be permanently deleted after 30 days.",
                "deleted_at": "2025-11-01T12:00:00Z",
                "data_retention_until": "2025-12-01T12:00:00Z"
            }
        }
    )