"""
EUNOMIA Legal AI Platform - Authentication Schemas
Pydantic schemas for authentication and authorization
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator
from enum import Enum

from app.models.user import UserRole


# ============================================================================
# TOKEN SCHEMAS
# ============================================================================
class TokenData(BaseModel):
    """
    JWT token payload data.
    
    Used internally for token validation and user identification.
    """
    user_id: int = Field(..., description="User ID from database")
    email: EmailStr = Field(..., description="User email address")
    role: UserRole = Field(..., description="User role for authorization")
    exp: datetime = Field(..., description="Token expiration timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "email": "john.doe@example.com",
                "role": "user",
                "exp": "2025-11-01T12:00:00Z"
            }
        }


class Token(BaseModel):
    """
    JWT access token response.
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }


class TokenPair(BaseModel):
    """
    JWT token pair (access + refresh).
    """
    access_token: str = Field(..., description="JWT access token (short-lived)")
    refresh_token: str = Field(..., description="JWT refresh token (long-lived)")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }


# ============================================================================
# LOGIN SCHEMAS
# ============================================================================
class LoginRequest(BaseModel):
    """
    User login request schema.
    
    Validates email and password format before authentication.
    """
    email: EmailStr = Field(
        ...,
        description="User email address",
        examples=["john.doe@example.com"]
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password (8-128 characters)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "john.doe@example.com",
                "password": "SecurePass123!"
            }
        }


class LoginResponse(BaseModel):
    """
    Successful login response with tokens and user info.
    """
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token TTL in seconds")
    user: dict = Field(..., description="Basic user information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "id": 1,
                    "email": "john.doe@example.com",
                    "full_name": "John Doe",
                    "role": "user"
                }
            }
        }


# ============================================================================
# REGISTRATION SCHEMAS
# ============================================================================
class RegisterRequest(BaseModel):
    """
    User registration request schema.
    
    Validates all required fields including GDPR consent.
    """
    email: EmailStr = Field(
        ...,
        description="User email address (must be unique)",
        examples=["john.doe@example.com"]
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Strong password (min 8 chars, 1 uppercase, 1 lowercase, 1 digit)"
    )
    password_confirm: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password confirmation (must match password)"
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="User's full name",
        examples=["John Doe"]
    )
    gdpr_consent: bool = Field(
        ...,
        description="GDPR consent acceptance (required, must be True)"
    )
    language: Optional[str] = Field(
        default="fr",
        max_length=10,
        description="Preferred language (fr, en, etc.)",
        examples=["fr", "en"]
    )
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Validate password strength requirements.
        
        Requirements:
        - At least 8 characters
        - At least 1 uppercase letter
        - At least 1 lowercase letter
        - At least 1 digit
        - At least 1 special character (optional but recommended)
        
        Args:
            v: Password string
            
        Returns:
            Validated password
            
        Raises:
            ValueError: If password doesn't meet requirements
        """
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least 1 uppercase letter")
        
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least 1 lowercase letter")
        
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least 1 digit")
        
        # Check for common weak passwords
        weak_passwords = [
            "password", "12345678", "qwerty123", "admin123",
            "letmein", "welcome1", "monkey123"
        ]
        if v.lower() in weak_passwords:
            raise ValueError("Password is too common, please choose a stronger one")
        
        return v
    
    @field_validator('password_confirm')
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """
        Validate that password and password_confirm match.
        
        Args:
            v: Password confirmation string
            info: Validation context with other field values
            
        Returns:
            Validated password confirmation
            
        Raises:
            ValueError: If passwords don't match
        """
        if 'password' in info.data and v != info.data['password']:
            raise ValueError("Passwords do not match")
        return v
    
    @field_validator('gdpr_consent')
    @classmethod
    def validate_gdpr_consent(cls, v: bool) -> bool:
        """
        Validate GDPR consent (must be True to register).
        
        Args:
            v: GDPR consent boolean
            
        Returns:
            Validated consent
            
        Raises:
            ValueError: If consent is not given
        """
        if not v:
            raise ValueError("GDPR consent is required to create an account")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "john.doe@example.com",
                "password": "SecurePass123!",
                "password_confirm": "SecurePass123!",
                "full_name": "John Doe",
                "gdpr_consent": True,
                "language": "fr"
            }
        }


class RegisterResponse(BaseModel):
    """
    Successful registration response.
    """
    message: str = Field(..., description="Success message")
    user: dict = Field(..., description="Created user information")
    email_verification_required: bool = Field(
        default=True,
        description="Whether email verification is required"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Account created successfully. Please check your email to verify your account.",
                "user": {
                    "id": 1,
                    "email": "john.doe@example.com",
                    "full_name": "John Doe",
                    "role": "user",
                    "is_active": True,
                    "is_verified": False
                },
                "email_verification_required": True
            }
        }


# ============================================================================
# TOKEN REFRESH SCHEMAS
# ============================================================================
class TokenRefreshRequest(BaseModel):
    """
    Token refresh request schema.
    """
    refresh_token: str = Field(
        ...,
        description="Valid refresh token",
        min_length=10
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TokenRefreshResponse(BaseModel):
    """
    Token refresh response with new access token.
    """
    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token TTL in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }


# ============================================================================
# PASSWORD RESET SCHEMAS
# ============================================================================
class PasswordResetRequest(BaseModel):
    """
    Password reset request schema (step 1).
    """
    email: EmailStr = Field(
        ...,
        description="User email address to reset password",
        examples=["john.doe@example.com"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "john.doe@example.com"
            }
        }


class PasswordResetConfirm(BaseModel):
    """
    Password reset confirmation schema (step 2).
    """
    token: str = Field(
        ...,
        description="Password reset token from email",
        min_length=20
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
        """Validate new password strength (same rules as registration)"""
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "new_password": "NewSecurePass123!",
                "new_password_confirm": "NewSecurePass123!"
            }
        }


class PasswordResetResponse(BaseModel):
    """
    Password reset response.
    """
    message: str = Field(..., description="Success message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Password reset email sent successfully"
            }
        }


# ============================================================================
# EMAIL VERIFICATION SCHEMAS
# ============================================================================
class EmailVerificationRequest(BaseModel):
    """
    Email verification request schema.
    """
    token: str = Field(
        ...,
        description="Email verification token from registration email",
        min_length=20
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "token": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
            }
        }


class EmailVerificationResponse(BaseModel):
    """
    Email verification response.
    """
    message: str = Field(..., description="Success message")
    user: dict = Field(..., description="Updated user information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Email verified successfully",
                "user": {
                    "id": 1,
                    "email": "john.doe@example.com",
                    "is_verified": True
                }
            }
        }


# ============================================================================
# API KEY SCHEMAS
# ============================================================================
class APIKeyCreateRequest(BaseModel):
    """
    API key creation request.
    """
    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="API key name/description",
        examples=["Production API", "Mobile App Key"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Production API Key"
            }
        }


class APIKeyCreateResponse(BaseModel):
    """
    API key creation response.
    
    WARNING: The API key is only shown once at creation.
    """
    api_key: str = Field(..., description="Generated API key (store securely!)")
    name: str = Field(..., description="API key name")
    created_at: datetime = Field(..., description="Creation timestamp")
    message: str = Field(
        default="API key created successfully. Store it securely - it won't be shown again.",
        description="Warning message"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "eun_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
                "name": "Production API Key",
                "created_at": "2025-11-01T12:00:00Z",
                "message": "API key created successfully. Store it securely - it won't be shown again."
            }
        }


class APIKeyRevokeResponse(BaseModel):
    """
    API key revocation response.
    """
    message: str = Field(..., description="Success message")
    revoked_at: datetime = Field(..., description="Revocation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "API key revoked successfully",
                "revoked_at": "2025-11-01T12:00:00Z"
            }
        }