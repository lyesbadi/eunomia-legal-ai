
# EUNOMIA Legal AI Platform - User Model
# SQLAlchemy model for user authentication and management

from typing import Optional, List
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Enum as SQLEnum, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


# ============================================================================
# USER ROLE ENUM
# ============================================================================
class UserRole(str, enum.Enum):

    # User role enumeration.
    
    # Roles hierarchy:
    # - ADMIN: Full system access, user management
    # - MANAGER: Can manage team documents and analyses
    # - USER: Standard user, can upload and analyze documents
    # - VIEWER: Read-only access to shared documents

    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"


# ============================================================================
# USER MODEL
# ============================================================================
class User(Base):

    # User model for authentication and authorization.
    
    # Features:
    # - Email-based authentication
    # - Password hashing (bcrypt)
    # - Role-based access control (RBAC)
    # - Account activation and verification
    # - GDPR compliance (data retention, consent tracking)
    # - Audit trail (login tracking, last activity)
    # - Soft delete capability
    
    # Relationships:
    # - documents: One-to-many with Document model
    # - audit_logs: One-to-many with AuditLog model

    
    # Primary Key
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # ========================================================================
    # AUTHENTICATION FIELDS
    # ========================================================================
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        doc="User email address (unique identifier)"
    )
    
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Bcrypt hashed password"
    )
    
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Account active status (False = disabled account)"
    )
    
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Email verification status"
    )
    
    verification_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Token for email verification"
    )
    
    # ========================================================================
    # USER PROFILE
    # ========================================================================
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="User's full name"
    )
    
    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole, name="user_role_enum", native_enum=False),
        default=UserRole.USER,
        nullable=False,
        index=True,
        doc="User role for RBAC"
    )
    
    company: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Company or organization name"
    )
    
    job_title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Job title or position"
    )
    
    # ========================================================================
    # ACTIVITY TRACKING
    # ========================================================================
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of last successful login"
    )
    
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of last API activity"
    )
    
    login_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        doc="Total number of successful logins"
    )
    
    failed_login_attempts: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        doc="Consecutive failed login attempts (reset on success)"
    )
    
    # ========================================================================
    # PASSWORD RESET
    # ========================================================================
    password_reset_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Token for password reset"
    )
    
    password_reset_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Expiration time for password reset token"
    )
    
    password_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of last password change"
    )
    
    # ========================================================================
    # API ACCESS
    # ========================================================================
    api_key_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Hashed API key for programmatic access"
    )
    
    api_key_created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="API key creation timestamp"
    )
    
    # ========================================================================
    # GDPR COMPLIANCE
    # ========================================================================
    gdpr_consent_given_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when GDPR consent was given"
    )
    
    gdpr_consent_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        doc="Version of terms accepted"
    )
    
    data_retention_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Date when user data should be deleted (GDPR compliance)"
    )
    
    anonymized: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether user data has been anonymized"
    )
    
    anonymized_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when data was anonymized"
    )
    
    # ========================================================================
    # PREFERENCES
    # ========================================================================
    language: Mapped[str] = mapped_column(
        String(10),
        default="fr",
        nullable=False,
        doc="Preferred language (fr, en, etc.)"
    )
    
    timezone: Mapped[str] = mapped_column(
        String(50),
        default="Europe/Paris",
        nullable=False,
        doc="User's timezone"
    )
    
    # ========================================================================
    # METADATA (TIMESTAMPS)
    # ========================================================================
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Account creation timestamp"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Last update timestamp"
    )
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Soft delete timestamp"
    )
    
    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================
    # Lazy loading to avoid circular imports
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select"
    )
    
    # ========================================================================
    # INDEXES
    # ========================================================================
    __table_args__ = (
        Index('ix_users_email_active', 'email', 'is_active'),
        Index('ix_users_role_active', 'role', 'is_active'),
        Index('ix_users_created_at', 'created_at'),
        Index('ix_users_last_activity', 'last_activity_at'),
    )
    
    # ========================================================================
    # METHODS
    # ========================================================================
    def __repr__(self) -> str:
        # String representation
        return f"<User(id={self.id}, email='{self.email}', role='{self.role.value}')>"
    
    @property
    def is_admin(self) -> bool:
        # Check if user is admin
        return self.role == UserRole.ADMIN
    
    @property
    def is_manager(self) -> bool:
        # Check if user is manager or higher
        return self.role in (UserRole.ADMIN, UserRole.MANAGER)
    
    @property
    def can_upload_documents(self) -> bool:
        # Check if user can upload documents
        return self.role in (UserRole.ADMIN, UserRole.MANAGER, UserRole.USER)
    
    @property
    def is_locked(self) -> bool:
        # Check if account is locked due to failed login attempts
        return self.failed_login_attempts >= 5
    
    @property
    def needs_password_change(self) -> bool:
        # Check if password should be changed (older than 90 days)
        if not self.password_changed_at:
            return False
        
        from datetime import timedelta
        ninety_days_ago = datetime.utcnow() - timedelta(days=90)
        return self.password_changed_at < ninety_days_ago
    
    @property
    def is_gdpr_compliant(self) -> bool:
        # Check if user has given GDPR consent
        return self.gdpr_consent_given_at is not None
    
    def update_last_login(self) -> None:
        # Update login tracking fields
        self.last_login_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()
        self.login_count += 1
        self.failed_login_attempts = 0  # Reset on successful login
    
    def increment_failed_login(self) -> None:
        # Increment failed login counter
        self.failed_login_attempts += 1
    
    def reset_failed_login(self) -> None:
        # Reset failed login counter
        self.failed_login_attempts = 0
    
    def update_activity(self) -> None:
        # Update last activity timestamp
        self.last_activity_at = datetime.utcnow()
    
    def soft_delete(self) -> None:
        # Soft delete user (GDPR right to be forgotten)
        self.deleted_at = datetime.utcnow()
        self.is_active = False
    
    def anonymize(self) -> None:

        # Anonymize user data for GDPR compliance.
        
        # Keeps minimal data for audit trail but removes PII.

        from app.core.security import anonymize_email
        
        # Anonymize email
        self.email = anonymize_email(self.email)
        
        # Clear personal data
        self.full_name = "Anonymized User"
        self.company = None
        self.job_title = None
        
        # Clear sensitive fields
        self.verification_token = None
        self.password_reset_token = None
        self.api_key_hash = None
        
        # Mark as anonymized
        self.anonymized = True
        self.anonymized_at = datetime.utcnow()
        self.is_active = False