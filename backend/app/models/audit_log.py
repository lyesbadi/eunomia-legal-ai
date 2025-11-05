# EUNOMIA Legal AI Platform - Audit Log Model
# SQLAlchemy model for GDPR compliance and security audit trail

from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, Index, Enum as SQLEnum, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


# ============================================================================
# ACTION TYPE ENUM
# ============================================================================
class ActionType(str, enum.Enum):
    """
    Types of actions to audit for GDPR compliance.
    """
    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET_REQUEST = "password_reset_request"
    PASSWORD_RESET_COMPLETE = "password_reset_complete"
    
    # User Management
    USER_REGISTER = "user_register"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_ANONYMIZE = "user_anonymize"
    
    # Document Operations
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_DOWNLOAD = "document_download"
    DOCUMENT_UPDATE = "document_update"
    DOCUMENT_DELETE = "document_delete"
    DOCUMENT_SHARE = "document_share"
    
    # Analysis Operations
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_FAILED = "analysis_failed"
    ANALYSIS_VIEW = "analysis_view"
    
    # Data Access (GDPR)
    DATA_EXPORT = "data_export"
    DATA_ACCESS_REQUEST = "data_access_request"
    DATA_DELETION_REQUEST = "data_deletion_request"
    DATA_ANONYMIZATION = "data_anonymization"
    
    # Admin Actions
    ADMIN_ACTION = "admin_action"
    SETTINGS_CHANGE = "settings_change"
    
    # Security Events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCESS_DENIED = "access_denied"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"


# ============================================================================
# RESOURCE TYPE ENUM
# ============================================================================
class ResourceType(str, enum.Enum):

    # Types of resources that can be audited.

    USER = "user"
    DOCUMENT = "document"
    ANALYSIS = "analysis"
    SYSTEM = "system"


# ============================================================================
# AUDIT LOG MODEL
# ============================================================================
class AuditLog(Base):
    # Audit log model for GDPR compliance and security monitoring.
    
    # Features:
    # - Comprehensive action logging
    # - IP address tracking
    # - User agent tracking
    # - Request metadata
    # - GDPR-compliant retention
    # - Tamper-evident design (append-only)
    
    # GDPR Requirements:
    # - Article 30: Records of processing activities
    # - Article 32: Security of processing
    # - Article 33: Notification of data breach
    
    # Relationships:
    # - user: Many-to-one with User model (nullable for system actions)

    
    # Primary Key
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # ========================================================================
    # USER RELATIONSHIP (nullable for system actions)
    # ========================================================================
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="User who performed the action (NULL for system actions)"
    )
    
    # ========================================================================
    # ACTION DETAILS
    # ========================================================================
    action: Mapped[ActionType] = mapped_column(
        SQLEnum(ActionType, name="action_type_enum", native_enum=False),
        nullable=False,
        index=True,
        doc="Type of action performed"
    )
    
    resource_type: Mapped[Optional[ResourceType]] = mapped_column(
        SQLEnum(ResourceType, name="resource_type_enum", native_enum=False),
        nullable=True,
        index=True,
        doc="Type of resource affected"
    )
    
    resource_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        doc="ID of the affected resource"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        default=None,
        doc="Human-readable description of the action"
    )
    
    # ========================================================================
    # REQUEST METADATA
    # ========================================================================
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),
        nullable=True,
        index=True,
        doc="IP address of the requester (IPv4 or IPv6)"
    )
    
    user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="User agent string from request"
    )
    
    request_method: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        doc="HTTP method (GET, POST, PUT, DELETE)"
    )
    
    request_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="API endpoint path"
    )
    
    request_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="Unique request identifier (for correlation)"
    )
    
    # ========================================================================
    # ACTION RESULT
    # ========================================================================
    success: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
        index=True,
        doc="Whether action succeeded"
    )
    
    status_code: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="HTTP status code"
    )
    
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Error message if action failed"
    )
    
    # ========================================================================
    # ADDITIONAL CONTEXT
    # ========================================================================
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Additional context as JSON (changes, filters, etc.)"
    )
    
    old_values: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Previous values (for update/delete operations)"
    )
    
    new_values: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="New values (for create/update operations)"
    )
    
    # ========================================================================
    # GDPR SPECIFIC FIELDS
    # ========================================================================
    is_gdpr_relevant: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        index=True,
        doc="Whether action is relevant for GDPR compliance"
    )
    
    contains_pii: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        doc="Whether log contains PII (for retention policies)"
    )
    
    legal_basis: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="GDPR legal basis (consent, legitimate_interest, etc.)"
    )
    
    # ========================================================================
    # SECURITY FLAGS
    # ========================================================================
    is_suspicious: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        index=True,
        doc="Whether activity is flagged as suspicious"
    )
    
    risk_score: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Risk score (0-100) for security monitoring"
    )
    
    requires_review: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        index=True,
        doc="Whether log requires human review"
    )
    
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When log was reviewed by admin"
    )
    
    reviewed_by: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Admin user ID who reviewed the log"
    )
    
    # ========================================================================
    # TIMING
    # ========================================================================
    duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Request duration in milliseconds"
    )
    
    # ========================================================================
    # METADATA (TIMESTAMP)
    # ========================================================================
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="When action was performed (immutable)"
    )
    
    # ========================================================================
    # RETENTION
    # ========================================================================
    retention_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Date when log can be deleted (GDPR retention)"
    )
    
    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="audit_logs",
        lazy="selectin"
    )
    
    # ========================================================================
    # INDEXES
    # ========================================================================
    __table_args__ = (
        Index('ix_audit_logs_user_action', 'user_id', 'action'),
        Index('ix_audit_logs_user_created', 'user_id', 'created_at'),
        Index('ix_audit_logs_action_created', 'action', 'created_at'),
        Index('ix_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('ix_audit_logs_ip_created', 'ip_address', 'created_at'),
        Index('ix_audit_logs_suspicious', 'is_suspicious', 'created_at'),
        Index('ix_audit_logs_gdpr', 'is_gdpr_relevant', 'created_at'),
        Index('ix_audit_logs_retention', 'retention_until'),
    )
    
    # ========================================================================
    # METHODS
    # ========================================================================
    def __repr__(self) -> str:
        # String representation
        return f"<AuditLog(id={self.id}, action='{self.action.value}', user_id={self.user_id})>"
    
    @property
    def is_authentication_event(self) -> bool:
        # Check if log is an authentication event
        return self.action in (
            ActionType.LOGIN_SUCCESS,
            ActionType.LOGIN_FAILED,
            ActionType.LOGOUT,
        )
    
    @property
    def is_data_access_event(self) -> bool:
        # Check if log is a data access event (GDPR relevant)
        return self.action in (
            ActionType.DOCUMENT_VIEW,
            ActionType.DOCUMENT_DOWNLOAD,
            ActionType.ANALYSIS_VIEW,
            ActionType.DATA_EXPORT,
        )
    
    @property
    def is_deletion_event(self) -> bool:
        # Check if log is a deletion event
        return self.action in (
            ActionType.USER_DELETE,
            ActionType.DOCUMENT_DELETE,
            ActionType.DATA_DELETION_REQUEST,
        )
    
    @property
    def is_high_risk(self) -> bool:
        # Check if log is high risk
        if self.risk_score is None:
            return False
        return self.risk_score >= 70
    
    @property
    def is_expired(self) -> bool:
        # Check if log retention period has expired
        if not self.retention_until:
            return False
        return datetime.utcnow() >= self.retention_until
    
    @classmethod
    def create_log(
        cls,
        action: ActionType,
        description: str,
        user_id: Optional[int] = None,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "AuditLog":

    #    Factory method to create audit log entries.
        
    #    Args:
    #        action: Type of action
    #        description: Human-readable description
    #        user_id: User who performed action
    #       resource_type: Type of resource affected
    #        resource_id: ID of resource affected
    #        ip_address: IP address
    #        user_agent: User agent string
    #        success: Whether action succeeded
    #        metadata: Additional context
    #        **kwargs: Additional fields
        
    #    Returns:
    #        AuditLog: New audit log instance

        # Determine GDPR relevance
        is_gdpr_relevant = action in (
            ActionType.USER_REGISTER,
            ActionType.USER_UPDATE,
            ActionType.USER_DELETE,
            ActionType.USER_ANONYMIZE,
            ActionType.DOCUMENT_VIEW,
            ActionType.DOCUMENT_DOWNLOAD,
            ActionType.DATA_EXPORT,
            ActionType.DATA_ACCESS_REQUEST,
            ActionType.DATA_DELETION_REQUEST,
            ActionType.DATA_ANONYMIZATION,
        )
        
        # Calculate retention period (7 years for GDPR)
        if is_gdpr_relevant:
            from datetime import timedelta
            retention_until = datetime.utcnow() + timedelta(days=2555)  # 7 years
        else:
            from datetime import timedelta
            retention_until = datetime.utcnow() + timedelta(days=365)  # 1 year
        
        return cls(
            action=action,
            description=description,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            metadata=metadata,
            is_gdpr_relevant=is_gdpr_relevant,
            retention_until=retention_until,
            **kwargs
        )
    
    def mark_suspicious(self, risk_score: int, reason: str) -> None:

        #Mark log as suspicious.
        
        # Args:
        #    risk_score: Risk score (0-100)
        #    reason: Reason for flagging

        self.is_suspicious = True
        self.risk_score = risk_score
        self.requires_review = True
        
        if not self.details:
            self.details = {}
        self.details["suspicious_reason"] = reason
    
    def mark_reviewed(self, reviewer_id: int) -> None:

        # Mark log as reviewed by admin.
        
        # Args:
        #     reviewer_id: Admin user ID

        self.requires_review = False
        self.reviewed_at = datetime.utcnow()
        self.reviewed_by = reviewer_id