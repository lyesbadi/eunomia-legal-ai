"""
EUNOMIA Legal AI Platform - Document Model
SQLAlchemy model for uploaded legal documents
"""
from typing import Optional, List
from datetime import datetime
from sqlalchemy import String, Integer, Boolean, DateTime, Enum as SQLEnum, Text, ForeignKey, Index, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


# ============================================================================
# DOCUMENT TYPE ENUM
# ============================================================================
class DocumentType(str, enum.Enum):
    """
    Legal document type classification.
    """
    CONTRACT = "contract"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    LEGAL_NOTICE = "legal_notice"
    COURT_DECISION = "court_decision"
    REGULATION = "regulation"
    LEGAL_OPINION = "legal_opinion"
    OTHER = "other"


# ============================================================================
# DOCUMENT STATUS ENUM
# ============================================================================
class DocumentStatus(str, enum.Enum):
    """
    Document processing status.
    """
    UPLOADED = "uploaded"  # Just uploaded, not yet processed
    PROCESSING = "processing"  # AI analysis in progress
    COMPLETED = "completed"  # Analysis completed successfully
    FAILED = "failed"  # Analysis failed
    DELETED = "deleted"  # Soft deleted


# ============================================================================
# DOCUMENT MODEL
# ============================================================================
class Document(Base):
    """
    Document model for uploaded legal documents.
    
    Features:
    - File metadata (name, size, type, hash)
    - Processing status tracking
    - Document classification
    - Ownership and access control
    - GDPR compliance (retention, encryption)
    - Soft delete capability
    
    Relationships:
    - owner: Many-to-one with User model
    - analysis: One-to-one with Analysis model
    """
    
    # Primary Key
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # ========================================================================
    # OWNERSHIP
    # ========================================================================
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Owner user ID"
    )
    
    # ========================================================================
    # FILE METADATA
    # ========================================================================
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Original filename"
    )
    
    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        unique=True,
        doc="Storage path (relative to UPLOAD_DIR)"
    )
    
    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="File size in bytes"
    )
    
    file_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="SHA-256 hash for deduplication and integrity"
    )
    
    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="MIME type (application/pdf, etc.)"
    )
    
    file_extension: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        doc="File extension (.pdf, .docx, etc.)"
    )
    
    # ========================================================================
    # DOCUMENT CLASSIFICATION
    # ========================================================================
    document_type: Mapped[Optional[DocumentType]] = mapped_column(
        SQLEnum(DocumentType, name="document_type_enum", native_enum=False),
        nullable=True,
        index=True,
        doc="Classified document type (set by AI)"
    )
    
    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Document title (extracted or provided)"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Document description or summary"
    )
    
    language: Mapped[str] = mapped_column(
        String(10),
        default="fr",
        nullable=False,
        doc="Detected document language (fr, en, etc.)"
    )
    
    # ========================================================================
    # PROCESSING STATUS
    # ========================================================================
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus, name="document_status_enum", native_enum=False),
        default=DocumentStatus.UPLOADED,
        nullable=False,
        index=True,
        doc="Current processing status"
    )
    
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When AI processing started"
    )
    
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When AI processing completed"
    )
    
    processing_duration_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Total processing time in seconds"
    )
    
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Error message if processing failed"
    )
    
    retry_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        doc="Number of processing retry attempts"
    )
    
    # ========================================================================
    # CONTENT METADATA
    # ========================================================================
    page_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of pages (for PDFs)"
    )
    
    word_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Approximate word count"
    )
    
    character_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Total character count"
    )
    
    extracted_text_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Path to extracted plain text file"
    )
    
    # ========================================================================
    # AI ANALYSIS METADATA
    # ========================================================================
    ai_confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Overall AI confidence score (0-1)"
    )
    
    contains_unfair_clauses: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        doc="Whether unfair clauses were detected"
    )
    
    risk_level: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        doc="Risk assessment level (low, medium, high)"
    )
    
    # ========================================================================
    # ACCESS CONTROL
    # ========================================================================
    is_public: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether document is publicly accessible"
    )
    
    is_shared: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether document is shared with others"
    )
    
    share_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        doc="Token for sharing document externally"
    )
    
    share_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Share link expiration time"
    )
    
    # ========================================================================
    # GDPR COMPLIANCE
    # ========================================================================
    contains_pii: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        doc="Whether document contains PII (detected by NER)"
    )
    
    anonymized: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether document has been anonymized"
    )
    
    anonymized_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When document was anonymized"
    )
    
    retention_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Date when document should be deleted (GDPR)"
    )
    
    encrypted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether file is encrypted at rest"
    )
    
    # ========================================================================
    # METADATA (TIMESTAMPS)
    # ========================================================================
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="Upload timestamp"
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
    
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last time document was viewed"
    )
    
    access_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        doc="Number of times document was accessed"
    )
    
    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="documents",
        lazy="selectin"
    )
    
    analysis: Mapped[Optional["Analysis"]] = relationship(
        "Analysis",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # ========================================================================
    # INDEXES
    # ========================================================================
    __table_args__ = (
        Index('ix_documents_user_status', 'user_id', 'status'),
        Index('ix_documents_user_uploaded', 'user_id', 'uploaded_at'),
        Index('ix_documents_hash', 'file_hash'),
        Index('ix_documents_type_status', 'document_type', 'status'),
        Index('ix_documents_retention', 'retention_until'),
    )
    
    # ========================================================================
    # METHODS
    # ========================================================================
    def __repr__(self) -> str:
        """String representation"""
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status.value}')>"
    
    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed"""
        return self.status == DocumentStatus.PROCESSING
    
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed"""
        return self.status == DocumentStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if processing failed"""
        return self.status == DocumentStatus.FAILED
    
    @property
    def can_retry(self) -> bool:
        """Check if processing can be retried"""
        return self.is_failed and self.retry_count < 3
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes"""
        return round(self.file_size / (1024 * 1024), 2)
    
    @property
    def is_expired(self) -> bool:
        """Check if document retention period has expired"""
        if not self.retention_until:
            return False
        return datetime.utcnow() >= self.retention_until
    
    @property
    def share_is_expired(self) -> bool:
        """Check if share link has expired"""
        if not self.share_expires_at:
            return False
        return datetime.utcnow() >= self.share_expires_at
    
    def start_processing(self) -> None:
        """Mark document as processing"""
        self.status = DocumentStatus.PROCESSING
        self.processing_started_at = datetime.utcnow()
    
    def complete_processing(self) -> None:
        """Mark document processing as completed"""
        self.status = DocumentStatus.COMPLETED
        self.processing_completed_at = datetime.utcnow()
        
        if self.processing_started_at:
            duration = datetime.utcnow() - self.processing_started_at
            self.processing_duration_seconds = duration.total_seconds()
    
    def fail_processing(self, error_message: str) -> None:
        """Mark document processing as failed"""
        self.status = DocumentStatus.FAILED
        self.error_message = error_message
        self.processing_completed_at = datetime.utcnow()
        self.retry_count += 1
    
    def increment_access(self) -> None:
        """Increment access counter"""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()
    
    def soft_delete(self) -> None:
        """Soft delete document"""
        self.status = DocumentStatus.DELETED
        self.deleted_at = datetime.utcnow()
    
    def anonymize(self) -> None:
        """Anonymize document metadata (GDPR)"""
        self.title = "Anonymized Document"
        self.description = None
        self.anonymized = True
        self.anonymized_at = datetime.utcnow()