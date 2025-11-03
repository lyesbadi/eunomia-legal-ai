"""
EUNOMIA Legal AI Platform - Document Schemas
Pydantic schemas for document management and upload
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

from app.models.document import DocumentType, DocumentStatus


# ============================================================================
# DOCUMENT BASE SCHEMAS
# ============================================================================
class DocumentBase(BaseModel):
    """
    Base document schema with common fields.
    """
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename"
    )
    title: Optional[str] = Field(
        None,
        max_length=500,
        description="Document title (extracted or user-provided)"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Document description or notes"
    )
    document_type: Optional[DocumentType] = Field(
        None,
        description="Classified document type (set by AI or user)"
    )
    language: str = Field(
        default="fr",
        max_length=10,
        description="Document language (ISO 639-1 code)"
    )
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """
        Validate language code.
        
        Args:
            v: Language code
            
        Returns:
            Validated language code
            
        Raises:
            ValueError: If language not supported
        """
        supported_languages = ["fr", "en", "de", "es", "it", "nl"]
        if v.lower() not in supported_languages:
            raise ValueError(
                f"Language '{v}' not supported. "
                f"Supported: {', '.join(supported_languages)}"
            )
        return v.lower()


# ============================================================================
# DOCUMENT UPLOAD SCHEMAS
# ============================================================================
class DocumentUploadRequest(BaseModel):
    """
    Document upload metadata (sent with multipart/form-data).
    
    The actual file is uploaded as multipart file.
    This schema validates the metadata fields.
    """
    title: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional document title"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional description or notes"
    )
    document_type: Optional[DocumentType] = Field(
        None,
        description="Optional pre-classification (will be verified by AI)"
    )
    language: str = Field(
        default="fr",
        description="Document language for analysis"
    )
    analyze_immediately: bool = Field(
        default=True,
        description="Start AI analysis immediately after upload"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Contrat de Location Commercial",
                "description": "Bail commercial pour local à Paris",
                "document_type": "contract",
                "language": "fr",
                "analyze_immediately": True
            }
        }
    )


class DocumentUploadResponse(BaseModel):
    """
    Document upload success response.
    """
    message: str = Field(..., description="Success message")
    document_id: int = Field(..., description="Created document ID")
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 file hash")
    status: DocumentStatus = Field(..., description="Processing status")
    analysis_started: bool = Field(..., description="Whether AI analysis was started")
    estimated_completion: Optional[str] = Field(
        None,
        description="Estimated analysis completion time"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Document uploaded successfully",
                "document_id": 42,
                "filename": "contrat_location.pdf",
                "file_size": 2457600,
                "file_hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6",
                "status": "processing",
                "analysis_started": True,
                "estimated_completion": "2-5 minutes"
            }
        }
    )


class DocumentUploadValidation(BaseModel):
    """
    File upload validation helper.
    
    Used internally to validate file before processing.
    """
    filename: str
    file_size: int
    mime_type: str
    file_extension: str
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """
        Validate file size (max 50 MB).
        
        Args:
            v: File size in bytes
            
        Returns:
            Validated file size
            
        Raises:
            ValueError: If file too large
        """
        max_size = 50 * 1024 * 1024  # 50 MB
        if v > max_size:
            raise ValueError(
                f"File too large ({v / 1024 / 1024:.2f} MB). "
                f"Maximum allowed: 50 MB"
            )
        if v == 0:
            raise ValueError("File is empty (0 bytes)")
        return v
    
    @field_validator('mime_type')
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """
        Validate MIME type against whitelist.
        
        Args:
            v: MIME type string
            
        Returns:
            Validated MIME type
            
        Raises:
            ValueError: If MIME type not allowed
        """
        allowed_mime_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown",
            "application/rtf"
        ]
        
        if v not in allowed_mime_types:
            raise ValueError(
                f"File type '{v}' not supported. "
                f"Allowed: PDF, DOCX, DOC, TXT, MD, RTF"
            )
        return v
    
    @field_validator('file_extension')
    @classmethod
    def validate_file_extension(cls, v: str) -> str:
        """
        Validate file extension.
        
        Args:
            v: File extension (with or without dot)
            
        Returns:
            Validated extension (lowercase, with dot)
            
        Raises:
            ValueError: If extension not allowed
        """
        # Normalize extension
        ext = v.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'
        
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf']
        
        if ext not in allowed_extensions:
            raise ValueError(
                f"File extension '{ext}' not supported. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )
        return ext


# ============================================================================
# DOCUMENT UPDATE SCHEMAS
# ============================================================================
class DocumentUpdate(BaseModel):
    """
    Schema for updating document metadata.
    
    All fields are optional. Only provided fields will be updated.
    """
    title: Optional[str] = Field(
        None,
        max_length=500,
        description="Document title"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Document description"
    )
    document_type: Optional[DocumentType] = Field(
        None,
        description="Document type classification"
    )
    language: Optional[str] = Field(
        None,
        max_length=10,
        description="Document language"
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
                f"Supported: {', '.join(supported_languages)}"
            )
        return v.lower()
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Contrat de Location Commerciale - Version Finale",
                "description": "Bail commercial signé le 15/10/2025",
                "document_type": "contract"
            }
        }
    )


# ============================================================================
# DOCUMENT RESPONSE SCHEMAS
# ============================================================================
class DocumentResponse(BaseModel):
    """
    Standard document response schema.
    """
    id: int = Field(..., description="Document ID")
    user_id: int = Field(..., description="Owner user ID")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    
    # File metadata
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 hash")
    mime_type: str = Field(..., description="MIME type")
    file_extension: str = Field(..., description="File extension")
    
    # Classification
    document_type: Optional[DocumentType] = Field(None, description="Document type")
    language: str = Field(..., description="Document language")
    
    # Processing status
    status: DocumentStatus = Field(..., description="Processing status")
    retry_count: int = Field(..., description="Number of processing retries")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Timestamps
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    processing_started_at: Optional[datetime] = Field(None, description="Processing start")
    processing_completed_at: Optional[datetime] = Field(None, description="Processing end")
    
    # GDPR
    encrypted: bool = Field(..., description="File encryption status")
    retention_until: Optional[datetime] = Field(None, description="Data retention date")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 42,
                "user_id": 1,
                "filename": "contrat_location.pdf",
                "title": "Contrat de Location Commercial",
                "description": "Bail commercial pour local à Paris",
                "file_size": 2457600,
                "file_hash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "mime_type": "application/pdf",
                "file_extension": ".pdf",
                "document_type": "contract",
                "language": "fr",
                "status": "completed",
                "retry_count": 0,
                "error_message": None,
                "uploaded_at": "2025-11-01T10:30:00Z",
                "processing_started_at": "2025-11-01T10:30:05Z",
                "processing_completed_at": "2025-11-01T10:32:15Z",
                "encrypted": True,
                "retention_until": "2028-11-01T10:30:00Z"
            }
        }
    )


class DocumentWithAnalysis(BaseModel):
    """
    Document response with embedded analysis results.
    """
    id: int = Field(..., description="Document ID")
    user_id: int = Field(..., description="Owner user ID")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    
    # File metadata
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type")
    file_extension: str = Field(..., description="File extension")
    
    # Classification
    document_type: Optional[DocumentType] = Field(None, description="Document type")
    language: str = Field(..., description="Document language")
    
    # Processing status
    status: DocumentStatus = Field(..., description="Processing status")
    
    # Timestamps
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    processing_completed_at: Optional[datetime] = Field(None, description="Processing end")
    
    # Analysis results (embedded)
    analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="AI analysis results (if completed)"
    )
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 42,
                "user_id": 1,
                "filename": "contrat_location.pdf",
                "title": "Contrat de Location Commercial",
                "file_size": 2457600,
                "mime_type": "application/pdf",
                "file_extension": ".pdf",
                "document_type": "contract",
                "language": "fr",
                "status": "completed",
                "uploaded_at": "2025-11-01T10:30:00Z",
                "processing_completed_at": "2025-11-01T10:32:15Z",
                "analysis": {
                    "classification": "contract",
                    "confidence": 0.97,
                    "risk_level": "medium",
                    "unfair_clauses_count": 2
                }
            }
        }
    )


class DocumentListItem(BaseModel):
    """
    Minimal document info for list views.
    """
    id: int = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    document_type: Optional[DocumentType] = Field(None, description="Document type")
    file_size: int = Field(..., description="File size in bytes")
    status: DocumentStatus = Field(..., description="Processing status")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    has_analysis: bool = Field(..., description="Whether analysis is available")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 42,
                "filename": "contrat_location.pdf",
                "title": "Contrat de Location Commercial",
                "document_type": "contract",
                "file_size": 2457600,
                "status": "completed",
                "uploaded_at": "2025-11-01T10:30:00Z",
                "has_analysis": True
            }
        }
    )


class DocumentListResponse(BaseModel):
    """
    Paginated document list response.
    """
    documents: List[DocumentListItem] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    total_size_mb: float = Field(..., description="Total storage used in MB")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {
                        "id": 42,
                        "filename": "contrat_location.pdf",
                        "title": "Contrat de Location Commercial",
                        "document_type": "contract",
                        "file_size": 2457600,
                        "status": "completed",
                        "uploaded_at": "2025-11-01T10:30:00Z",
                        "has_analysis": True
                    }
                ],
                "total": 15,
                "page": 1,
                "page_size": 20,
                "total_pages": 1,
                "total_size_mb": 47.3
            }
        }
    )


# ============================================================================
# DOCUMENT PROCESSING SCHEMAS
# ============================================================================
class DocumentReprocessRequest(BaseModel):
    """
    Request to reprocess a failed document.
    """
    force: bool = Field(
        default=False,
        description="Force reprocessing even if already completed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "force": False
            }
        }
    )


class DocumentProcessingStatus(BaseModel):
    """
    Real-time processing status response.
    """
    document_id: int = Field(..., description="Document ID")
    status: DocumentStatus = Field(..., description="Current processing status")
    progress_percentage: int = Field(
        ...,
        ge=0,
        le=100,
        description="Processing progress (0-100%)"
    )
    current_step: Optional[str] = Field(
        None,
        description="Current processing step description"
    )
    estimated_time_remaining: Optional[int] = Field(
        None,
        description="Estimated seconds remaining"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": 42,
                "status": "processing",
                "progress_percentage": 65,
                "current_step": "Extracting named entities (NER)",
                "estimated_time_remaining": 45,
                "error_message": None
            }
        }
    )


# ============================================================================
# DOCUMENT DOWNLOAD SCHEMAS
# ============================================================================
class DocumentDownloadResponse(BaseModel):
    """
    Document download metadata response.
    
    The actual file is served as binary stream.
    """
    filename: str = Field(..., description="Original filename")
    mime_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")
    download_url: str = Field(..., description="Temporary download URL")
    expires_in: int = Field(..., description="URL expiration in seconds")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "contrat_location.pdf",
                "mime_type": "application/pdf",
                "file_size": 2457600,
                "download_url": "/api/v1/documents/42/download?token=abc123",
                "expires_in": 3600
            }
        }
    )


# ============================================================================
# DOCUMENT DELETION SCHEMAS (GDPR)
# ============================================================================
class DocumentDeleteRequest(BaseModel):
    """
    Document deletion request.
    """
    confirm_deletion: bool = Field(
        ...,
        description="Explicit confirmation (must be True)"
    )
    delete_permanently: bool = Field(
        default=False,
        description="Permanently delete (bypass 30-day retention)"
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
                "confirm_deletion": True,
                "delete_permanently": False
            }
        }
    )


class DocumentDeleteResponse(BaseModel):
    """
    Document deletion response.
    """
    message: str = Field(..., description="Success message")
    document_id: int = Field(..., description="Deleted document ID")
    deleted_at: datetime = Field(..., description="Deletion timestamp")
    permanent_deletion_date: Optional[datetime] = Field(
        None,
        description="When data will be permanently deleted (if soft delete)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Document deleted successfully",
                "document_id": 42,
                "deleted_at": "2025-11-01T12:00:00Z",
                "permanent_deletion_date": "2025-12-01T12:00:00Z"
            }
        }
    )


# ============================================================================
# DOCUMENT STATISTICS SCHEMAS
# ============================================================================
class DocumentStatistics(BaseModel):
    """
    User document statistics.
    """
    total_documents: int = Field(..., description="Total documents uploaded")
    by_status: Dict[str, int] = Field(
        ...,
        description="Document count by status"
    )
    by_type: Dict[str, int] = Field(
        ...,
        description="Document count by type"
    )
    total_storage_mb: float = Field(..., description="Total storage used in MB")
    average_processing_time: Optional[float] = Field(
        None,
        description="Average processing time in seconds"
    )
    last_upload: Optional[datetime] = Field(
        None,
        description="Last document upload timestamp"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_documents": 15,
                "by_status": {
                    "uploaded": 1,
                    "processing": 2,
                    "completed": 12,
                    "failed": 0
                },
                "by_type": {
                    "contract": 8,
                    "tos": 4,
                    "privacy_policy": 2,
                    "other": 1
                },
                "total_storage_mb": 47.3,
                "average_processing_time": 125.5,
                "last_upload": "2025-11-01T10:30:00Z"
            }
        }
    )