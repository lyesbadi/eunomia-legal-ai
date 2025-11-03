"""
EUNOMIA Legal AI Platform - Document Management Routes
FastAPI routes for document upload, management, and download
"""
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import hashlib
import shutil
from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status, 
    UploadFile, 
    File, 
    Form,
    Query,
    BackgroundTasks
)
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc

from app.api.deps import (
    get_db,
    get_current_user,
    get_current_active_user,
    get_current_verified_user,
    verify_document_ownership,
    get_audit_logger,
    get_pagination,
    get_document_filters,
    AuditLogger,
    PaginationParams,
    DocumentFilters
)
from app.core.config import settings
from app.models.user import User
from app.models.document import Document, DocumentStatus, DocumentType
from app.models.audit_log import ActionType, ResourceType
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentWithAnalysis,
    DocumentListItem,
    DocumentListResponse,
    DocumentUpdate,
    DocumentReprocessRequest,
    DocumentProcessingStatus,
    DocumentDownloadResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    DocumentStatistics,
    DocumentUploadValidation
)
import logging


# ============================================================================
# ROUTER SETUP
# ============================================================================
router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hex digest of SHA-256 hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read in 8KB chunks
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


async def save_upload_file(
    upload_file: UploadFile,
    user_id: int
) -> tuple[Path, str, int]:
    """
    Save uploaded file to disk.
    
    Args:
        upload_file: FastAPI UploadFile
        user_id: User ID (for organizing files)
        
    Returns:
        Tuple of (file_path, file_hash, file_size)
    """
    # Create user directory
    user_dir = settings.UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{upload_file.filename}"
    file_path = user_dir / filename
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    
    # Calculate hash and size
    file_hash = calculate_file_hash(file_path)
    file_size = file_path.stat().st_size
    
    return file_path, file_hash, file_size


async def check_duplicate_file(
    db: AsyncSession,
    user_id: int,
    file_hash: str
) -> Optional[Document]:
    """
    Check if file with same hash already exists for user.
    
    Args:
        db: Database session
        user_id: User ID
        file_hash: File SHA-256 hash
        
    Returns:
        Existing document or None
    """
    result = await db.execute(
        select(Document).where(
            and_(
                Document.user_id == user_id,
                Document.file_hash == file_hash,
                Document.status != DocumentStatus.DELETED
            )
        )
    )
    return result.scalar_one_or_none()


async def trigger_document_analysis(document_id: int) -> None:
    """
    Trigger background analysis for document.
    
    Args:
        document_id: Document ID
        
    Note:
        This is a placeholder. In production, this would trigger a Celery task.
    """
    # TODO: Trigger Celery task
    # from app.tasks.analysis_tasks import analyze_document
    # analyze_document.delay(document_id)
    
    logger.info(f"Analysis triggered for document {document_id} (placeholder)")


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Original filename
        
    Returns:
        File extension (with dot)
    """
    return Path(filename).suffix.lower()


# ============================================================================
# DOCUMENT UPLOAD
# ============================================================================
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document",
    description="Upload legal document for AI analysis"
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (PDF, DOCX, etc.)"),
    title: Optional[str] = Form(None, description="Optional document title"),
    description: Optional[str] = Form(None, description="Optional description"),
    document_type: Optional[DocumentType] = Form(None, description="Optional document type"),
    language: str = Form("fr", description="Document language"),
    analyze_immediately: bool = Form(True, description="Start analysis immediately"),
    current_user: User = Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> DocumentUploadResponse:
    """
    Upload document for analysis.
    
    **Requirements:**
    - File size: max 50 MB
    - Supported formats: PDF, DOCX, DOC, TXT, MD, RTF
    - User must be verified
    
    **Process:**
    1. Validate file (size, type, extension)
    2. Check for duplicates (SHA-256 hash)
    3. Save file to storage
    4. Create document record in database
    5. Optionally trigger AI analysis
    
    Returns document information and processing status.
    """
    # Get file info
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    await file.seek(0)  # Reset file pointer
    
    file_extension = get_file_extension(file.filename)
    content_type = file.content_type or "application/octet-stream"
    
    # Validate file
    try:
        validation = DocumentUploadValidation(
            filename=file.filename,
            file_size=file_size,
            mime_type=content_type,
            file_extension=file_extension
        )
    except Exception as e:
        await audit.log(
            user_id=current_user.id,
            action=ActionType.DOCUMENT_UPLOAD,
            resource_type=ResourceType.DOCUMENT,
            success=False,
            error_message=f"Validation failed: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Save file to disk
    try:
        file_path, file_hash, actual_size = await save_upload_file(file, current_user.id)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        await audit.log(
            user_id=current_user.id,
            action=ActionType.DOCUMENT_UPLOAD,
            resource_type=ResourceType.DOCUMENT,
            success=False,
            error_message=f"File save failed: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file"
        )
    
    # Check for duplicate
    duplicate = await check_duplicate_file(db, current_user.id, file_hash)
    if duplicate:
        # Delete uploaded file (duplicate)
        file_path.unlink(missing_ok=True)
        
        logger.warning(
            f"Duplicate file detected for user {current_user.id}: "
            f"hash {file_hash} (existing document {duplicate.id})"
        )
        
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This file already exists (Document ID: {duplicate.id})"
        )
    
    # Create document record
    try:
        # Calculate relative path from UPLOAD_DIR
        relative_path = file_path.relative_to(settings.UPLOAD_DIR)
        
        document = Document(
            user_id=current_user.id,
            filename=file.filename,
            file_path=str(relative_path),
            file_size=actual_size,
            file_hash=file_hash,
            mime_type=content_type,
            file_extension=file_extension,
            title=title,
            description=description,
            document_type=document_type,
            language=language,
            status=DocumentStatus.UPLOADED,
            encrypted=False,  # TODO: Implement encryption
            retention_until=datetime.utcnow().replace(year=datetime.utcnow().year + 3)
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # Log upload
        await audit.log(
            user_id=current_user.id,
            action=ActionType.DOCUMENT_UPLOAD,
            resource_type=ResourceType.DOCUMENT,
            resource_id=document.id,
            details={
                "filename": file.filename,
                "size": actual_size,
                "type": document_type.value if document_type else None
            }
        )
        
        logger.info(
            f"Document uploaded: {file.filename} (ID: {document.id}) "
            f"by user {current_user.id}"
        )
        
        # Trigger analysis if requested
        analysis_started = False
        if analyze_immediately:
            document.status = DocumentStatus.PROCESSING
            document.processing_started_at = datetime.utcnow()
            await db.commit()
            
            background_tasks.add_task(trigger_document_analysis, document.id)
            analysis_started = True
            
            logger.info(f"Analysis triggered for document {document.id}")
        
        # Estimate completion time (2-5 minutes)
        estimated_completion = "2-5 minutes" if analysis_started else None
        
        return DocumentUploadResponse(
            message="Document uploaded successfully",
            document_id=document.id,
            filename=document.filename,
            file_size=document.file_size,
            file_hash=document.file_hash,
            status=document.status,
            analysis_started=analysis_started,
            estimated_completion=estimated_completion
        )
        
    except Exception as e:
        await db.rollback()
        
        # Delete uploaded file on error
        file_path.unlink(missing_ok=True)
        
        logger.error(f"Error creating document record: {e}")
        await audit.log(
            user_id=current_user.id,
            action=ActionType.DOCUMENT_UPLOAD,
            resource_type=ResourceType.DOCUMENT,
            success=False,
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create document record"
        )


# ============================================================================
# DOCUMENT LISTING & RETRIEVAL
# ============================================================================
@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List user documents",
    description="Get paginated list of user documents with filters"
)
async def list_documents(
    pagination: PaginationParams = Depends(get_pagination),
    filters: DocumentFilters = Depends(get_document_filters),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> DocumentListResponse:
    """
    List user documents.
    
    Supports:
    - Pagination (page, page_size)
    - Filtering (status, document_type, date_from, date_to)
    - Search (filename, title)
    
    Returns paginated list of documents.
    """
    # Build query
    query = select(Document).where(Document.user_id == current_user.id)
    
    # Apply filters
    filter_conditions = []
    
    if filters.status:
        filter_conditions.append(Document.status == DocumentStatus(filters.status))
    
    if filters.document_type:
        filter_conditions.append(Document.document_type == DocumentType(filters.document_type))
    
    if filters.search:
        search_pattern = f"%{filters.search}%"
        filter_conditions.append(
            or_(
                Document.filename.ilike(search_pattern),
                Document.title.ilike(search_pattern)
            )
        )
    
    if filters.date_from:
        filter_conditions.append(Document.uploaded_at >= filters.date_from)
    
    if filters.date_to:
        filter_conditions.append(Document.uploaded_at <= filters.date_to)
    
    if filter_conditions:
        query = query.where(and_(*filter_conditions))
    
    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.offset(pagination.offset).limit(pagination.limit)
    query = query.order_by(desc(Document.uploaded_at))
    
    # Execute query
    result = await db.execute(query)
    documents = result.scalars().all()
    
    # Convert to list items
    document_items = []
    for doc in documents:
        # Check if analysis exists
        has_analysis = doc.analysis is not None
        
        document_items.append(
            DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                title=doc.title,
                document_type=doc.document_type,
                file_size=doc.file_size,
                status=doc.status,
                uploaded_at=doc.uploaded_at,
                has_analysis=has_analysis
            )
        )
    
    # Calculate total storage
    total_size_bytes = sum(doc.file_size for doc in documents)
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    return DocumentListResponse(
        documents=document_items,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size,
        total_size_mb=round(total_size_mb, 2)
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Get detailed information about a document"
)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> DocumentResponse:
    """
    Get document details.
    
    Returns full document information including metadata and processing status.
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Log view
    await audit.log(
        user_id=current_user.id,
        action=ActionType.DOCUMENT_VIEW,
        resource_type=ResourceType.DOCUMENT,
        resource_id=document.id
    )
    
    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/with-analysis",
    response_model=DocumentWithAnalysis,
    summary="Get document with analysis",
    description="Get document with embedded analysis results"
)
async def get_document_with_analysis(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> DocumentWithAnalysis:
    """
    Get document with analysis results.
    
    Returns document information with embedded AI analysis if completed.
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get document with analysis
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Prepare analysis data
    analysis_data = None
    if document.analysis:
        analysis_data = {
            "id": document.analysis.id,
            "classification": document.analysis.document_class,
            "confidence": document.analysis.classification_confidence,
            "risk_level": document.analysis.risk_level,
            "risk_score": document.analysis.risk_score,
            "unfair_clauses_count": document.analysis.clause_count or 0,
            "entities_count": len(document.analysis.entities_detected) if document.analysis.entities_detected else 0,
            "completed_at": document.analysis.completed_at
        }
    
    # Convert to response
    doc_data = DocumentResponse.model_validate(document).model_dump()
    doc_data["analysis"] = analysis_data
    
    return DocumentWithAnalysis(**doc_data)


# ============================================================================
# DOCUMENT UPDATE
# ============================================================================
@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document metadata",
    description="Update document title, description, or classification"
)
async def update_document(
    document_id: int,
    data: DocumentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> DocumentResponse:
    """
    Update document metadata.
    
    - **title**: Document title
    - **description**: Document description
    - **document_type**: Document type classification
    - **language**: Document language
    
    All fields are optional.
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Track changes
    changes = {}
    
    # Update fields
    update_data = data.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(document, field):
            old_value = getattr(document, field)
            if old_value != value:
                setattr(document, field, value)
                changes[field] = {"old": str(old_value), "new": str(value)}
    
    if not changes:
        return DocumentResponse.model_validate(document)
    
    try:
        await db.commit()
        await db.refresh(document)
        
        # Log update
        await audit.log(
            user_id=current_user.id,
            action=ActionType.DOCUMENT_UPDATE,
            resource_type=ResourceType.DOCUMENT,
            resource_id=document.id,
            details={"changes": changes}
        )
        
        logger.info(f"Document {document.id} updated: {changes}")
        
        return DocumentResponse.model_validate(document)
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document"
        )


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentResponse,
    summary="Reprocess document",
    description="Retry AI analysis for failed or completed document"
)
async def reprocess_document(
    document_id: int,
    data: DocumentReprocessRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> DocumentResponse:
    """
    Reprocess document.
    
    - **force**: Force reprocessing even if already completed
    
    Useful for:
    - Retrying failed analyses
    - Re-analyzing with updated models
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Check if can retry
    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is currently being processed"
        )
    
    if document.status == DocumentStatus.COMPLETED and not data.force:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document already processed. Use force=true to reprocess."
        )
    
    # Update status
    document.status = DocumentStatus.PROCESSING
    document.processing_started_at = datetime.utcnow()
    document.processing_completed_at = None
    document.error_message = None
    
    if document.status == DocumentStatus.FAILED:
        document.retry_count += 1
    
    await db.commit()
    
    # Trigger analysis
    background_tasks.add_task(trigger_document_analysis, document.id)
    
    # Log reprocess
    await audit.log(
        user_id=current_user.id,
        action=ActionType.ANALYSIS_START,
        resource_type=ResourceType.DOCUMENT,
        resource_id=document.id,
        details={"reprocess": True, "force": data.force}
    )
    
    logger.info(f"Document {document.id} queued for reprocessing")
    
    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/status",
    response_model=DocumentProcessingStatus,
    summary="Get processing status",
    description="Get real-time processing status for document"
)
async def get_processing_status(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> DocumentProcessingStatus:
    """
    Get document processing status.
    
    Returns:
    - Current status
    - Progress percentage (0-100)
    - Current processing step
    - Estimated time remaining
    - Error message if failed
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Calculate progress
    progress = 0
    current_step = None
    estimated_time = None
    
    if document.status == DocumentStatus.UPLOADED:
        progress = 0
        current_step = "Waiting to start"
    elif document.status == DocumentStatus.PROCESSING:
        # TODO: Get actual progress from Celery task
        # For now, estimate based on time elapsed
        if document.processing_started_at:
            elapsed = (datetime.utcnow() - document.processing_started_at).seconds
            progress = min(int((elapsed / 180) * 100), 95)  # Max 95% until complete
            current_step = "Analyzing document with AI models"
            estimated_time = max(180 - elapsed, 0)
        else:
            progress = 10
            current_step = "Starting analysis"
    elif document.status == DocumentStatus.COMPLETED:
        progress = 100
        current_step = "Analysis completed"
    elif document.status == DocumentStatus.FAILED:
        progress = 0
        current_step = "Analysis failed"
    
    return DocumentProcessingStatus(
        document_id=document.id,
        status=document.status,
        progress_percentage=progress,
        current_step=current_step,
        estimated_time_remaining=estimated_time,
        error_message=document.error_message
    )


# ============================================================================
# DOCUMENT DOWNLOAD
# ============================================================================
@router.get(
    "/{document_id}/download",
    summary="Download document",
    description="Download original document file"
)
async def download_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
):
    """
    Download document file.
    
    Returns the original uploaded file with appropriate headers.
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Get file path
    file_path = settings.UPLOAD_DIR / document.file_path
    
    if not file_path.exists():
        logger.error(f"File not found on disk: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk"
        )
    
    # Log download
    await audit.log(
        user_id=current_user.id,
        action=ActionType.DOCUMENT_DOWNLOAD,
        resource_type=ResourceType.DOCUMENT,
        resource_id=document.id
    )
    
    logger.info(f"Document {document.id} downloaded by user {current_user.id}")
    
    # Return file
    return FileResponse(
        path=file_path,
        media_type=document.mime_type,
        filename=document.filename,
        headers={
            "Content-Disposition": f'attachment; filename="{document.filename}"'
        }
    )


# ============================================================================
# DOCUMENT DELETION (GDPR)
# ============================================================================
@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete document",
    description="Delete document and associated data (GDPR compliance)"
)
async def delete_document(
    document_id: int,
    data: DocumentDeleteRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> DocumentDeleteResponse:
    """
    Delete document (GDPR compliance).
    
    - **confirm_deletion**: Must be True
    - **delete_permanently**: If True, delete immediately; if False, soft delete
    
    Deletes:
    - Document record
    - Uploaded file
    - Analysis results
    - Vector embeddings
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    deleted_at = datetime.utcnow()
    
    try:
        if data.delete_permanently:
            # Delete file from disk
            file_path = settings.UPLOAD_DIR / document.file_path
            if file_path.exists():
                file_path.unlink()
            
            # Delete from database (cascade will delete analysis)
            await db.delete(document)
            await db.commit()
            
            # TODO: Delete from Qdrant vector store
            
            # Log deletion
            await audit.log(
                user_id=current_user.id,
                action=ActionType.DOCUMENT_DELETE,
                resource_type=ResourceType.DOCUMENT,
                resource_id=document_id,
                details={"permanent": True, "filename": document.filename}
            )
            
            logger.info(f"Document {document_id} permanently deleted by user {current_user.id}")
            
            return DocumentDeleteResponse(
                message="Document permanently deleted",
                document_id=document_id,
                deleted_at=deleted_at,
                permanent_deletion_date=deleted_at
            )
        else:
            # Soft delete
            document.status = DocumentStatus.DELETED
            await db.commit()
            
            # Log deletion
            await audit.log(
                user_id=current_user.id,
                action=ActionType.DOCUMENT_DELETE,
                resource_type=ResourceType.DOCUMENT,
                resource_id=document.id,
                details={"permanent": False}
            )
            
            logger.info(f"Document {document.id} soft deleted by user {current_user.id}")
            
            return DocumentDeleteResponse(
                message="Document deleted",
                document_id=document.id,
                deleted_at=deleted_at,
                permanent_deletion_date=None
            )
            
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


# ============================================================================
# DOCUMENT STATISTICS
# ============================================================================
@router.get(
    "/statistics",
    response_model=DocumentStatistics,
    summary="Get document statistics",
    description="Get user document statistics and usage metrics"
)
async def get_document_statistics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> DocumentStatistics:
    """
    Get document statistics for current user.
    
    Returns:
    - Total documents count
    - Count by status
    - Count by type
    - Total storage used
    - Average processing time
    - Last upload timestamp
    """
    # Total documents
    total_result = await db.execute(
        select(func.count(Document.id))
        .where(Document.user_id == current_user.id)
    )
    total_documents = total_result.scalar() or 0
    
    # By status
    by_status = {}
    for status_value in DocumentStatus:
        count_result = await db.execute(
            select(func.count(Document.id))
            .where(
                and_(
                    Document.user_id == current_user.id,
                    Document.status == status_value
                )
            )
        )
        count = count_result.scalar() or 0
        by_status[status_value.value] = count
    
    # By type
    by_type = {}
    for type_value in DocumentType:
        count_result = await db.execute(
            select(func.count(Document.id))
            .where(
                and_(
                    Document.user_id == current_user.id,
                    Document.document_type == type_value
                )
            )
        )
        count = count_result.scalar() or 0
        if count > 0:
            by_type[type_value.value] = count
    
    # Total storage
    storage_result = await db.execute(
        select(func.sum(Document.file_size))
        .where(Document.user_id == current_user.id)
    )
    storage_bytes = storage_result.scalar() or 0
    storage_mb = storage_bytes / (1024 * 1024)
    
    # Average processing time
    avg_time_result = await db.execute(
        select(
            func.avg(
                func.extract('epoch', Document.processing_completed_at - Document.processing_started_at)
            )
        )
        .where(
            and_(
                Document.user_id == current_user.id,
                Document.status == DocumentStatus.COMPLETED,
                Document.processing_started_at.isnot(None),
                Document.processing_completed_at.isnot(None)
            )
        )
    )
    avg_processing_time = avg_time_result.scalar()
    
    # Last upload
    last_upload_result = await db.execute(
        select(Document.uploaded_at)
        .where(Document.user_id == current_user.id)
        .order_by(desc(Document.uploaded_at))
        .limit(1)
    )
    last_upload = last_upload_result.scalar_one_or_none()
    
    return DocumentStatistics(
        total_documents=total_documents,
        by_status=by_status,
        by_type=by_type,
        total_storage_mb=round(storage_mb, 2),
        average_processing_time=round(avg_processing_time, 2) if avg_processing_time else None,
        last_upload=last_upload
    )