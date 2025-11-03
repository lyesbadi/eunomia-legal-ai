"""
EUNOMIA Legal AI Platform - Document Service
Business logic for document management and file operations
"""
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime
from pathlib import Path
import hashlib
import shutil
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc

from app.models.user import User
from app.models.document import Document, DocumentStatus, DocumentType
from app.core.config import settings
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# DOCUMENT SERVICE
# ============================================================================
class DocumentService:
    """
    Service for document management operations.
    
    Handles:
    - Document CRUD
    - File storage operations
    - File validation
    - Duplicate detection
    - Statistics
    """
    
    @staticmethod
    async def get_by_id(
        db: AsyncSession,
        document_id: int
    ) -> Optional[Document]:
        """
        Get document by ID.
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            Document object or None
        """
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_documents(
        db: AsyncSession,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Document], int]:
        """
        Get user documents with pagination and filters.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records
            filters: Optional filters (status, document_type, search, dates)
            
        Returns:
            Tuple of (documents list, total count)
        """
        query = select(Document).where(Document.user_id == user_id)
        
        # Apply filters
        if filters:
            conditions = []
            
            if "status" in filters:
                conditions.append(Document.status == filters["status"])
            
            if "document_type" in filters:
                conditions.append(Document.document_type == filters["document_type"])
            
            if "search" in filters:
                search_term = f"%{filters['search']}%"
                conditions.append(
                    or_(
                        Document.filename.ilike(search_term),
                        Document.title.ilike(search_term)
                    )
                )
            
            if "date_from" in filters:
                conditions.append(Document.uploaded_at >= filters["date_from"])
            
            if "date_to" in filters:
                conditions.append(Document.uploaded_at <= filters["date_to"])
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        query = query.order_by(desc(Document.uploaded_at))
        
        # Execute
        result = await db.execute(query)
        documents = result.scalars().all()
        
        return list(documents), total
    
    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: int,
        filename: str,
        file_path: str,
        file_size: int,
        file_hash: str,
        mime_type: str,
        **kwargs
    ) -> Document:
        """
        Create document record.
        
        Args:
            db: Database session
            user_id: Owner user ID
            filename: Original filename
            file_path: Storage path (relative)
            file_size: File size in bytes
            file_hash: SHA-256 hash
            mime_type: MIME type
            **kwargs: Additional fields
            
        Returns:
            Created document object
        """
        document = Document(
            user_id=user_id,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            mime_type=mime_type,
            file_extension=Path(filename).suffix.lower(),
            **kwargs
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"Document created: {filename} (ID: {document.id})")
        return document
    
    @staticmethod
    async def update(
        db: AsyncSession,
        document: Document,
        **kwargs
    ) -> Document:
        """
        Update document fields.
        
        Args:
            db: Database session
            document: Document to update
            **kwargs: Fields to update
            
        Returns:
            Updated document
        """
        for field, value in kwargs.items():
            if hasattr(document, field) and value is not None:
                setattr(document, field, value)
        
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"Document updated: ID {document.id}")
        return document
    
    @staticmethod
    async def delete(
        db: AsyncSession,
        document: Document,
        permanent: bool = False
    ) -> None:
        """
        Delete document.
        
        Args:
            db: Database session
            document: Document to delete
            permanent: If True, hard delete; if False, soft delete
        """
        if permanent:
            # Delete file from disk
            file_path = settings.UPLOAD_DIR / document.file_path
            if file_path.exists():
                file_path.unlink()
            
            # Delete from database
            await db.delete(document)
            await db.commit()
            logger.info(f"Document permanently deleted: ID {document.id}")
        else:
            # Soft delete
            document.status = DocumentStatus.DELETED
            await db.commit()
            logger.info(f"Document soft deleted: ID {document.id}")
    
    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of hash
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    @staticmethod
    def save_file(
        file_content: BinaryIO,
        user_id: int,
        filename: str
    ) -> tuple[Path, str, int]:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: File binary content
            user_id: User ID (for organizing files)
            filename: Original filename
            
        Returns:
            Tuple of (file_path, file_hash, file_size)
        """
        # Create user directory
        user_dir = settings.UPLOAD_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = user_dir / unique_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_content, buffer)
        
        # Calculate hash and size
        file_hash = DocumentService.calculate_file_hash(file_path)
        file_size = file_path.stat().st_size
        
        return file_path, file_hash, file_size
    
    @staticmethod
    async def check_duplicate(
        db: AsyncSession,
        user_id: int,
        file_hash: str
    ) -> Optional[Document]:
        """
        Check if file with same hash exists for user.
        
        Args:
            db: Database session
            user_id: User ID
            file_hash: SHA-256 hash
            
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
    
    @staticmethod
    def get_file_path(document: Document) -> Path:
        """
        Get absolute file path for document.
        
        Args:
            document: Document object
            
        Returns:
            Absolute path to file
        """
        return settings.UPLOAD_DIR / document.file_path
    
    @staticmethod
    def validate_file_size(size: int, max_size: int = 50 * 1024 * 1024) -> None:
        """
        Validate file size.
        
        Args:
            size: File size in bytes
            max_size: Maximum allowed size (default: 50 MB)
            
        Raises:
            ValueError: If file too large
        """
        if size > max_size:
            raise ValueError(
                f"File too large ({size / 1024 / 1024:.2f} MB). "
                f"Maximum: {max_size / 1024 / 1024:.0f} MB"
            )
        
        if size == 0:
            raise ValueError("File is empty")
    
    @staticmethod
    def validate_mime_type(mime_type: str) -> None:
        """
        Validate MIME type against whitelist.
        
        Args:
            mime_type: MIME type string
            
        Raises:
            ValueError: If MIME type not allowed
        """
        allowed_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown",
            "application/rtf"
        ]
        
        if mime_type not in allowed_types:
            raise ValueError(f"File type '{mime_type}' not supported")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    @staticmethod
    async def get_statistics(
        db: AsyncSession,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Get document statistics for user.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Dictionary with statistics
        """
        # Total documents
        total_result = await db.execute(
            select(func.count(Document.id))
            .where(Document.user_id == user_id)
        )
        total = total_result.scalar() or 0
        
        # By status
        by_status = {}
        for status in DocumentStatus:
            count_result = await db.execute(
                select(func.count(Document.id))
                .where(
                    and_(
                        Document.user_id == user_id,
                        Document.status == status
                    )
                )
            )
            by_status[status.value] = count_result.scalar() or 0
        
        # By type
        by_type = {}
        for doc_type in DocumentType:
            count_result = await db.execute(
                select(func.count(Document.id))
                .where(
                    and_(
                        Document.user_id == user_id,
                        Document.document_type == doc_type
                    )
                )
            )
            count = count_result.scalar() or 0
            if count > 0:
                by_type[doc_type.value] = count
        
        # Total storage
        storage_result = await db.execute(
            select(func.sum(Document.file_size))
            .where(Document.user_id == user_id)
        )
        storage_bytes = storage_result.scalar() or 0
        
        # Average processing time
        avg_time_result = await db.execute(
            select(
                func.avg(
                    func.extract('epoch', 
                        Document.processing_completed_at - Document.processing_started_at
                    )
                )
            )
            .where(
                and_(
                    Document.user_id == user_id,
                    Document.status == DocumentStatus.COMPLETED,
                    Document.processing_started_at.isnot(None),
                    Document.processing_completed_at.isnot(None)
                )
            )
        )
        avg_time = avg_time_result.scalar()
        
        # Last upload
        last_upload_result = await db.execute(
            select(Document.uploaded_at)
            .where(Document.user_id == user_id)
            .order_by(desc(Document.uploaded_at))
            .limit(1)
        )
        last_upload = last_upload_result.scalar_one_or_none()
        
        return {
            "total_documents": total,
            "by_status": by_status,
            "by_type": by_type,
            "total_storage_mb": round(storage_bytes / (1024 * 1024), 2),
            "average_processing_time": round(avg_time, 2) if avg_time else None,
            "last_upload": last_upload
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
async def get_document(db: AsyncSession, document_id: int) -> Optional[Document]:
    """Get document by ID."""
    return await DocumentService.get_by_id(db, document_id)


async def get_user_documents(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 100
) -> tuple[List[Document], int]:
    """Get user documents with pagination."""
    return await DocumentService.get_user_documents(db, user_id, skip, limit)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file."""
    return DocumentService.calculate_file_hash(file_path)