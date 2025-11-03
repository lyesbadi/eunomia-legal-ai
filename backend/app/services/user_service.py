"""
EUNOMIA Legal AI Platform - User Service
Business logic for user management and operations
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.exc import IntegrityError

from app.models.user import User, UserRole
from app.models.document import Document
from app.models.analysis import Analysis
from app.core.security import hash_password, verify_password
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# USER CRUD OPERATIONS
# ============================================================================
class UserService:
    """
    Service for user management operations.
    
    Handles:
    - User CRUD (Create, Read, Update, Delete)
    - Authentication helpers
    - User statistics
    - GDPR operations (anonymization, deletion)
    """
    
    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            User object or None
        """
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            db: Database session
            email: Email address
            
        Returns:
            User object or None
        """
        result = await db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_multi(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> tuple[List[User], int]:
        """
        Get multiple users with pagination and filters.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters (role, is_active, is_verified, search)
            
        Returns:
            Tuple of (users list, total count)
        """
        query = select(User)
        
        # Apply filters
        if filters:
            conditions = []
            
            if "role" in filters:
                conditions.append(User.role == filters["role"])
            
            if "is_active" in filters:
                conditions.append(User.is_active == filters["is_active"])
            
            if "is_verified" in filters:
                conditions.append(User.is_verified == filters["is_verified"])
            
            if "search" in filters:
                search_term = f"%{filters['search']}%"
                conditions.append(
                    or_(
                        User.email.ilike(search_term),
                        User.full_name.ilike(search_term)
                    )
                )
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        query = query.order_by(User.created_at.desc())
        
        # Execute
        result = await db.execute(query)
        users = result.scalars().all()
        
        return list(users), total
    
    @staticmethod
    async def create(
        db: AsyncSession,
        email: str,
        password: str,
        full_name: str,
        role: UserRole = UserRole.USER,
        **kwargs
    ) -> User:
        """
        Create new user.
        
        Args:
            db: Database session
            email: Email address
            password: Plain text password (will be hashed)
            full_name: User's full name
            role: User role (default: USER)
            **kwargs: Additional user fields
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If email already exists
        """
        # Check if email exists
        existing = await UserService.get_by_email(db, email)
        if existing:
            raise ValueError(f"Email {email} already registered")
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Create user
        user = User(
            email=email.lower(),
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
            **kwargs
        )
        
        try:
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            logger.info(f"User created: {email} (ID: {user.id})")
            return user
            
        except IntegrityError as e:
            await db.rollback()
            logger.error(f"Error creating user: {e}")
            raise ValueError("Failed to create user")
    
    @staticmethod
    async def update(
        db: AsyncSession,
        user: User,
        **kwargs
    ) -> User:
        """
        Update user fields.
        
        Args:
            db: Database session
            user: User object to update
            **kwargs: Fields to update
            
        Returns:
            Updated user object
        """
        for field, value in kwargs.items():
            if hasattr(user, field) and value is not None:
                setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        try:
            await db.commit()
            await db.refresh(user)
            
            logger.info(f"User updated: {user.email} (ID: {user.id})")
            return user
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating user: {e}")
            raise
    
    @staticmethod
    async def update_password(
        db: AsyncSession,
        user: User,
        new_password: str
    ) -> User:
        """
        Update user password.
        
        Args:
            db: Database session
            user: User object
            new_password: New plain text password
            
        Returns:
            Updated user object
        """
        user.hashed_password = hash_password(new_password)
        user.password_changed_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"Password changed for user: {user.email} (ID: {user.id})")
        return user
    
    @staticmethod
    async def delete(
        db: AsyncSession,
        user: User,
        soft: bool = True
    ) -> None:
        """
        Delete user account.
        
        Args:
            db: Database session
            user: User object to delete
            soft: If True, soft delete (anonymize); if False, hard delete
        """
        if soft:
            # Soft delete - anonymize user data
            user.is_active = False
            user.anonymized = True
            user.anonymized_at = datetime.utcnow()
            user.email = f"deleted_{user.id}@anonymized.local"
            user.full_name = None
            user.hashed_password = "DELETED"
            
            await db.commit()
            logger.info(f"User soft deleted (anonymized): ID {user.id}")
        else:
            # Hard delete - remove from database
            await db.delete(user)
            await db.commit()
            logger.info(f"User hard deleted: ID {user.id}")
    
    # ========================================================================
    # AUTHENTICATION HELPERS
    # ========================================================================
    @staticmethod
    async def authenticate(
        db: AsyncSession,
        email: str,
        password: str
    ) -> Optional[User]:
        """
        Authenticate user with email and password.
        
        Args:
            db: Database session
            email: Email address
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = await UserService.get_by_email(db, email)
        
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            user.last_failed_login = datetime.utcnow()
            await db.commit()
            return None
        
        # Reset failed attempts on success
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        user.last_activity = datetime.utcnow()
        await db.commit()
        
        return user
    
    @staticmethod
    async def is_email_available(db: AsyncSession, email: str) -> bool:
        """
        Check if email is available for registration.
        
        Args:
            db: Database session
            email: Email address to check
            
        Returns:
            True if email is available
        """
        user = await UserService.get_by_email(db, email)
        return user is None
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    @staticmethod
    async def get_statistics(
        db: AsyncSession,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Get user statistics.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Dictionary with statistics
        """
        # Documents count
        docs_result = await db.execute(
            select(func.count(Document.id))
            .where(Document.user_id == user_id)
        )
        documents_count = docs_result.scalar() or 0
        
        # Analyses count
        analyses_result = await db.execute(
            select(func.count(Analysis.id))
            .join(Document, Document.id == Analysis.document_id)
            .where(Document.user_id == user_id)
        )
        analyses_count = analyses_result.scalar() or 0
        
        # Storage used
        storage_result = await db.execute(
            select(func.sum(Document.file_size))
            .where(Document.user_id == user_id)
        )
        storage_bytes = storage_result.scalar() or 0
        storage_mb = storage_bytes / (1024 * 1024)
        
        # Last document upload
        last_doc_result = await db.execute(
            select(Document.uploaded_at)
            .where(Document.user_id == user_id)
            .order_by(Document.uploaded_at.desc())
            .limit(1)
        )
        last_upload = last_doc_result.scalar_one_or_none()
        
        return {
            "documents_count": documents_count,
            "analyses_count": analyses_count,
            "storage_used_mb": round(storage_mb, 2),
            "last_document_uploaded": last_upload
        }
    
    # ========================================================================
    # GDPR OPERATIONS
    # ========================================================================
    @staticmethod
    async def anonymize_user(
        db: AsyncSession,
        user: User
    ) -> None:
        """
        Anonymize user data (GDPR right to be forgotten).
        
        Args:
            db: Database session
            user: User to anonymize
        """
        user.anonymized = True
        user.anonymized_at = datetime.utcnow()
        user.email = f"anonymized_{user.id}@deleted.local"
        user.full_name = None
        user.hashed_password = "ANONYMIZED"
        user.is_active = False
        
        await db.commit()
        logger.info(f"User anonymized: ID {user.id}")
    
    @staticmethod
    async def export_user_data(
        db: AsyncSession,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Export all user data (GDPR data portability).
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Dictionary with all user data
        """
        # Get user
        user = await UserService.get_by_id(db, user_id)
        if not user:
            raise ValueError("User not found")
        
        # Get documents
        docs_result = await db.execute(
            select(Document).where(Document.user_id == user_id)
        )
        documents = docs_result.scalars().all()
        
        # Get analyses
        doc_ids = [doc.id for doc in documents]
        analyses_result = await db.execute(
            select(Analysis).where(Analysis.document_id.in_(doc_ids))
        )
        analyses = analyses_result.scalars().all()
        
        # Build export data
        export_data = {
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "language": user.language,
                "timezone": user.timezone,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            },
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "file_size": doc.file_size,
                    "document_type": doc.document_type.value if doc.document_type else None,
                    "uploaded_at": doc.uploaded_at.isoformat()
                }
                for doc in documents
            ],
            "analyses": [
                {
                    "id": analysis.id,
                    "document_id": analysis.document_id,
                    "classification": analysis.document_class,
                    "risk_level": analysis.risk_level,
                    "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None
                }
                for analysis in analyses
            ],
            "export_date": datetime.utcnow().isoformat()
        }
        
        return export_data


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    full_name: str,
    **kwargs
) -> User:
    """
    Convenience function to create user.
    
    Args:
        db: Database session
        email: Email address
        password: Password
        full_name: Full name
        **kwargs: Additional fields
        
    Returns:
        Created user
    """
    return await UserService.create(db, email, password, full_name, **kwargs)


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """
    Convenience function to get user by email.
    
    Args:
        db: Database session
        email: Email address
        
    Returns:
        User or None
    """
    return await UserService.get_by_email(db, email)


async def authenticate_user(
    db: AsyncSession,
    email: str,
    password: str
) -> Optional[User]:
    """
    Convenience function to authenticate user.
    
    Args:
        db: Database session
        email: Email address
        password: Password
        
    Returns:
        Authenticated user or None
    """
    return await UserService.authenticate(db, email, password)