"""
EUNOMIA Legal AI Platform - Database Management
SQLAlchemy 2.0 async engine, session management, and Base model
"""
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine
import logging
from datetime import datetime

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# NAMING CONVENTIONS FOR CONSTRAINTS
# ============================================================================
# Consistent naming for database constraints improves debugging and migrations
convention = {
    "ix": "ix_%(column_0_label)s",  # Index
    "uq": "uq_%(table_name)s_%(column_0_name)s",  # Unique constraint
    "ck": "ck_%(table_name)s_%(constraint_name)s",  # Check constraint
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",  # Foreign key
    "pk": "pk_%(table_name)s",  # Primary key
}

metadata = MetaData(naming_convention=convention)


# ============================================================================
# BASE MODEL CLASS
# ============================================================================
class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.
    
    Provides:
    - Automatic table name generation from class name
    - Common metadata with naming conventions
    - Shared metadata for all models
    """
    
    metadata = metadata
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Generate table name from class name.
        
        Converts CamelCase to snake_case and pluralizes.
        Example: UserDocument -> user_documents
        """
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        # Simple pluralization (add 's' if doesn't end with 's')
        if not name.endswith('s'):
            name += 's'
        return name


# ============================================================================
# DATABASE ENGINE
# ============================================================================
class DatabaseManager:
    """
    Manages database engine and session lifecycle.
    
    Singleton pattern ensures only one engine instance per application.
    """
    
    _engine: Optional[AsyncEngine] = None
    _session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    @classmethod
    def get_engine(cls) -> AsyncEngine:
        """
        Get or create async database engine.
        
        Configuration:
        - Connection pooling with pre-ping for connection health checks
        - Pool size and overflow based on settings
        - Echo SQL queries in debug mode
        
        Returns:
            AsyncEngine: SQLAlchemy async engine instance
        """
        if cls._engine is None:
            logger.info("Creating database engine...")
            
            # Engine configuration
            engine_config = {
                "echo": settings.DB_ECHO,
                "future": True,  # Use SQLAlchemy 2.0 style
                "pool_pre_ping": settings.DB_POOL_PRE_PING,
            }
            
            # Configure connection pool
            if settings.is_production:
                engine_config.update({
                    "poolclass": AsyncAdaptedQueuePool,
                    "pool_size": settings.DB_POOL_SIZE,
                    "max_overflow": settings.DB_MAX_OVERFLOW,
                    "pool_recycle": settings.DB_POOL_RECYCLE,
                })
            else:
                # Simpler pool for development
                engine_config["poolclass"] = NullPool
            
            cls._engine = create_async_engine(
                settings.DATABASE_URL,
                **engine_config
            )
            
            logger.info(
                f"Database engine created: {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
        
        return cls._engine
    
    @classmethod
    def get_session_factory(cls) -> async_sessionmaker[AsyncSession]:
        """
        Get or create session factory.
        
        Returns:
            async_sessionmaker: Factory for creating database sessions
        """
        if cls._session_factory is None:
            engine = cls.get_engine()
            cls._session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Don't expire objects after commit
                autocommit=False,
                autoflush=False,
            )
            logger.info("Database session factory created")
        
        return cls._session_factory
    
    @classmethod
    async def close(cls) -> None:
        """
        Close database engine and dispose of connection pool.
        
        Should be called on application shutdown.
        """
        if cls._engine is not None:
            logger.info("Closing database engine...")
            await cls._engine.dispose()
            cls._engine = None
            cls._session_factory = None
            logger.info("Database engine closed")
    
    @classmethod
    async def create_tables(cls) -> None:
        """
        Create all database tables.
        
        WARNING: Only use in development/testing.
        In production, use Alembic migrations.
        """
        if settings.is_production:
            logger.warning("create_tables() called in production - use Alembic migrations instead")
            return
        
        logger.info("Creating database tables...")
        engine = cls.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    
    @classmethod
    async def drop_tables(cls) -> None:
        """
        Drop all database tables.
        
        WARNING: DANGEROUS! Only use in development/testing.
        """
        if settings.is_production:
            logger.error("drop_tables() called in production - BLOCKED for safety")
            raise RuntimeError("Cannot drop tables in production environment")
        
        logger.warning("Dropping all database tables...")
        engine = cls.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All database tables dropped")


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database sessions in FastAPI routes.
    
    Usage:
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    
    Yields:
        AsyncSession: Database session
    
    Ensures:
        - Session is properly closed after request
        - Automatic rollback on exceptions
        - Connection returned to pool
    """
    session_factory = DatabaseManager.get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}", exc_info=True)
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI routes.
    
    Usage:
        async with get_db_context() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
    
    Yields:
        AsyncSession: Database session
    """
    session_factory = DatabaseManager.get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database context error: {e}", exc_info=True)
            raise
        finally:
            await session.close()


# ============================================================================
# DATABASE HEALTH CHECK
# ============================================================================
async def check_database_health() -> bool:
    """
    Check if database is accessible and healthy.
    
    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        from sqlalchemy import text
        
        async with get_db_context() as db:
            # Simple query to test connection
            result = await db.execute(text("SELECT 1"))
            result.scalar()
            
        logger.info("Database health check: OK")
        return True
    
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
async def init_database() -> None:
    """
    Initialize database connection and verify connectivity.
    
    Should be called on application startup.
    """
    logger.info("Initializing database connection...")
    
    # Create engine and session factory
    DatabaseManager.get_engine()
    DatabaseManager.get_session_factory()
    
    # Verify database connectivity
    is_healthy = await check_database_health()
    if not is_healthy:
        raise RuntimeError("Failed to connect to database")
    
    logger.info("Database initialized successfully")


async def close_database() -> None:
    """
    Close database connections and cleanup resources.
    
    Should be called on application shutdown.
    """
    logger.info("Closing database connections...")
    await DatabaseManager.close()
    logger.info("Database connections closed")


# ============================================================================
# TRANSACTION HELPERS
# ============================================================================
class TransactionManager:
    """
    Helper class for managing database transactions.
    
    Provides convenient methods for atomic operations.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize transaction manager.
        
        Args:
            session: Active database session
        """
        self.session = session
    
    async def commit(self) -> None:
        """Commit current transaction"""
        await self.session.commit()
        logger.debug("Transaction committed")
    
    async def rollback(self) -> None:
        """Rollback current transaction"""
        await self.session.rollback()
        logger.debug("Transaction rolled back")
    
    async def flush(self) -> None:
        """Flush pending changes without committing"""
        await self.session.flush()
        logger.debug("Session flushed")
    
    async def refresh(self, instance) -> None:
        """
        Refresh instance from database.
        
        Args:
            instance: SQLAlchemy model instance
        """
        await self.session.refresh(instance)
    
    async def execute_in_transaction(self, func, *args, **kwargs):
        """
        Execute function within a transaction.
        
        Automatically commits on success, rolls back on failure.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Result of func
        
        Raises:
            Exception: Re-raises any exception after rollback
        """
        try:
            result = await func(*args, **kwargs)
            await self.commit()
            return result
        except Exception as e:
            await self.rollback()
            logger.error(f"Transaction failed: {e}", exc_info=True)
            raise


# ============================================================================
# PAGINATION HELPER
# ============================================================================
class Pagination:
    """
    Helper class for paginating database queries.
    
    Usage:
        pagination = Pagination(page=1, page_size=20)
        query = select(User).limit(pagination.limit).offset(pagination.offset)
    """
    
    def __init__(self, page: int = 1, page_size: int = 20):
        """
        Initialize pagination.
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
        """
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), 100)  # Max 100 items per page
    
    @property
    def offset(self) -> int:
        """Calculate offset for SQL query"""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit for SQL query"""
        return self.page_size
    
    def get_metadata(self, total_count: int) -> dict:
        """
        Get pagination metadata.
        
        Args:
            total_count: Total number of items
        
        Returns:
            dict: Pagination metadata
        """
        total_pages = (total_count + self.page_size - 1) // self.page_size
        
        return {
            "page": self.page,
            "page_size": self.page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_previous": self.page > 1,
            "has_next": self.page < total_pages,
        }


# Export commonly used items
__all__ = [
    "Base",
    "get_db",
    "get_db_context",
    "DatabaseManager",
    "TransactionManager",
    "Pagination",
    "init_database",
    "close_database",
    "check_database_health",
]