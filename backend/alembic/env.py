"""
EUNOMIA Legal AI Platform - Alembic Environment Configuration
Async database migration environment for PostgreSQL with asyncpg
"""
import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ============================================================================
# ADD BACKEND TO PATH
# ============================================================================
# Add the backend directory to sys.path to import app modules
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# ============================================================================
# IMPORT APP COMPONENTS
# ============================================================================
from app.core.config import settings
from app.core.database import Base
# Import all models to ensure they're registered with Base.metadata
from app.models import User, Document, Analysis, AuditLog

# ============================================================================
# ALEMBIC CONFIG
# ============================================================================
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate
target_metadata = Base.metadata

# ============================================================================
# DATABASE URL FROM SETTINGS
# ============================================================================
def get_url():
    """
    Get database URL from application settings.
    
    This ensures consistency between app and migrations.
    Uses environment variables via settings.
    """
    return settings.DATABASE_URL


# Override the sqlalchemy.url in alembic.ini with our settings
config.set_main_option("sqlalchemy.url", get_url())


# ============================================================================
# MIGRATION FUNCTIONS
# ============================================================================
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect default value changes
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Run migrations with an established connection.
    
    Args:
        connection: SQLAlchemy connection object
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect default value changes
        # Include schemas
        include_schemas=True,
        # Render AS BATCH for SQLite compatibility (not needed for PostgreSQL)
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async engine.
    
    In this scenario we need to create an async Engine and associate
    a connection with the context.
    """
    # Create async engine configuration
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    # Create async engine
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Don't use connection pooling for migrations
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    Wrapper for async migration execution.
    """
    asyncio.run(run_async_migrations())


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()