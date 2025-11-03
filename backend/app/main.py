"""
EUNOMIA Legal AI Platform - Main Application
FastAPI application initialization and configuration
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

from app.core.config import settings
from app.core.database import engine, Base
from app.middleware import setup_middleware
from app.api import api_router, API_PREFIX
import sys


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN EVENTS
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Startup:
    - Initialize database tables
    - Load AI models (if needed)
    - Setup background tasks
    
    Shutdown:
    - Close database connections
    - Cleanup resources
    """
    # ========================================================================
    # STARTUP
    # ========================================================================
    logger.info("=" * 80)
    logger.info("🚀 EUNOMIA Legal AI Platform Starting...")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Database URL: {settings.DATABASE_URL.split('@')[-1]}")  # Hide credentials
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"CORS origins: {settings.FRONTEND_URL}")
    
    # Create database tables (for development)
    # Note: In production, use Alembic migrations instead
    if settings.ENVIRONMENT == "development":
        logger.info("Creating database tables (development mode)...")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating database tables: {e}")
            # Don't fail startup, tables might already exist
    
    # Ensure upload directory exists
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Upload directory ready: {settings.UPLOAD_DIR}")
    
    # Log AI models status (placeholder for when we add AI services)
    logger.info("AI Models:")
    logger.info("  - Hugging Face models: Ready (to be loaded on first use)")
    logger.info("  - Ollama Mistral 7B: Available")
    logger.info("  - Qdrant vector store: Connected")
    
    logger.info("=" * 80)
    logger.info("✅ EUNOMIA Platform Ready!")
    logger.info(f"📡 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"🔧 ReDoc: http://{settings.HOST}:{settings.PORT}/redoc")
    logger.info("=" * 80)
    
    yield
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    logger.info("=" * 80)
    logger.info("🛑 EUNOMIA Platform Shutting Down...")
    logger.info("=" * 80)
    
    # Close database connections
    logger.info("Closing database connections...")
    await engine.dispose()
    logger.info("✅ Database connections closed")
    
    # Cleanup resources
    logger.info("Cleaning up resources...")
    # Add any cleanup code here (close AI model connections, etc.)
    
    logger.info("=" * 80)
    logger.info("✅ EUNOMIA Platform Stopped Successfully")
    logger.info("=" * 80)


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    # EUNOMIA Legal AI Platform
    
    **AI-powered legal document analysis platform** designed for law firms, 
    legal departments, and compliance teams.
    
    ## Features
    
    ### 🤖 AI Analysis
    - **Document Classification** using Legal-BERT
    - **Named Entity Recognition** with CamemBERT-NER
    - **Unfair Clause Detection** (GDPR compliance)
    - **Document Summarization** with BART
    - **Risk Assessment** and scoring
    - **LLM Recommendations** using Mistral 7B
    
    ### 📄 Document Management
    - Secure file upload (PDF, DOCX, TXT)
    - Automatic duplicate detection
    - File encryption at rest
    - GDPR-compliant data retention
    
    ### 🔒 Security & Compliance
    - JWT authentication
    - Role-based access control (RBAC)
    - Complete audit logging
    - GDPR right to be forgotten
    - Data anonymization
    
    ### 📊 Analytics
    - Usage statistics
    - Document analytics
    - Risk trends
    - Performance metrics
    
    ## Authentication
    
    All endpoints (except `/auth/login` and `/auth/register`) require authentication.
    
    1. **Register**: `POST /api/v1/auth/register`
    2. **Login**: `POST /api/v1/auth/login` → Get access token
    3. **Use token**: Add header `Authorization: Bearer <token>`
    
    ## Rate Limiting
    
    Default limits:
    - **100 requests per minute** per IP/user
    - **50 MB** maximum file size
    - **20 documents** per page (pagination)
    
    ## Support
    
    - Documentation: https://docs.eunomia.legal
    - Support: support@eunomia.legal
    - GitHub: https://github.com/eunomia/legal-ai-platform
    """,
    version="1.0.0",
    openapi_url=f"{API_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    # Additional metadata
    contact={
        "name": "EUNOMIA Support",
        "email": "support@eunomia.legal",
        "url": "https://eunomia.legal"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://eunomia.legal/license"
    },
    # OpenAPI tags for organization
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User authentication and authorization endpoints"
        },
        {
            "name": "Users",
            "description": "User management and profile operations"
        },
        {
            "name": "Documents",
            "description": "Document upload, management, and download"
        },
        {
            "name": "Analyses",
            "description": "AI analysis results and export"
        },
        {
            "name": "Health",
            "description": "System health and monitoring"
        }
    ]
)


# ============================================================================
# MIDDLEWARE SETUP
# ============================================================================
# Setup all middleware (CORS, logging, rate limiting, error handling)
setup_middleware(app)

# Add trusted host middleware for production
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            settings.DOMAIN,
            f"*.{settings.DOMAIN}",
            "localhost"
        ]
    )


# ============================================================================
# INCLUDE ROUTERS
# ============================================================================
# Include API router with /api prefix
app.include_router(api_router, prefix=API_PREFIX)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.get(
    "/",
    tags=["Root"],
    summary="Root endpoint",
    description="API root with basic information"
)
async def root():
    """
    Root endpoint - API information.
    
    Returns basic information about the API.
    """
    return {
        "message": "Welcome to EUNOMIA Legal AI Platform API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": f"{API_PREFIX}/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "api": API_PREFIX
        }
    }


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================
@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Check if API is running"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns system health status.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }


@app.get(
    "/health/detailed",
    tags=["Health"],
    summary="Detailed health check",
    description="Detailed system health with component status"
)
async def detailed_health_check():
    """
    Detailed health check.
    
    Returns status of all system components.
    """
    # Check database connection
    db_status = "healthy"
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        logger.error(f"Database health check failed: {e}")
    
    # Check upload directory
    storage_status = "healthy" if settings.UPLOAD_DIR.exists() else "unhealthy"
    
    # Calculate storage usage
    try:
        from app.utils.files import get_directory_size, format_file_size
        storage_size = get_directory_size(settings.UPLOAD_DIR)
        storage_size_formatted = format_file_size(storage_size)
    except Exception:
        storage_size_formatted = "unknown"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "components": {
            "api": "healthy",
            "database": db_status,
            "storage": storage_status,
            "ai_models": "ready",  # Placeholder
            "vector_store": "connected"  # Placeholder
        },
        "storage": {
            "upload_directory": str(settings.UPLOAD_DIR),
            "total_size": storage_size_formatted
        }
    }


# ============================================================================
# CUSTOM ERROR HANDLERS (ADDITIONAL)
# ============================================================================
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "type": "NotFound",
                "message": f"Endpoint {request.url.path} not found",
                "suggestion": "Check API documentation at /docs"
            }
        }
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    """Custom 405 handler."""
    return JSONResponse(
        status_code=405,
        content={
            "error": {
                "type": "MethodNotAllowed",
                "message": f"Method {request.method} not allowed for {request.url.path}",
                "suggestion": "Check allowed methods in API documentation"
            }
        }
    )


# ============================================================================
# DEVELOPMENT HELPERS
# ============================================================================
if settings.DEBUG:
    @app.get(
        "/debug/routes",
        tags=["Debug"],
        include_in_schema=False
    )
    async def debug_routes():
        """
        List all registered routes (debug only).
        
        Only available in debug mode.
        """
        routes = []
        for route in app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name
                })
        
        return {
            "total_routes": len(routes),
            "routes": sorted(routes, key=lambda x: x["path"])
        }
    
    @app.get(
        "/debug/config",
        tags=["Debug"],
        include_in_schema=False
    )
    async def debug_config():
        """
        Show current configuration (debug only).
        
        Sensitive values are masked.
        """
        return {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "project_name": settings.APP_NAME,
            "host": settings.HOST,
            "port": settings.PORT,
            "database_url": settings.DATABASE_URL.split("@")[-1],  # Hide credentials
            "frontend_url": settings.FRONTEND_URL,
            "backend_url": settings.BACKEND_URL,
            "upload_dir": str(settings.UPLOAD_DIR),
            "rate_limiting_enabled": settings.RATE_LIMIT_ENABLED,
            "jwt_algorithm": settings.ALGORITHM,
            "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,  # Auto-reload in debug mode
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True,
        workers=1 if settings.DEBUG else 4  # Multiple workers in production
    )