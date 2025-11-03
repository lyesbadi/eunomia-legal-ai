"""
EUNOMIA Legal AI Platform - Middleware Package
Export all middleware components and setup functions
"""
from app.middleware.cors import setup_cors
from app.middleware.request_id import RequestIDMiddleware, get_request_id
from app.middleware.logging import LoggingMiddleware, configure_logging
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.error_handler import setup_exception_handlers


# ============================================================================
# MIDDLEWARE SETUP FUNCTION
# ============================================================================
def setup_middleware(app) -> None:
    """
    Configure all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        
    Middleware order (applied in reverse):
    1. Error handler (exception handlers)
    2. Rate limiting
    3. Logging
    4. Request ID
    5. CORS
    
    Example:
```python
        from fastapi import FastAPI
        from app.middleware import setup_middleware
        
        app = FastAPI()
        setup_middleware(app)
```
    """
    from app.core.config import settings
    
    # 1. Configure logging first
    configure_logging()
    
    # 2. Setup exception handlers
    setup_exception_handlers(app)
    
    # 3. Add middlewares (in reverse order of execution)
    
    # CORS (last, executes first)
    setup_cors(app)
    
    # Request ID
    app.add_middleware(RequestIDMiddleware)
    
    # Logging
    app.add_middleware(LoggingMiddleware)
    
    # Rate limiting (optional, can be disabled in dev)
    if settings.RATE_LIMIT_ENABLED:
        app.add_middleware(
            RateLimitMiddleware,
            requests=settings.RATE_LIMIT_PER_MINUTE,
            window=settings.RATE_LIMIT_WINDOW
        )


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # Setup functions
    "setup_middleware",
    "setup_cors",
    "setup_exception_handlers",
    "configure_logging",
    
    # Middleware classes
    "RequestIDMiddleware",
    "LoggingMiddleware",
    "RateLimitMiddleware",
    
    # Utilities
    "get_request_id",
]