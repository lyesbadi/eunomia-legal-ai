"""
EUNOMIA Legal AI Platform - Logging Middleware
Structured logging for all HTTP requests and responses
"""
from typing import Callable
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.middleware.request_id import get_request_id


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all HTTP requests and responses.
    
    Features:
    - Logs request method, path, query params
    - Logs response status code
    - Measures and logs request processing time
    - Includes request ID for correlation
    - Structured logging format (JSON-compatible)
    
    Example:
```python
        from fastapi import FastAPI
        from app.middleware.logging import LoggingMiddleware
        
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)
```
        
    Log Output Example:
```
        INFO: GET /api/v1/users/me | status=200 | time=0.045s | request_id=a1b2c3d4
```
    """
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] = None
    ):
        """
        Initialize middleware.
        
        Args:
            app: ASGI application
            exclude_paths: List of paths to exclude from logging (e.g., /health)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/healthz",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request and log details.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/route handler
            
        Returns:
            Response with X-Process-Time header
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get request ID
        request_id = get_request_id(request)
        
        # Start timer
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log incoming request
        logger.info(
            f"→ {method} {path} | "
            f"client={client_host} | "
            f"request_id={request_id}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add processing time to response headers
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            # Log response
            logger.info(
                f"← {method} {path} | "
                f"status={response.status_code} | "
                f"time={process_time:.3f}s | "
                f"request_id={request_id}"
            )
            
            # Log slow requests (> 1 second)
            if process_time > 1.0:
                logger.warning(
                    f"SLOW REQUEST: {method} {path} took {process_time:.3f}s | "
                    f"request_id={request_id}"
                )
            
            return response
            
        except Exception as e:
            # Log exceptions
            process_time = time.time() - start_time
            
            logger.error(
                f"✗ {method} {path} | "
                f"error={str(e)} | "
                f"time={process_time:.3f}s | "
                f"request_id={request_id}",
                exc_info=True
            )
            
            raise


def configure_logging() -> None:
    """
    Configure application-wide logging.
    
    Sets up:
    - Log format (timestamp, level, message)
    - Log level based on environment
    - Console and file handlers
    
    Example:
```python
        from app.middleware.logging import configure_logging
        
        # In main.py
        configure_logging()
```
    """
    from app.core.config import settings
    
    # Determine log level
    log_level = logging.INFO
    if settings.ENVIRONMENT == "development":
        log_level = logging.DEBUG
    elif settings.ENVIRONMENT == "production":
        log_level = logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s:%(funcName)s:%(lineno)d | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set levels for third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured for {settings.ENVIRONMENT} environment")