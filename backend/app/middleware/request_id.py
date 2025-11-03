"""
EUNOMIA Legal AI Platform - Request ID Middleware
Adds unique request ID to every request for distributed tracing
"""
from typing import Callable
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging


logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds a unique request ID to every request.
    
    Features:
    - Generates UUID4 for each request
    - Adds X-Request-ID header to request
    - Includes X-Request-ID in response headers
    - Makes request_id available in request.state
    
    Benefits:
    - Distributed tracing across services
    - Log correlation
    - Debugging production issues
    - Request tracking in monitoring systems
    
    Example:
```python
        from fastapi import FastAPI, Request
        from app.middleware.request_id import RequestIDMiddleware
        
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)
        
        @app.get("/test")
        async def test(request: Request):
            request_id = request.state.request_id
            return {"request_id": request_id}
```
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize middleware.
        
        Args:
            app: ASGI application
        """
        super().__init__(app)
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request and add request ID.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/route handler
            
        Returns:
            Response with X-Request-ID header
        """
        # Check if request already has an ID (from load balancer/proxy)
        request_id = request.headers.get("X-Request-ID")
        
        # Generate new ID if not present
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for access in routes
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


def get_request_id(request: Request) -> str:
    """
    Get request ID from request state.
    
    Args:
        request: FastAPI request
        
    Returns:
        Request ID string
        
    Example:
```python
        from fastapi import Request, Depends
        from app.middleware.request_id import get_request_id
        
        @app.get("/test")
        async def test(request_id: str = Depends(get_request_id)):
            logger.info(f"Processing request {request_id}")
            return {"request_id": request_id}
```
    """
    return getattr(request.state, "request_id", "unknown")