"""
EUNOMIA Legal AI Platform - Rate Limiting Middleware
Global rate limiting to protect against abuse and DDoS
"""
from typing import Callable
import time
from collections import defaultdict
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging


logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.
    
    Note: This is a basic implementation for single-server deployments.
    For production with multiple servers, use Redis-based rate limiting.
    """
    
    def __init__(self, requests: int = 100, window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests: Number of requests allowed per window
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.storage: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if request is allowed.
        
        Args:
            key: Client identifier (IP or user ID)
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window
        
        # Clean old requests
        self.storage[key] = [
            timestamp for timestamp in self.storage[key]
            if timestamp > window_start
        ]
        
        # Check if under limit
        if len(self.storage[key]) < self.requests:
            self.storage[key].append(now)
            return True, 0
        
        # Calculate retry-after
        oldest_request = min(self.storage[key])
        retry_after = int(oldest_request + self.window - now) + 1
        
        return False, retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Global rate limiting middleware.
    
    Features:
    - Per-IP rate limiting
    - Per-user rate limiting (when authenticated)
    - Configurable limits per environment
    - Returns 429 (Too Many Requests) when exceeded
    - Includes Retry-After header
    
    Example:
```python
        from fastapi import FastAPI
        from app.middleware.rate_limit import RateLimitMiddleware
        
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests=100,
            window=60
        )
```
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests: int = 100,
        window: int = 60,
        exclude_paths: list[str] = None
    ):
        """
        Initialize middleware.
        
        Args:
            app: ASGI application
            requests: Number of requests allowed per window
            window: Time window in seconds
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(requests, window)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/healthz",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
    
    def get_client_identifier(self, request: Request) -> str:
        """
        Get unique client identifier.
        
        Priority:
        1. User ID (if authenticated)
        2. IP address
        
        Args:
            request: FastAPI request
            
        Returns:
            Client identifier string
        """
        # Try to get user ID from request state (set by auth dependency)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user_{user_id}"
        
        # Fall back to IP address
        if request.client:
            return f"ip_{request.client.host}"
        
        return "unknown"
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request and check rate limit.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/route handler
            
        Returns:
            Response or 429 if rate limit exceeded
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get client identifier
        client_id = self.get_client_identifier(request)
        
        # Check rate limit
        is_allowed, retry_after = self.limiter.is_allowed(client_id)
        
        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded for {client_id} on {request.url.path}"
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit info to response headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests)
        response.headers["X-RateLimit-Window"] = str(self.limiter.window)
        response.headers["X-RateLimit-Remaining"] = str(
            self.limiter.requests - len(self.limiter.storage[client_id])
        )
        
        return response