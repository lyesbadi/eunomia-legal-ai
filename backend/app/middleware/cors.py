"""
EUNOMIA Legal AI Platform - CORS Middleware
Cross-Origin Resource Sharing configuration for frontend integration
"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from app.core.config import settings
import logging


logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        
    Configuration:
    - Allows requests from configured frontend origins
    - Supports credentials (cookies, authorization headers)
    - Allows all standard HTTP methods
    - Exposes necessary headers
    
    Example:
```python
        from fastapi import FastAPI
        from app.middleware.cors import setup_cors
        
        app = FastAPI()
        setup_cors(app)
```
    """
    # Get allowed origins from settings
    origins = []
    
    # Frontend URL (production)
    if settings.FRONTEND_URL:
        origins.append(settings.FRONTEND_URL)
    
    # Backend URL (for Swagger UI)
    if settings.BACKEND_URL:
        origins.append(settings.BACKEND_URL)
    
    # Development origins
    if settings.ENVIRONMENT == "development":
        origins.extend([
            "http://localhost:3000",      # React dev server
            "http://localhost:5173",      # Vite dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ])
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],  # GET, POST, PUT, DELETE, PATCH, OPTIONS
        allow_headers=["*"],  # All headers including Authorization
        expose_headers=[
            "Content-Length",
            "Content-Type",
            "X-Request-ID",
            "X-Process-Time"
        ],
        max_age=600  # Cache preflight requests for 10 minutes
    )
    
    logger.info(f"CORS configured with allowed origins: {origins}")


# ============================================================================
# CORS CONFIGURATION DETAILS
# ============================================================================
"""
CORS Configuration Breakdown:

1. **allow_origins**: List of allowed origins
   - Production: settings.FRONTEND_URL (e.g., https://eunomia.legal)
   - Development: localhost:3000, localhost:5173
   - Backend: For Swagger UI access

2. **allow_credentials**: True
   - Allows cookies and Authorization headers
   - Required for JWT tokens
   - Frontend must use withCredentials: true

3. **allow_methods**: ["*"]
   - GET, POST, PUT, DELETE, PATCH, OPTIONS
   - All standard HTTP methods

4. **allow_headers**: ["*"]
   - Authorization (Bearer tokens)
   - Content-Type
   - Accept
   - X-Request-ID
   - Custom headers

5. **expose_headers**: List of headers exposed to frontend
   - X-Request-ID: For request tracking
   - X-Process-Time: For performance monitoring

6. **max_age**: 600 seconds (10 minutes)
   - Browser caches preflight (OPTIONS) requests
   - Reduces number of OPTIONS calls

Frontend Usage:
```javascript
// Axios configuration
axios.defaults.withCredentials = true;

// Fetch configuration
fetch(url, {
    credentials: 'include',
    headers: {
        'Authorization': `Bearer ${token}`
    }
});
```
"""