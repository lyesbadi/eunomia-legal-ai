"""
EUNOMIA Legal AI Platform - Error Handler Middleware
Centralized exception handling for consistent error responses
"""
from typing import Union
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError
import logging

from app.middleware.request_id import get_request_id


logger = logging.getLogger(__name__)


# ============================================================================
# ERROR RESPONSE MODELS
# ============================================================================
def create_error_response(
    request: Request,
    status_code: int,
    error_type: str,
    message: str,
    details: Union[dict, list, None] = None
) -> JSONResponse:
    """
    Create standardized error response.
    
    Args:
        request: FastAPI request
        status_code: HTTP status code
        error_type: Error type (e.g., "ValidationError", "NotFound")
        message: Human-readable error message
        details: Additional error details
        
    Returns:
        JSONResponse with error information
    """
    request_id = get_request_id(request)
    
    error_response = {
        "error": {
            "type": error_type,
            "message": message,
            "request_id": request_id,
            "path": request.url.path
        }
    }
    
    if details:
        error_response["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handle HTTP exceptions (4xx, 5xx).
    
    Args:
        request: FastAPI request
        exc: HTTP exception
        
    Returns:
        JSON error response
    """
    return create_error_response(
        request=request,
        status_code=exc.status_code,
        error_type="HTTPException",
        message=exc.detail
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors (Pydantic).
    
    Args:
        request: FastAPI request
        exc: Validation error
        
    Returns:
        JSON error response with validation details
    """
    # Extract validation errors
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        f"Validation error on {request.url.path}: {errors}"
    )
    
    return create_error_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_type="ValidationError",
        message="Request validation failed",
        details=errors
    )


async def sqlalchemy_exception_handler(
    request: Request,
    exc: SQLAlchemyError
) -> JSONResponse:
    """
    Handle SQLAlchemy database errors.
    
    Args:
        request: FastAPI request
        exc: SQLAlchemy exception
        
    Returns:
        JSON error response
    """
    request_id = get_request_id(request)
    
    logger.error(
        f"Database error on {request.url.path} | "
        f"request_id={request_id} | "
        f"error={str(exc)}",
        exc_info=True
    )
    
    # Don't expose internal database errors to users
    return create_error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="DatabaseError",
        message="A database error occurred. Please try again later."
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    Args:
        request: FastAPI request
        exc: Any exception
        
    Returns:
        JSON error response
    """
    request_id = get_request_id(request)
    
    # Log full traceback for debugging
    logger.error(
        f"Unhandled exception on {request.url.path} | "
        f"request_id={request_id} | "
        f"error={str(exc)}\n"
        f"{''.join(traceback.format_tb(exc.__traceback__))}",
        exc_info=True
    )
    
    # Return generic error to user (don't expose internals)
    return create_error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="InternalServerError",
        message="An unexpected error occurred. Please try again later."
    )


# ============================================================================
# SETUP FUNCTION
# ============================================================================
def setup_exception_handlers(app) -> None:
    """
    Register all exception handlers with FastAPI app.
    
    Args:
        app: FastAPI application instance
        
    Example:
```python
        from fastapi import FastAPI
        from app.middleware.error_handler import setup_exception_handlers
        
        app = FastAPI()
        setup_exception_handlers(app)
```
    """
    # HTTP exceptions
    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler
    )
    
    # Validation errors
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler
    )
    
    # Database errors
    app.add_exception_handler(
        SQLAlchemyError,
        sqlalchemy_exception_handler
    )
    
    # Catch-all for unexpected errors
    app.add_exception_handler(
        Exception,
        general_exception_handler
    )
    
    logger.info("Exception handlers configured")