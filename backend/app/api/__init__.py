"""
EUNOMIA Legal AI Platform - API Package
Main API router and utilities
"""
from fastapi import APIRouter

from app.api.v1 import api_router as api_v1_router


# ============================================================================
# MAIN API ROUTER
# ============================================================================
api_router = APIRouter()

# Include API v1 with prefix
api_router.include_router(
    api_v1_router,
    prefix="/v1"
)


# ============================================================================
# API VERSION INFO
# ============================================================================
API_VERSION = "1.0.0"
API_PREFIX = "/api"

# Available API versions
API_VERSIONS = {
    "v1": {
        "version": "1.0.0",
        "status": "stable",
        "endpoints": [
            "/api/v1/auth",
            "/api/v1/users",
            "/api/v1/documents",
            "/api/v1/analyses"
        ]
    }
}


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    "api_router",
    "API_VERSION",
    "API_PREFIX",
    "API_VERSIONS"
]