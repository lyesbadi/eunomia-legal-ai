"""
EUNOMIA Legal AI Platform - API v1 Router Aggregation
Combines all v1 API routers into a single router
"""
from fastapi import APIRouter

from app.api.v1 import auth, users, documents, analyses


# ============================================================================
# API V1 ROUTER
# ============================================================================
api_router = APIRouter()

# Include all v1 routers
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"]
)

api_router.include_router(
    analyses.router,
    prefix="/analyses",
    tags=["Analyses"]
)


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = ["api_router"]