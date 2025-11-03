"""
EUNOMIA Legal AI Platform - Services Package
Business logic layer for the application
"""
from app.services.user_service import (
    UserService,
    create_user,
    get_user_by_email,
    authenticate_user
)
from app.services.document_service import (
    DocumentService,
    get_document,
    get_user_documents,
    calculate_file_hash
)
from app.services.audit_service import (
    AuditService,
    log_action,
    get_user_audit_logs
)


from app.services.ai_service import ai_service, AIService
from app.services.llm_service import llm_service, LLMService
from app.services.vector_service import vector_service, VectorService


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # Service classes
    "UserService",
    "DocumentService",
    "AuditService",
    
    # User convenience functions
    "create_user",
    "get_user_by_email",
    "authenticate_user",
    
    # Document convenience functions
    "get_document",
    "get_user_documents",
    "calculate_file_hash",
    
    # Audit convenience functions
    "log_action",
    "get_user_audit_logs",

    # AI convience functions
    "ai_service",
    "AIService",
    "llm_service",
    "LLMService",
    "vector_service",
    "VectorService",
]