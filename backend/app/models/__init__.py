
# EUNOMIA Legal AI Platform - Models Package
# Export all SQLAlchemy models for easy imports

from app.core.database import Base

# Import all models (order matters for relationships)
from app.models.user import User, UserRole
from app.models.document import Document, DocumentType, DocumentStatus
from app.models.analysis import Analysis
from app.models.audit_log import AuditLog, ActionType, ResourceType

# Export all models
__all__ = [
    # Base
    "Base",
    
    # User
    "User",
    "UserRole",
    
    # Document
    "Document",
    "DocumentType",
    "DocumentStatus",
    
    # Analysis
    "Analysis",
    
    # Audit Log
    "AuditLog",
    "ActionType",
    "ResourceType",
]