"""
EUNOMIA Legal AI Platform - Schemas Package
Export all Pydantic schemas for easy imports across the application
"""

# ============================================================================
# AUTHENTICATION SCHEMAS
# ============================================================================
from app.schemas.auth import (
    # Token Schemas
    TokenData,
    Token,
    TokenPair,
    
    # Login Schemas
    LoginRequest,
    LoginResponse,
    
    # Registration Schemas
    RegisterRequest,
    RegisterResponse,
    
    # Token Refresh Schemas
    TokenRefreshRequest,
    TokenRefreshResponse,
    
    # Password Reset Schemas
    PasswordResetRequest,
    PasswordResetConfirm,
    PasswordResetResponse,
    
    # Email Verification Schemas
    EmailVerificationRequest,
    EmailVerificationResponse,
    
    # API Key Schemas
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyRevokeResponse,
)


# ============================================================================
# USER SCHEMAS
# ============================================================================
from app.schemas.user import (
    # Base Schemas
    UserBase,
    
    # Create/Update Schemas
    UserCreate,
    UserUpdate,
    UserUpdateAdmin,
    UserUpdatePassword,
    
    # Response Schemas
    UserResponse,
    UserPublic,
    UserWithStats,
    UserDetail,
    
    # List Schemas
    UserListResponse,
    
    # Deletion Schemas (GDPR)
    UserDeleteRequest,
    UserDeleteResponse,
)


# ============================================================================
# DOCUMENT SCHEMAS
# ============================================================================
from app.schemas.document import (
    # Base Schemas
    DocumentBase,
    
    # Upload Schemas
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentUploadValidation,
    
    # Update Schemas
    DocumentUpdate,
    
    # Response Schemas
    DocumentResponse,
    DocumentWithAnalysis,
    DocumentListItem,
    DocumentListResponse,
    
    # Processing Schemas
    DocumentReprocessRequest,
    DocumentProcessingStatus,
    
    # Download Schemas
    DocumentDownloadResponse,
    
    # Deletion Schemas (GDPR)
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    
    # Statistics Schemas
    DocumentStatistics,
)


# ============================================================================
# ANALYSIS SCHEMAS
# ============================================================================
from app.schemas.analysis import (
    # Classification Schemas
    ClassificationLabel,
    ClassificationResult,
    
    # Named Entity Recognition Schemas
    EntityType,
    NamedEntity,
    NERResult,
    
    # Unfair Clause Detection Schemas
    ClauseSeverity,
    UnfairClause,
    ClauseDetectionResult,
    
    # Summarization Schemas
    SummaryResult,
    
    # Risk Assessment Schemas
    RiskLevel,
    RiskFactor,
    RiskAssessment,
    
    # LLM Recommendation Schemas
    LLMRecommendation,
    
    # Question-Answering Schemas
    QAPair,
    QAResult,
    
    # Vector Embeddings Schemas
    VectorEmbeddingInfo,
    
    # Complete Analysis Schemas
    AnalysisResponse,
    AnalysisListItem,
    AnalysisListResponse,
)


# ============================================================================
# EXPORTS - ORGANIZED BY CATEGORY
# ============================================================================
__all__ = [
    # ========================================================================
    # AUTHENTICATION
    # ========================================================================
    # Tokens
    "TokenData",
    "Token",
    "TokenPair",
    
    # Login
    "LoginRequest",
    "LoginResponse",
    
    # Registration
    "RegisterRequest",
    "RegisterResponse",
    
    # Token Refresh
    "TokenRefreshRequest",
    "TokenRefreshResponse",
    
    # Password Reset
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "PasswordResetResponse",
    
    # Email Verification
    "EmailVerificationRequest",
    "EmailVerificationResponse",
    
    # API Keys
    "APIKeyCreateRequest",
    "APIKeyCreateResponse",
    "APIKeyRevokeResponse",
    
    # ========================================================================
    # USER MANAGEMENT
    # ========================================================================
    # Base
    "UserBase",
    
    # Create/Update
    "UserCreate",
    "UserUpdate",
    "UserUpdateAdmin",
    "UserUpdatePassword",
    
    # Responses
    "UserResponse",
    "UserPublic",
    "UserWithStats",
    "UserDetail",
    
    # Lists
    "UserListResponse",
    
    # Deletion (GDPR)
    "UserDeleteRequest",
    "UserDeleteResponse",
    
    # ========================================================================
    # DOCUMENT MANAGEMENT
    # ========================================================================
    # Base
    "DocumentBase",
    
    # Upload
    "DocumentUploadRequest",
    "DocumentUploadResponse",
    "DocumentUploadValidation",
    
    # Update
    "DocumentUpdate",
    
    # Responses
    "DocumentResponse",
    "DocumentWithAnalysis",
    "DocumentListItem",
    "DocumentListResponse",
    
    # Processing
    "DocumentReprocessRequest",
    "DocumentProcessingStatus",
    
    # Download
    "DocumentDownloadResponse",
    
    # Deletion (GDPR)
    "DocumentDeleteRequest",
    "DocumentDeleteResponse",
    
    # Statistics
    "DocumentStatistics",
    
    # ========================================================================
    # AI ANALYSIS
    # ========================================================================
    # Classification
    "ClassificationLabel",
    "ClassificationResult",
    
    # Named Entity Recognition
    "EntityType",
    "NamedEntity",
    "NERResult",
    
    # Unfair Clause Detection
    "ClauseSeverity",
    "UnfairClause",
    "ClauseDetectionResult",
    
    # Summarization
    "SummaryResult",
    
    # Risk Assessment
    "RiskLevel",
    "RiskFactor",
    "RiskAssessment",
    
    # LLM Recommendations
    "LLMRecommendation",
    
    # Question-Answering
    "QAPair",
    "QAResult",
    
    # Vector Embeddings
    "VectorEmbeddingInfo",
    
    # Complete Analysis
    "AnalysisResponse",
    "AnalysisListItem",
    "AnalysisListResponse",
]


# ============================================================================
# SCHEMA GROUPS - FOR CONVENIENCE
# ============================================================================

# Authentication schemas group
AUTH_SCHEMAS = [
    "TokenData", "Token", "TokenPair",
    "LoginRequest", "LoginResponse",
    "RegisterRequest", "RegisterResponse",
    "TokenRefreshRequest", "TokenRefreshResponse",
    "PasswordResetRequest", "PasswordResetConfirm", "PasswordResetResponse",
    "EmailVerificationRequest", "EmailVerificationResponse",
    "APIKeyCreateRequest", "APIKeyCreateResponse", "APIKeyRevokeResponse",
]

# User management schemas group
USER_SCHEMAS = [
    "UserBase", "UserCreate", "UserUpdate", "UserUpdateAdmin", "UserUpdatePassword",
    "UserResponse", "UserPublic", "UserWithStats", "UserDetail",
    "UserListResponse",
    "UserDeleteRequest", "UserDeleteResponse",
]

# Document management schemas group
DOCUMENT_SCHEMAS = [
    "DocumentBase",
    "DocumentUploadRequest", "DocumentUploadResponse", "DocumentUploadValidation",
    "DocumentUpdate",
    "DocumentResponse", "DocumentWithAnalysis", "DocumentListItem", "DocumentListResponse",
    "DocumentReprocessRequest", "DocumentProcessingStatus",
    "DocumentDownloadResponse",
    "DocumentDeleteRequest", "DocumentDeleteResponse",
    "DocumentStatistics",
]

# Analysis schemas group
ANALYSIS_SCHEMAS = [
    "ClassificationLabel", "ClassificationResult",
    "EntityType", "NamedEntity", "NERResult",
    "ClauseSeverity", "UnfairClause", "ClauseDetectionResult",
    "SummaryResult",
    "RiskLevel", "RiskFactor", "RiskAssessment",
    "LLMRecommendation",
    "QAPair", "QAResult",
    "VectorEmbeddingInfo",
    "AnalysisResponse", "AnalysisListItem", "AnalysisListResponse",
]


# ============================================================================
# HELPER FUNCTIONS FOR SCHEMA INTROSPECTION
# ============================================================================

def get_all_schemas() -> list[str]:
    """
    Get list of all available schema names.
    
    Returns:
        List of schema class names
        
    Example:
        >>> from app.schemas import get_all_schemas
        >>> schemas = get_all_schemas()
        >>> print(len(schemas))
        85
    """
    return __all__


def get_schemas_by_category(category: str) -> list[str]:
    """
    Get schemas filtered by category.
    
    Args:
        category: One of 'auth', 'user', 'document', 'analysis'
        
    Returns:
        List of schema names in that category
        
    Raises:
        ValueError: If category is invalid
        
    Example:
        >>> from app.schemas import get_schemas_by_category
        >>> auth_schemas = get_schemas_by_category('auth')
        >>> print(auth_schemas[:3])
        ['TokenData', 'Token', 'TokenPair']
    """
    categories = {
        'auth': AUTH_SCHEMAS,
        'user': USER_SCHEMAS,
        'document': DOCUMENT_SCHEMAS,
        'analysis': ANALYSIS_SCHEMAS,
    }
    
    if category not in categories:
        raise ValueError(
            f"Invalid category '{category}'. "
            f"Valid categories: {', '.join(categories.keys())}"
        )
    
    return categories[category]


def get_request_schemas() -> list[str]:
    """
    Get all request schemas (schemas used for API input).
    
    Returns:
        List of request schema names
        
    Example:
        >>> from app.schemas import get_request_schemas
        >>> request_schemas = get_request_schemas()
        >>> 'LoginRequest' in request_schemas
        True
    """
    return [s for s in __all__ if 'Request' in s]


def get_response_schemas() -> list[str]:
    """
    Get all response schemas (schemas used for API output).
    
    Returns:
        List of response schema names
        
    Example:
        >>> from app.schemas import get_response_schemas
        >>> response_schemas = get_response_schemas()
        >>> 'LoginResponse' in response_schemas
        True
    """
    return [s for s in __all__ if 'Response' in s]


# ============================================================================
# VERSION & METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "EUNOMIA Legal AI Platform"
__description__ = "Pydantic schemas for API request/response validation"

# Total schemas count
TOTAL_SCHEMAS = len(__all__)

# Schemas by type
SCHEMAS_BY_TYPE = {
    "Authentication": len(AUTH_SCHEMAS),
    "User Management": len(USER_SCHEMAS),
    "Document Management": len(DOCUMENT_SCHEMAS),
    "AI Analysis": len(ANALYSIS_SCHEMAS),
}


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def _validate_exports():
    """
    Internal validation to ensure all schemas are properly exported.
    
    Raises:
        RuntimeError: If schema export validation fails
    """
    # Verify all schemas in __all__ can be imported
    import sys
    current_module = sys.modules[__name__]
    
    for schema_name in __all__:
        if not hasattr(current_module, schema_name):
            raise RuntimeError(
                f"Schema '{schema_name}' listed in __all__ but not imported"
            )
    
    # Verify schema groups match __all__
    all_group_schemas = (
        AUTH_SCHEMAS + 
        USER_SCHEMAS + 
        DOCUMENT_SCHEMAS + 
        ANALYSIS_SCHEMAS
    )
    
    if set(all_group_schemas) != set(__all__):
        missing = set(__all__) - set(all_group_schemas)
        extra = set(all_group_schemas) - set(__all__)
        
        error_msg = []
        if missing:
            error_msg.append(f"Missing from groups: {missing}")
        if extra:
            error_msg.append(f"Extra in groups: {extra}")
        
        raise RuntimeError(
            f"Schema group mismatch: {'; '.join(error_msg)}"
        )


# Run validation on import (only in development)
import os
if os.getenv("ENVIRONMENT", "development") == "development":
    try:
        _validate_exports()
    except RuntimeError as e:
        import warnings
        warnings.warn(f"Schema export validation warning: {e}")


# ============================================================================
# USAGE EXAMPLES (FOR DOCUMENTATION)
# ============================================================================

"""
USAGE EXAMPLES
==============

1. Import specific schemas:
```python
   from app.schemas import LoginRequest, LoginResponse, UserResponse
```

2. Import by category:
```python
   from app.schemas.auth import LoginRequest, RegisterRequest
   from app.schemas.user import UserResponse, UserWithStats
   from app.schemas.document import DocumentUploadRequest, DocumentResponse
   from app.schemas.analysis import AnalysisResponse, ClassificationResult
```

3. Import all schemas (not recommended):
```python
   from app.schemas import *
```

4. Use in FastAPI routes:
```python
   from fastapi import APIRouter
   from app.schemas import LoginRequest, LoginResponse
   
   router = APIRouter()
   
   @router.post("/login", response_model=LoginResponse)
   async def login(data: LoginRequest):
       # data is automatically validated
       return LoginResponse(...)
```

5. Use helper functions:
```python
   from app.schemas import get_schemas_by_category, get_request_schemas
   
   # Get all auth schemas
   auth_schemas = get_schemas_by_category('auth')
   
   # Get all request schemas
   requests = get_request_schemas()
```

6. Schema validation in services:
```python
   from app.schemas import UserCreate, UserResponse
   
   def create_user(data: UserCreate) -> UserResponse:
       # data is already validated by Pydantic
       user = User(**data.model_dump())
       db.add(user)
       db.commit()
       return UserResponse.model_validate(user)
```

7. Generate OpenAPI documentation:
```python
   from fastapi import FastAPI
   from app.schemas import LoginRequest, LoginResponse
   
   app = FastAPI()
   
   @app.post("/api/v1/auth/login", response_model=LoginResponse)
   async def login(data: LoginRequest):
       # OpenAPI schema automatically generated from Pydantic models
       pass
```

SCHEMA ORGANIZATION
===================

Authentication (17 schemas)
├── Tokens (3): TokenData, Token, TokenPair
├── Login (2): LoginRequest, LoginResponse
├── Registration (2): RegisterRequest, RegisterResponse
├── Token Refresh (2): TokenRefreshRequest, TokenRefreshResponse
├── Password Reset (3): PasswordResetRequest, PasswordResetConfirm, PasswordResetResponse
├── Email Verification (2): EmailVerificationRequest, EmailVerificationResponse
└── API Keys (3): APIKeyCreateRequest, APIKeyCreateResponse, APIKeyRevokeResponse

User Management (12 schemas)
├── Base (1): UserBase
├── Create/Update (4): UserCreate, UserUpdate, UserUpdateAdmin, UserUpdatePassword
├── Responses (4): UserResponse, UserPublic, UserWithStats, UserDetail
├── Lists (1): UserListResponse
└── Deletion (2): UserDeleteRequest, UserDeleteResponse

Document Management (15 schemas)
├── Base (1): DocumentBase
├── Upload (3): DocumentUploadRequest, DocumentUploadResponse, DocumentUploadValidation
├── Update (1): DocumentUpdate
├── Responses (4): DocumentResponse, DocumentWithAnalysis, DocumentListItem, DocumentListResponse
├── Processing (2): DocumentReprocessRequest, DocumentProcessingStatus
├── Download (1): DocumentDownloadResponse
├── Deletion (2): DocumentDeleteRequest, DocumentDeleteResponse
└── Statistics (1): DocumentStatistics

AI Analysis (41 schemas)
├── Classification (2): ClassificationLabel, ClassificationResult
├── NER (3): EntityType, NamedEntity, NERResult
├── Clauses (3): ClauseSeverity, UnfairClause, ClauseDetectionResult
├── Summarization (1): SummaryResult
├── Risk (3): RiskLevel, RiskFactor, RiskAssessment
├── LLM (1): LLMRecommendation
├── Q&A (2): QAPair, QAResult
├── Embeddings (1): VectorEmbeddingInfo
└── Complete (3): AnalysisResponse, AnalysisListItem, AnalysisListResponse

TOTAL: 85 SCHEMAS
"""