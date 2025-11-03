"""
EUNOMIA Legal AI Platform - Configuration Management
Centralized configuration using Pydantic Settings with environment variables
"""
from typing import Optional, List, Any
from pydantic import Field, validator, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All sensitive data (passwords, tokens) must be provided via .env file.
    Defaults are provided for development only.
    """
    
    # =========================================================================
    # APPLICATION SETTINGS
    # =========================================================================
    APP_NAME: str = "EUNOMIA Legal AI Platform"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "European Sovereign Legal Analysis Platform"
    ENVIRONMENT: str = Field(default="production", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Debug mode (disable in production)")
    INSTANCE_ID: str = Field(default="1", description="Instance ID for load balancing")
    HOST: str = Field(default="0.0.0.0", description="Host to bind server")
    PORT: int = Field(default=8000, description="Port to bind server")
    DOMAIN: str = Field(default="localhost", description="Application domain name")
    
    # URLs (for CORS, emails, etc.)
    FRONTEND_URL: str = Field(default="http://localhost:3000", description="Main URL for the frontend")
    BACKEND_URL: str = Field(default="http://localhost:8000", description="Main URL for the backend")

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    API_DOCS_URL: Optional[str] = "/api/docs"
    API_REDOC_URL: Optional[str] = "/api/redoc"
    
    # =========================================================================
    # SECURITY SETTINGS
    # =========================================================================
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT encoding (MUST be changed in production)"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="JWT access token lifetime")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="JWT refresh token lifetime")
    
    # Password hashing
    BCRYPT_ROUNDS: int = Field(default=12, description="Bcrypt hashing rounds (higher = more secure but slower)")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Max requests per minute per IP")
    
    # =========================================================================
    # DATABASE SETTINGS (PostgreSQL)
    # =========================================================================
    POSTGRES_SERVER: str = Field(default="postgres", description="PostgreSQL server hostname")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL server port")
    POSTGRES_USER: str = Field(default="eunomia_user", description="PostgreSQL username")
    POSTGRES_PASSWORD: str = Field(default="changeme", description="PostgreSQL password")
    POSTGRES_DB: str = Field(default="eunomia_db", description="PostgreSQL database name")
    
    DATABASE_URL: Optional[str] = None
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        """Construct PostgreSQL async connection URL from components"""
        if isinstance(v, str) and v:
            return v
        
        values = info.data
        return (
            f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:"
            f"{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}:"
            f"{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"
        )
    
    # Database Pool Settings
    DB_POOL_SIZE: int = Field(default=20, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=10, description="Max overflow connections")
    DB_POOL_RECYCLE: int = Field(default=3600, description="Connection recycle time (seconds)")
    DB_POOL_PRE_PING: bool = Field(default=True, description="Test connections before use")
    DB_ECHO: bool = Field(default=False, description="Echo SQL queries (debug only)")
    
    # =========================================================================
    # REDIS SETTINGS
    # =========================================================================
    REDIS_HOST: str = Field(default="redis", description="Redis server hostname")
    REDIS_PORT: int = Field(default=6379, description="Redis server port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password (optional)")
    
    REDIS_URL: Optional[str] = None
    
    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: Optional[str], info) -> str:
        """Construct Redis connection URL from components"""
        if isinstance(v, str) and v:
            return v
        
        values = info.data
        password = values.get('REDIS_PASSWORD')
        auth = f":{password}@" if password else ""
        
        return (
            f"redis://{auth}{values.get('REDIS_HOST')}:"
            f"{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
        )
    
    # Redis Cache Settings
    CACHE_TTL_SECONDS: int = Field(default=3600, description="Default cache TTL")
    CACHE_KEY_PREFIX: str = Field(default="eunomia", description="Redis key prefix")
    
    # =========================================================================
    # CELERY SETTINGS
    # =========================================================================
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_TIME_LIMIT: int = Field(default=300, description="Task time limit (seconds)")

    @field_validator("CELERY_BROKER_URL", mode="before")
    @classmethod
    def set_celery_broker(cls, v: Optional[str], info) -> str:
        """Use Redis as Celery broker"""
        if isinstance(v, str) and v:
            return v
        return info.data.get('REDIS_URL', 'redis://redis:6379/0')
    
    @field_validator("CELERY_RESULT_BACKEND", mode="before")
    @classmethod
    def set_celery_backend(cls, v: Optional[str], info) -> str:
        """Use Redis as Celery result backend"""
        if isinstance(v, str) and v:
            return v
        return info.data.get('REDIS_URL', 'redis://redis:6379/0')
    
    # =========================================================================
    # QDRANT VECTOR DATABASE
    # =========================================================================
    QDRANT_HOST: str = Field(default="qdrant", description="Qdrant server hostname")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant HTTP port")
    QDRANT_GRPC_PORT: int = Field(default=6334, description="Qdrant gRPC port")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key (optional)")
    
    QDRANT_URL: Optional[str] = None
    
    @field_validator("QDRANT_URL", mode="before")
    @classmethod
    def assemble_qdrant_url(cls, v: Optional[str], info) -> str:
        """Construct Qdrant HTTP URL"""
        if isinstance(v, str) and v:
            return v
        
        values = info.data
        return f"http://{values.get('QDRANT_HOST')}:{values.get('QDRANT_PORT')}"
    
    # Qdrant Collections
    QDRANT_COLLECTION_DOCUMENTS: str = "legal_documents"
    QDRANT_COLLECTION_CHUNKS: str = "document_chunks"
    QDRANT_VECTOR_SIZE: int = Field(default=768, description="Embedding vector size (sentence-transformers)")
    
    # =========================================================================
    # OLLAMA LLM SETTINGS
    # =========================================================================
    OLLAMA_HOST: str = Field(default="ollama", description="Ollama server hostname")
    OLLAMA_PORT: int = Field(default=11434, description="Ollama server port")
    OLLAMA_MODEL: str = Field(default="eurollm-9b:latest", description="Ollama model name (EuroLLM)")
    OLLAMA_URL: Optional[str] = None
    
    @field_validator("OLLAMA_URL", mode="before")
    @classmethod
    def assemble_ollama_url(cls, v: Optional[str], info) -> str:
        """Construct Ollama API URL"""
        if isinstance(v, str) and v:
            return v
        
        values = info.data
        return f"http://{values.get('OLLAMA_HOST')}:{values.get('OLLAMA_PORT')}"
    
    # LLM Generation Settings
    OLLAMA_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    OLLAMA_MAX_TOKENS: int = Field(default=2048, description="Max tokens per generation")
    OLLAMA_TIMEOUT: int = Field(default=120, description="Request timeout (seconds)")
    
    # =========================================================================
    # HUGGING FACE MODELS
    # =========================================================================
    HF_CACHE_DIR: str = Field(default="/root/.cache/huggingface", description="Hugging Face cache directory")
    HF_TOKEN: Optional[str] = Field(default=None, description="Hugging Face API token (optional)")
    
    # Model Names
    HF_MODEL_LEGAL_BERT: str = "nlpaueb/legal-bert-base-uncased"
    HF_MODEL_CAMEMBERT_NER: str = "Jean-Baptiste/camembert-ner"
    HF_MODEL_UNFAIR_TOS: str = "CodeHima/TOSBertV2"
    HF_MODEL_SUMMARIZATION: str = "facebook/bart-large-cnn"
    HF_MODEL_EMBEDDINGS: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_MODEL_QA: str = "deepset/roberta-base-squad2"
    
    # Model Loading Settings
    HF_DEVICE: str = Field(default="cpu", description="Device for model inference (cpu/cuda)")
    HF_MAX_LENGTH: int = Field(default=512, description="Max sequence length for models")
    
    # =========================================================================
    # FILE STORAGE SETTINGS
    # =========================================================================
    UPLOAD_DIR: Path = Field(default="/app/uploads", description="Directory for uploaded files")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, description="Max file upload size (MB)")
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md"],
        description="Allowed file extensions"
    )
    
    # =========================================================================
    # LOGGING SETTINGS
    # =========================================================================
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = "json"  # json or text
    LOG_DIR: str = Field(default="/app/logs", description="Log files directory")
    LOG_ROTATION: str = Field(default="100 MB", description="Log rotation size")
    LOG_RETENTION: str = Field(default="30 days", description="Log retention period")
    
    # =========================================================================
    # MONITORING & OBSERVABILITY
    # =========================================================================
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = Field(default=9090, description="Prometheus metrics port")
    
    # Sentry (Optional)
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    SENTRY_ENVIRONMENT: Optional[str] = None
    
    @field_validator("SENTRY_ENVIRONMENT", mode="before")
    @classmethod
    def set_sentry_env(cls, v: Optional[str], info) -> Optional[str]:
        """Set Sentry environment from ENVIRONMENT"""
        if v:
            return v
        return info.data.get('ENVIRONMENT')
    
    # =========================================================================
    # GDPR COMPLIANCE SETTINGS
    # =========================================================================
    GDPR_AUDIT_ENABLED: bool = Field(default=True, description="Enable GDPR audit logging")
    GDPR_DATA_RETENTION_DAYS: int = Field(default=2555, description="Data retention period (7 years)")
    GDPR_ANONYMIZATION_ENABLED: bool = Field(default=True, description="Enable automatic PII anonymization")
    
    # =========================================================================
    # FEATURE FLAGS
    # =========================================================================
    FEATURE_DOCUMENT_ANALYSIS: bool = True
    FEATURE_NER_EXTRACTION: bool = True
    FEATURE_UNFAIR_CLAUSE_DETECTION: bool = True
    FEATURE_SUMMARIZATION: bool = True
    FEATURE_QA: bool = True
    FEATURE_LLM_GENERATION: bool = True
    
    # =========================================================================
    # PYDANTIC SETTINGS CONFIGURATION
    # =========================================================================
    model_config = SettingsConfigDict(
        env_file=".env.production",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL (for Alembic migrations)"""
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as list"""
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return self.CORS_ORIGINS


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once
    and reused across the application.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Export settings instance
settings = get_settings()