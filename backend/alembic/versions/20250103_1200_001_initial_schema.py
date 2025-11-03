"""Initial schema with users, documents, analyses, audit_logs

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-01-03 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # ========================================================================
    # USERS TABLE
    # ========================================================================
    op.create_table(
        'users',
        # Primary Key
        sa.Column('id', sa.Integer(), nullable=False),
        
        # Authentication
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        
        # Profile
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('company', sa.String(length=255), nullable=True),
        sa.Column('role', sa.Enum('ADMIN', 'MANAGER', 'USER', 'VIEWER', name='userrole', native_enum=False), nullable=False, server_default='USER'),
        
        # Status
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        
        # GDPR Compliance
        sa.Column('anonymized', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('anonymized_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('data_retention_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('gdpr_consent_date', sa.DateTime(timezone=True), nullable=True),
        
        # Constraints
        sa.PrimaryKeyConstraint('id', name='pk_users'),
        sa.UniqueConstraint('email', name='uq_users_email')
    )
    
    # Indexes for users
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_role', 'users', ['role'])
    op.create_index('ix_users_is_active', 'users', ['is_active'])
    op.create_index('ix_users_created_at', 'users', ['created_at'])
    
    # ========================================================================
    # DOCUMENTS TABLE
    # ========================================================================
    op.create_table(
        'documents',
        # Primary Key
        sa.Column('id', sa.Integer(), nullable=False),
        
        # User relationship
        sa.Column('user_id', sa.Integer(), nullable=False),
        
        # File metadata
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('storage_path', sa.String(length=500), nullable=False),
        
        # Document info
        sa.Column('title', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('document_type', sa.Enum('CONTRACT', 'TERMS_OF_SERVICE', 'PRIVACY_POLICY', 'LEGAL_NOTICE', 'COURT_DECISION', 'REGULATION', 'LEGAL_OPINION', 'OTHER', name='documenttype', native_enum=False), nullable=True),
        
        # Processing status
        sa.Column('status', sa.Enum('UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED', 'DELETED', name='documentstatus', native_enum=False), nullable=False, server_default='UPLOADED'),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        
        # Content
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        
        # Constraints
        sa.PrimaryKeyConstraint('id', name='pk_documents'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_documents_user_id_users', ondelete='CASCADE')
    )
    
    # Indexes for documents
    op.create_index('ix_documents_user_id', 'documents', ['user_id'])
    op.create_index('ix_documents_status', 'documents', ['status'])
    op.create_index('ix_documents_file_hash', 'documents', ['file_hash'])
    op.create_index('ix_documents_created_at', 'documents', ['created_at'])
    op.create_index('ix_documents_document_type', 'documents', ['document_type'])
    
    # ========================================================================
    # ANALYSES TABLE
    # ========================================================================
    op.create_table(
        'analyses',
        # Primary Key
        sa.Column('id', sa.Integer(), nullable=False),
        
        # Document relationship (one-to-one)
        sa.Column('document_id', sa.Integer(), nullable=False),
        
        # Document Classification (Legal-BERT)
        sa.Column('document_class', sa.String(length=100), nullable=True),
        sa.Column('classification_confidence', sa.Float(), nullable=True),
        sa.Column('classification_labels', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Named Entity Recognition (CamemBERT-NER)
        sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('entity_count', sa.Integer(), nullable=True),
        
        # Unfair Clause Detection (Unfair-ToS)
        sa.Column('unfair_clauses', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('unfair_clause_count', sa.Integer(), nullable=True),
        sa.Column('fairness_score', sa.Float(), nullable=True),
        
        # Summarization (BART)
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('summary_length', sa.Integer(), nullable=True),
        
        # Risk Assessment
        sa.Column('risk_level', sa.String(length=20), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=True),
        sa.Column('risk_factors', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # LLM Recommendations (Ollama Mistral)
        sa.Column('recommendations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('llm_model_version', sa.String(length=50), nullable=True),
        
        # Question-Answering
        sa.Column('qa_pairs', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Vector Embeddings Info
        sa.Column('embedding_model', sa.String(length=100), nullable=True),
        sa.Column('vector_dimension', sa.Integer(), nullable=True),
        sa.Column('qdrant_point_id', sa.String(length=100), nullable=True),
        
        # Processing metadata
        sa.Column('processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('model_versions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        
        # Constraints
        sa.PrimaryKeyConstraint('id', name='pk_analyses'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name='fk_analyses_document_id_documents', ondelete='CASCADE'),
        sa.UniqueConstraint('document_id', name='uq_analyses_document_id')
    )
    
    # Indexes for analyses
    op.create_index('ix_analyses_document_id', 'analyses', ['document_id'])
    op.create_index('ix_analyses_document_class', 'analyses', ['document_class'])
    op.create_index('ix_analyses_risk_level', 'analyses', ['risk_level'])
    op.create_index('ix_analyses_created_at', 'analyses', ['created_at'])
    
    # ========================================================================
    # AUDIT_LOGS TABLE
    # ========================================================================
    op.create_table(
        'audit_logs',
        # Primary Key
        sa.Column('id', sa.Integer(), nullable=False),
        
        # User relationship (nullable for system actions)
        sa.Column('user_id', sa.Integer(), nullable=True),
        
        # Action details
        sa.Column('action', sa.Enum(
            'LOGIN_SUCCESS', 'LOGIN_FAILED', 'LOGOUT', 'PASSWORD_CHANGE', 'PASSWORD_RESET_REQUEST', 'PASSWORD_RESET_COMPLETE',
            'USER_REGISTER', 'USER_UPDATE', 'USER_DELETE', 'USER_ANONYMIZE',
            'DOCUMENT_UPLOAD', 'DOCUMENT_VIEW', 'DOCUMENT_DOWNLOAD', 'DOCUMENT_UPDATE', 'DOCUMENT_DELETE', 'DOCUMENT_SHARE',
            'ANALYSIS_START', 'ANALYSIS_COMPLETE', 'ANALYSIS_FAILED', 'ANALYSIS_VIEW',
            'DATA_EXPORT', 'DATA_ACCESS_REQUEST', 'DATA_DELETION_REQUEST', 'DATA_ANONYMIZATION',
            'ADMIN_ACTION', 'SETTINGS_CHANGE',
            'SUSPICIOUS_ACTIVITY', 'ACCESS_DENIED', 'API_KEY_CREATED', 'API_KEY_REVOKED',
            name='actiontype', native_enum=False
        ), nullable=False),
        
        sa.Column('resource_type', sa.Enum('USER', 'DOCUMENT', 'ANALYSIS', 'SYSTEM', name='resourcetype', native_enum=False), nullable=True),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        
        # Request metadata
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('request_method', sa.String(length=10), nullable=True),
        sa.Column('request_path', sa.String(length=500), nullable=True),
        
        # Result
        sa.Column('success', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('error_message', sa.Text(), nullable=True),
        
        # Additional data
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Timestamp
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        
        # Constraints
        sa.PrimaryKeyConstraint('id', name='pk_audit_logs'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_audit_logs_user_id_users', ondelete='SET NULL')
    )
    
    # Indexes for audit_logs
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_resource_type', 'audit_logs', ['resource_type'])
    op.create_index('ix_audit_logs_resource_id', 'audit_logs', ['resource_id'])
    op.create_index('ix_audit_logs_timestamp', 'audit_logs', ['timestamp'])
    op.create_index('ix_audit_logs_ip_address', 'audit_logs', ['ip_address'])


def downgrade() -> None:
    """Drop all tables."""
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('audit_logs')
    op.drop_table('analyses')
    op.drop_table('documents')
    op.drop_table('users')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS actiontype')
    op.execute('DROP TYPE IF EXISTS resourcetype')
    op.execute('DROP TYPE IF EXISTS documentstatus')
    op.execute('DROP TYPE IF EXISTS documenttype')
    op.execute('DROP TYPE IF EXISTS userrole')