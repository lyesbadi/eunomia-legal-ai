"""
EUNOMIA Legal AI Platform - Analysis Model
SQLAlchemy model for AI analysis results
"""
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import String, Integer, Boolean, DateTime, Text, ForeignKey, Index, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


# ============================================================================
# ANALYSIS MODEL
# ============================================================================
class Analysis(Base):
    """
    Analysis model for AI-generated document analysis results.
    
    Stores results from:
    - Document classification (Legal-BERT)
    - Named Entity Recognition (CamemBERT-NER)
    - Unfair clause detection (Unfair-ToS)
    - Document summarization (BART)
    - Question-Answering capabilities
    - LLM-generated recommendations (Ollama Mistral)
    
    Features:
    - Structured JSON storage for complex results
    - Confidence scores for each analysis type
    - Processing metadata
    - Versioning for model updates
    
    Relationships:
    - document: One-to-one with Document model
    """
    
    # Primary Key
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # ========================================================================
    # DOCUMENT RELATIONSHIP
    # ========================================================================
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        doc="Associated document ID (one-to-one)"
    )
    
    # ========================================================================
    # DOCUMENT CLASSIFICATION
    # ========================================================================
    document_class: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Classified document type (contract, ToS, etc.)"
    )
    
    classification_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Classification confidence score (0-1)"
    )
    
    classification_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="All classification scores as JSON {class: score}"
    )
    
    # ========================================================================
    # NAMED ENTITY RECOGNITION (NER)
    # ========================================================================
    entities_detected: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Total number of entities detected"
    )
    
    entities: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Detected entities as JSON [{text, label, start, end, confidence}]"
    )
    
    persons_detected: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of person names detected"
    )
    
    organizations_detected: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of organizations detected"
    )
    
    locations_detected: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of locations detected"
    )
    
    dates_detected: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of dates detected"
    )
    
    # ========================================================================
    # UNFAIR CLAUSE DETECTION
    # ========================================================================
    unfair_clauses_detected: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of unfair clauses detected"
    )
    
    unfair_clauses: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Detected unfair clauses as JSON [{text, type, severity, position}]"
    )
    
    has_unfair_clauses: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        doc="Whether any unfair clauses were detected"
    )
    
    unfair_clause_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Overall unfair clause score (0-1, higher = more unfair)"
    )
    
    # ========================================================================
    # DOCUMENT SUMMARIZATION
    # ========================================================================
    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="AI-generated document summary"
    )
    
    summary_length: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Summary length in characters"
    )
    
    key_points: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Extracted key points as JSON array"
    )
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    risk_level: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        doc="Overall risk level (low, medium, high, critical)"
    )
    
    risk_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Numerical risk score (0-1)"
    )
    
    risk_factors: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Detected risk factors as JSON [{factor, severity, description}]"
    )
    
    # ========================================================================
    # LLM RECOMMENDATIONS (OLLAMA)
    # ========================================================================
    recommendations: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="LLM-generated recommendations"
    )
    
    custom_clauses: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="LLM-generated custom clauses"
    )
    
    legal_explanation: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Vulgarized legal explanation"
    )
    
    # ========================================================================
    # QUESTION-ANSWERING
    # ========================================================================
    qa_pairs: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Pre-computed Q&A pairs as JSON [{question, answer, confidence}]"
    )
    
    # ========================================================================
    # VECTOR EMBEDDINGS
    # ========================================================================
    embedding_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether embeddings were generated and stored in Qdrant"
    )
    
    qdrant_point_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Qdrant vector point ID for document embeddings"
    )
    
    chunks_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of text chunks embedded"
    )
    
    # ========================================================================
    # LANGUAGE & CONTENT ANALYSIS
    # ========================================================================
    detected_language: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        doc="Detected document language"
    )
    
    language_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Language detection confidence"
    )
    
    readability_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Readability score (Flesch-Kincaid or similar)"
    )
    
    complexity_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Legal complexity score (0-1)"
    )
    
    # ========================================================================
    # PROCESSING METADATA
    # ========================================================================
    model_versions: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Versions of models used as JSON {model_name: version}"
    )
    
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Total AI processing time"
    )
    
    gpu_used: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether GPU was used for processing"
    )
    
    # ========================================================================
    # CONFIDENCE & QUALITY METRICS
    # ========================================================================
    overall_confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Overall analysis confidence (0-1)"
    )
    
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Analysis quality score (0-1)"
    )
    
    needs_human_review: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether analysis needs human review (low confidence)"
    )
    
    # ========================================================================
    # CUSTOM FIELDS
    # ========================================================================
    custom_tags: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="User-defined custom tags"
    )
    
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="User notes about the analysis"
    )
    
    # ========================================================================
    # METADATA (TIMESTAMPS)
    # ========================================================================
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="Analysis completion timestamp"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Last update timestamp"
    )
    
    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="analysis",
        lazy="selectin"
    )
    
    # ========================================================================
    # INDEXES
    # ========================================================================
    __table_args__ = (
        Index('ix_analysis_document', 'document_id'),
        Index('ix_analysis_risk_level', 'risk_level'),
        Index('ix_analysis_has_unfair', 'has_unfair_clauses'),
        Index('ix_analysis_analyzed_at', 'analyzed_at'),
    )
    
    # ========================================================================
    # METHODS
    # ========================================================================
    def __repr__(self) -> str:
        """String representation"""
        return f"<Analysis(id={self.id}, document_id={self.document_id}, risk_level='{self.risk_level}')>"
    
    @property
    def is_high_risk(self) -> bool:
        """Check if document is high risk"""
        return self.risk_level in ("high", "critical")
    
    @property
    def is_low_confidence(self) -> bool:
        """Check if analysis has low confidence"""
        if self.overall_confidence is None:
            return False
        return self.overall_confidence < 0.7
    
    @property
    def has_entities(self) -> bool:
        """Check if any entities were detected"""
        return self.entities_detected is not None and self.entities_detected > 0
    
    @property
    def has_pii(self) -> bool:
        """Check if PII (persons) was detected"""
        return self.persons_detected is not None and self.persons_detected > 0
    
    def get_entity_by_type(self, entity_type: str) -> list:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Entity type (PERSON, ORG, LOC, etc.)
        
        Returns:
            list: List of entities matching the type
        """
        if not self.entities:
            return []
        
        return [
            entity for entity in self.entities
            if entity.get("label") == entity_type
        ]
    
    def get_unfair_clauses_by_severity(self, min_severity: str = "medium") -> list:
        """
        Get unfair clauses filtered by severity.
        
        Args:
            min_severity: Minimum severity (low, medium, high, critical)
        
        Returns:
            list: Filtered unfair clauses
        """
        if not self.unfair_clauses:
            return []
        
        severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_levels.get(min_severity, 0)
        
        return [
            clause for clause in self.unfair_clauses
            if severity_levels.get(clause.get("severity", "low"), 0) >= min_level
        ]
    
    def to_dict_summary(self) -> Dict[str, Any]:
        """
        Get summary dict for API responses.
        
        Returns:
            dict: Analysis summary
        """
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_class": self.document_class,
            "risk_level": self.risk_level,
            "has_unfair_clauses": self.has_unfair_clauses,
            "unfair_clauses_count": self.unfair_clauses_detected or 0,
            "entities_count": self.entities_detected or 0,
            "overall_confidence": self.overall_confidence,
            "needs_human_review": self.needs_human_review,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
        }