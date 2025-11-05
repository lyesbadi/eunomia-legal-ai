"""
EUNOMIA Legal AI Platform - Analysis Schemas
Pydantic schemas for AI analysis results and structured data
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


# ============================================================================
# CLASSIFICATION SCHEMAS
# ============================================================================
class ClassificationLabel(BaseModel):
    """
    Single classification label with confidence score.
    """
    label: str = Field(..., description="Classification label")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "contract",
                "score": 0.97
            }
        }
    )


class ClassificationResult(BaseModel):
    """
    Document classification result from Legal-BERT.
    """
    primary_class: str = Field(..., description="Primary classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Primary confidence score")
    top_predictions: List[ClassificationLabel] = Field(
        ...,
        max_length=5,
        description="Top 3-5 predictions with scores"
    )
    model_used: str = Field(
        default="nlpaueb/legal-bert-base-uncased",
        description="Model identifier"
    )
    
    @field_validator('top_predictions')
    @classmethod
    def validate_predictions(cls, v: List[ClassificationLabel]) -> List[ClassificationLabel]:
        """
        Validate predictions are sorted by score descending.
        
        Args:
            v: List of predictions
            
        Returns:
            Validated and sorted predictions
        """
        if not v:
            raise ValueError("At least one prediction required")
        
        # Sort by score descending
        sorted_predictions = sorted(v, key=lambda x: x.score, reverse=True)
        return sorted_predictions
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "primary_class": "contract",
                "confidence": 0.97,
                "top_predictions": [
                    {"label": "contract", "score": 0.97},
                    {"label": "tos", "score": 0.02},
                    {"label": "privacy_policy", "score": 0.01}
                ],
                "model_used": "nlpaueb/legal-bert-base-uncased"
            }
        }
    )


# ============================================================================
# NAMED ENTITY RECOGNITION SCHEMAS
# ============================================================================
class EntityType(str, Enum):
    """
    Common legal entity types.
    """
    PERSON = "PER"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    MISC = "MISC"


class NamedEntity(BaseModel):
    """
    Single named entity extracted from document.
    """
    text: str = Field(..., description="Entity text")
    entity_type: str = Field(..., description="Entity type (PER, ORG, LOC, etc.)")
    start_pos: int = Field(..., ge=0, description="Start position in text")
    end_pos: int = Field(..., ge=0, description="End position in text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    @field_validator('end_pos')
    @classmethod
    def validate_positions(cls, v: int, info) -> int:
        """Validate end_pos > start_pos"""
        if 'start_pos' in info.data and v <= info.data['start_pos']:
            raise ValueError("end_pos must be greater than start_pos")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Paris",
                "entity_type": "LOC",
                "start_pos": 245,
                "end_pos": 250,
                "confidence": 0.95
            }
        }
    )


class NERResult(BaseModel):
    """
    Named Entity Recognition results from CamemBERT-NER.
    """
    entities: List[NamedEntity] = Field(
        ...,
        description="List of extracted entities"
    )
    total_count: int = Field(..., ge=0, description="Total number of entities")
    by_type: Dict[str, int] = Field(
        ...,
        description="Count of entities by type"
    )
    average_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence score"
    )
    model_used: str = Field(
        default="Jean-Baptiste/camembert-ner",
        description="Model identifier"
    )
    
    @field_validator('total_count')
    @classmethod
    def validate_count(cls, v: int, info) -> int:
        """Validate total_count matches entities length"""
        if 'entities' in info.data and v != len(info.data['entities']):
            raise ValueError("total_count must match entities list length")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entities": [
                    {
                        "text": "SARL ABC",
                        "entity_type": "ORG",
                        "start_pos": 120,
                        "end_pos": 128,
                        "confidence": 0.98
                    },
                    {
                        "text": "Paris",
                        "entity_type": "LOC",
                        "start_pos": 245,
                        "end_pos": 250,
                        "confidence": 0.95
                    }
                ],
                "total_count": 2,
                "by_type": {
                    "ORG": 1,
                    "LOC": 1
                },
                "average_confidence": 0.965,
                "model_used": "Jean-Baptiste/camembert-ner"
            }
        }
    )


# ============================================================================
# UNFAIR CLAUSE DETECTION SCHEMAS
# ============================================================================
class ClauseSeverity(str, Enum):
    """
    Severity levels for unfair clauses.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UnfairClause(BaseModel):
    """
    Single unfair or abusive clause detected.
    """
    clause_text: str = Field(..., description="Text of the unfair clause")
    clause_type: str = Field(..., description="Type of unfairness")
    severity: ClauseSeverity = Field(..., description="Severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    explanation: str = Field(..., description="Why this clause is unfair")
    recommendation: str = Field(..., description="Recommended action")
    start_pos: int = Field(..., ge=0, description="Start position in document")
    end_pos: int = Field(..., ge=0, description="End position in document")
    
    @field_validator('clause_text')
    @classmethod
    def validate_clause_length(cls, v: str) -> str:
        """Validate clause is not empty and not too long"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Clause text cannot be empty")
        if len(v) > 5000:
            raise ValueError("Clause text too long (max 5000 chars)")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "clause_text": "Le prestataire se réserve le droit de modifier les conditions sans préavis.",
                "clause_type": "unilateral_modification",
                "severity": "high",
                "confidence": 0.92,
                "explanation": "Cette clause permet au prestataire de modifier unilatéralement le contrat sans préavis, ce qui est abusif selon le droit de la consommation.",
                "recommendation": "Exiger un délai de préavis minimum de 30 jours et un droit de résiliation.",
                "start_pos": 1250,
                "end_pos": 1330
            }
        }
    )


class ClauseDetectionResult(BaseModel):
    """
    Unfair clause detection results from Unfair-ToS model.
    """
    unfair_clauses: List[UnfairClause] = Field(
        ...,
        description="List of detected unfair clauses"
    )
    total_count: int = Field(..., ge=0, description="Total number of unfair clauses")
    by_severity: Dict[str, int] = Field(
        ...,
        description="Count by severity level"
    )
    by_type: Dict[str, int] = Field(
        ...,
        description="Count by clause type"
    )
    overall_fairness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall document fairness score (0=unfair, 1=fair)"
    )
    model_used: str = Field(
        default="coastalcph/unfair-tos",
        description="Model identifier"
    )
    
    @field_validator('total_count')
    @classmethod
    def validate_count(cls, v: int, info) -> int:
        """Validate total_count matches clauses length"""
        if 'unfair_clauses' in info.data and v != len(info.data['unfair_clauses']):
            raise ValueError("total_count must match unfair_clauses list length")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "unfair_clauses": [
                    {
                        "clause_text": "Le prestataire se réserve le droit...",
                        "clause_type": "unilateral_modification",
                        "severity": "high",
                        "confidence": 0.92,
                        "explanation": "Modification unilatérale sans préavis",
                        "recommendation": "Exiger un délai de préavis",
                        "start_pos": 1250,
                        "end_pos": 1330
                    }
                ],
                "total_count": 1,
                "by_severity": {
                    "high": 1
                },
                "by_type": {
                    "unilateral_modification": 1
                },
                "overall_fairness_score": 0.65,
                "model_used": "coastalcph/unfair-tos"
            }
        }
    )


# ============================================================================
# SUMMARIZATION SCHEMAS
# ============================================================================
class SummaryResult(BaseModel):
    """
    Document summarization result from BART.
    """
    summary_text: str = Field(..., description="Generated summary")
    summary_length: int = Field(..., ge=0, description="Summary length in characters")
    original_length: int = Field(..., ge=0, description="Original document length")
    compression_ratio: float = Field(..., ge=0.0, description="Compression ratio")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Summary quality score")
    key_points: List[str] = Field(
        ...,
        max_length=10,
        description="Extracted key points (bullet points)"
    )
    model_used: str = Field(
        default="facebook/bart-large-cnn",
        description="Model identifier"
    )
    
    @field_validator('summary_text')
    @classmethod
    def validate_summary(cls, v: str) -> str:
        """Validate summary is not empty"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Summary cannot be empty")
        return v
    
    @field_validator('compression_ratio')
    @classmethod
    def calculate_compression(cls, v: float, info) -> float:
        """Calculate and validate compression ratio"""
        if 'summary_length' in info.data and 'original_length' in info.data:
            original = info.data['original_length']
            summary = info.data['summary_length']
            if original > 0:
                calculated = summary / original
                return round(calculated, 2)
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary_text": "Contrat de location commerciale pour un local de 120m² situé à Paris. Durée 3 ans renouvelable. Loyer mensuel de 2500€ HT avec révision annuelle selon l'indice ICC. Dépôt de garantie de 7500€.",
                "summary_length": 185,
                "original_length": 12450,
                "compression_ratio": 0.01,
                "confidence": 0.88,
                "key_points": [
                    "Location commerciale 120m²",
                    "Durée: 3 ans renouvelable",
                    "Loyer: 2500€ HT/mois",
                    "Révision annuelle ICC",
                    "Dépôt de garantie: 7500€"
                ],
                "model_used": "facebook/bart-large-cnn"
            }
        }
    )


# ============================================================================
# RISK ASSESSMENT SCHEMAS
# ============================================================================
class RiskLevel(str, Enum):
    """
    Document risk assessment levels.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFactor(BaseModel):
    """
    Individual risk factor identified in document.
    """
    category: str = Field(..., description="Risk category")
    description: str = Field(..., description="Risk description")
    severity: RiskLevel = Field(..., description="Risk severity")
    impact: str = Field(..., description="Potential impact")
    mitigation: str = Field(..., description="Recommended mitigation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "Clause abusive",
                "description": "Modification unilatérale sans préavis",
                "severity": "high",
                "impact": "Le contrat peut être modifié sans votre consentement",
                "mitigation": "Négocier un délai de préavis de 30 jours minimum"
            }
        }
    )


class RiskAssessment(BaseModel):
    """
    Overall document risk assessment.
    """
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numerical risk score (0=safe, 1=dangerous)"
    )
    risk_factors: List[RiskFactor] = Field(
        ...,
        description="Identified risk factors"
    )
    total_risks: int = Field(..., ge=0, description="Total number of risks")
    by_severity: Dict[str, int] = Field(
        ...,
        description="Count of risks by severity"
    )
    recommendation: str = Field(
        ...,
        description="Overall recommendation"
    )
    requires_legal_review: bool = Field(
        ...,
        description="Whether professional legal review is recommended"
    )
    
    @field_validator('total_risks')
    @classmethod
    def validate_count(cls, v: int, info) -> int:
        """Validate total_risks matches factors length"""
        if 'risk_factors' in info.data and v != len(info.data['risk_factors']):
            raise ValueError("total_risks must match risk_factors list length")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "risk_level": "medium",
                "risk_score": 0.62,
                "risk_factors": [
                    {
                        "category": "Clause abusive",
                        "description": "Modification unilatérale",
                        "severity": "high",
                        "impact": "Modification sans consentement",
                        "mitigation": "Négocier préavis 30 jours"
                    }
                ],
                "total_risks": 1,
                "by_severity": {
                    "high": 1
                },
                "recommendation": "Ce document présente des risques modérés. Nous recommandons de négocier certaines clauses avant signature.",
                "requires_legal_review": True
            }
        }
    )


# ============================================================================
# LLM RECOMMENDATIONS SCHEMAS
# ============================================================================
class LLMRecommendation(BaseModel):
    """
    AI-generated recommendation from Ollama Mistral.
    """
    recommendation_text: str = Field(
        ...,
        description="Generated recommendation in plain language"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Recommendation confidence"
    )
    action_items: List[str] = Field(
        ...,
        max_length=10,
        description="Concrete action items"
    )
    estimated_priority: str = Field(
        ...,
        description="Priority level (low, medium, high)"
    )
    model_used: str = Field(
        default="eurollm-9b:latest",
        description="LLM model identifier"
    )
    
    @field_validator('recommendation_text')
    @classmethod
    def validate_recommendation(cls, v: str) -> str:
        """Validate recommendation is substantive"""
        if not v or len(v.strip()) < 50:
            raise ValueError("Recommendation must be at least 50 characters")
        if len(v) > 5000:
            raise ValueError("Recommendation too long (max 5000 chars)")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendation_text": "Ce contrat de location commerciale présente plusieurs points nécessitant votre attention. La clause de révision du loyer basée sur l'ICC est standard, mais assurez-vous qu'elle inclut une clause de plafonnement. Le délai de préavis de 6 mois est conforme à la loi Pinel.",
                "confidence": 0.85,
                "action_items": [
                    "Vérifier la clause de plafonnement du loyer",
                    "Confirmer le délai de préavis",
                    "Négocier les charges récupérables"
                ],
                "estimated_priority": "medium",
                "model_used": "eurollm-9b:latest"
            }
        }
    )


# ============================================================================
# QUESTION-ANSWERING SCHEMAS
# ============================================================================
class QAPair(BaseModel):
    """
    Question-Answer pair generated from document.
    """
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Answer extracted from document")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Answer confidence score"
    )
    start_pos: int = Field(..., ge=0, description="Answer start position in text")
    end_pos: int = Field(..., ge=0, description="Answer end position in text")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Quel est le montant du loyer mensuel ?",
                "answer": "2500 euros HT",
                "confidence": 0.96,
                "start_pos": 1850,
                "end_pos": 1863
            }
        }
    )


class QAResult(BaseModel):
    """
    Question-Answering results from RoBERTa.
    """
    qa_pairs: List[QAPair] = Field(
        ...,
        description="Pre-generated Q&A pairs"
    )
    total_pairs: int = Field(..., ge=0, description="Total number of Q&A pairs")
    average_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence score"
    )
    model_used: str = Field(
        default="deepset/roberta-base-squad2",
        description="Model identifier"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "qa_pairs": [
                    {
                        "question": "Quel est le montant du loyer ?",
                        "answer": "2500 euros HT",
                        "confidence": 0.96,
                        "start_pos": 1850,
                        "end_pos": 1863
                    },
                    {
                        "question": "Quelle est la durée du bail ?",
                        "answer": "3 ans renouvelable",
                        "confidence": 0.94,
                        "start_pos": 920,
                        "end_pos": 938
                    }
                ],
                "total_pairs": 2,
                "average_confidence": 0.95,
                "model_used": "deepset/roberta-base-squad2"
            }
        }
    )


# ============================================================================
# VECTOR EMBEDDINGS SCHEMAS
# ============================================================================
class VectorEmbeddingInfo(BaseModel):
    """
    Vector embedding generation information.
    """
    embedding_generated: bool = Field(..., description="Whether embeddings were generated")
    qdrant_point_id: Optional[str] = Field(None, description="Qdrant vector point ID")
    chunks_count: int = Field(..., ge=0, description="Number of text chunks embedded")
    vector_dimension: int = Field(..., description="Embedding vector dimension")
    model_used: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model identifier"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embedding_generated": True,
                "qdrant_point_id": "doc_42_chunks",
                "chunks_count": 15,
                "vector_dimension": 384,
                "model_used": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    )


# ============================================================================
# COMPLETE ANALYSIS RESPONSE SCHEMA
# ============================================================================
class AnalysisResponse(BaseModel):
    """
    Complete AI analysis response with all results.
    
    This is the main response schema returned after document processing.
    """
    id: int = Field(..., description="Analysis ID")
    document_id: int = Field(..., description="Associated document ID")
    
    # Classification
    classification: Optional[ClassificationResult] = Field(
        None,
        description="Document classification results"
    )
    
    # Named Entity Recognition
    ner: Optional[NERResult] = Field(
        None,
        description="Named entity recognition results"
    )
    
    # Unfair Clause Detection
    clauses: Optional[ClauseDetectionResult] = Field(
        None,
        description="Unfair clause detection results"
    )
    
    # Summarization
    summary: Optional[SummaryResult] = Field(
        None,
        description="Document summarization results"
    )
    
    # Risk Assessment
    risk: Optional[RiskAssessment] = Field(
        None,
        description="Risk assessment results"
    )
    
    # LLM Recommendations
    recommendations: Optional[LLMRecommendation] = Field(
        None,
        description="AI-generated recommendations"
    )
    
    # Question-Answering
    qa: Optional[QAResult] = Field(
        None,
        description="Question-answering pairs"
    )
    
    # Vector Embeddings
    embeddings: Optional[VectorEmbeddingInfo] = Field(
        None,
        description="Vector embedding information"
    )
    
    # Language & Content
    detected_language: Optional[str] = Field(None, description="Detected language")
    language_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    readability_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    complexity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Processing Metadata
    processing_time_seconds: float = Field(..., ge=0.0, description="Total processing time")
    models_used: List[str] = Field(..., description="List of models used in analysis")
    completed_at: datetime = Field(..., description="Analysis completion timestamp")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 100,
                "document_id": 42,
                "classification": {
                    "primary_class": "contract",
                    "confidence": 0.97,
                    "top_predictions": [
                        {"label": "contract", "score": 0.97}
                    ],
                    "model_used": "nlpaueb/legal-bert-base-uncased"
                },
                "ner": {
                    "entities": [],
                    "total_count": 0,
                    "by_type": {},
                    "average_confidence": 0.0,
                    "model_used": "Jean-Baptiste/camembert-ner"
                },
                "clauses": {
                    "unfair_clauses": [],
                    "total_count": 0,
                    "by_severity": {},
                    "by_type": {},
                    "overall_fairness_score": 0.85,
                    "model_used": "coastalcph/unfair-tos"
                },
                "summary": {
                    "summary_text": "Contrat de location commerciale...",
                    "summary_length": 185,
                    "original_length": 12450,
                    "compression_ratio": 0.01,
                    "confidence": 0.88,
                    "key_points": [],
                    "model_used": "facebook/bart-large-cnn"
                },
                "risk": {
                    "risk_level": "low",
                    "risk_score": 0.25,
                    "risk_factors": [],
                    "total_risks": 0,
                    "by_severity": {},
                    "recommendation": "Document appears safe",
                    "requires_legal_review": False
                },
                "detected_language": "fr",
                "language_confidence": 0.99,
                "readability_score": 0.65,
                "complexity_score": 0.72,
                "processing_time_seconds": 125.5,
                "models_used": [
                    "nlpaueb/legal-bert-base-uncased",
                    "Jean-Baptiste/camembert-ner",
                    "coastalcph/unfair-tos"
                ],
                "completed_at": "2025-11-01T10:32:15Z"
            }
        }
    )


# ============================================================================
# ANALYSIS LIST SCHEMAS
# ============================================================================
class AnalysisListItem(BaseModel):
    """
    Minimal analysis info for list views.
    """
    id: int = Field(..., description="Analysis ID")
    document_id: int = Field(..., description="Associated document ID")
    document_filename: str = Field(..., description="Document filename")
    primary_classification: Optional[str] = Field(None, description="Primary class")
    risk_level: Optional[str] = Field(None, description="Risk level")
    unfair_clauses_count: int = Field(default=0, description="Number of unfair clauses")
    completed_at: datetime = Field(..., description="Completion timestamp")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 100,
                "document_id": 42,
                "document_filename": "contrat_location.pdf",
                "primary_classification": "contract",
                "risk_level": "medium",
                "unfair_clauses_count": 2,
                "completed_at": "2025-11-01T10:32:15Z"
            }
        }
    )


class AnalysisListResponse(BaseModel):
    """
    Paginated analysis list response.
    """
    analyses: List[AnalysisListItem] = Field(..., description="List of analyses")
    total: int = Field(..., description="Total number of analyses")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analyses": [
                    {
                        "id": 100,
                        "document_id": 42,
                        "document_filename": "contrat_location.pdf",
                        "primary_classification": "contract",
                        "risk_level": "medium",
                        "unfair_clauses_count": 2,
                        "completed_at": "2025-11-01T10:32:15Z"
                    }
                ],
                "total": 15,
                "page": 1,
                "page_size": 20,
                "total_pages": 1
            }
        }
    )