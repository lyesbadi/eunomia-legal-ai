"""
EUNOMIA Legal AI Platform - Analysis Routes
FastAPI routes for AI analysis results retrieval and export
"""
from typing import Optional, List
from datetime import datetime
from io import BytesIO
from app.models.document import Document
import json
from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status,
    Query,
    Response
)
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc

from app.api.deps import (
    get_db,
    get_current_user,
    get_current_active_user,
    verify_document_ownership,
    get_audit_logger,
    get_pagination,
    AuditLogger,
    PaginationParams
)
from app.models.user import User
from app.models.document import Document
from app.models.analysis import Analysis
from app.models.audit_log import ActionType, ResourceType
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisListItem,
    AnalysisListResponse,
    ClassificationResult,
    NERResult,
    ClauseDetectionResult,
    SummaryResult,
    RiskAssessment,
    LLMRecommendation,
    QAResult,
    VectorEmbeddingInfo
)
import logging


# ============================================================================
# ROUTER SETUP
# ============================================================================
router = APIRouter(tags=["Analyses"])
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
async def build_analysis_response(
    analysis: Analysis,
    db: AsyncSession
) -> AnalysisResponse:
    """
    Build complete analysis response from database model.
    
    Args:
        analysis: Analysis database model
        db: Database session (to fetch related document)
        
    Returns:
        AnalysisResponse with all structured results
    """
    # Fetch related document for original_length
    doc_result = await db.execute(
        select(Document).where(Document.id == analysis.document_id)
    )
    document = doc_result.scalar_one_or_none()
    
    # Parse classification
    classification = None
    if analysis.classification_details:
        classification = ClassificationResult(**analysis.classification_details)
    
    # Parse NER
    ner = None
    if analysis.entities_detected:
        ner = NERResult(**analysis.entities_detected)
    
    # Parse clauses
    clauses = None
    if analysis.unfair_clauses:
        clauses = ClauseDetectionResult(**analysis.unfair_clauses)
    
    # Parse summary with real original_length
    summary = None
    if analysis.summary_text:
        # Get original text length from document
        original_length = 0
        if document:
            # If document has extracted text content
            if hasattr(document, 'content') and document.content:
                original_length = len(document.content)
            # Or estimate from file size (for binary files)
            elif document.file_size:
                # Rough estimate: 1 char ≈ 1 byte for text
                original_length = document.file_size
        
        # Extract key points from summary (split by sentences)
        key_points = []
        if analysis.summary_text:
            # Simple sentence splitting (can be improved with NLP)
            sentences = [s.strip() for s in analysis.summary_text.split('.') if s.strip()]
            # Take first 3 sentences as key points
            key_points = sentences[:3]
        
        # Calculate compression ratio
        summary_length = len(analysis.summary_text)
        compression_ratio = (
            (1 - (summary_length / original_length)) * 100
            if original_length > 0 else 0.0
        )
        
        summary = SummaryResult(
            summary_text=analysis.summary_text,
            summary_length=summary_length,
            original_length=original_length,  # ✅ Valeur réelle
            compression_ratio=round(compression_ratio, 2),
            confidence=analysis.summary_confidence or 0.85,
            key_points=key_points,  # ✅ Points clés extraits
            model_used="facebook/bart-large-cnn"
        )
    
    # Parse risk assessment
    risk = None
    if analysis.risk_level:
        risk = RiskAssessment(
            risk_level=analysis.risk_level,
            risk_score=analysis.risk_score or 0.0,
            risk_factors=analysis.risk_factors or [],
            total_risks=len(analysis.risk_factors) if analysis.risk_factors else 0,
            by_severity={},
            recommendation=analysis.recommendations_text or "No recommendations available",
            requires_legal_review=analysis.risk_level in ["high", "critical"]
        )
    
    # Parse LLM recommendations
    recommendations = None
    if analysis.recommendations_text and analysis.llm_generated:
        recommendations = LLMRecommendation(
            recommendation_text=analysis.recommendations_text,
            confidence=analysis.llm_confidence or 0.85,  # ✅ Utilise la vraie valeur si stockée
            action_items=[],
            estimated_priority="medium",
            model_used="eurollm-9b:latest"
        )
    
    # Parse Q&A
    qa = None
    if analysis.qa_pairs:
        qa = QAResult(**analysis.qa_pairs)
    
    # Parse embeddings info
    embeddings = None
    if analysis.embedding_generated:
        embeddings = VectorEmbeddingInfo(
            embedding_generated=True,
            qdrant_point_id=analysis.qdrant_point_id,
            chunks_count=analysis.chunks_count or 0,
            vector_dimension=384,
            model_used="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Calculate processing time
    processing_time = 0.0
    if analysis.processing_started_at and analysis.completed_at:
        delta = analysis.completed_at - analysis.processing_started_at
        processing_time = delta.total_seconds()
    
    # Build models used list
    models_used = []
    if classification:
        models_used.append("nlpaueb/legal-bert-base-uncased")
    if ner:
        models_used.append("Jean-Baptiste/camembert-ner")
    if clauses:
        models_used.append("claudiolemos/unfair-tos-bert")
    if summary:
        models_used.append("facebook/bart-large-cnn")
    if recommendations:
        models_used.append("eurollm-9b:latest")
    if embeddings:
        models_used.append("sentence-transformers/all-MiniLM-L6-v2")
    
    # Build complete response
    return AnalysisResponse(
        analysis_id=analysis.id,
        document_id=analysis.document_id,
        status=analysis.status,
        classification=classification,
        ner=ner,
        unfair_clauses=clauses,
        summary=summary,
        risk_assessment=risk,
        recommendations=recommendations,
        qa_results=qa,
        embeddings_info=embeddings,
        processing_started_at=analysis.processing_started_at,
        completed_at=analysis.completed_at,
        processing_time_seconds=processing_time,
        models_used=models_used,
        error_message=analysis.error_message
    )


async def generate_analysis_json(analysis: Analysis) -> dict:
    """
    Generate JSON export of analysis.
    
    Args:
        analysis: Analysis database model
        
    Returns:
        Dictionary with all analysis data
    """
    response = build_analysis_response(analysis)
    return response.model_dump(mode='json', exclude_none=True)


async def generate_analysis_pdf(analysis: Analysis, document: Document) -> bytes:
    """
    Generate PDF export of analysis (placeholder).
    
    Args:
        analysis: Analysis database model
        document: Document database model
        
    Returns:
        PDF bytes
        
    Note:
        This is a placeholder. In production, integrate with PDF generation library
        like ReportLab or WeasyPrint.
    """
    # TODO: Implement PDF generation
    # For now, return a simple text file
    
    content = f"""
EUNOMIA Legal AI Analysis Report
Generated: {datetime.utcnow().isoformat()}

Document: {document.filename}
Document Type: {document.document_type.value if document.document_type else 'Unknown'}
Analyzed: {analysis.completed_at.isoformat() if analysis.completed_at else 'N/A'}

=== CLASSIFICATION ===
Class: {analysis.document_class or 'N/A'}
Confidence: {analysis.classification_confidence or 0:.2%}

=== RISK ASSESSMENT ===
Risk Level: {analysis.risk_level or 'N/A'}
Risk Score: {analysis.risk_score or 0:.2f}

=== SUMMARY ===
{analysis.summary_text or 'No summary available'}

=== RECOMMENDATIONS ===
{analysis.recommendations_text or 'No recommendations available'}

---
This is a placeholder PDF. Implement proper PDF generation in production.
Generated by EUNOMIA Legal AI Platform - eunomia.legal
    """.strip()
    
    return content.encode('utf-8')


# ============================================================================
# ANALYSIS LISTING
# ============================================================================
@router.get(
    "",
    response_model=AnalysisListResponse,
    summary="List user analyses",
    description="Get paginated list of user analyses with filters"
)
async def list_analyses(
    pagination: PaginationParams = Depends(get_pagination),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    date_from: Optional[datetime] = Query(None, description="Filter by completion date (from)"),
    date_to: Optional[datetime] = Query(None, description="Filter by completion date (to)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> AnalysisListResponse:
    """
    List user analyses.
    
    Supports:
    - Pagination (page, page_size)
    - Filtering (document_type, risk_level, date_from, date_to)
    
    Returns paginated list of analyses with minimal information.
    """
    # Build query - join with Document to filter by user
    query = select(Analysis).join(
        Document, Document.id == Analysis.document_id
    ).where(
        Document.user_id == current_user.id
    )
    
    # Apply filters
    filter_conditions = []
    
    if document_type:
        filter_conditions.append(Analysis.document_class == document_type)
    
    if risk_level:
        filter_conditions.append(Analysis.risk_level == risk_level)
    
    if date_from:
        filter_conditions.append(Analysis.completed_at >= date_from)
    
    if date_to:
        filter_conditions.append(Analysis.completed_at <= date_to)
    
    if filter_conditions:
        query = query.where(and_(*filter_conditions))
    
    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.offset(pagination.offset).limit(pagination.limit)
    query = query.order_by(desc(Analysis.completed_at))
    
    # Execute query
    result = await db.execute(query)
    analyses = result.scalars().all()
    
    # Convert to list items
    analysis_items = []
    for analysis in analyses:
        # Get document info
        doc_result = await db.execute(
            select(Document).where(Document.id == analysis.document_id)
        )
        document = doc_result.scalar_one_or_none()
        
        analysis_items.append(
            AnalysisListItem(
                id=analysis.id,
                document_id=analysis.document_id,
                document_filename=document.filename if document else "Unknown",
                primary_classification=analysis.document_class,
                risk_level=analysis.risk_level,
                unfair_clauses_count=analysis.clause_count or 0,
                completed_at=analysis.completed_at
            )
        )
    
    return AnalysisListResponse(
        analyses=analysis_items,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size
    )


# ============================================================================
# ANALYSIS RETRIEVAL
# ============================================================================
@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get analysis details",
    description="Get complete AI analysis results"
)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> AnalysisResponse:
    """
    Get complete analysis results.
    
    Returns:
    - Classification results (Legal-BERT)
    - Named entities (CamemBERT-NER)
    - Unfair clauses (Unfair-ToS)
    - Document summary (BART)
    - Risk assessment
    - LLM recommendations (Mistral 7B)
    - Q&A pairs (RoBERTa)
    - Vector embeddings info
    """
    # Get analysis with document
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found"
        )
    
    # Verify ownership via document
    await verify_document_ownership(analysis.document_id, current_user, db)
    
    # Log view
    await audit.log(
        user_id=current_user.id,
        action=ActionType.ANALYSIS_VIEW,
        resource_type=ResourceType.ANALYSIS,
        resource_id=analysis.id
    )
    
    # Build and return response
    return await build_analysis_response(analysis, db)


@router.get(
    "/document/{document_id}",
    response_model=AnalysisResponse,
    summary="Get analysis by document ID",
    description="Get AI analysis results for a specific document"
)
async def get_analysis_by_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> AnalysisResponse:
    """
    Get analysis by document ID.
    
    Convenience endpoint to get analysis for a document without knowing analysis ID.
    """
    # Verify ownership
    await verify_document_ownership(document_id, current_user, db)
    
    # Get analysis for document
    result = await db.execute(
        select(Analysis).where(Analysis.document_id == document_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found for document {document_id}"
        )
    
    # Log view
    await audit.log(
        user_id=current_user.id,
        action=ActionType.ANALYSIS_VIEW,
        resource_type=ResourceType.ANALYSIS,
        resource_id=analysis.id
    )
    
    # Build and return response
    return await build_analysis_response(analysis, db)


# ============================================================================
# ANALYSIS EXPORT
# ============================================================================
@router.get(
    "/{analysis_id}/export/json",
    summary="Export analysis as JSON",
    description="Download complete analysis results as JSON file"
)
async def export_analysis_json(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> Response:
    """
    Export analysis as JSON.
    
    Downloads complete analysis results in structured JSON format.
    Useful for:
    - Integration with other systems
    - Backup/archiving
    - Further processing
    """
    # Get analysis
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found"
        )
    
    # Verify ownership
    await verify_document_ownership(analysis.document_id, current_user, db)
    
    # Generate JSON export
    analysis_data = await generate_analysis_json(analysis)
    
    # Get document for filename
    doc_result = await db.execute(
        select(Document).where(Document.id == analysis.document_id)
    )
    document = doc_result.scalar_one()
    
    # Generate filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{analysis.id}_{timestamp}.json"
    
    # Log export
    await audit.log(
        user_id=current_user.id,
        action=ActionType.DATA_EXPORT,
        resource_type=ResourceType.ANALYSIS,
        resource_id=analysis.id,
        details={"format": "json"}
    )
    
    logger.info(f"Analysis {analysis.id} exported as JSON by user {current_user.id}")
    
    # Return JSON file
    json_str = json.dumps(analysis_data, indent=2, ensure_ascii=False)
    
    return Response(
        content=json_str,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get(
    "/{analysis_id}/export/pdf",
    summary="Export analysis as PDF",
    description="Download analysis report as PDF file"
)
async def export_analysis_pdf(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit_logger)
) -> StreamingResponse:
    """
    Export analysis as PDF.
    
    Downloads formatted analysis report as PDF document.
    Useful for:
    - Sharing with clients
    - Legal documentation
    - Professional reports
    
    Note: This is a placeholder implementation.
    In production, integrate with PDF generation library.
    """
    # Get analysis
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found"
        )
    
    # Verify ownership
    await verify_document_ownership(analysis.document_id, current_user, db)
    
    # Get document
    doc_result = await db.execute(
        select(Document).where(Document.id == analysis.document_id)
    )
    document = doc_result.scalar_one()
    
    # Generate PDF
    pdf_bytes = await generate_analysis_pdf(analysis, document)
    
    # Generate filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_report_{analysis.id}_{timestamp}.pdf"
    
    # Log export
    await audit.log(
        user_id=current_user.id,
        action=ActionType.DATA_EXPORT,
        resource_type=ResourceType.ANALYSIS,
        resource_id=analysis.id,
        details={"format": "pdf"}
    )
    
    logger.info(f"Analysis {analysis.id} exported as PDF by user {current_user.id}")
    
    # Return PDF file
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# ============================================================================
# ANALYSIS STATISTICS
# ============================================================================
@router.get(
    "/statistics/summary",
    summary="Get analysis statistics",
    description="Get user analysis statistics and metrics"
)
async def get_analysis_statistics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get analysis statistics for current user.
    
    Returns:
    - Total analyses count
    - Count by document type
    - Count by risk level
    - Average confidence scores
    - Common detected entities
    - Average processing time
    """
    # Total analyses
    total_result = await db.execute(
        select(func.count(Analysis.id))
        .join(Document, Document.id == Analysis.document_id)
        .where(Document.user_id == current_user.id)
    )
    total_analyses = total_result.scalar() or 0
    
    # By document type
    by_type = {}
    type_result = await db.execute(
        select(
            Analysis.document_class,
            func.count(Analysis.id)
        )
        .join(Document, Document.id == Analysis.document_id)
        .where(Document.user_id == current_user.id)
        .group_by(Analysis.document_class)
    )
    for doc_type, count in type_result:
        if doc_type:
            by_type[doc_type] = count
    
    # By risk level
    by_risk = {}
    risk_result = await db.execute(
        select(
            Analysis.risk_level,
            func.count(Analysis.id)
        )
        .join(Document, Document.id == Analysis.document_id)
        .where(Document.user_id == current_user.id)
        .group_by(Analysis.risk_level)
    )
    for risk_level, count in risk_result:
        if risk_level:
            by_risk[risk_level] = count
    
    # Average classification confidence
    avg_confidence_result = await db.execute(
        select(func.avg(Analysis.classification_confidence))
        .join(Document, Document.id == Analysis.document_id)
        .where(Document.user_id == current_user.id)
    )
    avg_confidence = avg_confidence_result.scalar() or 0.0
    
    # Average processing time
    avg_time_result = await db.execute(
        select(
            func.avg(
                func.extract('epoch', Analysis.completed_at - Analysis.processing_started_at)
            )
        )
        .join(Document, Document.id == Analysis.document_id)
        .where(
            and_(
                Document.user_id == current_user.id,
                Analysis.processing_started_at.isnot(None),
                Analysis.completed_at.isnot(None)
            )
        )
    )
    avg_processing_time = avg_time_result.scalar() or 0.0
    
    # Total unfair clauses detected
    total_clauses_result = await db.execute(
        select(func.sum(Analysis.clause_count))
        .join(Document, Document.id == Analysis.document_id)
        .where(Document.user_id == current_user.id)
    )
    total_clauses = total_clauses_result.scalar() or 0
    
    return {
        "total_analyses": total_analyses,
        "by_document_type": by_type,
        "by_risk_level": by_risk,
        "average_confidence": round(avg_confidence, 3) if avg_confidence else 0.0,
        "average_processing_time_seconds": round(avg_processing_time, 2) if avg_processing_time else 0.0,
        "total_unfair_clauses_detected": total_clauses,
        "analyses_with_high_risk": by_risk.get("high", 0) + by_risk.get("critical", 0)
    }


# ============================================================================
# ANALYSIS COMPARISON (BONUS)
# ============================================================================
@router.get(
    "/compare",
    summary="Compare multiple analyses",
    description="Compare analysis results for multiple documents"
)
async def compare_analyses(
    analysis_ids: str = Query(..., description="Comma-separated analysis IDs (max 5)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Compare multiple analyses side-by-side.
    
    - **analysis_ids**: Comma-separated list of analysis IDs (max 5)
    
    Returns comparison of:
    - Classifications
    - Risk levels
    - Unfair clauses counts
    - Processing times
    
    Useful for comparing different versions of a document or similar documents.
    """
    # Parse analysis IDs
    try:
        ids = [int(id.strip()) for id in analysis_ids.split(",")]
        if len(ids) > 5:
            raise ValueError("Maximum 5 analyses can be compared")
        if len(ids) < 2:
            raise ValueError("At least 2 analyses required for comparison")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Get all analyses
    result = await db.execute(
        select(Analysis).where(Analysis.id.in_(ids))
    )
    analyses = result.scalars().all()
    
    if len(analyses) != len(ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more analyses not found"
        )
    
    # Verify ownership for all
    for analysis in analyses:
        await verify_document_ownership(analysis.document_id, current_user, db)
    
    # Build comparison
    comparison = {
        "analyses_count": len(analyses),
        "analyses": []
    }
    
    for analysis in analyses:
        # Get document
        doc_result = await db.execute(
            select(Document).where(Document.id == analysis.document_id)
        )
        document = doc_result.scalar_one()
        
        comparison["analyses"].append({
            "analysis_id": analysis.id,
            "document_id": analysis.document_id,
            "document_filename": document.filename,
            "classification": analysis.document_class,
            "confidence": analysis.classification_confidence,
            "risk_level": analysis.risk_level,
            "risk_score": analysis.risk_score,
            "unfair_clauses_count": analysis.clause_count or 0,
            "entities_count": len(analysis.entities_detected) if analysis.entities_detected else 0,
            "completed_at": analysis.completed_at
        })
    
    # Add summary statistics
    comparison["summary"] = {
        "most_common_classification": max(
            set(a.document_class for a in analyses if a.document_class),
            key=lambda x: sum(1 for a in analyses if a.document_class == x),
            default=None
        ),
        "highest_risk": max((a.risk_level for a in analyses if a.risk_level), default=None),
        "total_unfair_clauses": sum(a.clause_count or 0 for a in analyses),
        "average_confidence": sum(a.classification_confidence or 0 for a in analyses) / len(analyses)
    }
    
    return comparison