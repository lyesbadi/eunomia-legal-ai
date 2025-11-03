"""
EUNOMIA Legal AI Platform - Celery Analysis Tasks
Asynchronous tasks for complete document AI analysis pipeline
"""
from typing import Dict, Any, Optional, List
import asyncio
import logging
from datetime import datetime
from celery import Task
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.database import DatabaseManager
from app.models.document import Document, DocumentStatus
from app.models.analysis import Analysis
from app.models.audit_log import AuditLog, ActionType, ResourceType
from app.services import ai_service, llm_service, vector_service
from app.services.ai_service import AIService
from app.services.llm_service import LLMService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM TASK BASE CLASS
# ============================================================================
class CallbackTask(Task):
    """
    Custom Celery task with error handling and logging.
    """
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Error handler called when task fails.
        
        Args:
            exc: Exception raised
            task_id: Unique task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        logger.error(f"❌ Task {self.name}[{task_id}] failed: {exc}")
        logger.error(f"   Args: {args}")
        logger.error(f"   Exception info: {einfo}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """
        Success handler called when task completes.
        
        Args:
            retval: Return value of task
            task_id: Unique task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
        """
        logger.info(f"✅ Task {self.name}[{task_id}] completed successfully")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """
        Retry handler called when task is retried.
        
        Args:
            exc: Exception that caused retry
            task_id: Unique task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        logger.warning(f"🔄 Task {self.name}[{task_id}] retrying due to: {exc}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def run_async(coro):
    """
    Run async coroutine in sync context.
    
    Args:
        coro: Async coroutine to run
        
    Returns:
        Result of coroutine
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def get_db_session() -> AsyncSession:
    """
    Get async database session for tasks.
    
    Returns:
        AsyncSession instance
    """
    session_factory = DatabaseManager.get_session_factory()
    return session_factory()


async def update_document_status(
    session: AsyncSession,
    document_id: int,
    status: DocumentStatus,
    error_message: Optional[str] = None
) -> None:
    """
    Update document processing status.
    
    Args:
        session: Database session
        document_id: Document ID
        status: New status
        error_message: Error message if failed
    """
    document = await session.get(Document, document_id)
    if document:
        document.status = status
        if error_message:
            document.error_message = error_message
        if status == DocumentStatus.PROCESSING:
            document.processing_started_at = datetime.utcnow()
        elif status in [DocumentStatus.COMPLETED, DocumentStatus.FAILED]:
            document.processing_completed_at = datetime.utcnow()
        await session.commit()


async def create_audit_log(
    session: AsyncSession,
    user_id: int,
    action: ActionType,
    resource_type: ResourceType,
    resource_id: int,
    success: bool = True,
    error_message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create audit log entry.
    
    Args:
        session: Database session
        user_id: User ID
        action: Action type
        resource_type: Resource type
        resource_id: Resource ID
        success: Whether action succeeded
        error_message: Error message if failed
        details: Additional details
    """
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        success=success,
        error_message=error_message,
        details=details,
        description=f"Analysis task: {action.value}"
    )
    session.add(audit_log)
    await session.commit()


# ============================================================================
# MAIN ANALYSIS TASK
# ============================================================================
@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="tasks.analyze_document",
    max_retries=3,
    default_retry_delay=60
)
def analyze_document(self, document_id: int, user_id: int) -> Dict[str, Any]:
    """
    Complete document analysis pipeline.
    
    Pipeline steps:
    1. Document classification (Legal-BERT)
    2. Named Entity Recognition (CamemBERT-NER)
    3. Unfair clause detection (Unfair-ToS)
    4. Document summarization (BART)
    5. Risk assessment calculation
    6. LLM recommendations generation (EuroLLM-9B)
    7. Vector embeddings & indexing (Qdrant)
    8. Save results to database
    
    Args:
        document_id: Document ID to analyze
        user_id: User ID who requested analysis
        
    Returns:
        Analysis results summary
        
    Raises:
        Exception: If analysis fails
    """
    return run_async(_analyze_document_async(self, document_id, user_id))


async def _analyze_document_async(
    task: Task,
    document_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Async implementation of document analysis.
    
    Args:
        task: Celery task instance
        document_id: Document ID
        user_id: User ID
        
    Returns:
        Analysis results
    """
    session = await get_db_session()
    start_time = datetime.utcnow()
    
    try:
        logger.info("=" * 80)
        logger.info(f"🚀 Starting analysis for document {document_id}")
        logger.info("=" * 80)
        
        # ====================================================================
        # STEP 0: Load document and update status
        # ====================================================================
        document = await session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        if not document.extracted_text:
            raise ValueError(f"Document {document_id} has no extracted text")
        
        await update_document_status(session, document_id, DocumentStatus.PROCESSING)
        
        text = document.extracted_text
        logger.info(f"📄 Document loaded: {len(text)} characters")
        
        # ====================================================================
        # STEP 1: Document Classification (Legal-BERT)
        # ====================================================================
        logger.info("1️⃣ Running document classification...")
        classification_result = await ai_service.classify_document(text)
        logger.info(f"✅ Classification: {classification_result['primary_class']} ({classification_result['confidence']:.2%})")
        
        # ====================================================================
        # STEP 2: Named Entity Recognition (CamemBERT-NER)
        # ====================================================================
        logger.info("2️⃣ Running Named Entity Recognition...")
        ner_result = await ai_service.extract_entities(text)
        logger.info(f"✅ Entities found: {ner_result['entity_count']}")
        
        # ====================================================================
        # STEP 3: Unfair Clause Detection (Unfair-ToS)
        # ====================================================================
        logger.info("3️⃣ Detecting unfair clauses...")
        unfair_clauses_result = await ai_service.detect_unfair_clauses(text)
        logger.info(f"✅ Unfair clauses: {unfair_clauses_result['unfair_clause_count']}")
        
        # ====================================================================
        # STEP 4: Document Summarization (BART)
        # ====================================================================
        logger.info("4️⃣ Generating document summary...")
        summary_result = await ai_service.summarize_document(text)
        logger.info(f"✅ Summary generated: {len(summary_result['summary_text'])} characters")
        
        # ====================================================================
        # STEP 5: Risk Assessment Calculation
        # ====================================================================
        logger.info("5️⃣ Calculating risk assessment...")
        risk_assessment = await _calculate_risk_assessment(
            unfair_clauses_result,
            ner_result,
            classification_result
        )
        logger.info(f"✅ Risk level: {risk_assessment['risk_level']} (score: {risk_assessment['risk_score']:.2f})")
        
        # ====================================================================
        # STEP 6: LLM Recommendations (EuroLLM-9B)
        # ====================================================================
        logger.info("6️⃣ Generating LLM recommendations...")
        llm_recommendations = await llm_service.generate_recommendations(
            document_text=text,
            document_type=classification_result['primary_class'],
            classification=classification_result,
            unfair_clauses=unfair_clauses_result['unfair_clauses'],
            risk_assessment=risk_assessment
        )
        logger.info(f"✅ Recommendations generated: {len(llm_recommendations['action_items'])} action items")
        
        # ====================================================================
        # STEP 7: Language Detection
        # ====================================================================
        logger.info("7️⃣ Detecting document language...")
        detected_language, language_confidence = await ai_service.detect_language(text)
        logger.info(f"✅ Language: {detected_language} ({language_confidence:.2%})")
        
        # ====================================================================
        # STEP 8: Vector Embeddings & Indexing (Qdrant)
        # ====================================================================
        logger.info("8️⃣ Generating embeddings and indexing in Qdrant...")
        
        # Index full document
        qdrant_point_id = await vector_service.index_document(
            document_id=document_id,
            text=text,
            metadata={
                "user_id": user_id,
                "document_type": classification_result['primary_class'],
                "risk_level": risk_assessment['risk_level'],
                "language": detected_language
            }
        )
        
        # Index document chunks for precise search
        chunks = _split_text_into_chunks(text, chunk_size=500)
        chunk_point_ids = await vector_service.index_document_chunks(
            document_id=document_id,
            chunks=chunks,
            metadata={
                "user_id": user_id,
                "document_type": classification_result['primary_class']
            }
        )
        
        logger.info(f"✅ Indexed in Qdrant: 1 full doc + {len(chunk_point_ids)} chunks")
        
        # ====================================================================
        # STEP 9: Save Analysis Results to Database
        # ====================================================================
        logger.info("9️⃣ Saving analysis results to database...")
        
        analysis = Analysis(
            document_id=document_id,
            
            # Classification
            document_class=classification_result['primary_class'],
            classification_confidence=classification_result['confidence'],
            classification_details=classification_result,
            
            # NER
            entities_detected=ner_result,
            entity_count=ner_result['entity_count'],
            
            # Unfair Clauses
            unfair_clauses=unfair_clauses_result,
            has_unfair_clauses=unfair_clauses_result['unfair_clause_count'] > 0,
            unfair_clause_score=1.0 - unfair_clauses_result['fairness_score'],
            
            # Summary
            summary_text=summary_result['summary_text'],
            summary_length=summary_result['summary_length'],
            summary_confidence=summary_result['compression_ratio'],
            
            # Risk Assessment
            risk_level=risk_assessment['risk_level'],
            risk_score=risk_assessment['risk_score'],
            risk_factors=risk_assessment['risk_factors'],
            
            # LLM Recommendations
            recommendations_text=llm_recommendations['recommendation_text'],
            llm_generated=True,
            llm_model_version=llm_recommendations['model_used'],
            
            # Language
            detected_language=detected_language,
            language_confidence=language_confidence,
            
            # Embeddings
            embedding_generated=True,
            qdrant_point_id=qdrant_point_id,
            chunks_count=len(chunk_point_ids),
            
            # Processing metadata
            processing_started_at=start_time,
            completed_at=datetime.utcnow()
        )
        
        session.add(analysis)
        await session.commit()
        await session.refresh(analysis)
        
        logger.info(f"✅ Analysis saved with ID: {analysis.id}")
        
        # ====================================================================
        # STEP 10: Update document status to COMPLETED
        # ====================================================================
        await update_document_status(session, document_id, DocumentStatus.COMPLETED)
        
        # ====================================================================
        # STEP 11: Create audit log
        # ====================================================================
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        await create_audit_log(
            session=session,
            user_id=user_id,
            action=ActionType.ANALYSIS_COMPLETE,
            resource_type=ResourceType.ANALYSIS,
            resource_id=analysis.id,
            success=True,
            details={
                "document_id": document_id,
                "document_type": classification_result['primary_class'],
                "risk_level": risk_assessment['risk_level'],
                "unfair_clauses": unfair_clauses_result['unfair_clause_count'],
                "processing_time_seconds": processing_time,
                "models_used": [
                    "legal-bert",
                    "camembert-ner",
                    "unfair-tos",
                    "bart",
                    "sentence-transformers",
                    "eurollm-9b"
                ]
            }
        )
        
        logger.info("=" * 80)
        logger.info(f"✅ Analysis COMPLETED in {processing_time:.2f}s")
        logger.info("=" * 80)
        
        # Return summary
        return {
            "status": "completed",
            "analysis_id": analysis.id,
            "document_id": document_id,
            "document_type": classification_result['primary_class'],
            "risk_level": risk_assessment['risk_level'],
            "unfair_clauses_count": unfair_clauses_result['unfair_clause_count'],
            "entities_count": ner_result['entity_count'],
            "processing_time_seconds": processing_time
        }
    
    except Exception as e:
        logger.error(f"❌ Analysis failed for document {document_id}: {e}", exc_info=True)
        
        # Update document status to FAILED
        await update_document_status(
            session,
            document_id,
            DocumentStatus.FAILED,
            error_message=str(e)
        )
        
        # Create audit log for failure
        await create_audit_log(
            session=session,
            user_id=user_id,
            action=ActionType.ANALYSIS_FAILED,
            resource_type=ResourceType.DOCUMENT,
            resource_id=document_id,
            success=False,
            error_message=str(e)
        )
        
        # Retry task if not max retries
        if task.request.retries < task.max_retries:
            raise task.retry(exc=e, countdown=60)
        
        raise
    
    finally:
        await session.close()


# ============================================================================
# HELPER FUNCTIONS FOR ANALYSIS
# ============================================================================
async def _calculate_risk_assessment(
    unfair_clauses_result: Dict[str, Any],
    ner_result: Dict[str, Any],
    classification_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate overall risk assessment based on analysis results.
    
    Args:
        unfair_clauses_result: Unfair clause detection results
        ner_result: NER results
        classification_result: Classification results
        
    Returns:
        Risk assessment dictionary
    """
    risk_factors = []
    risk_score = 0.0
    
    # Factor 1: Unfair clauses
    unfair_count = unfair_clauses_result['unfair_clause_count']
    fairness_score = unfair_clauses_result['fairness_score']
    
    if unfair_count > 0:
        unfairness_weight = min(unfair_count / 5.0, 1.0) * 0.5  # Max 50% weight
        risk_score += unfairness_weight
        
        severity_distribution = {}
        for clause in unfair_clauses_result['unfair_clauses']:
            severity = clause['severity']
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        risk_factors.append({
            "factor": "unfair_clauses",
            "severity": "high" if unfair_count >= 3 else "medium",
            "score": unfairness_weight,
            "description": f"{unfair_count} clauses potentiellement abusives détectées",
            "details": severity_distribution
        })
    
    # Factor 2: Document type risk
    doc_type = classification_result['primary_class']
    type_risk_map = {
        'terms_of_service': 0.3,
        'privacy_policy': 0.2,
        'contract': 0.25,
        'legal_notice': 0.1,
        'other': 0.15
    }
    
    type_risk = type_risk_map.get(doc_type, 0.15)
    risk_score += type_risk
    
    if type_risk >= 0.2:
        risk_factors.append({
            "factor": "document_type",
            "severity": "medium",
            "score": type_risk,
            "description": f"Type de document '{doc_type}' nécessite attention particulière",
            "details": {"document_type": doc_type}
        })
    
    # Factor 3: Entity complexity
    entity_count = ner_result['entity_count']
    if entity_count > 20:
        complexity_risk = 0.15
        risk_score += complexity_risk
        
        risk_factors.append({
            "factor": "complexity",
            "severity": "low",
            "score": complexity_risk,
            "description": f"Document complexe avec {entity_count} entités identifiées",
            "details": ner_result['by_type']
        })
    
    # Normalize risk score (0-1)
    risk_score = min(risk_score, 1.0)
    
    # Determine risk level
    if risk_score >= 0.75:
        risk_level = "critical"
    elif risk_score >= 0.5:
        risk_level = "high"
    elif risk_score >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "total_risks": len(risk_factors),
        "recommendation": _get_risk_recommendation(risk_level),
        "requires_legal_review": risk_level in ["high", "critical"]
    }


def _get_risk_recommendation(risk_level: str) -> str:
    """
    Get recommendation based on risk level.
    
    Args:
        risk_level: Risk level (low, medium, high, critical)
        
    Returns:
        Recommendation text
    """
    recommendations = {
        "low": "Ce document présente peu de risques. Une lecture attentive suffit.",
        "medium": "Ce document nécessite une attention particulière sur certains points. Recommandé de consulter les recommandations détaillées.",
        "high": "Ce document présente des risques significatifs. Consultation d'un avocat fortement recommandée avant signature.",
        "critical": "⚠️ ATTENTION : Ce document présente des risques majeurs. Ne PAS signer sans revue approfondie par un avocat."
    }
    
    return recommendations.get(risk_level, recommendations["medium"])


def _split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for indexing.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks


# ============================================================================
# EMBEDDINGS-ONLY TASK (for re-indexing)
# ============================================================================
@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="tasks.generate_embeddings",
    max_retries=3
)
def generate_embeddings(self, document_id: int) -> Dict[str, Any]:
    """
    Generate and index embeddings for a document (without full analysis).
    
    Used for:
    - Re-indexing existing documents
    - Updating embeddings after model change
    
    Args:
        document_id: Document ID
        
    Returns:
        Indexing results
    """
    return run_async(_generate_embeddings_async(self, document_id))


async def _generate_embeddings_async(task: Task, document_id: int) -> Dict[str, Any]:
    """
    Async implementation of embeddings generation.
    
    Args:
        task: Celery task instance
        document_id: Document ID
        
    Returns:
        Indexing results
    """
    session = await get_db_session()
    
    try:
        logger.info(f"🔍 Generating embeddings for document {document_id}")
        
        # Load document
        document = await session.get(Document, document_id)
        if not document or not document.extracted_text:
            raise ValueError(f"Document {document_id} not found or has no text")
        
        # Generate embeddings and index
        qdrant_point_id = await vector_service.index_document(
            document_id=document_id,
            text=document.extracted_text,
            metadata={
                "user_id": document.user_id,
                "filename": document.filename
            }
        )
        
        # Index chunks
        chunks = _split_text_into_chunks(document.extracted_text)
        chunk_ids = await vector_service.index_document_chunks(
            document_id=document_id,
            chunks=chunks,
            metadata={"user_id": document.user_id}
        )
        
        logger.info(f"✅ Embeddings generated: 1 doc + {len(chunk_ids)} chunks")
        
        return {
            "status": "completed",
            "document_id": document_id,
            "qdrant_point_id": qdrant_point_id,
            "chunks_indexed": len(chunk_ids)
        }
    
    except Exception as e:
        logger.error(f"❌ Embeddings generation failed: {e}")
        if task.request.retries < task.max_retries:
            raise task.retry(exc=e)
        raise
    
    finally:
        await session.close()


# ============================================================================
# BATCH ANALYSIS TASK
# ============================================================================
@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="tasks.analyze_documents_batch"
)
def analyze_documents_batch(self, document_ids: List[int], user_id: int) -> Dict[str, Any]:
    """
    Analyze multiple documents in batch.
    
    Args:
        document_ids: List of document IDs
        user_id: User ID
        
    Returns:
        Batch analysis results
    """
    logger.info(f"📦 Batch analysis started: {len(document_ids)} documents")
    
    results = []
    for doc_id in document_ids:
        try:
            result = analyze_document.delay(doc_id, user_id)
            results.append({
                "document_id": doc_id,
                "task_id": result.id,
                "status": "queued"
            })
        except Exception as e:
            logger.error(f"❌ Failed to queue document {doc_id}: {e}")
            results.append({
                "document_id": doc_id,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "total_documents": len(document_ids),
        "queued": len([r for r in results if r['status'] == 'queued']),
        "failed": len([r for r in results if r['status'] == 'error']),
        "results": results
    }