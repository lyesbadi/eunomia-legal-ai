"""
EUNOMIA Legal AI Platform - Celery Tasks
Async tasks for document analysis pipeline
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from celery import Task
from celery.signals import task_prerun, task_postrun, task_failure
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.database import get_db_session
from app.core.config import settings
from app.models.document import Document, DocumentStatus
from app.models.analysis import Analysis, AnalysisStatus
from app.models.audit_log import AuditLog, AuditAction, ResourceType
from app.services.ai_service import ai_service
from app.services.llm_service import llm_service
from app.services.vector_service import vector_service
from app.utils.text import extract_text
import asyncio
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# CELERY TASK BASE CLASS
# ============================================================================
class CallbackTask(Task):
    """Base task with callbacks for monitoring."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f" Task {task_id} succeeded: {self.name}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f" Task {task_id} failed: {self.name}")
        logger.error(f"Exception: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f" Task {task_id} retrying: {self.name}")


# ============================================================================
# ASYNC HELPERS
# ============================================================================
def run_async(coro):
    """
    Run async coroutine in sync context.
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Coroutine result
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


async def update_document_status(
    session: AsyncSession,
    document_id: int,
    status: DocumentStatus
) -> None:
    """
    Update document status.
    
    Args:
        session: Database session
        document_id: Document ID
        status: New status
    """
    document = await session.get(Document, document_id)
    if document:
        document.status = status
        await session.commit()


async def create_analysis_record(
    session: AsyncSession,
    document_id: int,
    user_id: int
) -> Analysis:
    """
    Create initial analysis record.
    
    Args:
        session: Database session
        document_id: Document ID
        user_id: User ID
        
    Returns:
        Created analysis record
    """
    analysis = Analysis(
        document_id=document_id,
        user_id=user_id,
        status=AnalysisStatus.PENDING
    )
    session.add(analysis)
    await session.commit()
    await session.refresh(analysis)
    return analysis


async def update_analysis_results(
    session: AsyncSession,
    analysis: Analysis,
    results: Dict[str, Any]
) -> None:
    """
    Update analysis with results.
    
    Args:
        session: Database session
        analysis: Analysis record
        results: Analysis results
    """
    analysis.document_type = results.get("document_type")
    analysis.classification_confidence = results.get("classification_confidence")
    analysis.entities = results.get("entities", {})
    analysis.unfair_clauses = results.get("unfair_clauses", [])
    analysis.summary = results.get("summary")
    analysis.risk_score = results.get("risk_score")
    analysis.recommendations = results.get("recommendations", [])
    analysis.processing_time = results.get("processing_time")
    analysis.status = AnalysisStatus.COMPLETED
    analysis.completed_at = datetime.utcnow()
    
    await session.commit()


async def log_audit_event(
    session: AsyncSession,
    user_id: int,
    action: AuditAction,
    resource_type: ResourceType,
    resource_id: int,
    success: bool = True,
    error_message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log audit event.
    
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
        logger.info(f" Starting analysis for document {document_id}")
        logger.info("=" * 80)
        
        # ====================================================================
        # STEP 0: Load document and update status
        # ====================================================================
        document = await session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        await update_document_status(session, document_id, DocumentStatus.PROCESSING)
        
        # ====================================================================
        # STEP 0.5: Extract text if not already done
        # ====================================================================
        text: str
        
        if not document.extracted_text_path or document.extracted_text_path == "":
            logger.info(" No extracted text found, extracting from file...")
            
            # Construire le chemin absolu du fichier original
            original_file_path = Path(settings.UPLOAD_DIR) / document.file_path
            
            if not original_file_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_file_path}")
            
            # Extraire le texte
            logger.info(f" Extracting text from: {original_file_path}")
            text = extract_text(original_file_path)
            
            if not text or len(text.strip()) < 50:
                raise ValueError(f"Extracted text too short ({len(text)} chars), possible extraction error")
            
            logger.info(f" Text extracted: {len(text)} characters")
            
            # Sauvegarder le texte extrait dans un fichier .txt
            # Format: uploads/user_123/contract_abc123_extracted.txt
            extracted_filename = original_file_path.stem + "_extracted.txt"
            extracted_file_path = original_file_path.parent / extracted_filename
            
            # Écrire le fichier texte
            with open(extracted_file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f" Text saved to: {extracted_file_path}")
            
            # Mettre à jour le document dans la BDD avec le chemin RELATIF
            # Extraire le chemin relatif depuis UPLOAD_DIR
            relative_extracted_path = extracted_file_path.relative_to(settings.UPLOAD_DIR)
            document.extracted_text_path = str(relative_extracted_path)
            
            # Commit pour sauvegarder le chemin
            await session.commit()
            await session.refresh(document)
            
            logger.info(f" Database updated: extracted_text_path = {document.extracted_text_path}")
        
        else:
            # Le texte a déjà été extrait, le charger
            logger.info(f" Loading existing extracted text: {document.extracted_text_path}")
            
            extracted_file_path = Path(settings.UPLOAD_DIR) / document.extracted_text_path
            
            if not extracted_file_path.exists():
                raise FileNotFoundError(f"Extracted text file not found: {extracted_file_path}")
            
            with open(extracted_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f" Text loaded: {len(text)} characters")
        
        # Valider que le texte est disponible
        if not text or len(text.strip()) < 50:
            raise ValueError(f"Document text too short for analysis ({len(text)} chars)")
        
        logger.info(f" Document ready for analysis: {len(text)} characters")
        
        # ====================================================================
        # STEP 1: Document Classification (Legal-BERT)
        # ====================================================================
        logger.info(" Running document classification...")
        classification_result = await ai_service.classify_document(text)
        logger.info(f" Classification: {classification_result['primary_class']} ({classification_result['confidence']:.2%})")
        
        # ====================================================================
        # STEP 2: Named Entity Recognition (CamemBERT-NER)
        # ====================================================================
        logger.info(" Running Named Entity Recognition...")
        ner_result = await ai_service.extract_entities(text)
        logger.info(f" Entities found: {ner_result['entity_count']}")
        
        # ====================================================================
        # STEP 3: Unfair Clause Detection (Unfair-ToS)
        # ====================================================================
        logger.info(" Detecting unfair clauses...")
        unfair_clauses_result = await ai_service.detect_unfair_clauses(text)
        logger.info(f" Unfair clauses: {len(unfair_clauses_result['clauses'])}")
        
        # ====================================================================
        # STEP 4: Document Summarization (BART)
        # ====================================================================
        logger.info(" Generating summary...")
        summary_result = await ai_service.summarize_document(text)
        logger.info(f" Summary generated: {len(summary_result['summary'])} chars")
        
        # ====================================================================
        # STEP 5: Risk Assessment
        # ====================================================================
        logger.info(" Calculating risk score...")
        risk_score = _calculate_risk_score(
            unfair_clauses=unfair_clauses_result['clauses'],
            classification=classification_result
        )
        logger.info(f" Risk score: {risk_score:.2f}/100")
        
        # ====================================================================
        # STEP 6: LLM Recommendations (Ollama Mistral)
        # ====================================================================
        logger.info(" Generating recommendations...")
        recommendations_result = await llm_service.generate_recommendations(
            document_type=classification_result['primary_class'],
            entities=ner_result['entities'],
            unfair_clauses=unfair_clauses_result['clauses'],
            risk_score=risk_score
        )
        logger.info(f" Recommendations: {len(recommendations_result['recommendations'])}")
        
        # ====================================================================
        # STEP 7: Vector Embeddings & Indexing (Qdrant)
        # ====================================================================
        logger.info(" Indexing document in vector database...")
        qdrant_point_id = await vector_service.index_document(
            document_id=document_id,
            text=text,
            metadata={
                "user_id": user_id,
                "filename": document.filename,
                "document_type": classification_result['primary_class']
            }
        )
        
        # Index document chunks for semantic search
        chunks = _split_text_into_chunks(text)
        chunk_ids = await vector_service.index_document_chunks(
            document_id=document_id,
            chunks=chunks,
            metadata={"user_id": user_id}
        )
        logger.info(f" Indexed: 1 document + {len(chunk_ids)} chunks")
        
        # ====================================================================
        # STEP 8: Save results to database
        # ====================================================================
        logger.info(" Saving analysis results...")
        
        analysis = await create_analysis_record(session, document_id, user_id)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        results = {
            "document_type": classification_result['primary_class'],
            "classification_confidence": classification_result['confidence'],
            "entities": ner_result['entities'],
            "unfair_clauses": unfair_clauses_result['clauses'],
            "summary": summary_result['summary'],
            "risk_score": risk_score,
            "recommendations": recommendations_result['recommendations'],
            "processing_time": processing_time,
            "qdrant_point_id": qdrant_point_id,
            "chunks_count": len(chunk_ids)
        }
        
        await update_analysis_results(session, analysis, results)
        await update_document_status(session, document_id, DocumentStatus.COMPLETED)
        
        # Log audit event
        await log_audit_event(
            session=session,
            user_id=user_id,
            action=AuditAction.ANALYSIS_COMPLETED,
            resource_type=ResourceType.DOCUMENT,
            resource_id=document_id,
            success=True,
            details={"processing_time": processing_time, "risk_score": risk_score}
        )
        
        logger.info("=" * 80)
        logger.info(f" Analysis completed in {processing_time:.2f}s")
        logger.info("=" * 80)
        
        return {
            "status": "completed",
            "document_id": document_id,
            "analysis_id": analysis.id,
            "processing_time": processing_time,
            "results": results
        }
    
    except Exception as e:
        logger.error(f" Analysis failed for document {document_id}: {e}")
        
        # Update statuses
        await update_document_status(session, document_id, DocumentStatus.FAILED)
        
        # Log audit event
        await log_audit_event(
            session=session,
            user_id=user_id,
            action=AuditAction.ANALYSIS_FAILED,
            resource_type=ResourceType.DOCUMENT,
            resource_id=document_id,
            success=False,
            error_message=str(e)
        )
        
        # Retry if possible
        if task.request.retries < task.max_retries:
            logger.info(f" Retrying analysis (attempt {task.request.retries + 1}/{task.max_retries})")
            raise task.retry(exc=e)
        
        raise
    
    finally:
        await session.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _calculate_risk_score(
    unfair_clauses: List[Dict[str, Any]],
    classification: Dict[str, Any]
) -> float:
    """
    Calculate risk score based on analysis results.
    
    Args:
        unfair_clauses: Detected unfair clauses
        classification: Document classification
        
    Returns:
        Risk score (0-100)
    """
    score = 0.0
    
    # Base score from unfair clauses (max 60 points)
    if unfair_clauses:
        clause_scores = [c.get('severity', 0.5) for c in unfair_clauses]
        avg_severity = sum(clause_scores) / len(clause_scores)
        score += avg_severity * 60
    
    # Add score based on document type (max 20 points)
    high_risk_types = ['bail_commercial', 'contrat_travail', 'vente']
    if classification['primary_class'] in high_risk_types:
        score += 20
    
    # Add score based on classification confidence (max 20 points)
    # Lower confidence = higher risk
    confidence_penalty = (1 - classification['confidence']) * 20
    score += confidence_penalty
    
    return min(score, 100.0)


def _split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks for vector indexing.
    
    Args:
        text: Text to split
        chunk_size: Approximate chunk size in words
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


# ============================================================================
# EMBEDDINGS GENERATION TASK
# ============================================================================
@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="tasks.generate_embeddings",
    max_retries=3
)
def generate_embeddings(self, document_id: int) -> Dict[str, Any]:
    """
    Generate and index embeddings for a document.
    
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
        logger.info(f" Generating embeddings for document {document_id}")
        
        # Load document
        document = await session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        
        # Load extracted text
        if not document.extracted_text_path:
            raise ValueError(f"Document {document_id} has no extracted text")
        
        extracted_file_path = Path(settings.UPLOAD_DIR) / document.extracted_text_path
        
        if not extracted_file_path.exists():
            raise FileNotFoundError(f"Extracted text file not found: {extracted_file_path}")
        
        with open(extracted_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Generate embeddings and index
        qdrant_point_id = await vector_service.index_document(
            document_id=document_id,
            text=text,
            metadata={
                "user_id": document.user_id,
                "filename": document.filename
            }
        )
        
        # Index chunks
        chunks = _split_text_into_chunks(text)
        chunk_ids = await vector_service.index_document_chunks(
            document_id=document_id,
            chunks=chunks,
            metadata={"user_id": document.user_id}
        )
        
        logger.info(f" Embeddings generated: 1 doc + {len(chunk_ids)} chunks")
        
        return {
            "status": "completed",
            "document_id": document_id,
            "qdrant_point_id": qdrant_point_id,
            "chunks_indexed": len(chunk_ids)
        }
    
    except Exception as e:
        logger.error(f" Embeddings generation failed: {e}")
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
    return run_async(_analyze_documents_batch_async(self, document_ids, user_id))


async def _analyze_documents_batch_async(
    task: Task,
    document_ids: List[int],
    user_id: int
) -> Dict[str, Any]:
    """
    Async implementation of batch analysis.
    
    Args:
        task: Celery task instance
        document_ids: List of document IDs
        user_id: User ID
        
    Returns:
        Batch results
    """
    logger.info(f" Starting batch analysis for {len(document_ids)} documents")
    
    results = {
        "total": len(document_ids),
        "succeeded": 0,
        "failed": 0,
        "errors": []
    }
    
    for doc_id in document_ids:
        try:
            await _analyze_document_async(task, doc_id, user_id)
            results["succeeded"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "document_id": doc_id,
                "error": str(e)
            })
            logger.error(f" Failed to analyze document {doc_id}: {e}")
    
    logger.info(f" Batch completed: {results['succeeded']}/{results['total']} succeeded")
    
    return results


# ============================================================================
# CELERY SIGNAL HANDLERS
# ============================================================================
@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extra):
    """Log when task starts."""
    logger.info(f"  Task started: {task.name} (ID: {task_id})")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, **extra):
    """Log when task completes."""
    logger.info(f"  Task finished: {task.name} (ID: {task_id})")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **extra):
    """Log when task fails."""
    logger.error(f" Task failed: {task_id}")
    logger.error(f"Exception: {exception}")