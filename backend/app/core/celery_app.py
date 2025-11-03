"""
EUNOMIA Legal AI Platform - Celery Application
Celery configuration and task queue setup
"""
from celery import Celery
from kombu import Queue, Exchange

from app.core.config import settings

# ============================================================================
# CELERY APP INITIALIZATION
# ============================================================================
celery_app = Celery(
    "eunomia",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.analysis_tasks"]
)

# ============================================================================
# CELERY CONFIGURATION
# ============================================================================
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,  # 5 minutes hard limit
    task_soft_time_limit=settings.CELERY_TASK_TIME_LIMIT - 30,  # 4m30s soft limit
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch 1 task at a time (important for memory)
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    worker_disable_rate_limits=False,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Queue settings
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    
    # Define queues with priorities
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default", priority=5),
        Queue("llm_queue", Exchange("llm"), routing_key="llm", priority=3),
        Queue("embeddings_queue", Exchange("embeddings"), routing_key="embeddings", priority=7),
    ),
    
    # Task routes
    task_routes={
        "tasks.analyze_document": {"queue": "default"},
        "tasks.generate_embeddings": {"queue": "embeddings_queue"},
        "tasks.analyze_documents_batch": {"queue": "default"},
    },
    
    # Retry settings
    task_acks_late=True,  # Task acknowledged after completion (important for reliability)
    task_reject_on_worker_lost=True,
    
    # Beat schedule (for periodic tasks - if needed later)
    beat_schedule={
        # Example: Clean old results every day
        # "cleanup-old-results": {
        #     "task": "tasks.cleanup_old_results",
        #     "schedule": crontab(hour=2, minute=0),
        # }
    }
)

# ============================================================================
# CELERY SIGNALS (for logging)
# ============================================================================
from celery.signals import task_prerun, task_postrun, task_failure
import logging

logger = logging.getLogger(__name__)


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Log when task starts."""
    logger.info(f"🚀 Task starting: {task.name}[{task_id}]")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, **extra):
    """Log when task completes."""
    logger.info(f"✅ Task completed: {task.name}[{task_id}]")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **extra):
    """Log when task fails."""
    logger.error(f"❌ Task failed: {sender.name}[{task_id}] - {exception}")