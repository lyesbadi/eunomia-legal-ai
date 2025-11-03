"""
EUNOMIA Legal AI Platform - Vector Service
Qdrant vector database service for semantic search and document embeddings
"""
from typing import Optional, List, Dict, Any, Tuple
import logging
from datetime import datetime
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest
)
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)


# ============================================================================
# VECTOR SERVICE
# ============================================================================
class VectorService:
    """
    Vector Service for Qdrant interactions.
    
    Features:
    - Document embedding storage
    - Semantic search
    - Similar document retrieval
    - Chunked document indexing
    
    Collections:
    - legal_documents: Full document embeddings
    - document_chunks: Chunked document embeddings for precise search
    """
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            timeout=30.0
        )
        
        self.collection_documents = settings.QDRANT_COLLECTION_DOCUMENTS
        self.collection_chunks = settings.QDRANT_COLLECTION_CHUNKS
        self.vector_size = settings.QDRANT_VECTOR_SIZE
        
        logger.info(f"🔍 Vector Service initialized: {settings.QDRANT_URL}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close Qdrant client."""
        await self.client.close()
    
    # ========================================================================
    # COLLECTION MANAGEMENT
    # ========================================================================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def ensure_collections_exist(self) -> None:
        """
        Ensure required Qdrant collections exist.
        
        Creates collections if they don't exist:
        - legal_documents: Full document embeddings (384-dim)
        - document_chunks: Chunked embeddings (384-dim)
        """
        try:
            # Check if documents collection exists
            collections = await self.client.get_collections()
            existing_names = {c.name for c in collections.collections}
            
            # Create documents collection if needed
            if self.collection_documents not in existing_names:
                logger.info(f" Creating collection: {self.collection_documents}")
                await self.client.create_collection(
                    collection_name=self.collection_documents,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f" Collection created: {self.collection_documents}")
            
            # Create chunks collection if needed
            if self.collection_chunks not in existing_names:
                logger.info(f" Creating collection: {self.collection_chunks}")
                await self.client.create_collection(
                    collection_name=self.collection_chunks,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f" Collection created: {self.collection_chunks}")
        
        except Exception as e:
            logger.error(f" Failed to ensure collections: {e}")
            raise
    
    # ========================================================================
    # DOCUMENT INDEXING
    # ========================================================================
    async def index_document(
        self,
        document_id: int,
        text: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Index full document in Qdrant.
        
        Args:
            document_id: Document ID from database
            text: Full document text
            metadata: Document metadata (type, user_id, etc.)
            
        Returns:
            Qdrant point ID (UUID)
        """
        start_time = datetime.now()
        
        try:
            # Generate embedding
            embeddings = await ai_service.generate_embeddings([text])
            embedding = embeddings[0]
            
            # Create point
            point_id = f"doc_{document_id}"
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "text": text[:1000],  # Store first 1000 chars for preview
                    "indexed_at": datetime.utcnow().isoformat(),
                    **metadata
                }
            )
            
            # Upsert to Qdrant
            await self.client.upsert(
                collection_name=self.collection_documents,
                points=[point]
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f" Document {document_id} indexed in {elapsed:.2f}s")
            
            return point_id
        
        except Exception as e:
            logger.error(f" Failed to index document {document_id}: {e}")
            raise
    
    async def index_document_chunks(
        self,
        document_id: int,
        chunks: List[str],
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Index document chunks for precise semantic search.
        
        Args:
            document_id: Document ID from database
            chunks: List of text chunks
            metadata: Document metadata
            
        Returns:
            List of Qdrant point IDs
        """
        start_time = datetime.now()
        
        try:
            # Generate embeddings for all chunks
            embeddings = await ai_service.generate_embeddings(chunks)
            
            # Create points
            points = []
            point_ids = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"doc_{document_id}_chunk_{i}"
                point_ids.append(point_id)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "chunk_index": i,
                        "text": chunk,
                        "indexed_at": datetime.utcnow().isoformat(),
                        **metadata
                    }
                )
                points.append(point)
            
            # Batch upsert
            await self.client.upsert(
                collection_name=self.collection_chunks,
                points=points
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f" Document {document_id}: {len(chunks)} chunks indexed in {elapsed:.2f}s")
            
            return point_ids
        
        except Exception as e:
            logger.error(f" Failed to index chunks for document {document_id}: {e}")
            raise
    
    # ========================================================================
    # SEMANTIC SEARCH
    # ========================================================================
    async def search_similar_documents(
        self,
        query: str,
        user_id: Optional[int] = None,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            user_id: Filter by user ID (optional)
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar documents with scores
        """
        start_time = datetime.now()
        
        try:
            # Generate query embedding
            query_embeddings = await ai_service.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Build filter
            filter_conditions = None
            if user_id is not None:
                filter_conditions = Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            
            # Search
            search_results = await self.client.search(
                collection_name=self.collection_documents,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "document_id": hit.payload.get("document_id"),
                    "score": hit.score,
                    "preview": hit.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in hit.payload.items()
                        if k not in ["document_id", "text"]
                    }
                })
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f" Found {len(results)} similar documents in {elapsed:.2f}s")
            
            return results
        
        except Exception as e:
            logger.error(f" Semantic search error: {e}")
            raise
    
    async def search_in_chunks(
        self,
        query: str,
        document_id: Optional[int] = None,
        user_id: Optional[int] = None,
        limit: int = 20,
        score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search in document chunks for precise results.
        
        Args:
            query: Search query
            document_id: Specific document ID (optional)
            user_id: Filter by user (optional)
            limit: Max results
            score_threshold: Min similarity score
            
        Returns:
            List of matching chunks with scores
        """
        start_time = datetime.now()
        
        try:
            # Generate query embedding
            query_embeddings = await ai_service.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Build filter
            conditions = []
            if document_id is not None:
                conditions.append(
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                )
            if user_id is not None:
                conditions.append(
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                )
            
            filter_conditions = Filter(must=conditions) if conditions else None
            
            # Search
            search_results = await self.client.search(
                collection_name=self.collection_chunks,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "document_id": hit.payload.get("document_id"),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "text": hit.payload.get("text"),
                    "score": hit.score,
                    "metadata": {
                        k: v for k, v in hit.payload.items()
                        if k not in ["document_id", "chunk_index", "text"]
                    }
                })
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f" Found {len(results)} matching chunks in {elapsed:.2f}s")
            
            return results
        
        except Exception as e:
            logger.error(f" Chunk search error: {e}")
            raise
    
    # ========================================================================
    # DOCUMENT DELETION
    # ========================================================================
    async def delete_document_vectors(self, document_id: int) -> None:
        """
        Delete all vectors for a document (full doc + chunks).
        
        Args:
            document_id: Document ID to delete
        """
        try:
            # Delete from documents collection
            await self.client.delete(
                collection_name=self.collection_documents,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            # Delete from chunks collection
            await self.client.delete(
                collection_name=self.collection_chunks,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            logger.info(f" Deleted vectors for document {document_id}")
        
        except Exception as e:
            logger.error(f" Failed to delete vectors for document {document_id}: {e}")
            raise
    
    # ========================================================================
    # HEALTH CHECK
    # ========================================================================
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Qdrant service health.
        
        Returns:
            Health status dictionary
        """
        try:
            collections = await self.client.get_collections()
            
            # Get collection stats
            stats = {}
            for collection in collections.collections:
                if collection.name in [self.collection_documents, self.collection_chunks]:
                    collection_info = await self.client.get_collection(collection.name)
                    stats[collection.name] = {
                        "vectors_count": collection_info.points_count,
                        "status": "ok"
                    }
            
            return {
                "status": "healthy",
                "qdrant_url": settings.QDRANT_URL,
                "collections": stats
            }
        
        except Exception as e:
            logger.error(f" Qdrant health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================
vector_service = VectorService()