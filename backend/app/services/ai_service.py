"""
EUNOMIA Legal AI Platform - AI Service
Hugging Face models service for legal document analysis (Singleton pattern)
"""
from typing import Optional, List, Dict, Any, Tuple
import asyncio
from functools import lru_cache
import logging
from datetime import datetime
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Pipeline
)
from sentence_transformers import SentenceTransformer
import spacy
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# AI SERVICE SINGLETON
# ============================================================================
class AIService:
    """
    AI Service managing all Hugging Face models.
    
    Singleton pattern ensures models are loaded only once at startup.
    
    Models loaded:
    - Legal-BERT: Document classification
    - CamemBERT-NER: Named Entity Recognition (French)
    - Unfair-ToS: Unfair clause detection
    - BART: Document summarization
    - Sentence Transformers: Embeddings generation
    - Spacy: Text preprocessing
    
    Memory allocation (total ~8 GB):
    - Legal-BERT: ~450 MB
    - CamemBERT-NER: ~450 MB
    - Unfair-ToS: ~450 MB
    - BART: ~1.6 GB
    - Sentence Transformers: ~100 MB
    - Spacy: ~500 MB
    """
    
    _instance: Optional['AIService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern - only one instance allowed."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize AI models (called only once)."""
        if self._initialized:
            return
        
        logger.info("=" * 80)
        logger.info("🤖 Initializing EUNOMIA AI Service...")
        logger.info("=" * 80)
        
        # Determine device
        self.device = self._get_device()
        logger.info(f"🖥️  Device: {self.device}")
        
        # Initialize models
        self.legal_bert_classifier: Optional[Pipeline] = None
        self.camembert_ner: Optional[Pipeline] = None
        self.unfair_tos_classifier: Optional[Pipeline] = None
        self.summarizer: Optional[Pipeline] = None
        self.sentence_transformer: Optional[SentenceTransformer] = None
        self.spacy_nlp: Optional[spacy.Language] = None
        
        # Model versions (for tracking)
        self.model_versions = {
            "legal_bert": settings.HF_MODEL_LEGAL_BERT,
            "camembert_ner": settings.HF_MODEL_CAMEMBERT_NER,
            "unfair_tos": settings.HF_MODEL_UNFAIR_TOS,
            "summarization": settings.HF_MODEL_SUMMARIZATION,
            "embeddings": settings.HF_MODEL_EMBEDDINGS,
            "spacy": "fr_core_news_md-3.8.0"
        }
        
        # Load all models
        self._load_all_models()
        
        self._initialized = True
        logger.info("=" * 80)
        logger.info("AI Service Initialized Successfully")
        logger.info("=" * 80)
    
    def _get_device(self) -> str:
        """
        Determine optimal device (CPU/CUDA/MPS).
        
        Returns:
            Device string for PyTorch
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Silicon MPS available")
        else:
            device = "cpu"
            logger.info("ℹ️  Using CPU (no GPU detected)")
        
        return device
    
    def _load_all_models(self) -> None:
        """Load all AI models sequentially with error handling."""
        
        try:
            # 1. Legal-BERT (Document Classification)
            logger.info(" Loading Legal-BERT...")
            start_time = datetime.now()
            self.legal_bert_classifier = pipeline(
                "text-classification",
                model=settings.HF_MODEL_LEGAL_BERT,
                device=self.device if self.device != "cpu" else -1,
                top_k=5  # Return top 5 predictions
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Legal-BERT loaded ({elapsed:.2f}s)")
        
        except Exception as e:
            logger.error(f" Failed to load Legal-BERT: {e}")
            self.legal_bert_classifier = None
        
        try:
            # 2. CamemBERT-NER (Named Entity Recognition)
            logger.info(" Loading CamemBERT-NER...")
            start_time = datetime.now()
            self.camembert_ner = pipeline(
                "ner",
                model=settings.HF_MODEL_CAMEMBERT_NER,
                device=self.device if self.device != "cpu" else -1,
                aggregation_strategy="simple"  # Merge tokens
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"CamemBERT-NER loaded ({elapsed:.2f}s)")
        
        except Exception as e:
            logger.error(f" Failed to load CamemBERT-NER: {e}")
            self.camembert_ner = None
        
        try:
            # 3. Unfair-ToS Classifier (Clause Detection)
            logger.info(" Loading Unfair-ToS Classifier...")
            start_time = datetime.now()
            self.unfair_tos_classifier = pipeline(
                "text-classification",
                model=settings.HF_MODEL_UNFAIR_TOS,
                device=self.device if self.device != "cpu" else -1,
                top_k=None  # Return all labels with scores
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Unfair-ToS Classifier loaded ({elapsed:.2f}s)")
        
        except Exception as e:
            logger.error(f" Failed to load Unfair-ToS: {e}")
            self.unfair_tos_classifier = None
        
        try:
            # 4. BART Summarizer
            logger.info(" Loading BART Summarizer...")
            start_time = datetime.now()
            self.summarizer = pipeline(
                "summarization",
                model=settings.HF_MODEL_SUMMARIZATION,
                device=self.device if self.device != "cpu" else -1
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"BART Summarizer loaded ({elapsed:.2f}s)")
        
        except Exception as e:
            logger.error(f" Failed to load BART: {e}")
            self.summarizer = None
        
        try:
            # 5. Sentence Transformers (Embeddings)
            logger.info(" Loading Sentence Transformers...")
            start_time = datetime.now()
            self.sentence_transformer = SentenceTransformer(
                settings.HF_MODEL_EMBEDDINGS,
                device=self.device
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Sentence Transformers loaded ({elapsed:.2f}s)")
        
        except Exception as e:
            logger.error(f" Failed to load Sentence Transformers: {e}")
            self.sentence_transformer = None
        
        try:
            # 6. Spacy (French NLP)
            logger.info(" Loading Spacy French model...")
            start_time = datetime.now()
            self.spacy_nlp = spacy.load("fr_core_news_md")
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Spacy loaded ({elapsed:.2f}s)")
        
        except Exception as e:
            logger.error(f" Failed to load Spacy: {e}")
            logger.info("Run: python -m spacy download fr_core_news_md")
            self.spacy_nlp = None
    
    # ========================================================================
    # DOCUMENT CLASSIFICATION
    # ========================================================================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def classify_document(
        self,
        text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Classify legal document type using Legal-BERT.
        
        Args:
            text: Document text (max 512 tokens)
            top_k: Number of top predictions to return
            
        Returns:
            Classification result with predictions and confidence
            
        Example:
            {
                "primary_class": "contract",
                "confidence": 0.97,
                "top_predictions": [
                    {"label": "contract", "score": 0.97},
                    {"label": "tos", "score": 0.02}
                ],
                "model_used": "nlpaueb/legal-bert-base-uncased"
            }
        """
        if not self.legal_bert_classifier:
            raise RuntimeError("Legal-BERT model not loaded")
        
        start_time = datetime.now()
        
        try:
            # Truncate text if too long (Legal-BERT max 512 tokens)
            text_truncated = text[:2048]  # Approximately 512 tokens
            
            # Run classification in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None,
                self.legal_bert_classifier,
                text_truncated
            )
            
            # Extract results
            if isinstance(predictions, list) and len(predictions) > 0:
                results = predictions[0] if isinstance(predictions[0], list) else predictions
            else:
                results = predictions
            
            # Format response
            top_predictions = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
            
            result = {
                "primary_class": top_predictions[0]['label'],
                "confidence": top_predictions[0]['score'],
                "top_predictions": [
                    {
                        "label": pred['label'],
                        "score": pred['score']
                    }
                    for pred in top_predictions
                ],
                "model_used": settings.HF_MODEL_LEGAL_BERT
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Classification completed in {elapsed:.2f}s")
            
            return result
        
        except Exception as e:
            logger.error(f" Classification error: {e}")
            raise
    
    # ========================================================================
    # NAMED ENTITY RECOGNITION
    # ========================================================================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def extract_entities(
        self,
        text: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract named entities using CamemBERT-NER.
        
        Args:
            text: Document text
            confidence_threshold: Minimum confidence score (0-1)
            
        Returns:
            NER result with entities by type
            
        Example:
            {
                "entities": [
                    {
                        "text": "Jean Dupont",
                        "type": "PER",
                        "start": 10,
                        "end": 21,
                        "confidence": 0.98
                    }
                ],
                "entity_count": 15,
                "by_type": {
                    "PER": 5,
                    "ORG": 3,
                    "LOC": 7
                },
                "model_used": "Jean-Baptiste/camembert-ner"
            }
        """
        if not self.camembert_ner:
            raise RuntimeError("CamemBERT-NER model not loaded")
        
        start_time = datetime.now()
        
        try:
            # Split text into chunks if too long (max 512 tokens per chunk)
            max_chunk_length = 2000  # Approximately 512 tokens
            text_chunks = [
                text[i:i + max_chunk_length]
                for i in range(0, len(text), max_chunk_length)
            ]
            
            # Process all chunks
            all_entities = []
            offset = 0
            
            for chunk in text_chunks:
                # Run NER in thread pool
                loop = asyncio.get_event_loop()
                chunk_entities = await loop.run_in_executor(
                    None,
                    self.camembert_ner,
                    chunk
                )
                
                # Adjust entity positions and filter by confidence
                for entity in chunk_entities:
                    if entity['score'] >= confidence_threshold:
                        all_entities.append({
                            "text": entity['word'],
                            "type": entity['entity_group'],
                            "start": entity['start'] + offset,
                            "end": entity['end'] + offset,
                            "confidence": entity['score']
                        })
                
                offset += len(chunk)
            
            # Count by type
            by_type = {}
            for entity in all_entities:
                entity_type = entity['type']
                by_type[entity_type] = by_type.get(entity_type, 0) + 1
            
            result = {
                "entities": all_entities,
                "entity_count": len(all_entities),
                "by_type": by_type,
                "model_used": settings.HF_MODEL_CAMEMBERT_NER
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"NER completed: {len(all_entities)} entities in {elapsed:.2f}s")
            
            return result
        
        except Exception as e:
            logger.error(f" NER error: {e}")
            raise
    
    # ========================================================================
    # UNFAIR CLAUSE DETECTION
    # ========================================================================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def detect_unfair_clauses(
        self,
        text: str,
        chunk_size: int = 500,
        unfair_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Detect potentially unfair clauses in Terms of Service.
        
        Args:
            text: Document text
            chunk_size: Characters per clause chunk
            unfair_threshold: Minimum score to flag as unfair (0-1)
            
        Returns:
            Unfair clause detection result
            
        Example:
            {
                "unfair_clauses": [
                    {
                        "text": "We may terminate your account...",
                        "category": "termination",
                        "severity": "high",
                        "score": 0.89,
                        "explanation": "Unilateral termination clause"
                    }
                ],
                "unfair_clause_count": 3,
                "fairness_score": 0.35,
                "model_used": "dennlinger/bert-base-uncased-unfair-tos"
            }
        """
        if not self.unfair_tos_classifier:
            raise RuntimeError("Unfair-ToS model not loaded")
        
        start_time = datetime.now()
        
        try:
            # Split text into clause-sized chunks
            chunks = self._split_into_clauses(text, chunk_size)
            
            unfair_clauses = []
            total_score = 0.0
            
            # Analyze each chunk
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Run classification in thread pool
                loop = asyncio.get_event_loop()
                predictions = await loop.run_in_executor(
                    None,
                    self.unfair_tos_classifier,
                    chunk
                )
                
                # Check if any unfair category has high score
                if isinstance(predictions, list) and len(predictions) > 0:
                    clause_results = predictions[0]
                    
                    # Find highest unfair score
                    max_unfair_score = 0.0
                    unfair_category = None
                    
                    for pred in clause_results:
                        label = pred['label'].lower()
                        score = pred['score']
                        
                        # Categories considered "unfair"
                        unfair_categories = [
                            'unfair',
                            'potentially_unfair',
                            'unilateral_termination',
                            'unilateral_change',
                            'content_removal',
                            'jurisdiction',
                            'limitation_of_liability'
                        ]
                        
                        if any(cat in label for cat in unfair_categories) and score > max_unfair_score:
                            max_unfair_score = score
                            unfair_category = label
                    
                    total_score += max_unfair_score
                    
                    # Flag if above threshold
                    if max_unfair_score >= unfair_threshold:
                        # Determine severity
                        if max_unfair_score >= 0.85:
                            severity = "critical"
                        elif max_unfair_score >= 0.75:
                            severity = "high"
                        elif max_unfair_score >= 0.65:
                            severity = "medium"
                        else:
                            severity = "low"
                        
                        unfair_clauses.append({
                            "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            "category": unfair_category or "unknown",
                            "severity": severity,
                            "score": max_unfair_score,
                            "position": i,
                            "explanation": self._get_clause_explanation(unfair_category)
                        })
            
            # Calculate overall fairness score (inverse of average unfairness)
            avg_unfair_score = total_score / len(chunks) if chunks else 0.0
            fairness_score = 1.0 - avg_unfair_score
            
            result = {
                "unfair_clauses": unfair_clauses,
                "unfair_clause_count": len(unfair_clauses),
                "fairness_score": fairness_score,
                "total_clauses_analyzed": len(chunks),
                "model_used": settings.HF_MODEL_UNFAIR_TOS
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Unfair clause detection completed: {len(unfair_clauses)} found in {elapsed:.2f}s")
            
            return result
        
        except Exception as e:
            logger.error(f" Unfair clause detection error: {e}")
            raise
    
    def _split_into_clauses(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into clause-sized chunks.
        
        Args:
            text: Document text
            chunk_size: Target chunk size in characters
            
        Returns:
            List of text chunks
        """
        # Split by sentences if Spacy available
        if self.spacy_nlp:
            try:
                doc = self.spacy_nlp(text)
                sentences = [sent.text for sent in doc.sents]
                
                # Group sentences into chunks
                chunks = []
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    if current_length + sentence_length > chunk_size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                return chunks
            
            except Exception as e:
                logger.warning(f"Spacy sentence splitting failed: {e}")
        
        # Fallback: simple character chunking
        return [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]
    
    def _get_clause_explanation(self, category: Optional[str]) -> str:
        """
        Get human-readable explanation for unfair clause category.
        
        Args:
            category: Clause category
            
        Returns:
            Explanation text
        """
        explanations = {
            "unilateral_termination": "Cette clause permet la résiliation unilatérale sans préavis",
            "unilateral_change": "Cette clause autorise des modifications unilatérales des conditions",
            "content_removal": "Cette clause permet la suppression de contenu sans justification",
            "jurisdiction": "Cette clause impose un tribunal potentiellement défavorable",
            "limitation_of_liability": "Cette clause limite excessivement la responsabilité du fournisseur",
            "unfair": "Cette clause présente des caractéristiques potentiellement abusives",
            "potentially_unfair": "Cette clause nécessite un examen juridique approfondi"
        }
        
        return explanations.get(category, "Clause nécessitant une attention particulière")
    
    # ========================================================================
    # DOCUMENT SUMMARIZATION
    # ========================================================================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def summarize_document(
        self,
        text: str,
        max_length: int = 500,
        min_length: int = 100
    ) -> Dict[str, Any]:
        """
        Generate document summary using BART.
        
        Args:
            text: Document text
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            
        Returns:
            Summary result
            
        Example:
            {
                "summary_text": "Ce contrat de location...",
                "summary_length": 245,
                "original_length": 5420,
                "compression_ratio": 0.045,
                "model_used": "facebook/bart-large-cnn"
            }
        """
        if not self.summarizer:
            raise RuntimeError("BART summarizer not loaded")
        
        start_time = datetime.now()
        
        try:
            # BART has max input length of 1024 tokens (~4096 chars)
            max_input_length = 4000
            text_truncated = text[:max_input_length]
            
            # Run summarization in thread pool
            loop = asyncio.get_event_loop()
            summary_result = await loop.run_in_executor(
                None,
                lambda: self.summarizer(
                    text_truncated,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
            )
            
            summary_text = summary_result[0]['summary_text']
            
            result = {
                "summary_text": summary_text,
                "summary_length": len(summary_text),
                "original_length": len(text),
                "compression_ratio": len(summary_text) / len(text) if text else 0.0,
                "model_used": settings.HF_MODEL_SUMMARIZATION
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Summarization completed in {elapsed:.2f}s")
            
            return result
        
        except Exception as e:
            logger.error(f" Summarization error: {e}")
            raise
    
    # ========================================================================
    # EMBEDDINGS GENERATION
    # ========================================================================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate_embeddings(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate vector embeddings using Sentence Transformers.
        
        Args:
            texts: List of text strings to embed
            normalize: Whether to L2-normalize vectors
            
        Returns:
            List of embedding vectors (384-dimensional)
        """
        if not self.sentence_transformer:
            raise RuntimeError("Sentence Transformer model not loaded")
        
        start_time = datetime.now()
        
        try:
            # Run embedding in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.sentence_transformer.encode(
                    texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False
                )
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Generated {len(embeddings_list)} embeddings in {elapsed:.2f}s")
            
            return embeddings_list
        
        except Exception as e:
            logger.error(f" Embedding generation error: {e}")
            raise
    
    # ========================================================================
    # LANGUAGE DETECTION
    # ========================================================================
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect document language using Spacy.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not self.spacy_nlp:
            return ("fr", 0.5)  # Default to French
        
        try:
            # Use first 1000 characters for detection
            sample = text[:1000]
            doc = self.spacy_nlp(sample)
            
            # Count French-specific tokens/patterns
            french_indicators = 0
            total_tokens = 0
            
            for token in doc:
                if not token.is_punct and not token.is_space:
                    total_tokens += 1
                    # Check if word exists in French vocabulary
                    if token.is_alpha and doc.vocab.has_vector(token.text.lower()):
                        french_indicators += 1
            
            confidence = french_indicators / total_tokens if total_tokens > 0 else 0.5
            
            return ("fr", min(confidence, 1.0))
        
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return ("fr", 0.5)
    
    # ========================================================================
    # HEALTH CHECK
    # ========================================================================
    def health_check(self) -> Dict[str, Any]:
        """
        Check health status of all models.
        
        Returns:
            Health status dictionary
        """
        return {
            "initialized": self._initialized,
            "device": self.device,
            "models": {
                "legal_bert": self.legal_bert_classifier is not None,
                "camembert_ner": self.camembert_ner is not None,
                "unfair_tos": self.unfair_tos_classifier is not None,
                "summarizer": self.summarizer is not None,
                "sentence_transformer": self.sentence_transformer is not None,
                "spacy": self.spacy_nlp is not None
            },
            "model_versions": self.model_versions
        }


# ============================================================================
# GLOBAL SINGLETON INSTANCE
# ============================================================================
@lru_cache()
def get_ai_service() -> AIService:
    """
    Get singleton AI service instance.
    
    Returns:
        AIService instance (cached)
    """
    return AIService()


# Create instance at module import (lazy loading)
ai_service = get_ai_service()