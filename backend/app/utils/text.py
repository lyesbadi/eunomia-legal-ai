"""
EUNOMIA Legal AI Platform - Text Utilities
Text extraction and processing from various file formats
"""
from typing import Optional, List
from pathlib import Path
import re

# Text extraction libraries
try:
    import PyPDF2
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

import logging


logger = logging.getLogger(__name__)


# ============================================================================
# TEXT EXTRACTION
# ============================================================================
def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from PDF file.
    
    Tries PyPDF2 first, falls back to pdfminer if needed.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text
        
    Raises:
        RuntimeError: If PDF libraries not available
        ValueError: If extraction fails
        
    Example:
        >>> text = extract_text_from_pdf(Path("document.pdf"))
        >>> print(len(text))
        15000
    """
    if not PDF_AVAILABLE:
        raise RuntimeError("PDF extraction libraries not installed")
    
    try:
        # Try PyPDF2 first (faster)
        text_parts = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
        
        text = '\n'.join(text_parts)
        
        # If PyPDF2 returns empty text, try pdfminer
        if not text.strip():
            logger.info(f"PyPDF2 returned empty text, trying pdfminer for {file_path}")
            text = pdfminer_extract(file_path)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        raise ValueError(f"Failed to extract text from PDF: {e}")


def extract_text_from_docx(file_path: Path) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        Extracted text
        
    Raises:
        RuntimeError: If python-docx not available
        ValueError: If extraction fails
        
    Example:
        >>> text = extract_text_from_docx(Path("contract.docx"))
    """
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx library not installed")
    
    try:
        doc = DocxDocument(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return '\n'.join(paragraphs).strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        raise ValueError(f"Failed to extract text from DOCX: {e}")


def extract_text_from_txt(file_path: Path) -> str:
    """
    Extract text from plain text file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        File content
        
    Example:
        >>> text = extract_text_from_txt(Path("notes.txt"))
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Try with latin-1 encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read().strip()


def extract_text(file_path: Path) -> str:
    """
    Extract text from file based on extension.
    
    Supports: PDF, DOCX, DOC, TXT, MD, RTF
    
    Args:
        file_path: Path to file
        
    Returns:
        Extracted text
        
    Raises:
        ValueError: If file type not supported
        
    Example:
        >>> text = extract_text(Path("document.pdf"))
    """
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif extension in ['.txt', '.md', '.rtf']:
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


# ============================================================================
# TEXT PROCESSING
# ============================================================================
def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    - Removes extra whitespace
    - Normalizes line breaks
    - Removes special characters
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
        
    Example:
        >>> clean_text("Hello   world\\n\\n\\nTest")
        'Hello world\\nTest'
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    return text.strip()


def truncate_text(
    text: str,
    max_length: int = 1000,
    suffix: str = "..."
) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
        
    Example:
        >>> truncate_text("Very long text...", max_length=10)
        'Very lo...'
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Useful for processing with LLMs or embeddings.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
        
    Example:
        >>> chunks = chunk_text("Long document text...", chunk_size=100)
        >>> len(chunks)
        15
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Text to count
        
    Returns:
        Word count
        
    Example:
        >>> count_words("Hello world! This is a test.")
        6
    """
    return len(re.findall(r'\b\w+\b', text))


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Text to process
        
    Returns:
        List of sentences
        
    Example:
        >>> sentences = extract_sentences("Hello. How are you? I'm fine!")
        >>> len(sentences)
        3
    """
    # Simple sentence splitting (can be improved with nltk)
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# TEXT ANALYSIS
# ============================================================================
def calculate_readability_score(text: str) -> float:
    """
    Calculate simple readability score (0-1).
    
    Based on average word and sentence length.
    Lower score = easier to read
    
    Args:
        text: Text to analyze
        
    Returns:
        Readability score (0 = very easy, 1 = very difficult)
        
    Example:
        >>> score = calculate_readability_score("Simple text here.")
        >>> score < 0.5
        True
    """
    sentences = extract_sentences(text)
    if not sentences:
        return 0.5
    
    words = text.split()
    if not words:
        return 0.5
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Normalize to 0-1 scale
    # Typical values: 10-20 words/sentence, 4-6 chars/word
    sentence_score = min(avg_sentence_length / 30, 1.0)
    word_score = min(avg_word_length / 10, 1.0)
    
    return (sentence_score + word_score) / 2


def detect_language_simple(text: str) -> str:
    """
    Simple language detection (French vs English).
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code ('fr' or 'en')
        
    Note:
        This is a simple heuristic. For production, use langdetect or similar.
        
    Example:
        >>> detect_language_simple("Bonjour le monde")
        'fr'
        >>> detect_language_simple("Hello world")
        'en'
    """
    # Common French words
    french_words = {'le', 'la', 'les', 'de', 'et', 'un', 'une', 'dans', 'pour', 'que'}
    
    # Common English words
    english_words = {'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'it'}
    
    words = set(text.lower().split())
    
    french_count = len(words & french_words)
    english_count = len(words & english_words)
    
    return 'fr' if french_count > english_count else 'en'