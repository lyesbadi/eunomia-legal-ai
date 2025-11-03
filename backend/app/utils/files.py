"""
EUNOMIA Legal AI Platform - File Utilities
Helper functions for file operations and management
"""
from typing import Optional, BinaryIO
from pathlib import Path
import hashlib
import mimetypes
import magic
import shutil
from datetime import datetime

from app.core.config import settings
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# FILE VALIDATION
# ============================================================================
def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Original filename
        
    Returns:
        File extension with dot (e.g., ".pdf")
        
    Example:
        >>> get_file_extension("document.pdf")
        '.pdf'
        >>> get_file_extension("contract.DOCX")
        '.docx'
    """
    return Path(filename).suffix.lower()


def get_mime_type(file_path: Path) -> str:
    """
    Detect MIME type of file using python-magic.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string
        
    Example:
        >>> get_mime_type(Path("document.pdf"))
        'application/pdf'
    """
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(str(file_path))
    except Exception as e:
        logger.warning(f"Failed to detect MIME type with magic: {e}")
        # Fallback to mimetypes library
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"


def is_allowed_file_type(filename: str, mime_type: str) -> bool:
    """
    Check if file type is allowed.
    
    Args:
        filename: Original filename
        mime_type: MIME type
        
    Returns:
        True if file type is allowed
        
    Example:
        >>> is_allowed_file_type("doc.pdf", "application/pdf")
        True
        >>> is_allowed_file_type("script.exe", "application/x-executable")
        False
    """
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.rtf'}
    allowed_mimes = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/markdown',
        'application/rtf'
    }
    
    extension = get_file_extension(filename)
    
    return extension in allowed_extensions and mime_type in allowed_mimes


def validate_file_size(size: int, max_size: Optional[int] = None) -> None:
    """
    Validate file size against maximum allowed.
    
    Args:
        size: File size in bytes
        max_size: Maximum size in bytes (default from settings)
        
    Raises:
        ValueError: If file too large or empty
        
    Example:
        >>> validate_file_size(1024 * 1024)  # 1 MB - OK
        >>> validate_file_size(100 * 1024 * 1024)  # 100 MB - Raises ValueError
    """
    if max_size is None:
        max_size = 50 * 1024 * 1024  # 50 MB default
    
    if size == 0:
        raise ValueError("File is empty (0 bytes)")
    
    if size > max_size:
        size_mb = size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise ValueError(
            f"File too large ({size_mb:.2f} MB). Maximum allowed: {max_mb:.0f} MB"
        )


# ============================================================================
# FILE HASHING
# ============================================================================
def calculate_file_hash(
    file_path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    Calculate cryptographic hash of file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, sha1)
        chunk_size: Read chunk size in bytes (default: 8KB)
        
    Returns:
        Hex digest of hash
        
    Example:
        >>> calculate_file_hash(Path("document.pdf"))
        'a1b2c3d4e5f6...'
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def calculate_stream_hash(
    stream: BinaryIO,
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    Calculate hash of binary stream.
    
    Args:
        stream: Binary stream (file-like object)
        algorithm: Hash algorithm
        chunk_size: Read chunk size in bytes
        
    Returns:
        Hex digest of hash
        
    Note:
        Stream position is reset to 0 after hashing
    """
    hash_func = hashlib.new(algorithm)
    
    # Save original position
    original_pos = stream.tell()
    stream.seek(0)
    
    # Calculate hash
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        hash_func.update(chunk)
    
    # Reset position
    stream.seek(original_pos)
    
    return hash_func.hexdigest()


# ============================================================================
# FILE STORAGE
# ============================================================================
def get_user_upload_dir(user_id: int) -> Path:
    """
    Get upload directory for user.
    
    Creates directory if it doesn't exist.
    
    Args:
        user_id: User ID
        
    Returns:
        Path to user's upload directory
        
    Example:
        >>> get_user_upload_dir(123)
        Path('/uploads/123')
    """
    user_dir = settings.UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate unique filename with timestamp.
    
    Args:
        original_filename: Original filename
        
    Returns:
        Unique filename with timestamp prefix
        
    Example:
        >>> generate_unique_filename("document.pdf")
        '20251101_123045_document.pdf'
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{original_filename}"


def save_upload_file(
    file_content: BinaryIO,
    destination: Path
) -> int:
    """
    Save uploaded file to destination.
    
    Args:
        file_content: File binary content
        destination: Destination path
        
    Returns:
        File size in bytes
        
    Example:
        >>> size = save_upload_file(file.file, Path("/uploads/doc.pdf"))
    """
    # Ensure directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Save file
    with open(destination, 'wb') as buffer:
        shutil.copyfileobj(file_content, buffer)
    
    return destination.stat().st_size


def delete_file(file_path: Path) -> bool:
    """
    Delete file from disk.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if deleted, False if file didn't exist
        
    Example:
        >>> delete_file(Path("/uploads/old_document.pdf"))
        True
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"File deleted: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


# ============================================================================
# FILE INFO
# ============================================================================
def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return file_path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "2.5 MB")
        
    Example:
        >>> format_file_size(1024)
        '1.0 KB'
        >>> format_file_size(2_500_000)
        '2.4 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_file_info(file_path: Path) -> dict:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
        
    Example:
        >>> info = get_file_info(Path("document.pdf"))
        >>> print(info['size_mb'])
        2.4
    """
    stat = file_path.stat()
    
    return {
        'name': file_path.name,
        'extension': file_path.suffix.lower(),
        'size_bytes': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'mime_type': get_mime_type(file_path),
        'hash': calculate_file_hash(file_path)
    }


# ============================================================================
# CLEANUP
# ============================================================================
def cleanup_old_files(
    directory: Path,
    days: int = 30,
    pattern: str = "*"
) -> int:
    """
    Delete files older than specified days.
    
    Args:
        directory: Directory to clean
        days: Delete files older than this many days
        pattern: File pattern (glob)
        
    Returns:
        Number of files deleted
        
    Example:
        >>> cleanup_old_files(Path("/uploads/temp"), days=7)
        15
    """
    if not directory.exists():
        return 0
    
    cutoff_time = datetime.utcnow().timestamp() - (days * 86400)
    deleted = 0
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
    
    logger.info(f"Cleaned up {deleted} files older than {days} days from {directory}")
    return deleted


def get_directory_size(directory: Path) -> int:
    """
    Calculate total size of directory recursively.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
        
    Example:
        >>> size = get_directory_size(Path("/uploads/123"))
        >>> print(format_file_size(size))
        '47.3 MB'
    """
    total_size = 0
    
    for item in directory.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    
    return total_size