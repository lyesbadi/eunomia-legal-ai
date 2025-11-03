"""
EUNOMIA Legal AI Platform - Custom Validators
Validation functions for data integrity and security
"""
from typing import Optional, List
import re
from datetime import datetime
from pathlib import Path

import logging


logger = logging.getLogger(__name__)


# ============================================================================
# EMAIL VALIDATION
# ============================================================================
def is_valid_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
        
    Example:
        >>> is_valid_email("user@example.com")
        True
        >>> is_valid_email("invalid.email")
        False
    """
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    return bool(email_pattern.match(email))


def is_disposable_email(email: str) -> bool:
    """
    Check if email is from a disposable email provider.
    
    Args:
        email: Email address
        
    Returns:
        True if disposable email
        
    Example:
        >>> is_disposable_email("user@tempmail.com")
        True
        >>> is_disposable_email("user@gmail.com")
        False
    """
    disposable_domains = {
        'tempmail.com',
        'guerrillamail.com',
        '10minutemail.com',
        'throwaway.email',
        'mailinator.com',
        'maildrop.cc',
        'temp-mail.org',
        'fakeinbox.com'
    }
    
    domain = email.split('@')[-1].lower()
    return domain in disposable_domains


# ============================================================================
# PASSWORD VALIDATION
# ============================================================================
def validate_password_strength(password: str) -> tuple[bool, List[str]]:
    """
    Validate password strength against security requirements.
    
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Example:
        >>> valid, errors = validate_password_strength("Weak123")
        >>> valid
        False
        >>> "special character" in errors[0].lower()
        True
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one digit")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    
    return len(errors) == 0, errors


def is_common_password(password: str) -> bool:
    """
    Check if password is in common passwords list.
    
    Args:
        password: Password to check
        
    Returns:
        True if password is too common
        
    Example:
        >>> is_common_password("password123")
        True
        >>> is_common_password("K9@xPz#mQ2vL")
        False
    """
    common_passwords = {
        'password', 'password123', '123456', '12345678',
        'qwerty', 'abc123', 'password1', 'admin', 'letmein',
        'welcome', 'monkey', '1234567890', 'password!',
        'Welcome1', 'Password1', 'Admin123'
    }
    
    return password.lower() in common_passwords


# ============================================================================
# FILE VALIDATION
# ============================================================================
def validate_filename(filename: str) -> tuple[bool, Optional[str]]:
    """
    Validate filename for security.
    
    Checks for:
    - Path traversal attempts (../)
    - Invalid characters
    - Reserved names (Windows)
    - Length limits
    
    Args:
        filename: Filename to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> valid, error = validate_filename("document.pdf")
        >>> valid
        True
        >>> valid, error = validate_filename("../../etc/passwd")
        >>> valid
        False
    """
    # Check for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Filename contains invalid path characters"
    
    # Check for invalid characters
    invalid_chars = '<>:"|?*\x00'
    if any(char in filename for char in invalid_chars):
        return False, "Filename contains invalid characters"
    
    # Check for Windows reserved names
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = Path(filename).stem.upper()
    if name_without_ext in reserved_names:
        return False, "Filename uses reserved system name"
    
    # Check length
    if len(filename) > 255:
        return False, "Filename too long (max 255 characters)"
    
    if len(filename) == 0:
        return False, "Filename cannot be empty"
    
    return True, None


def validate_file_extension(filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate file extension against whitelist.
    
    Args:
        filename: Filename to validate
        allowed_extensions: List of allowed extensions (with dots)
        
    Returns:
        True if extension is allowed
        
    Example:
        >>> validate_file_extension("doc.pdf", [".pdf", ".docx"])
        True
        >>> validate_file_extension("script.exe", [".pdf", ".docx"])
        False
    """
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf']
    
    extension = Path(filename).suffix.lower()
    return extension in allowed_extensions


# ============================================================================
# TEXT VALIDATION
# ============================================================================
def contains_sql_injection(text: str) -> bool:
    """
    Check if text contains potential SQL injection patterns.
    
    Args:
        text: Text to check
        
    Returns:
        True if suspicious patterns detected
        
    Example:
        >>> contains_sql_injection("SELECT * FROM users")
        True
        >>> contains_sql_injection("Normal user input")
        False
    """
    sql_patterns = [
        r"('|(\\')|(--)|(/\\*)|(\\*/)|(\bOR\b)|(\bAND\b))",
        r"(\bUNION\b)|(\bSELECT\b)|(\bINSERT\b)|(\bUPDATE\b)|(\bDELETE\b)",
        r"(\bDROP\b)|(\bCREATE\b)|(\bALTER\b)|(\bEXEC\b)",
        r"(\bSCRIPT\b)|(\bJAVASCRIPT\b)|(\bONLOAD\b)|(\bONERROR\b)"
    ]
    
    text_upper = text.upper()
    
    for pattern in sql_patterns:
        if re.search(pattern, text_upper, re.IGNORECASE):
            return True
    
    return False


def contains_xss(text: str) -> bool:
    """
    Check if text contains potential XSS patterns.
    
    Args:
        text: Text to check
        
    Returns:
        True if suspicious patterns detected
        
    Example:
        >>> contains_xss("<script>alert('xss')</script>")
        True
        >>> contains_xss("Normal text")
        False
    """
    xss_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'onerror\s*=',
        r'onload\s*=',
        r'onclick\s*=',
        r'<iframe',
        r'<object',
        r'<embed'
    ]
    
    for pattern in xss_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input by removing potentially dangerous content.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Example:
        >>> sanitize_input("<script>alert('test')</script>Hello")
        'Hello'
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Truncate
    if len(text) > max_length:
        text = text[:max_length]
    
    # Strip whitespace
    text = text.strip()
    
    return text


# ============================================================================
# BUSINESS LOGIC VALIDATION
# ============================================================================
def validate_date_range(
    start_date: datetime,
    end_date: datetime,
    max_range_days: int = 365
) -> tuple[bool, Optional[str]]:
    """
    Validate date range.
    
    Args:
        start_date: Start date
        end_date: End date
        max_range_days: Maximum allowed range in days
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> from datetime import datetime, timedelta
        >>> start = datetime(2025, 1, 1)
        >>> end = datetime(2025, 1, 31)
        >>> valid, error = validate_date_range(start, end)
        >>> valid
        True
    """
    if start_date > end_date:
        return False, "Start date must be before end date"
    
    date_diff = (end_date - start_date).days
    
    if date_diff > max_range_days:
        return False, f"Date range exceeds maximum of {max_range_days} days"
    
    return True, None


def validate_pagination(page: int, page_size: int) -> tuple[bool, Optional[str]]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> valid, error = validate_pagination(1, 20)
        >>> valid
        True
        >>> valid, error = validate_pagination(0, 20)
        >>> valid
        False
    """
    if page < 1:
        return False, "Page number must be at least 1"
    
    if page_size < 1:
        return False, "Page size must be at least 1"
    
    if page_size > 100:
        return False, "Page size cannot exceed 100"
    
    return True, None


def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID format.
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid UUID format
        
    Example:
        >>> validate_uuid("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        True
        >>> validate_uuid("invalid-uuid")
        False
    """
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(uuid_string))


# ============================================================================
# API KEY VALIDATION
# ============================================================================
def validate_api_key_format(api_key: str) -> bool:
    """
    Validate API key format.
    
    Expected format: eun_live_<40_hex_chars> or eun_test_<40_hex_chars>
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid format
        
    Example:
        >>> validate_api_key_format("eun_live_a1b2c3d4e5f6...")
        True
        >>> validate_api_key_format("invalid_key")
        False
    """
    pattern = re.compile(r'^eun_(live|test)_[0-9a-f]{40}$')
    return bool(pattern.match(api_key))


# ============================================================================
# PHONE NUMBER VALIDATION
# ============================================================================
def validate_phone_number(phone: str, country_code: str = "FR") -> bool:
    """
    Simple phone number validation.
    
    Args:
        phone: Phone number to validate
        country_code: Country code (FR or US)
        
    Returns:
        True if valid format
        
    Example:
        >>> validate_phone_number("+33612345678", "FR")
        True
        >>> validate_phone_number("0612345678", "FR")
        True
    """
    # Remove spaces and dashes
    phone = re.sub(r'[\s\-\(\)]', '', phone)
    
    if country_code == "FR":
        # French mobile: +33 6/7 or 06/07
        pattern = re.compile(r'^(\+33|0)[67]\d{8}$')
    elif country_code == "US":
        # US: +1 or 1 followed by 10 digits
        pattern = re.compile(r'^(\+1|1)?\d{10}$')
    else:
        # Generic: 8-15 digits
        pattern = re.compile(r'^\+?\d{8,15}$')
    
    return bool(pattern.match(phone))


# ============================================================================
# BATCH VALIDATION
# ============================================================================
def validate_batch(
    values: List[str],
    validator_func,
    stop_on_first_error: bool = False
) -> tuple[List[bool], List[Optional[str]]]:
    """
    Validate a batch of values.
    
    Args:
        values: List of values to validate
        validator_func: Validation function
        stop_on_first_error: Stop on first error
        
    Returns:
        Tuple of (list of validation results, list of error messages)
        
    Example:
        >>> emails = ["valid@test.com", "invalid", "ok@domain.com"]
        >>> results, errors = validate_batch(emails, is_valid_email)
        >>> results
        [True, False, True]
    """
    results = []
    errors = []
    
    for value in values:
        try:
            is_valid = validator_func(value)
            results.append(is_valid)
            errors.append(None if is_valid else f"Invalid: {value}")
            
            if not is_valid and stop_on_first_error:
                break
        except Exception as e:
            results.append(False)
            errors.append(str(e))
            
            if stop_on_first_error:
                break
    
    return results, errors