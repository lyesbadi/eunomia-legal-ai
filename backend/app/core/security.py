"""
EUNOMIA Legal AI Platform - Security Module
Password hashing, JWT token management, and security utilities
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import jwt, JWTError
import secrets
import logging

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# PASSWORD HASHING
# ============================================================================
# Configure password hashing context with bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS,
)


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    
    Args:
        password: Plaintext password to hash
    
    Returns:
        str: Hashed password
    
    Example:
        >>> hashed = hash_password("MySecurePassword123!")
        >>> print(hashed)
        $2b$12$...
    """
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    
    hashed = pwd_context.hash(password)
    logger.debug("Password hashed successfully")
    return hashed


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a hashed password.
    
    Args:
        plain_password: Plaintext password to verify
        hashed_password: Previously hashed password
    
    Returns:
        bool: True if password matches, False otherwise
    
    Example:
        >>> hashed = hash_password("MyPassword")
        >>> verify_password("MyPassword", hashed)
        True
        >>> verify_password("WrongPassword", hashed)
        False
    """
    try:
        is_valid = pwd_context.verify(plain_password, hashed_password)
        if is_valid:
            logger.debug("Password verification successful")
        else:
            logger.debug("Password verification failed")
        return is_valid
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def needs_rehash(hashed_password: str) -> bool:
    """
    Check if a hashed password needs to be rehashed.
    
    Useful when bcrypt rounds are increased in settings.
    
    Args:
        hashed_password: Previously hashed password
    
    Returns:
        bool: True if password should be rehashed
    """
    return pwd_context.needs_update(hashed_password)


# ============================================================================
# JWT TOKEN MANAGEMENT
# ============================================================================
def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Payload data to encode in token (typically {"sub": user_id})
        expires_delta: Custom expiration time (default: from settings)
    
    Returns:
        str: Encoded JWT token
    
    Example:
        >>> token = create_access_token({"sub": "user@example.com", "role": "admin"})
        >>> print(token)
        eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),  # Issued at
        "type": "access"
    })
    
    # Encode JWT
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    logger.debug(f"Access token created (expires: {expire})")
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token.
    
    Refresh tokens have longer lifetime and are used to obtain new access tokens.
    
    Args:
        data: Payload data to encode in token (typically {"sub": user_id})
        expires_delta: Custom expiration time (default: from settings)
    
    Returns:
        str: Encoded JWT refresh token
    
    Example:
        >>> token = create_refresh_token({"sub": "user@example.com"})
    """
    to_encode = data.copy()
    
    # Set expiration time (longer than access token)
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    # Encode JWT
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    logger.debug(f"Refresh token created (expires: {expire})")
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token to decode
    
    Returns:
        dict: Decoded token payload if valid, None otherwise
    
    Raises:
        JWTError: If token is invalid or expired
    
    Example:
        >>> token = create_access_token({"sub": "user@example.com"})
        >>> payload = decode_token(token)
        >>> print(payload["sub"])
        user@example.com
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Validate token type (access or refresh)
        token_type = payload.get("type")
        if token_type not in ["access", "refresh"]:
            logger.warning(f"Invalid token type: {token_type}")
            return None
        
        logger.debug(f"Token decoded successfully (type: {token_type})")
        return payload
    
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise JWTError("Token has expired")
    
    except jwt.JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise JWTError(f"Could not validate credentials: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {e}", exc_info=True)
        raise JWTError("Invalid token")


def verify_token_type(token: str, expected_type: str) -> bool:
    """
    Verify that a token has the expected type (access or refresh).
    
    Args:
        token: JWT token to verify
        expected_type: Expected token type ("access" or "refresh")
    
    Returns:
        bool: True if token type matches, False otherwise
    """
    try:
        payload = decode_token(token)
        if payload is None:
            return False
        
        token_type = payload.get("type")
        return token_type == expected_type
    
    except JWTError:
        return False


def get_token_subject(token: str) -> Optional[str]:
    """
    Extract the subject (user identifier) from a token.
    
    Args:
        token: JWT token
    
    Returns:
        str: Token subject (typically user email or ID), None if invalid
    
    Example:
        >>> token = create_access_token({"sub": "user@example.com"})
        >>> subject = get_token_subject(token)
        >>> print(subject)
        user@example.com
    """
    try:
        payload = decode_token(token)
        if payload is None:
            return None
        
        subject = payload.get("sub")
        return subject
    
    except JWTError:
        return None


# ============================================================================
# API KEY GENERATION
# ============================================================================
def generate_api_key(prefix: str = "eunomia", length: int = 32) -> str:
    """
    Generate a secure random API key.
    
    Args:
        prefix: Prefix for the API key (for identification)
        length: Length of random part (bytes)
    
    Returns:
        str: Generated API key
    
    Example:
        >>> api_key = generate_api_key()
        >>> print(api_key)
        eunomia_abc123def456...
    """
    random_part = secrets.token_urlsafe(length)
    api_key = f"{prefix}_{random_part}"
    
    logger.info("API key generated")
    return api_key


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against its stored hash.
    
    Args:
        api_key: API key to verify
        stored_hash: Hashed API key from database
    
    Returns:
        bool: True if API key matches hash
    """
    try:
        return pwd_context.verify(api_key, stored_hash)
    except Exception as e:
        logger.error(f"API key verification error: {e}")
        return False


# ============================================================================
# SECURITY VALIDATORS
# ============================================================================
def validate_password_strength(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength according to security policy.
    
    Requirements:
    - At least 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 digit
    - At least 1 special character
    
    Args:
        password: Password to validate
    
    Returns:
        tuple: (is_valid, error_message)
    
    Example:
        >>> is_valid, error = validate_password_strength("Weak")
        >>> print(is_valid, error)
        False "Password must be at least 8 characters long"
    """
    if not password:
        return False, "Password cannot be empty"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
    
    Returns:
        str: Sanitized filename
    
    Example:
        >>> sanitize_filename("../../etc/passwd")
        "passwd"
        >>> sanitize_filename("my document.pdf")
        "my_document.pdf"
    """
    import re
    import os
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace spaces and special chars with underscores
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = re.sub(r'[\s]+', '_', filename)
    
    # Remove leading dots (hidden files)
    filename = filename.lstrip('.')
    
    # Ensure filename is not empty
    if not filename:
        filename = f"file_{secrets.token_hex(8)}"
    
    return filename


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Useful for:
    - Password reset tokens
    - Email verification tokens
    - Session tokens
    
    Args:
        length: Length of token in bytes
    
    Returns:
        str: URL-safe random token
    
    Example:
        >>> token = generate_secure_token()
        >>> print(len(token))
        43  # Base64 encoding increases length
    """
    return secrets.token_urlsafe(length)


# ============================================================================
# RATE LIMITING HELPERS
# ============================================================================
def generate_rate_limit_key(identifier: str, endpoint: str) -> str:
    """
    Generate a Redis key for rate limiting.
    
    Args:
        identifier: User ID, IP address, or API key
        endpoint: API endpoint path
    
    Returns:
        str: Redis key for rate limiting
    
    Example:
        >>> key = generate_rate_limit_key("192.168.1.1", "/api/v1/documents")
        >>> print(key)
        eunomia:ratelimit:192.168.1.1:/api/v1/documents
    """
    return f"{settings.CACHE_KEY_PREFIX}:ratelimit:{identifier}:{endpoint}"


# ============================================================================
# GDPR COMPLIANCE HELPERS
# ============================================================================
def anonymize_email(email: str) -> str:
    """
    Anonymize email address for GDPR compliance.
    
    Keeps first 2 chars of username and domain for audit trails.
    
    Args:
        email: Email address to anonymize
    
    Returns:
        str: Anonymized email
    
    Example:
        >>> anonymized = anonymize_email("john.doe@example.com")
        >>> print(anonymized)
        jo***@ex***.com
    """
    try:
        username, domain = email.split('@')
        
        # Anonymize username (keep first 2 chars)
        if len(username) > 2:
            anon_username = username[:2] + '*' * (len(username) - 2)
        else:
            anon_username = '*' * len(username)
        
        # Anonymize domain (keep first 2 chars of domain name)
        domain_parts = domain.split('.')
        if len(domain_parts[0]) > 2:
            anon_domain_name = domain_parts[0][:2] + '*' * (len(domain_parts[0]) - 2)
        else:
            anon_domain_name = '*' * len(domain_parts[0])
        
        anon_domain = f"{anon_domain_name}.{'.'.join(domain_parts[1:])}"
        
        return f"{anon_username}@{anon_domain}"
    
    except Exception as e:
        logger.error(f"Error anonymizing email: {e}")
        return "***@***.***"


def generate_gdpr_deletion_token() -> str:
    """
    Generate a secure token for GDPR data deletion requests.
    
    Returns:
        str: Secure deletion token
    """
    return generate_secure_token(48)


# Export commonly used items
__all__ = [
    "hash_password",
    "verify_password",
    "needs_rehash",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "verify_token_type",
    "get_token_subject",
    "generate_api_key",
    "verify_api_key",
    "validate_password_strength",
    "sanitize_filename",
    "generate_secure_token",
    "generate_rate_limit_key",
    "anonymize_email",
    "generate_gdpr_deletion_token",
]