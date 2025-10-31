"""
Unit tests for security module
Run with: pytest tests/test_security.py -v
"""
import pytest
from datetime import timedelta
from jose import JWTError

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token_type,
    get_token_subject,
    validate_password_strength,
    sanitize_filename,
    anonymize_email,
    generate_secure_token,
)


class TestPasswordHashing:
    """Test password hashing and verification"""
    
    def test_hash_password(self):
        """Test password hashing"""
        password = "MySecurePassword123!"
        hashed = hash_password(password)
        
        assert hashed != password
        assert hashed.startswith("$2b$")
        assert len(hashed) == 60  # Bcrypt hash length
    
    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "MySecurePassword123!"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "MySecurePassword123!"
        hashed = hash_password(password)
        
        assert verify_password("WrongPassword", hashed) is False
    
    def test_hash_password_too_short(self):
        """Test that short passwords raise error"""
        with pytest.raises(ValueError):
            hash_password("short")


class TestJWTTokens:
    """Test JWT token creation and validation"""
    
    def test_create_access_token(self):
        """Test access token creation"""
        data = {"sub": "test@example.com", "role": "user"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 50
    
    def test_create_refresh_token(self):
        """Test refresh token creation"""
        data = {"sub": "test@example.com"}
        token = create_refresh_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 50
    
    def test_decode_token(self):
        """Test token decoding"""
        data = {"sub": "test@example.com", "role": "admin"}
        token = create_access_token(data)
        payload = decode_token(token)
        
        assert payload["sub"] == "test@example.com"
        assert payload["role"] == "admin"
        assert payload["type"] == "access"
    
    def test_verify_token_type_access(self):
        """Test token type verification for access token"""
        token = create_access_token({"sub": "test@example.com"})
        
        assert verify_token_type(token, "access") is True
        assert verify_token_type(token, "refresh") is False
    
    def test_verify_token_type_refresh(self):
        """Test token type verification for refresh token"""
        token = create_refresh_token({"sub": "test@example.com"})
        
        assert verify_token_type(token, "refresh") is True
        assert verify_token_type(token, "access") is False
    
    def test_get_token_subject(self):
        """Test extracting subject from token"""
        email = "test@example.com"
        token = create_access_token({"sub": email})
        
        subject = get_token_subject(token)
        assert subject == email
    
    def test_decode_invalid_token(self):
        """Test decoding invalid token raises error"""
        with pytest.raises(JWTError):
            decode_token("invalid.token.here")


class TestPasswordStrength:
    """Test password strength validation"""
    
    def test_valid_strong_password(self):
        """Test that strong password passes validation"""
        is_valid, error = validate_password_strength("MySecure123!")
        assert is_valid is True
        assert error is None
    
    def test_password_too_short(self):
        """Test that short password fails"""
        is_valid, error = validate_password_strength("Short1!")
        assert is_valid is False
        assert "8 characters" in error
    
    def test_password_no_uppercase(self):
        """Test that password without uppercase fails"""
        is_valid, error = validate_password_strength("mysecure123!")
        assert is_valid is False
        assert "uppercase" in error
    
    def test_password_no_lowercase(self):
        """Test that password without lowercase fails"""
        is_valid, error = validate_password_strength("MYSECURE123!")
        assert is_valid is False
        assert "lowercase" in error
    
    def test_password_no_digit(self):
        """Test that password without digit fails"""
        is_valid, error = validate_password_strength("MySecurePass!")
        assert is_valid is False
        assert "digit" in error
    
    def test_password_no_special(self):
        """Test that password without special char fails"""
        is_valid, error = validate_password_strength("MySecure123")
        assert is_valid is False
        assert "special" in error


class TestFilenameValidation:
    """Test filename sanitization"""
    
    def test_sanitize_normal_filename(self):
        """Test sanitizing normal filename"""
        result = sanitize_filename("document.pdf")
        assert result == "document.pdf"
    
    def test_sanitize_filename_with_spaces(self):
        """Test sanitizing filename with spaces"""
        result = sanitize_filename("my document.pdf")
        assert result == "my_document.pdf"
    
    def test_sanitize_path_traversal(self):
        """Test preventing path traversal"""
        result = sanitize_filename("../../etc/passwd")
        assert result == "passwd"
        assert ".." not in result
    
    def test_sanitize_special_chars(self):
        """Test removing special characters"""
        result = sanitize_filename("doc@#$%^&*.pdf")
        assert "@" not in result
        assert "#" not in result


class TestGDPRHelpers:
    """Test GDPR compliance helpers"""
    
    def test_anonymize_email(self):
        """Test email anonymization"""
        email = "john.doe@example.com"
        anonymized = anonymize_email(email)
        
        assert anonymized.startswith("jo")
        assert "*" in anonymized
        assert "@" in anonymized
        assert ".com" in anonymized
        assert email != anonymized
    
    def test_anonymize_short_email(self):
        """Test anonymizing short email"""
        email = "ab@cd.com"
        anonymized = anonymize_email(email)
        
        assert "*" in anonymized
        assert "@" in anonymized


class TestTokenGeneration:
    """Test secure token generation"""
    
    def test_generate_secure_token(self):
        """Test generating secure random token"""
        token1 = generate_secure_token()
        token2 = generate_secure_token()
        
        assert isinstance(token1, str)
        assert isinstance(token2, str)
        assert len(token1) > 30
        assert token1 != token2  # Should be random
    
    def test_generate_secure_token_custom_length(self):
        """Test generating token with custom length"""
        token = generate_secure_token(length=16)
        assert isinstance(token, str)
        # Base64 encoding increases length
        assert len(token) > 16