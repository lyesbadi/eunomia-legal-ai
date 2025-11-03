"""
EUNOMIA Legal AI Platform - Formatters
Formatting utilities for various data types
"""
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta, date
from decimal import Decimal
import json

import logging


logger = logging.getLogger(__name__)


# ============================================================================
# DATE & TIME FORMATTERS
# ============================================================================
def format_datetime(
    dt: datetime,
    format_type: str = "iso"
) -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object
        format_type: Format type ('iso', 'human', 'short', 'date_only')
        
    Returns:
        Formatted datetime string
        
    Example:
        >>> dt = datetime(2025, 11, 1, 14, 30)
        >>> format_datetime(dt, "iso")
        '2025-11-01T14:30:00'
        >>> format_datetime(dt, "human")
        'November 1, 2025 at 2:30 PM'
        >>> format_datetime(dt, "short")
        '01/11/2025 14:30'
    """
    if format_type == "iso":
        return dt.isoformat()
    elif format_type == "human":
        return dt.strftime("%B %d, %Y at %I:%M %p")
    elif format_type == "short":
        return dt.strftime("%d/%m/%Y %H:%M")
    elif format_type == "date_only":
        return dt.strftime("%Y-%m-%d")
    else:
        return dt.isoformat()


def format_relative_time(dt: datetime) -> str:
    """
    Format datetime as relative time (e.g., "2 hours ago").
    
    Args:
        dt: Datetime object
        
    Returns:
        Relative time string
        
    Example:
        >>> now = datetime.utcnow()
        >>> past = now - timedelta(hours=2)
        >>> format_relative_time(past)
        '2 hours ago'
    """
    now = datetime.utcnow()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif seconds < 2592000:
        weeks = int(seconds / 604800)
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif seconds < 31536000:
        months = int(seconds / 2592000)
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = int(seconds / 31536000)
        return f"{years} year{'s' if years > 1 else ''} ago"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration
        
    Example:
        >>> format_duration(125.5)
        '2m 5s'
        >>> format_duration(3665)
        '1h 1m 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {secs}s"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    return f"{hours}h {mins}m {secs}s"


# ============================================================================
# NUMBER FORMATTERS
# ============================================================================
def format_number(
    number: float,
    decimals: int = 2,
    thousands_sep: str = ",",
    decimal_sep: str = "."
) -> str:
    """
    Format number with separators.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        thousands_sep: Thousands separator
        decimal_sep: Decimal separator
        
    Returns:
        Formatted number string
        
    Example:
        >>> format_number(1234567.89)
        '1,234,567.89'
        >>> format_number(1234567.89, thousands_sep=" ", decimal_sep=",")
        '1 234 567,89'
    """
    # Format with decimals
    formatted = f"{number:,.{decimals}f}"
    
    # Replace default separators if needed
    if thousands_sep != ",":
        formatted = formatted.replace(",", "TEMP")
        formatted = formatted.replace(".", ",")
        formatted = formatted.replace("TEMP", thousands_sep)
    
    if decimal_sep != ".":
        formatted = formatted.replace(".", decimal_sep)
    
    return formatted


def format_percentage(
    value: float,
    decimals: int = 1,
    include_sign: bool = True
) -> str:
    """
    Format number as percentage.
    
    Args:
        value: Value (0.0 to 1.0 or 0 to 100)
        decimals: Decimal places
        include_sign: Include % sign
        
    Returns:
        Formatted percentage
        
    Example:
        >>> format_percentage(0.8567)
        '85.7%'
        >>> format_percentage(85.67, include_sign=False)
        '85.7'
    """
    # If value is between 0-1, convert to percentage
    if 0 <= value <= 1:
        value = value * 100
    
    formatted = f"{value:.{decimals}f}"
    
    if include_sign:
        formatted += "%"
    
    return formatted


def format_file_size(size_bytes: int, precision: int = 1) -> str:
    """
    Format file size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        precision: Decimal precision
        
    Returns:
        Formatted size string
        
    Example:
        >>> format_file_size(1024)
        '1.0 KB'
        >>> format_file_size(2_500_000)
        '2.4 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.{precision}f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.{precision}f} EB"


def format_currency(
    amount: float,
    currency: str = "EUR",
    locale: str = "fr_FR"
) -> str:
    """
    Format amount as currency.
    
    Args:
        amount: Amount to format
        currency: Currency code (EUR, USD, GBP)
        locale: Locale for formatting
        
    Returns:
        Formatted currency string
        
    Example:
        >>> format_currency(1234.56, "EUR")
        '1 234,56 €'
        >>> format_currency(1234.56, "USD", "en_US")
        '$1,234.56'
    """
    if locale.startswith("fr"):
        # French formatting
        formatted = format_number(amount, decimals=2, thousands_sep=" ", decimal_sep=",")
        
        if currency == "EUR":
            return f"{formatted} €"
        elif currency == "USD":
            return f"{formatted} $"
        elif currency == "GBP":
            return f"{formatted} £"
    else:
        # English formatting
        formatted = format_number(amount, decimals=2, thousands_sep=",", decimal_sep=".")
        
        if currency == "EUR":
            return f"€{formatted}"
        elif currency == "USD":
            return f"${formatted}"
        elif currency == "GBP":
            return f"£{formatted}"
    
    return f"{formatted} {currency}"


# ============================================================================
# STRING FORMATTERS
# ============================================================================
def format_phone_number(phone: str, country_code: str = "FR") -> str:
    """
    Format phone number for display.
    
    Args:
        phone: Phone number
        country_code: Country code
        
    Returns:
        Formatted phone number
        
    Example:
        >>> format_phone_number("0612345678", "FR")
        '06 12 34 56 78'
        >>> format_phone_number("+33612345678", "FR")
        '+33 6 12 34 56 78'
    """
    # Remove all non-digit characters except +
    clean = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    if country_code == "FR":
        if clean.startswith('+33'):
            # +33 6 12 34 56 78
            return f"+33 {clean[3]} {clean[4:6]} {clean[6:8]} {clean[8:10]} {clean[10:12]}"
        elif clean.startswith('0'):
            # 06 12 34 56 78
            return f"{clean[0:2]} {clean[2:4]} {clean[4:6]} {clean[6:8]} {clean[8:10]}"
    
    return phone


def truncate_string(
    text: str,
    max_length: int = 50,
    suffix: str = "..."
) -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
        
    Example:
        >>> truncate_string("This is a very long text", max_length=10)
        'This is...'
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_list(
    items: List[str],
    separator: str = ", ",
    last_separator: str = " and "
) -> str:
    """
    Format list of items to readable string.
    
    Args:
        items: List of items
        separator: Separator between items
        last_separator: Separator before last item
        
    Returns:
        Formatted string
        
    Example:
        >>> format_list(["apple", "banana", "cherry"])
        'apple, banana and cherry'
    """
    if not items:
        return ""
    
    if len(items) == 1:
        return items[0]
    
    if len(items) == 2:
        return f"{items[0]}{last_separator}{items[1]}"
    
    return f"{separator.join(items[:-1])}{last_separator}{items[-1]}"


# ============================================================================
# JSON FORMATTERS
# ============================================================================
def format_json(
    data: Any,
    indent: int = 2,
    sort_keys: bool = True
) -> str:
    """
    Format data as pretty JSON.
    
    Args:
        data: Data to format
        indent: Indentation spaces
        sort_keys: Sort dictionary keys
        
    Returns:
        Formatted JSON string
        
    Example:
        >>> data = {"name": "John", "age": 30}
        >>> print(format_json(data))
        {
          "age": 30,
          "name": "John"
        }
    """
    return json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)


def format_dict_for_display(
    data: Dict[str, Any],
    key_width: int = 20
) -> str:
    """
    Format dictionary for console display.
    
    Args:
        data: Dictionary to format
        key_width: Width for key column
        
    Returns:
        Formatted string
        
    Example:
        >>> data = {"name": "John", "age": 30}
        >>> print(format_dict_for_display(data))
        name                : John
        age                 : 30
    """
    lines = []
    for key, value in data.items():
        key_str = str(key).ljust(key_width)
        lines.append(f"{key_str}: {value}")
    
    return "\n".join(lines)


# ============================================================================
# TEMPLATE FORMATTERS
# ============================================================================
def format_template(
    template: str,
    **kwargs
) -> str:
    """
    Simple template formatting with keyword arguments.
    
    Args:
        template: Template string with {placeholders}
        **kwargs: Values to fill
        
    Returns:
        Formatted string
        
    Example:
        >>> template = "Hello {name}, you are {age} years old"
        >>> format_template(template, name="John", age=30)
        'Hello John, you are 30 years old'
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing template variable: {e}")
        return template


def format_email_subject(
    subject_type: str,
    **kwargs
) -> str:
    """
    Format email subject based on type.
    
    Args:
        subject_type: Type of email
        **kwargs: Template variables
        
    Returns:
        Formatted subject
        
    Example:
        >>> format_email_subject("welcome", name="John")
        'Welcome to EUNOMIA, John!'
    """
    templates = {
        "welcome": "Welcome to EUNOMIA, {name}!",
        "password_reset": "EUNOMIA - Password Reset Request",
        "verification": "EUNOMIA - Verify Your Email",
        "analysis_complete": "Your document analysis is ready",
        "document_uploaded": "Document '{filename}' uploaded successfully"
    }
    
    template = templates.get(subject_type, "EUNOMIA - Notification")
    return format_template(template, **kwargs)


# ============================================================================
# SPECIAL FORMATTERS
# ============================================================================
def format_initials(name: str) -> str:
    """
    Get initials from full name.
    
    Args:
        name: Full name
        
    Returns:
        Initials (max 2 letters)
        
    Example:
        >>> format_initials("John Doe")
        'JD'
        >>> format_initials("Marie-Claire Dubois")
        'MD'
    """
    parts = name.split()
    if not parts:
        return "?"
    
    if len(parts) == 1:
        return parts[0][0].upper()
    
    return (parts[0][0] + parts[-1][0]).upper()


def format_slug(text: str) -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to convert
        
    Returns:
        URL-friendly slug
        
    Example:
        >>> format_slug("Hello World! This is a test.")
        'hello-world-this-is-a-test'
    """
    import unicodedata
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces with hyphens
    text = re.sub(r'\s+', '-', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\-]', '', text)
    
    # Remove multiple consecutive hyphens
    text = re.sub(r'\-+', '-', text)
    
    # Strip hyphens from ends
    text = text.strip('-')
    
    return text


import re