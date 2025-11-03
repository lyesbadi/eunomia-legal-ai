"""
EUNOMIA Legal AI Platform - Utils Package
Utility functions and helpers for the application
"""
from app.utils.files import (
    get_file_extension,
    get_mime_type,
    is_allowed_file_type,
    validate_file_size,
    calculate_file_hash,
    calculate_stream_hash,
    get_user_upload_dir,
    generate_unique_filename,
    save_upload_file,
    delete_file,
    get_file_size,
    format_file_size,
    get_file_info,
    cleanup_old_files,
    get_directory_size
)

from app.utils.text import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text,
    clean_text,
    truncate_text,
    chunk_text,
    count_words,
    extract_sentences,
    calculate_readability_score,
    detect_language_simple
)

from app.utils.validators import (
    is_valid_email,
    is_disposable_email,
    validate_password_strength,
    is_common_password,
    validate_filename,
    validate_file_extension,
    contains_sql_injection,
    contains_xss,
    sanitize_input,
    validate_date_range,
    validate_pagination,
    validate_uuid,
    validate_api_key_format,
    validate_phone_number,
    validate_batch
)

from app.utils.formatters import (
    format_datetime,
    format_relative_time,
    format_duration,
    format_number,
    format_percentage,
    format_file_size as format_file_size_formatter,  # Alias to avoid conflict
    format_currency,
    format_phone_number,
    truncate_string,
    format_list,
    format_json,
    format_dict_for_display,
    format_template,
    format_email_subject,
    format_initials,
    format_slug
)


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # Files utilities
    "get_file_extension",
    "get_mime_type",
    "is_allowed_file_type",
    "validate_file_size",
    "calculate_file_hash",
    "calculate_stream_hash",
    "get_user_upload_dir",
    "generate_unique_filename",
    "save_upload_file",
    "delete_file",
    "get_file_size",
    "format_file_size",
    "get_file_info",
    "cleanup_old_files",
    "get_directory_size",
    
    # Text utilities
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_txt",
    "extract_text",
    "clean_text",
    "truncate_text",
    "chunk_text",
    "count_words",
    "extract_sentences",
    "calculate_readability_score",
    "detect_language_simple",
    
    # Validators
    "is_valid_email",
    "is_disposable_email",
    "validate_password_strength",
    "is_common_password",
    "validate_filename",
    "validate_file_extension",
    "contains_sql_injection",
    "contains_xss",
    "sanitize_input",
    "validate_date_range",
    "validate_pagination",
    "validate_uuid",
    "validate_api_key_format",
    "validate_phone_number",
    "validate_batch",
    
    # Formatters
    "format_datetime",
    "format_relative_time",
    "format_duration",
    "format_number",
    "format_percentage",
    "format_file_size_formatter",
    "format_currency",
    "format_phone_number",
    "truncate_string",
    "format_list",
    "format_json",
    "format_dict_for_display",
    "format_template",
    "format_email_subject",
    "format_initials",
    "format_slug",
]