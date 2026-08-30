import re

def validate_and_clean_text(text: str, min_chars: int = 10) -> str:
    """
    Validates input type and applies non-destructive cleaning.
    Preserves casing, punctuation, and structural markers.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize excessive whitespace while preserving raw character structure
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    if not cleaned_text:
        raise ValueError("Text input cannot be empty or pure whitespace.")

    if len(cleaned_text) < min_chars:
        raise ValueError(f"Text is too short for reliable detection (minimum {min_chars} characters).")

    return cleaned_text