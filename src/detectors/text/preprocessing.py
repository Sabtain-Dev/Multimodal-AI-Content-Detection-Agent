import re
from typing import List

def validate_and_clean_text(text: str) -> str:
    """Validates string input and normalizes excessive whitespace."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    if not cleaned_text:
        raise ValueError("Text input cannot be empty or pure whitespace.")

    return cleaned_text

def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int = 480,
    overlap: int = 32
) -> List[str]:
    """
    Splits input text into overlapping chunks based on tokenizer tokens.
    Reserves capacity for model special tokens (e.g., [CLS], [SEP]).
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly smaller than chunk_size.")

    # Encode without adding special tokens to perform clean sliding window
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return [text]

    # If text fits within a single chunk
    if len(token_ids) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(token_ids):
        end = start + chunk_size
        chunk_token_ids = token_ids[start:end]
        
        # Decode tokens back into text chunk
        decoded_chunk = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
        chunks.append(decoded_chunk.strip())

        if end >= len(token_ids):
            break

        start += step

    return chunks