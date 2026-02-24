"""
Shared Solidity processing utilities for SCAR data pipeline.
Used across all notebooks for consistent normalization, hashing, and filtering.
"""

import re
import hashlib


def normalize_solidity(source: str) -> str:
    """Strip comments and collapse whitespace for dedup hashing.

    Removes:
    - Single-line comments (// ...)
    - Multi-line comments (/* ... */)
    - Leading/trailing whitespace per line
    - Blank lines
    - SPDX license identifiers
    - Pragma statements (version-dependent, not semantic)
    """
    # Remove multi-line comments
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    # Remove single-line comments
    source = re.sub(r'//.*$', '', source, flags=re.MULTILINE)
    # Remove SPDX identifiers
    source = re.sub(r'SPDX-License-Identifier:.*$', '', source, flags=re.MULTILINE)
    # Remove pragma statements
    source = re.sub(r'pragma\s+solidity.*?;', '', source)
    # Collapse whitespace: strip each line, remove blanks
    lines = [line.strip() for line in source.splitlines()]
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def hash_contract(source: str) -> str:
    """SHA256 hash of normalized source for dedup."""
    normalized = normalize_solidity(source)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def count_lines(source: str) -> int:
    """Count non-empty lines in raw source."""
    return len([line for line in source.splitlines() if line.strip()])


def is_import_only(source: str, threshold: float = 0.8) -> bool:
    """Check if >threshold of non-empty lines are import statements.

    Used to filter OpenZeppelin import-only wrapper files that are
    boilerplate, not meaningful signal for SAE training.
    """
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines:
        return True
    import_lines = [line for line in lines if line.startswith('import ') or line.startswith('import"')]
    return len(import_lines) / len(lines) > threshold


def passes_quality_filter(source: str, min_lines: int = 50) -> bool:
    """Check if a contract passes all quality filters.

    Returns True if contract should be KEPT.
    Filters:
    - Must have >= min_lines non-empty lines
    - Must not be import-only (>80% imports)
    """
    if count_lines(source) < min_lines:
        return False
    if is_import_only(source):
        return False
    return True


def extract_solidity_from_file(filepath: str) -> str | None:
    """Read a .sol file, return contents or None if unreadable."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception:
        return None


def truncate_description(text: str, max_chars: int = 300) -> str:
    """Truncate description to max_chars, breaking at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Break at last space to avoid cutting mid-word
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:  # Only break at space if not too far back
        truncated = truncated[:last_space]
    return truncated.strip() + '...'
