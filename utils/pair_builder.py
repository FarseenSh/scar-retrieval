"""
Contrastive pair construction utilities for SCAR data pipeline.
Builds query-positive-hard_negative triples for InfoNCE training.
"""

import random
from typing import Optional


def format_query(title: str, description: str, severity: str) -> str:
    """Format a finding into the standard query format.

    Output: "{SEVERITY} severity: {title}. {description_300char}"
    """
    severity_upper = severity.strip().upper()
    if severity_upper not in ("HIGH", "MEDIUM"):
        severity_upper = "MEDIUM"  # Default for ambiguous

    # Clean title
    title = title.strip().rstrip('.')

    # Truncate description
    desc = description.strip()
    if len(desc) > 300:
        cut = desc[:300]
        last_space = cut.rfind(' ')
        if last_space > 210:
            cut = cut[:last_space]
        desc = cut.strip() + '...'

    return f"{severity_upper} severity: {title}. {desc}"


def build_hard_negative_same_protocol(
    findings: list[dict],
    current_idx: int
) -> Optional[str]:
    """Find a hard negative from the same protocol but different vulnerability.

    Args:
        findings: List of finding dicts with keys: protocol, vuln_type, code
        current_idx: Index of the current finding

    Returns:
        Code string of the hard negative, or None if no suitable candidate.
    """
    current = findings[current_idx]
    candidates = [
        f for i, f in enumerate(findings)
        if i != current_idx
        and f.get('protocol') == current.get('protocol')
        and f.get('vuln_type') != current.get('vuln_type')
        and f.get('code', '').strip()
    ]
    if candidates:
        return random.choice(candidates)['code']
    return None


def build_hard_negative_same_report(
    findings: list[dict],
    current_idx: int
) -> Optional[str]:
    """Find a hard negative from the same audit report but different finding.

    Args:
        findings: List of finding dicts with keys: report_name, code, title
        current_idx: Index of the current finding

    Returns:
        Code string of the hard negative, or None if no suitable candidate.
    """
    current = findings[current_idx]
    candidates = [
        f for i, f in enumerate(findings)
        if i != current_idx
        and f.get('report_name') == current.get('report_name')
        and f.get('title') != current.get('title')
        and f.get('code', '').strip()
    ]
    if candidates:
        return random.choice(candidates)['code']
    return None


def build_hard_negative_same_firm(
    findings: list[dict],
    current_idx: int
) -> Optional[str]:
    """Find a hard negative from the same audit firm but different report.

    Args:
        findings: List of finding dicts with keys: firm, vuln_type, code
        current_idx: Index of the current finding

    Returns:
        Code string of the hard negative, or None.
    """
    current = findings[current_idx]
    candidates = [
        f for i, f in enumerate(findings)
        if i != current_idx
        and f.get('firm') == current.get('firm')
        and f.get('vuln_type') != current.get('vuln_type')
        and f.get('code', '').strip()
    ]
    if candidates:
        return random.choice(candidates)['code']
    return None


def build_random_negative(
    all_codes: list[str],
    exclude_code: str
) -> Optional[str]:
    """Pick a random code snippet as fallback negative.

    Args:
        all_codes: List of all available code snippets
        exclude_code: The positive code to exclude

    Returns:
        A random code string different from the positive.
    """
    candidates = [c for c in all_codes if c != exclude_code and c.strip()]
    if candidates:
        return random.choice(candidates)
    return None


def classify_negative_type(
    negative_code: Optional[str],
    strategy_used: str
) -> str:
    """Return the negative type label for the dataset schema."""
    if negative_code is None:
        return "none"
    return strategy_used  # "same_protocol_diff_vuln", "same_report_diff_finding", "same_firm_safe", "random"


def build_pair_record(
    query: str,
    positive_code: str,
    hard_negative_code: Optional[str],
    negative_type: str,
    source: str,
    severity: str,
    vuln_type: str = "",
    quality_tier: int = 2
) -> dict:
    """Build a single contrastive pair record matching the HF dataset schema."""
    return {
        "query": query,
        "positive": positive_code,
        "hard_negative": hard_negative_code or "",
        "negative_type": negative_type,
        "source": source,
        "severity": severity.upper(),
        "vuln_type": vuln_type,
        "quality_tier": quality_tier,
    }
