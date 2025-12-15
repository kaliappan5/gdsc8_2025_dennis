# src/modules/knowledge_graph/domain_utils.py
from typing import Optional
from src.config import DOMAIN_ALIAS
from .utils import _norm_key, _norm_text

def canonical_domain(value: Optional[str]) -> str:
    """
    Return canonical domain as present in KNOWN_JOB_DOMAINS if resolvable via DOMAIN_ALIAS.
    Else return the stripped text (non-empty) or "".
    """
    if not value:
        return ""
    key = _norm_key(value)
    return DOMAIN_ALIAS.get(key, _norm_text(value))

def domains_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return canonical_domain(a) == canonical_domain(b)
