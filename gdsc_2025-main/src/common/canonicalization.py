# src/common/canonicalization.py

from __future__ import annotations
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

from src.config import (
    EDU_LEVELS, EDU_RANK, RANK_TO_EDU,
    KNOWN_JOB_DOMAINS, DOMAIN_ALIAS, WorkType,
)

# --- small helpers ------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _norm_lite(s: str) -> str:
    """Lowercase, strip, replace underscores with hyphens (for language + work-type)."""
    return s.strip().lower().replace("_", "-")

def _normalize_domain_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[()&,-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# --- language canonicalization ------------------------------------------------

_LANGUAGE_MAP: Dict[str, str] = {
    "pt": "pt-br", "pt-br": "pt-br", "pt br": "pt-br",
    "portuguese (br)": "pt-br", "portugues (br)": "pt-br",
    "português (br)": "pt-br", "portugues-br": "pt-br",
    "en": "en", "english": "en", "inglês": "en", "ingles": "en",
    "es": "es", "spanish": "es", "espanol": "es", "español": "es",
}

def canonicalize_language(code_or_name: Optional[str]) -> Optional[str]:
    if not code_or_name:
        return None
    s = _norm_lite(code_or_name)
    return _LANGUAGE_MAP.get(s, s)

def canonicalize_language_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    if value is None:
        return []
    tokens = value if isinstance(value, list) else re.split(r"[,;\s/]+", value.strip())
    out: List[str] = []
    for t in tokens:
        if not t:
            continue
        c = canonicalize_language(t)
        if c:
            out.append(c)
    return out

# --- work type canonicalization ----------------------------------------------

def canonicalize_work_type(raw: Optional[str]) -> WorkType:
    if not raw:
        return "unspecified"
    s = _norm_lite(raw)
    if any(w in s for w in ["remote", "home-office", "home office", "remoto"]):
        return "remote"
    if any(w in s for w in ["hybrid", "híbrido", "hibrido"]):
        return "hybrid"
    if any(w in s for w in ["onsite", "on-site", "in-person", "in person", "presencial", "field", "terminal"]):
        return "onsite"
    if s in {"either", "both"}:
        return "either"
    return "unspecified"

# --- skill level (BIA ladder) -------------------------------------------------

def to_bia_skill_level(level_text: Optional[str]) -> str:
    """
    Map free-text to one of SKILL_LEVELS (BIA ladder).
    """
    if not level_text:
        return "Basic"
    lt = level_text.strip().lower()
    if any(k in lt for k in ["basic", "beginner", "intro", "foundation", "foundational", "essentials"]):
        return "Basic"
    if any(k in lt for k in ["intermediate", "practitioner", "applied"]):
        return "Intermediate"
    if any(k in lt for k in ["advanced", "expert", "specialist", "master"]):
        return "Advanced"
    return "Basic"

# --- education normalization (pt-BR + EN synonyms) ---------------------------

# NOTE: These are moved verbatim from models.py to keep behavior identical.
_EDU_PT_SYNS = {
    "ensino fundamental": EDU_RANK["Ensino Fundamental"],
    "fundamental completo": EDU_RANK["Ensino Fundamental"],
    "ensino medio": EDU_RANK["Ensino Médio"],
    "colegio": EDU_RANK["Ensino Médio"],
    "segundo grau": EDU_RANK["Ensino Médio"],
    "tecnico": EDU_RANK["Técnico"],
    "curso tecnico": EDU_RANK["Técnico"],
    "tecnologo": EDU_RANK["Tecnólogo"],
    "graduacao": EDU_RANK["Graduação"],
    "graduacao completa": EDU_RANK["Graduação"],
    "superior completo": EDU_RANK["Graduação"],
    "bacharel": EDU_RANK["Bacharelado"],
    "bacharelado": EDU_RANK["Bacharelado"],
    "licenciatura": EDU_RANK["Licenciatura"],
    "pos graduacao": EDU_RANK["Pós-graduação"],
    "pos-graduacao": EDU_RANK["Pós-graduação"],
    "lato sensu": EDU_RANK["Pós-graduação"],
    "especializacao": EDU_RANK["Especialização"],
    "mba": EDU_RANK["MBA"],
    "mestrado": EDU_RANK["Mestrado"],
    "stricto sensu": EDU_RANK["Mestrado"],
    "doutorado": EDU_RANK["Doutorado"],
    "phd": EDU_RANK["Doutorado"],
}
_EDU_EN_SYNS = {
    "elementary school": EDU_RANK["Ensino Fundamental"],
    "high school": EDU_RANK["Ensino Médio"],
    "technical": EDU_RANK["Técnico"],
    "technologist": EDU_RANK["Tecnólogo"],
    "undergraduate": EDU_RANK["Graduação"],
    "bachelor": EDU_RANK["Bacharelado"],
    "licentiate": EDU_RANK["Licenciatura"],
    "postgraduate": EDU_RANK["Pós-graduação"],
    "specialization": EDU_RANK["Especialização"],
    "master": EDU_RANK["Mestrado"],
    "mba": EDU_RANK["MBA"],
    "doctorate": EDU_RANK["Doutorado"],
    "phd": EDU_RANK["Doutorado"],
}

def normalize_education_level(raw: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Return (rank, canonical_label) if recognized; else (None, None).
    """
    if not raw:
        return None, None
    t = _strip_accents(raw).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t).strip()

    # Exact match against official list
    for i, lvl in enumerate(EDU_LEVELS, start=1):
        if _strip_accents(lvl).lower() == t:
            return i, lvl

    # Synonyms (pt then en)
    for key, code in _EDU_PT_SYNS.items():
        if key in t:
            return code, RANK_TO_EDU[code]
    for key, code in _EDU_EN_SYNS.items():
        if key in t:
            return code, RANK_TO_EDU[code]

    return None, None

def canonicalize_edu_level(raw: Optional[str]) -> str:
    """
    Strictly return one of EDU_LEVELS (case-insensitive). Raises on unknown.
    """
    if not raw:
        raise ValueError("education_level is required and must be one of EDU_LEVELS.")
    if raw in EDU_LEVELS:
        return raw
    raw_lower = raw.strip().lower()
    for lvl in EDU_LEVELS:
        if lvl.lower() == raw_lower:
            return lvl
    raise ValueError(f"education_level '{raw}' is not in EDU_LEVELS: {EDU_LEVELS}")

# --- domains -----------------------------------------------------------------

def canonicalize_domain_or_raise(raw: Optional[str]) -> str:
    """
    Return the canonical domain via DOMAIN_ALIAS, or raise if not known.
    """
    if not raw:
        raise ValueError("top_domain is required and must be a known domain.")
    key = _normalize_domain_key(raw)
    dom = DOMAIN_ALIAS.get(key)
    if not dom:
        if raw in KNOWN_JOB_DOMAINS:
            return raw
        raise ValueError(
            "Unknown domain '{raw}'. Must be one of known domains. "
            f"Tip: check spelling/casing. Known examples start like: {', '.join(KNOWN_JOB_DOMAINS[:5])} ..."
        )
    return dom
