"""
Skill canonicalization and catalog management for training extraction.
"""

from __future__ import annotations
import json
import re
import difflib
from pathlib import Path
from typing import List

# Stopwords and acronyms for canonicalization
STOPWORDS_LEVEL = {
    "basic", "intermediate", "advanced", "beginner", "intro", "introduction",
    "fundamentals", "essentials", "course", "training", "workshop", "overview",
    "masterclass", "bootcamp",
}
ACRONYMS = {
    "AI","ML","NLP","SQL","AWS","GCP","API","BI","ETL","CI","CD","KPI","OKR",
    "UX","UI","QA","RPA","GPU","IoT","API","DAX","C#","C++","HR","HSE","R&D",
}


def _normalize_for_canonical(name: str) -> str:
    s = re.sub(r"\(.*?\)", " ", name)           # remove parenthetical
    s = re.sub(r"[-_/]", " ", s)                # unify separators
    s = re.sub(r"\blevel\s*\d+\b", " ", s, flags=re.I)
    s = s.lower()
    tokens = [t for t in re.split(r"\s+", s) if t]
    tokens = [t for t in tokens if t not in STOPWORDS_LEVEL]
    tokens = [t for t in tokens if t not in {"basics", "advanced", "intermediate"}]
    return " ".join(tokens).strip()


def _smart_capitalize(skill: str) -> str:
    out: List[str] = []
    for tok in skill.split():
        if tok.upper() in ACRONYMS or (tok.isalpha() and len(tok) <= 4 and tok.isupper()):
            out.append(tok.upper())
        elif tok.lower() in {"and", "of", "for", "with", "in", "on", "to", "&"}:
            out.append(tok.lower())
        else:
            out.append(tok.capitalize())
    return " ".join(out)


def canonicalize_skill(raw_name: str) -> str:
    """
    Normalize a skill to a canonical textual form (level-agnostic).
    """
    cand = _normalize_for_canonical(raw_name)
    if not cand:
        cand = raw_name.strip()
    return _smart_capitalize(cand)


def load_skill_catalog(path: Path) -> list[str]:
    if path.exists():
        try:
            return sorted(set(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            return []
    return []


def save_skill_catalog(path: Path, skills: list[str]) -> None:
    skills_sorted = sorted(set(skills))
    path.write_text(
        json.dumps(skills_sorted, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def map_to_catalog(canonical_skill: str, catalog: list[str], cutoff: float = 0.92) -> str:
    """
    Return closest catalog skill above similarity cutoff, else the input canonical_skill.
    """
    if not catalog:
        return canonical_skill
    matches = difflib.get_close_matches(canonical_skill, catalog, n=1, cutoff=cutoff)
    return matches[0] if matches else canonical_skill
