
import os
import csv
import html
import re
from typing import Dict, List, Literal, Optional

WorkType = Literal["onsite", "remote", "hybrid", "either", "unspecified"]

SKILL_LEVELS: List[str] = ["Basic", "Intermediate", "Advanced"]
SKILL_RANK = {"Basic": 1, "Intermediate": 2, "Advanced": 3}
RANK_TO_SKILL = {v: k for k, v in SKILL_RANK.items()}

EDU_LEVELS: List[str] = [
    "Ensino Fundamental",
    "Ensino Médio",
    "Técnico",
    "Tecnólogo",
    "Graduação",
    "Bacharelado",
    "Licenciatura",
    "Pós-graduação",
    "Especialização",
    "Mestrado",
    "MBA",
    "Doutorado",
]
EDU_RANK = {lvl: i + 1 for i, lvl in enumerate(EDU_LEVELS)}
RANK_TO_EDU = {v: k for k, v in EDU_RANK.items()}

_domains_path = os.path.join(
    os.path.dirname(__file__), "..", "processed_data", "allowed_domains.txt"
)

def _norm(s: str) -> str:
    s = s.lower().strip()
    import re

    s = re.sub(r"[()&,-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[()&,-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

with open(_domains_path, encoding="utf-8") as f:
    KNOWN_JOB_DOMAINS: List[str] = [line.strip() for line in f if line.strip()]

DOMAIN_ALIAS: Dict[str, str] = {}
for dom in KNOWN_JOB_DOMAINS:
    norm = _normalize(dom)
    DOMAIN_ALIAS[norm] = dom
    DOMAIN_ALIAS[norm.replace(" and ", " & ")] = dom
    DOMAIN_ALIAS[norm.replace(" & ", " and ")] = dom
    DOMAIN_ALIAS[norm.replace(" ", "")] = dom

CurrentFocus = Literal["jobs_and_training", "training_only", "awareness"]


ALLOWED_SKILLS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "processed_data", "domain_allowed_skills_2025-10-21_adjusted.csv"
)

def _coerce_domain_to_known(raw: str) -> Optional[str]:
    """
    Map a raw domain label from CSV into the canonical entry in KNOWN_JOB_DOMAINS.
    Uses DOMAIN_ALIAS (normalized) and exact matches.
    """
    if not raw:
        return None
    s = html.unescape(raw).strip()
    if s in KNOWN_JOB_DOMAINS:
        return s
    # try normalized lookup
    key = _norm(s)  # same normalizer used to build DOMAIN_ALIAS
    return DOMAIN_ALIAS.get(key)

def _load_allowed_skills_from_csv(csv_path: str) -> Dict[str, List[str]]:
    """
    Read domain->skill list from a CSV (columns: domain, skill_name).
    - Canonicalizes domain to KNOWN_JOB_DOMAINS via DOMAIN_ALIAS.
    - HTML-unescapes both domain and skill names.
    - De-duplicates; sorts alphabetically per-domain for stability.
    """
    out: Dict[str, List[str]] = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # allow alternate column name 'skill' just in case
        for row in reader:
            dom_raw = row.get("domain") or row.get("Domain") or ""
            skill_raw = row.get("skill_name") or row.get("skill") or row.get("Skill") or ""
            dom_can = _coerce_domain_to_known(dom_raw)
            if not dom_can:
                # silently skip rows whose domain we cannot map to a known domain
                continue
            skill = html.unescape(skill_raw).strip()
            if not skill:
                continue
            arr = out.setdefault(dom_can, [])
            if skill not in arr:
                arr.append(skill)
    # sort for determinism
    for d in out:
        out[d].sort()
    return out

# --- Public: allowed skills map ---
ALLOWED_SKILLS: Dict[str, List[str]] = _load_allowed_skills_from_csv(ALLOWED_SKILLS_CSV)

def get_allowed_skills_for(domain: str) -> List[str]:
    """Helper: safely fetch allowed skills for a domain (empty list if unknown)."""
    return ALLOWED_SKILLS.get(domain, [])
