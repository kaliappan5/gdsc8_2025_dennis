"""
Validation and data processing functions for PersonaInfo extraction.
Extracted from PersonaInfo_Extraction.ipynb notebook.
"""

from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from .models import PersonaInfo
from src.config import EDU_LEVELS, ALLOWED_SKILLS

# Focus constants for routing
FOCUS_JOB = "jobs+trainings"
FOCUS_TRAINING = "training_only"
FOCUS_AWARENESS = "awareness"

# Level mapping for technical skills
LEVEL_MAP = {
    "basic": "Basic",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
    "1": "Basic", "2": "Intermediate", "3": "Advanced",
    "junior": "Basic", "mid": "Intermediate", "senior": "Advanced",
}

# Language code mapping
LANG_CODE_MAP = {
    # Portuguese
    "portuguese": "pt-br", "pt": "pt-br", "pt-br": "pt-br", "ptbr": "pt-br", "br portuguese": "pt-br",
    "brazilian portuguese": "pt-br", "português": "pt-br", "português do brasil": "pt-br",
    # English
    "english": "en", "en": "en", "inglês": "en",
    # Spanish (use if needed)
    "spanish": "es", "español": "es", "es": "es",
    # Add more as needed...
}


def normalize_focus(text: Optional[str]) -> Optional[str]:
    """Map natural phrases to canonical routing tokens. Keep user-facing tone elsewhere."""
    if not text:
        return None
    t = (text or "").strip().lower()
    if any(k in t for k in ["job", "full-time", "full time", "hire", "hiring", "find a role", "finding a job"]):
        return FOCUS_JOB
    if "training_only" in t or "training only" in t or "upskill" in t or "train" in t:
        return FOCUS_TRAINING
    if "awareness" in t or "explore" in t or "just looking" in t:
        return FOCUS_AWARENESS
    if t in {FOCUS_JOB, FOCUS_TRAINING, FOCUS_AWARENESS}:
        return t
    return None


def _coerce_level(s: Any) -> str:
    """Coerce skill level to Basic/Intermediate/Advanced."""
    if not s:
        return "Basic"
    t = str(s).strip().lower()
    return LEVEL_MAP.get(t, "Basic")


def _normalize_skill_name(s: str) -> str:
    """Normalize skill name for comparison."""
    return " ".join(str(s).strip().lower().split())


def _dedup_preserve_order(items: List[Any]) -> List[Any]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _filter_to_allowed_skills(
    domain: str,
    skills_in: List[str],
    allowed_skills: Dict[str, List[str]] = None,
) -> List[str]:
    """Return only skills present in ALLOWED_SKILLS[domain]; empty list if domain unknown."""
    if allowed_skills is None:
        allowed_skills = ALLOWED_SKILLS
    allowed = allowed_skills.get(domain, [])
    if not allowed:
        return []
    allowed_norm = {_normalize_skill_name(x): x for x in allowed}
    out: List[str] = []
    for raw in skills_in or []:
        norm = _normalize_skill_name(raw)
        canon = allowed_norm.get(norm)
        if canon:
            out.append(canon)
    return _dedup_preserve_order(out)


def _filter_technical_skills_to_allowed(
    domain: str,
    tech_in: List[Dict[str, Any]],
    allowed_skills: Dict[str, List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Keep only items whose 'name' in ALLOWED_SKILLS[domain]; coerce 'level' to Basic/Intermediate/Advanced.
    Accepts either {'name':..., 'level':...} or {"Skill": 2} shapes. Empty if domain unknown.
    """
    if allowed_skills is None:
        allowed_skills = ALLOWED_SKILLS
    allowed = allowed_skills.get(domain, [])
    if not allowed:
        return []
    allowed_norm = {_normalize_skill_name(x): x for x in allowed}
    tmp: List[Dict[str, str]] = []
    for item in tech_in or []:
        if isinstance(item, dict) and "name" in item:
            name = str(item["name"]).strip()
            level = _coerce_level(item.get("level"))
        elif isinstance(item, dict) and len(item) == 1:
            # shape like {"Risk Documentation": 2}
            k, v = next(iter(item.items()))
            name, level = str(k), _coerce_level(v)
        else:
            # ignore strings here; technical_skills should be dicts after extraction
            continue
        norm = _normalize_skill_name(name)
        canon = allowed_norm.get(norm)
        if canon:
            tmp.append({"name": canon, "level": level})
    # de-dup by name (keep last or first; here we keep last)
    by_name: Dict[str, Dict[str, str]] = {}
    for d in tmp:
        by_name[d["name"]] = d
    return list(by_name.values())


def _canon_work_type(v: Any) -> str:
    """Canonicalize preferred work type values."""
    if v is None or v == "":
        return "unspecified"
    t = str(v).strip().lower()
    t = {"on-site": "onsite", "on site": "onsite"}.get(t, t)
    return t if t in {"onsite", "remote", "hybrid", "either", "unspecified"} else "unspecified"


def _complete_personainfo_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict with all PersonaInfo fields present and valid defaults applied."""
    try:
        normalized = PersonaInfo(**(d or {})).model_dump()
    except Exception:
        normalized = PersonaInfo().model_dump()
    default_full = PersonaInfo().model_dump()
    for k, v in default_full.items():
        if k not in normalized:
            normalized[k] = v
    return normalized


def to_language_codes(items: list[str]) -> list[str]:
    """
    Map free-text languages to canonical short codes. Dedup while preserving order.
    Unknown entries are kept as-is if they already look like a short code, otherwise dropped.
    """
    out, seen = [], set()
    for raw in items or []:
        s = (raw or "").strip()
        if not s:
            continue
        key = s.lower()
        code = LANG_CODE_MAP.get(key)
        if code is None:
            # Keep short-looking codes such as "en-GB" or "de" if user typed them.
            if 2 <= len(s) <= 5 or "-" in s:
                code = s.lower()
            else:
                # Skip long names we don't recognize, to avoid noisy values.
                continue
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def sanitize_domain(val: str | None, known_domains: list[str]) -> str:
    """
    Accept domain only if it matches exactly one of known_domains.
    Anything else returns 'unspecified'.
    """
    v = (val or "").strip()
    return v if v in known_domains else "unspecified"


def _postprocess_persona(
    profile: Dict[str, Any],
    domain: str,
    allowed_skills: Dict[str, List[str]] = None,
    edu_levels: List[str] = None,
    *,
    intent_in_state: Optional[str] = None,
    focus_in_state: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize and strictly filter a PersonaInfo-shaped dict:
      - Enforce allowed-skills-only for training_motivation and technical_skills.
      - Normalize education_level against EDU_LEVELS (localized).
      - Normalize current_focus to canonical routing tokens.
      - Coerce levels, languages, roles, work type, years_experience.
      - Overlay 'intent' and 'current_focus' from state if provided.
      - Ensure COMPLETE PersonaInfo JSON is returned.

    If domain is empty/unknown, both technical_skills and training_motivation are set to [] (strict rule).
    """
    if allowed_skills is None:
        allowed_skills = ALLOWED_SKILLS
    if edu_levels is None:
        edu_levels = EDU_LEVELS
        
    p = dict(profile or {})

    # Prefer state overlays if provided
    if intent_in_state:
        p["intent"] = intent_in_state
    if focus_in_state:
        p["current_focus"] = focus_in_state

    # Domain resolution
    dom = (domain or p.get("top_domain") or "").strip()

    # current_focus canonicalization (routing tokens only)
    p["current_focus"] = normalize_focus(p.get("current_focus")) or FOCUS_AWARENESS
    
    # education_level must match one of EDU_LEVELS
    edu = (p.get("education_level") or "").strip()
    if edu:
        if edu not in edu_levels:
            match = next((e for e in edu_levels if e.lower() == edu.lower()), None)
            p["education_level"] = match if match else ""
    else:
        p["education_level"] = ""

    # preferred_work_type canonical
    p["preferred_work_type"] = _canon_work_type(p.get("preferred_work_type"))

    # years_experience numeric or None
    y = p.get("years_experience")
    if y is not None:
        try:
            p["years_experience"] = float(y)
        except Exception:
            p["years_experience"] = None

    # languages as list[str]
    langs = p.get("languages") or []
    if isinstance(langs, str):
        langs = [x.strip() for x in langs.replace(";", ",").split(",") if x.strip()]
    p["languages"] = to_language_codes([str(x).strip() for x in langs if str(x).strip()])

    # desired_job_roles as list[str]
    roles = p.get("desired_job_roles") or []
    if isinstance(roles, str):
        roles = [x.strip() for x in roles.split(",") if x.strip()]
    p["desired_job_roles"] = _dedup_preserve_order([str(x).strip() for x in roles if str(x).strip()])

    # technical_skills normalization + STRICT FILTER to domain's allowed skills
    raw_ts = p.get("technical_skills") or []
    p["technical_skills"] = _filter_technical_skills_to_allowed(dom, raw_ts, allowed_skills) if dom else []

    # training_motivation normalization + STRICT FILTER to domain's allowed skills
    tm = p.get("training_motivation") or []
    if isinstance(tm, str):
        tm = [x.strip() for x in tm.split(",") if x.strip()]
    tm = [str(x).strip() for x in tm if str(x).strip()]
    p["training_motivation"] = _filter_to_allowed_skills(dom, tm, allowed_skills) if dom else []

    # location_city (keep "unknown" if not provided)
    p["location_city"] = (p.get("location_city") or "unknown").strip() or "unknown"

    # open_to_relocate boolean default False (only relevant for job path; harmless to keep default)
    p["open_to_relocate"] = bool(p.get("open_to_relocate") or False)

    # Final schema completion & validation
    return _complete_personainfo_dict(p)