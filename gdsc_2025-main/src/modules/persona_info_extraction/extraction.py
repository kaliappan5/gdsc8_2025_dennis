"""
Profile extraction and merging functions for persona info extraction.
Extracted from PersonaInfo_Extraction.ipynb notebook.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from .models import PersonaInfo
from .validation import (
    _complete_personainfo_dict,
    _coerce_level,
    _dedup_preserve_order,
    _normalize_skill_name,
    normalize_focus,
    sanitize_domain,
    to_language_codes,
    FOCUS_AWARENESS
)


def _filter_to_allowed_skills(
    skill_list: List[str], 
    domain_allowed: List[str],
    normalize_fn = _normalize_skill_name,
    strict: bool = True
) -> List[str]:
    """
    Filters skills to only those allowed for the domain.
    If strict=True and domain_allowed is empty, return empty list.
    """
    if strict and not domain_allowed:
        return []
    
    if not skill_list:
        return []
    
    # Normalize allowed skills for comparison
    allowed_normalized = {normalize_fn(s): s for s in domain_allowed}
    
    result = []
    for skill in skill_list:
        normalized = normalize_fn(skill)
        if normalized in allowed_normalized:
            result.append(allowed_normalized[normalized])
    
    return _dedup_preserve_order(result)


def _extract_profile(extractor_agent, transcript: str) -> dict:
    """
    Structured extraction via PersonaInfo; fallback to inline JSON; then complete.
    """
    try:
        obj = extractor_agent.structured_output(
            PersonaInfo,
            prompt=(
                "Transcript between Assistant and User follows. "
                "Return ONE PersonaInfo object with ALL fields present and respecting the constraints.. "
                "If unknown, use empty string for text, null for numbers, [] for lists. No prose.\n\n"
                + transcript
            ),
        )
        try:
            profile = obj.model_dump()
        except AttributeError:
            profile = dict(obj) if isinstance(obj, dict) else {}
    except Exception:
        profile = {}

    # Fallback: try to parse inline JSON if structured extraction failed
    if not profile:
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', transcript, re.DOTALL)
            if json_match:
                inline = json.loads(json_match.group())
                profile = inline
        except Exception:
            profile = {}

    return _complete_personainfo_dict(profile)


def _summarize_intent(intent_agent, transcript: str, *, persona_id: str, model_id: str) -> str:
    """
    Free‑text, one‑sentence summary of the user's current goal (use user's own words; no classification).
    Uses call_strands_agent_logged for metrics + emissions logging.
    """
    from .llm_client import call_strands_agent_logged
    
    prompt = (
        "Summarize the user's current career or training goal in ONE short sentence. "
        "Prefer their own words. Do NOT classify. Do NOT invent details.\n\n"
        f"Transcript:\n{transcript}"
    )

    # Use the logged wrapper (phase tag = 'intent')
    result_text = call_strands_agent_logged(
        intent_agent,
        prompt,
        phase="intent",
        persona_id=persona_id,
        model_id=model_id
    )

    # Clean up quotes and whitespace
    return result_text.strip().strip('"').strip("'").strip()


def _merge_profile(base: dict, incr: dict, state: dict) -> dict:
    """
    Conservative merge: prefer non-empty from 'incr'. Overlay free‑text intent, domain, current_focus, age.
    """
    merged = dict(base or {})
    incr = incr or {}

    def _nz(v):
        if v is None:
            return False
        if isinstance(v, (list, dict, str)) and len(v) == 0:
            return False
        return True

    for k, v in incr.items():
        if _nz(v):
            merged[k] = v

    # Overlay authoritative signals from state if present
    if state.get("intent"):
        merged["intent"] = state["intent"]
    if state.get("domain") and state["domain"] != "unspecified":
        merged["top_domain"] = state["domain"]
    if state.get("current_focus"):
        merged["current_focus"] = state["current_focus"]
    if state.get("age") is not None:
        merged["age"] = state["age"]

    return _complete_personainfo_dict(merged)


def _postprocess_persona_extraction(
    profile: Dict[str, Any],
    domain: str,
    allowed_skills: Dict[str, List[str]],
    edu_levels: List[str],
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
            p["education_level"] = match
    
    # Skill filtering based on domain
    domain_allowed = allowed_skills.get(dom, [])
    
    # Filter training_motivation
    raw_motivation = p.get("training_motivation", [])
    if isinstance(raw_motivation, str):
        raw_motivation = [s.strip() for s in raw_motivation.replace(";", ",").split(",") if s.strip()]
    p["training_motivation"] = _filter_to_allowed_skills(raw_motivation, domain_allowed)
    
    # Filter and normalize technical_skills
    raw_skills = p.get("technical_skills", [])
    normalized_skills = []
    for skill in raw_skills:
        if isinstance(skill, dict):
            name = skill.get("name", "").strip()
            level = _coerce_level(skill.get("level"))
            if name and name in domain_allowed:
                normalized_skills.append({"name": name, "level": level})
        elif isinstance(skill, str):
            name = skill.strip()
            if name in domain_allowed:
                normalized_skills.append({"name": name, "level": "Basic"})
    p["technical_skills"] = normalized_skills
    
    # Normalize languages
    p["languages"] = to_language_codes(p.get("languages", []))
    
    # Normalize years_experience
    years = p.get("years_experience")
    if years is not None:
        try:
            years = float(years)
            p["years_experience"] = max(0.0, years)
        except (ValueError, TypeError):
            p["years_experience"] = None
    
    # Ensure complete PersonaInfo structure
    return _complete_personainfo_dict(p)


def extract_and_merge_profile(
    extractor_agent, 
    intent_agent,
    transcript: str, 
    existing_profile: dict = None, 
    state: dict = None,
    persona_id: str = "default",
    model_id: str = "default",
    allowed_skills: dict = None,
    edu_levels: list = None,
    known_job_domains: list = None
) -> dict:
    """
    Complete workflow: extract profile, summarize intent, merge with existing, and postprocess.
    """
    # Set defaults
    existing_profile = existing_profile or {}
    state = state or {}
    allowed_skills = allowed_skills or {}
    edu_levels = edu_levels or []
    known_job_domains = known_job_domains or []
    
    # Extract new profile
    new_profile = _extract_profile(extractor_agent, transcript)
    
    # Extract intent
    if intent_agent:
        intent = _summarize_intent(intent_agent, transcript, persona_id=persona_id, model_id=model_id)
        state = {**state, "intent": intent}
    
    # Merge profiles
    merged_profile = _merge_profile(existing_profile, new_profile, state)
    
    # Post-process
    domain = state.get("domain", merged_profile.get("top_domain", "unspecified"))
    domain = sanitize_domain(domain, known_job_domains)
    
    final_profile = _postprocess_persona_extraction(
        merged_profile,
        domain,
        allowed_skills,
        edu_levels,
        intent_in_state=state.get("intent"),
        focus_in_state=state.get("current_focus"),
    )
    
    return final_profile