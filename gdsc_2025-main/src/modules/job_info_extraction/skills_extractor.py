# src/modules/job_info_extraction/skills_extractor.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from string import Template

from src.agents import call_mistral
from src.modules.training_info_extraction.parsing import json_from_llm_text
from src.common.canonicalization import to_bia_skill_level

# ---------------------------
# Loading helpers
# ---------------------------

def load_allowed_skills(csv_path: Path) -> Dict[str, List[str]]:
    """
    Load a CSV with columns: domain,skill
    Returns: {canonical_domain: [canonical_skill, ...]}
    """
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")
    required = {"domain", "skill"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must contain columns: {required}. Found: {df.columns.tolist()}")
    by_dom = (
        df.groupby("domain")["skill"]
          .apply(lambda s: sorted(set(x.strip() for x in s.tolist() if str(x).strip())))
          .to_dict()
    )
    return by_dom

def load_domain_settings(settings_path: Optional[Path]) -> Dict[str, Any]:
    """
    Optional domain-settings override (JSON).
    This is *not* used for matching; it's provided to enable future enrichment
    flows built from persona replies. Safe to be absent.
    """
    if not settings_path:
        return {}
    if not Path(settings_path).exists():
        return {}
    try:
        return json.loads(Path(settings_path).read_text(encoding="utf-8"))
    except Exception:
        return {}

# ---------------------------
# Prompt (from your TXT)
# ---------------------------

JOB_REQ_PROMPT_T = Template(r'''
You are a precise recruiter's assistant.
Task
----
Given a JOB DESCRIPTION and the job's DOMAIN, select REQUIRED SKILLS **only from ALLOWED_SKILLS** for that domain.
A skill qualifies **only if both** are true:
1) It is explicitly **required** (must-have) in the job text (not "preferred", "nice to have", "optional", "bonus", "plus").
2) There is an explicit **level phrase** near the skill mention. Allowed level indicators:
  - Level words/phrases: mainly "basic", "foundational", "intermediate", sometimes also "working knowledge", "solid", "strong", "proficient", "advanced", "expert", "senior", "lead", "principal", "deep", "hands-on"
Return ONLY a strict JSON object (no markdown, no commentary):
{
  "job_title": "string",
  "domain": "$domain",
  "skills": [
    {
      "name": "string", // EXACT string from ALLOWED_SKILLS
      "level": "Basic|Intermediate|Advanced",
      "evidence": "string" // <=160 chars; includes the level phrase + skill mention, verbatim from JD
    }
  ],
  "confidence": 0.0
}
Rules
-----
- Use EXACT strings from ALLOWED_SKILLS; do NOT invent new skills or variants.
- Many job descriptions have 1 or 0 required skills. Prefer fewer over more. If uncertain, exclude.
- Ignore soft skills and generic responsibilities (e.g. strong analytical thinking, attention to detail, leadership)
- Ignore any items under or marked as: "preferred", "nice to have", "bonus", "plus", "optional".
- **Evidence must include** the level phrase and the skill mention, trimmed to ≤160 chars.
- Level mapping from phrases (no years/certs):
  - "Exposure to", "familiarity with" → Basic
  - "Working knowledge", "solid understanding" → Intermediate
  - "Hands-on", "deep expertise", "strong", "proficient", "advanced", "expert", "lead/architect/principal" → Advanced
- ALWAYS ignore the "Responsibilities & Duties" section or task descriptions. NEVER deduce required skills from that.

Section Prioritization (when present)
-------------------------------------
- Prioritize: "Required", "Minimum Qualifications", "Must Have", "What you need", "We require".
- Deprioritize/ignore: "Preferred", "Nice to Have", "Bonus", "Good to have", "Optional", "Plus".

PROVIDED_DOMAIN = "$domain"
ALLOWED_SKILLS (for the provided domain) = $allowed_skills_json
JOB DESCRIPTION (Markdown):
"""$job_md"""
''')

def _build_required_skills_prompt(domain: str, allowed_skills: List[str], job_md: str) -> str:
    MAX_ALLOWED = 200  # keep token usage in check
    allowed = allowed_skills[:MAX_ALLOWED]
    return JOB_REQ_PROMPT_T.substitute(
        domain=domain,
        allowed_skills_json=json.dumps(allowed, ensure_ascii=False),
        job_md=job_md
    )

# ---------------------------
# Core extraction
# ---------------------------

@dataclass
class RequiredSkill:
    name: str
    min_level: Optional[str] = None
    evidence: Optional[str] = None  # stored for JSONL debugging; ignored by JobInfo schema

def extract_required_skills_for_job(
    job_md: str,
    domain: str,
    allowed_skills_for_domain: List[str],
    model: str = "mistral-medium-latest",
) -> Tuple[List[RequiredSkill], Dict[str, Any]]:
    """
    - Calls the LLM with a strict allowed list
    - Parses JSON
    - Filters strictly to allowed skills (case-insensitive)
    - Maps 'level' → BIA ladder (Basic|Intermediate|Advanced)
    Returns: (required_skills_list, raw_object_for_debug)
    """
    if not allowed_skills_for_domain:
        return [], {"note": "no allowed skills for domain"}

    prompt = _build_required_skills_prompt(domain, allowed_skills_for_domain, job_md)
    res = call_mistral(prompt, model=model)  # your agents wrapper
    txt = res.get("content", "") if isinstance(res, dict) else ""
    obj = json_from_llm_text(txt) if txt else None
    if not isinstance(obj, dict):
        obj = {"domain": domain, "skills": [], "confidence": 0.0}

    # Strictly filter to allowed set (case-insensitive), and clamp level
    allowed_lower = {s.lower(): s for s in allowed_skills_for_domain}
    out: List[RequiredSkill] = []
    for s in obj.get("skills") or []:
        raw_name = (s.get("name") or "").strip()
        if raw_name.lower() in allowed_lower:
            canonical_name = allowed_lower[raw_name.lower()]
            level = s.get("level") or ""
            level_bia = to_bia_skill_level(level or "Intermediate")  # conservative default
            evidence = (s.get("evidence") or "")[:160] if s.get("evidence") else None
            out.append(RequiredSkill(name=canonical_name, min_level=level_bia, evidence=evidence))

    # Deduplicate by name, keep highest level (Advanced > Intermediate > Basic)
    rank = {"Basic": 1, "Intermediate": 2, "Advanced": 3}
    merged: Dict[str, RequiredSkill] = {}
    for rs in out:
        k = rs.name
        if k not in merged or rank.get(rs.min_level or "Basic", 1) > rank.get(merged[k].min_level or "Basic", 1):
            merged[k] = rs

    return list(merged.values()), obj
