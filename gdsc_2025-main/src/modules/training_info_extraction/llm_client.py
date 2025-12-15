"""
LLM client and prompt builders for training info extraction flows.
"""

from __future__ import annotations
import json
from typing import Iterable, List
from src.agents import call_mistral
from src.common.env import load_project_dotenv
load_project_dotenv()



def call_llm(prompt: str, *, model: str = "mistral-medium-latest") -> dict | None:
    """
    Thin wrapper around src.agents.call_mistral to keep the pipeline decoupled from the client.
    Returns the raw dict response or None on failure.
    """
    try:
        return call_mistral(prompt, model=model)
    except Exception:
        return None


def build_extraction_prompt(md_text: str, allowed_domains: Iterable[str], known_skills: List[str]) -> str:
    """
    Prompt for the one-skill extraction step (strict JSON output).
    Mirrors the notebook’s intent: ALLOWED_DOMAINS & KNOWN_SKILLS guidance. 
    """
    # Keep KNOWN_SKILLS modest if needed (callers can slice upstream)
    allowed = list(allowed_domains)
    tmpl = f"""
You are a meticulous information extraction assistant.
Return ONLY a strict JSON object with fields:
{{
  "course_title": str,  // best title you can infer
  "domain": str,        // EXACTLY one from ALLOWED_DOMAINS
  "skill_name": str,    // a SINGLE skill taught in this course
  "skill_level": str,   // EXACTLY one of: "Basic" | "Intermediate" | "Advanced"
  "evidence": str,      // <=160 chars quote supporting skill & level
  "confidence": float   // 0.0–1.0
}}
Rules:
- The training covers EXACTLY ONE skill. Identify it precisely (e.g., "Python", "Power BI DAX", "Docker", "Excel", "SQL").
- ALLOWED_DOMAINS = {allowed}
- LEVELS = ["Basic","Intermediate","Advanced"] — choose exactly one.
- KNOWN_SKILLS = {known_skills}
  If the extracted skill matches (or is equivalent to) an entry in KNOWN_SKILLS, REUSE the exact string from KNOWN_SKILLS for "skill_name".
- Prefer concise, widely-used skill names (no trailing words like "basics", "course", "workshop", or level indicators).
- Be concrete and testable; pick the main skill, not a broad umbrella domain.
- "evidence" must be a short quote from the text (<=160 chars).
- Return JSON only. No markdown, no explanations.

COURSE DESCRIPTION (Markdown):
\"\"\"
{md_text}
\"\"\"
""".strip()
    return tmpl


def build_triplet_prompt(triplet_items: list[dict], domain_name: str, avoid_skills: list[str]) -> str:
    """
    Prompt used in the triplet labeling step.
    triplet_items: [{'index': int, 'title': str}, ...] exactly 3 items.
    """
    guidance = f"""
You are labeling THREE course titles from the SAME domain: "{domain_name}".
They form a BASIC → INTERMEDIATE → ADVANCED progression for ONE skill.
RETURN a concise SKILL (1–4 words), preferably a phrase present in the titles.
The SKILL must be DISTINCT inside this domain and MUST NOT duplicate or trivially overlap with any of:
{json.dumps(avoid_skills, ensure_ascii=False)}

If your initial skill would conflict, add a 1–3 word qualifier FROM THE TITLES to distinguish it.
Examples: "(Batch Size)", "(Ingredient Yields)", "(Costing)".

Rules:
- If titles contain level hints (e.g., "Foundations", "201", "Advanced"), use them.
- If none, assume order: 1st=Basic, 2nd=Intermediate, 3rd=Advanced.
- Do NOT add extra fields or commentary.

Return ONLY this JSON:
{{
  "skill": "<1-4 word skill (qualify if needed)>",
  "mapping": [
    {{"index": <int>, "skill_level": "Basic"}},
    {{"index": <int>, "skill_level": "Intermediate"}},
    {{"index": <int>, "skill_level": "Advanced"}}
  ]
}}
""".strip()
    json_block = json.dumps(triplet_items, ensure_ascii=False)
    return f"{guidance}\n\nCOURSES:\n{json_block}"
