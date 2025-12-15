# src/modules/job_info_extraction/backfill.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


from src.models import JobInfo
from src.modules.training_info_extraction.parsing import json_from_llm_text
from src.common.canonicalization import (
    canonicalize_work_type,
    canonicalize_language_list,
    canonicalize_edu_level,      # strict when present
    to_bia_skill_level,
)
from .llm_client import call_llm_job_fields
from .io import ensure_parent

# ----------------------------
# Utilities
# ----------------------------
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _compact_required_skills(skills: Any) -> List[Dict[str, Any]]:
    """
    Normalize to [{name, min_level?}]
    """
    out: List[Dict[str, Any]] = []
    if not skills:
        return out
    if isinstance(skills, dict):
        for k, v in skills.items():
            out.append({"name": k, "min_level": to_bia_skill_level(v)})
        return out
    if isinstance(skills, list):
        for item in skills:
            if isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, dict):
                name = item.get("name") or item.get("skill") or item.get("id") or ""
                level = item.get("min_level") or item.get("level") or item.get("proficiency")
                row: Dict[str, Any] = {"name": name}
                if level:
                    row["min_level"] = to_bia_skill_level(level)
                if name:
                    out.append(row)
    return out

def _fields_missing_in_record(record: Dict[str, Any], candidate_fields: List[str]) -> List[str]:
    missing = []
    for k in candidate_fields:
        v = record.get(k, None)
        if v is None or v == "" or v == [] or v == {}:
            missing.append(k)
    return missing

# ----------------------------
# Backfill (LLM pass on missing fields only)
# ----------------------------
@dataclass
class BackfillConfig:
    job_bundle_path: Path                 # input primary bundle JSON ({job_id: dict})
    job_md_dir: Path                      # directory with job markdown files (e.g., data/jobs)
    out_path: Path                        # output merged bundle path
    model: str = "mistral-small-latest"
    fields_considered: Optional[List[str]] = None  # if None, use a sensible default
    resume_json: Optional[Path] = None    # if set, keeps interim per-job enrichment JSON (resume-friendly)

DEFAULT_FIELDS = [
    "company",
    "location_city",
    "work_type",
    "languages_required",
    "education_level_required",
    "years_of_experience_required",
    "required_skills",
]

def _job_md_for_id(job_id: str, job_md_dir: Path) -> Optional[Path]:
    """
    Heuristic: job_id like "j70" -> look for "j70.md" in job_md_dir.
    """
    base = re.sub(r"[^0-9]", "", job_id)  # "70"
    candidates = [
        job_md_dir / f"j{base}.md",
        job_md_dir / f"{job_id}.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    # last resort: any file whose stem equals job_id
    alt = job_md_dir / f"{job_id}.md"
    return alt if alt.exists() else None

def _merge_missing_fields_only(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Copy a field from `secondary` if and only if the field exists in `primary` AND primary[field] is None/""
    """
    merged = dict(primary)
    filled = 0
    for k, v2 in (secondary or {}).items():
        if k in merged and (merged[k] is None or merged[k] == "" or merged[k] == [] or merged[k] == {}):
            if v2 not in (None, "", [], {}):
                merged[k] = v2
                filled += 1
    return merged, filled

def _canonicalize_job_fields(partial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply canonicalizers for fields we backfill.
    """
    out = dict(partial)
    if "work_type" in out and out["work_type"] not in (None, "", [], {}):
        out["work_type"] = canonicalize_work_type(out["work_type"])
    if "languages_required" in out and out["languages_required"] not in (None, "", [], {}):
        out["languages_required"] = canonicalize_language_list(out["languages_required"])
    if "education_level_required" in out:
        raw = out["education_level_required"]
        out["education_level_required"] = canonicalize_edu_level(raw) if raw else None
    if "required_skills" in out and out["required_skills"] not in (None, "", [], {}):
        out["required_skills"] = _compact_required_skills(out["required_skills"])
    return out

def backfill_missing_fields(cfg: BackfillConfig) -> Dict[str, Any]:
    """
    Loads a primary bundle, reads job markdowns, calls the LLM only for missing fields,
    canonicalizes results, merges them into the records, and writes an output bundle.

    This follows the pattern you demonstrated with:
    - make_job_extraction_prompt(fields)
    - merging only missing fields
    - language/work_type canonicalization
    """
    bundle = json.loads(cfg.job_bundle_path.read_text(encoding="utf-8"))
    fields = cfg.fields_considered or list(DEFAULT_FIELDS)
    resume: Dict[str, Any] = {}
    if cfg.resume_json and cfg.resume_json.exists():
        try:
            resume = json.loads(cfg.resume_json.read_text(encoding="utf-8"))
        except Exception:
            resume = {}

    updated: Dict[str, Any] = {}
    total_filled = 0

    for jid, rec in bundle.items():
        rec = dict(rec or {})
        missing = _fields_missing_in_record(rec, fields)
        if not missing:
            updated[jid] = rec
            continue

        # resume support: if we already enriched this id, reuse
        if cfg.resume_json and jid in resume:
            enriched = resume[jid]
        else:
            # read job md text
            md_path = _job_md_for_id(jid, cfg.job_md_dir)
            if not md_path or not md_path.exists():
                updated[jid] = rec
                continue
            md_text = _read_text(md_path)

            # call LLM for the missing fields only
            raw_text = call_llm_job_fields(md_text, fields=missing, model=cfg.model)
            parsed = {}
            try:
                parsed = json_from_llm_text(raw_text) or {}
            except Exception:
                parsed = {}

            # canonicalize partial result
            enriched = _canonicalize_job_fields(parsed)

            # optional resume write
            if cfg.resume_json:
                ensure_parent(cfg.resume_json)
                resume[jid] = enriched
                cfg.resume_json.write_text(json.dumps(resume, ensure_ascii=False, indent=2), encoding="utf-8")

        # merge into primary (only where primary has empty / None)
        merged, filled = _merge_missing_fields_only(rec, enriched)
        total_filled += filled

        # validate against JobInfo (tolerant)
        try:
            merged = JobInfo.model_validate(merged).model_dump()
        except Exception:
            # keep as-is; at worst, downstream will ignore extras
            pass

        updated[jid] = merged

    # write output
    ensure_parent(cfg.out_path)
    cfg.out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "input": str(cfg.job_bundle_path),
        "output": str(cfg.out_path),
        "resume": str(cfg.resume_json) if cfg.resume_json else None,
        "jobs": len(updated),
        "fields_filled": total_filled,
    }

# ----------------------------
# Bundle merge (primary <- secondary; only if primary has None)
# ----------------------------
def _coerce_record(value: Any) -> Optional[Dict[str, Any]]:
    """
    If values were double-encoded in JSON (stringified dict), coerce safely.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None

def merge_bundles_only_if_none(primary_path: Path,
                               secondary_path: Path,
                               out_path: Path,
                               include_secondary_only: bool = False) -> Dict[str, Any]:
    """
    Merge two {id: dict} JSON files. Copy a field from secondary if and only if
    primary[id] exists AND primary[id][field] is None/empty.
    If 'include_secondary_only' is True, records that exist only in secondary are kept.
    """
    primary_raw = json.loads(primary_path.read_text(encoding="utf-8"))
    secondary_raw = json.loads(secondary_path.read_text(encoding="utf-8"))

    primary = {k: _coerce_record(v) for k, v in primary_raw.items()}
    secondary = {k: _coerce_record(v) for k, v in secondary_raw.items()}

    p_ids = set(primary.keys())
    s_ids = set(secondary.keys())
    ids = (p_ids | s_ids) if include_secondary_only else p_ids

    out: Dict[str, Any] = {}
    filled_total = 0
    added_from_secondary = 0
    skipped_uncoercible = 0

    for jid in sorted(ids):
        p = primary.get(jid)
        s = secondary.get(jid)
        if p is None and s is None:
            skipped_uncoercible += 1
            continue
        if p is None:
            if include_secondary_only and s is not None:
                out[jid] = s
                added_from_secondary += 1
            continue
        # merge
        merged, filled = _merge_missing_fields_only(p, s or {})
        out[jid] = merged
        filled_total += filled

    ensure_parent(out_path)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "primary": str(primary_path),
        "secondary": str(secondary_path),
        "output": str(out_path),
        "ids_primary": len(p_ids),
        "ids_secondary": len(s_ids),
        "ids_output": len(out),
        "fields_filled": filled_total,
        "added_from_secondary_only": added_from_secondary if include_secondary_only else 0,
        "skipped_uncoercible": skipped_uncoercible,
    }
