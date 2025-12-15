# src/modules/job_info_extraction/prefill.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from src.models import JobInfo
from src.common.canonicalization import (
    canonicalize_domain_or_raise,
    to_bia_skill_level,
)
from .io import ensure_parent

@dataclass
class PrefillInputs:
    jobs_labels_csv: Path  # ../processed_data/jobs_extracted_labels_en_blocked.csv
    jobs_skills_csv: Optional[Path] = None  # ../processed_data/jobs_skills_YYYY-MM-DD.csv

def _parse_required_skills_from_row(skills_val: Any, levels_val: Any) -> List[Dict[str, Optional[str]]]:
    """
    Your CSV stores consolidated skills + levels with line breaks. In the original notebook,
    it split by `" \n "`. We generalize to splitlines() and trim.
    """
    def _split(v: Any) -> List[str]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []
        s = str(v)
        # split on newline or " \n "
        parts = [p.strip() for p in re.split(r"\s*\n\s*", s) if p.strip()]
        return parts

    names = _split(skills_val)
    levels = _split(levels_val)
    if len(levels) < len(names):
        levels += [None] * (len(names) - len(levels))

    out: List[Dict[str, Optional[str]]] = []
    for name, lvl in zip(names, levels):
        entry = {"name": name}
        if lvl and str(lvl).strip():
            entry["min_level"] = to_bia_skill_level(str(lvl))
        out.append(entry)
    return out

def prefill_bundle_from_csvs(inputs: PrefillInputs,
                             out_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Build {job_id: JobInfo-like dict} from the 'blocked' labels CSV and optional skills CSV.

    - job_id: 'j{job_index}'
    - domain: taken from English domain label ('libelle_domaine_professionel_en'), with fallback.
    - job_role: optional from 'libelle_rome_en'.
    - required_skills: from the skills CSV if provided.
    """
    jobs_df = pd.read_csv(inputs.jobs_labels_csv, dtype=str, encoding="utf-8").fillna("")
    skills_df = pd.read_csv(inputs.jobs_skills_csv, dtype=str, encoding="utf-8").fillna("") \
        if inputs.jobs_skills_csv and inputs.jobs_skills_csv.exists() else None

    # Normalize the minimal columns we rely on
    # Expect 'job_index' or 'index', a title, and English domain name
    idx_col = "job_index" if "job_index" in jobs_df.columns else ("index" if "index" in jobs_df.columns else None)
    if not idx_col:
        raise ValueError("jobs_labels_csv must contain 'job_index' (or 'index').")

    # Skills lookup by job index (if available)
    skills_by_idx: Dict[str, Dict[str, Any]] = {}
    if skills_df is not None:
        idx_col2 = "index" if "index" in skills_df.columns else ("job_index" if "job_index" in skills_df.columns else None)
        if not idx_col2:
            # Accept silently, just no skills
            idx_col2 = None
        if idx_col2:
            for _, r in skills_df.iterrows():
                skills_by_idx[str(r.get(idx_col2))] = {
                    "skills": r.get("skills"),
                    "levels": r.get("levels"),
                }

    bundle: Dict[str, Dict[str, Any]] = {}

    for _, r in jobs_df.iterrows():
        j_idx = str(r.get(idx_col)).strip()
        if not j_idx:
            continue
        job_id = f"j{j_idx}"
        title = r.get("title") or ""
        # English domain label column:
        dom_en = r.get("libelle_domaine_professionel_en") or ""
        # Try to canonicalize; if fails, leave as given (JobInfo allows Optional[str])
        try:
            domain = canonicalize_domain_or_raise(dom_en) if dom_en else None
        except Exception:
            domain = dom_en or None

        job_role = r.get("libelle_rome_en") or None

        # Required skills if present in the companion CSV
        rs_list: List[Dict[str, Optional[str]]] = []
        srow = skills_by_idx.get(j_idx)
        if srow:
            rs_list = _parse_required_skills_from_row(srow.get("skills"), srow.get("levels"))

        # Compose a minimal JobInfo-like dict (fields JobInfo doesn't have are ignored by pydantic)
        record = {
            "job_id": job_id,
            "title": title,
            "domain": domain,
            "job_role": job_role,
            "location_city": None,
            "work_type": None,
            "languages_required": None,
            "education_level_required": None,
            "years_of_experience_required": None,
            "required_skills": rs_list or None,
        }

        # Validate to catch drift early; keep tolerant about missing optional fields
        try:
            record = JobInfo.model_validate(record).model_dump()
        except Exception:
            # keep as-is; downstream can still enrich
            pass

        bundle[job_id] = record

    # Write bundle if requested (use today’s date naming)
    if out_path:
        ensure_parent(out_path)
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    return bundle
