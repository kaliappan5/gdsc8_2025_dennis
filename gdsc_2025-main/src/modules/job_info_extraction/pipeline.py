# src/modules/job_info_extraction/pipeline.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable

from .io import discover_sources, load_resume_set, append_jsonl, append_csv, ensure_empty_csv
from .parsing import json_from_llm_text
from .models import validate_job_info
from .llm_client import call_llm_job

from .domain_picker import (
    prepare_taxonomy, compute_content_key,
    pick_domain_with_bias, pick_rome_within_domain,
)
from .skills_extractor import (
    load_allowed_skills, load_domain_settings,
    extract_required_skills_for_job,
)

from src.common.canonicalization import (
    canonicalize_domain_or_raise,
    canonicalize_language_list,
    canonicalize_work_type,
    canonicalize_edu_level,
    to_bia_skill_level,
)

@dataclass
class ExtractJobsConfig:
    input_dir: Optional[Path] = None
    inputs: Optional[Iterable[Path]] = None
    out_dir: Path = Path("processed_data/outputs/jobs")
    model: str = "mistral-small-latest"
    resume: bool = True
    max_files: Optional[int] = None
    sleep_seconds: float = 0.0
    allowed_domains: Optional[List[str]] = None
    taxonomy_csv: Optional[Path] = None              # domain/ROME tournament (optional)
    extract_required_skills: bool = True             # turn on the domain-scoped skill extractor
    allowed_skills_csv: Optional[Path] = None        # CSV with columns: domain,skill (from enrichment/personas)
    domain_settings_json: Optional[Path] = None      # future enrichment knobs built from persona replies

def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _source_key_for(path: Path) -> str:
    return path.stem

def _soft_domain(value: Optional[str]) -> str:
    if not value:
        return "unspecified"
    try:
        return canonicalize_domain_or_raise(value)
    except Exception:
        return "unspecified"

def _compact_required_skills(skills: Any) -> List[Dict[str, Any]]:
    """
    Normalize to [{name, min_level?}]
    (Used only if we *don't* run the domain-scoped extraction.)
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

def _row_for_csv(valid: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(valid)
    if isinstance(row.get("languages_required"), list):
        row["languages_required"] = ",".join(row["languages_required"])
    if isinstance(row.get("required_skills"), list):
        row["required_skills"] = "; ".join(
            f"{s.get('name')}({s.get('min_level','unspecified')})" for s in row["required_skills"]
        )
    return row

def extract_jobs(config: ExtractJobsConfig) -> Dict[str, Any]:
    out_dir = config.out_dir
    jsonl_path = out_dir / "jobs.jsonl"
    csv_path = out_dir / "jobs.csv"

    sources = discover_sources(config.input_dir, config.inputs, config.max_files)
    processed = load_resume_set(jsonl_path) if config.resume else set()

    headers = [
        "job_id","source_key","title","domain","location_city","work_type",
        "languages_required","education_level_required","years_of_experience_required",
        "required_skills","job_role","domain_confidence","rome_title_en","rome_confidence","source_path"
    ]
    ensure_empty_csv(csv_path, headers=headers)

    taxonomy = None
    if config.taxonomy_csv and Path(config.taxonomy_csv).exists():
        taxonomy = prepare_taxonomy(Path(config.taxonomy_csv))

    # Load allowed skills table (from trainings+personas enrichment) — optional but recommended
    allowed_skills_by_domain: Dict[str, List[str]] = {}
    if config.allowed_skills_csv and Path(config.allowed_skills_csv).exists():
        allowed_skills_by_domain = load_allowed_skills(Path(config.allowed_skills_csv))

    # Domain settings are optional; currently unused in per-job extraction.
    _ = load_domain_settings(config.domain_settings_json)

    written = 0
    for path in sources:
        source_key = _source_key_for(path)
        if config.resume and source_key in processed:
            continue

        raw_text = _read_source(path)
        content_key = compute_content_key(path, raw_text)
        excerpt = raw_text if len(raw_text) <= 1400 else " ".join(raw_text.split())[:1400]

        # 1) Generic LLM extraction (title, location, etc.)
        try:
            txt = call_llm_job(raw_text, model=config.model, fields_to_extract=None)
            parsed = json_from_llm_text(txt)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}

        # 2) Domain via taxonomy tournament (EN), then canonicalize for KG
        domain_en, domain_conf = ("", 0.0)
        if taxonomy:
            domain_en, domain_conf = pick_domain_with_bias(content_key, excerpt, taxonomy, model=config.model)
        domain_raw = domain_en or parsed.get("domain") or "unspecified"
        domain = _soft_domain(domain_raw)

        # 3) Optional ROME inside domain (for analysis only)
        rome_en, rome_conf = ("", 0.0)
        if taxonomy and domain_en:
            rome_en, rome_conf = pick_rome_within_domain(content_key, excerpt, domain_en, taxonomy, model=config.model)

        # 4) Canonicalize other fields to match JobInfo
        job_id = parsed.get("job_id") or f"job_{source_key}"
        title = parsed.get("title") or "Untitled role"
        location_city = parsed.get("location_city") or parsed.get("city") or ""
        work_type = canonicalize_work_type(parsed.get("work_type") or "unspecified")
        languages_required = canonicalize_language_list(parsed.get("languages_required") or parsed.get("languages") or [])
        raw_edu = parsed.get("education_level_required") or parsed.get("education_level")
        education_level_required = canonicalize_edu_level(raw_edu) if raw_edu else None
        years_of_experience_required = parsed.get("years_of_experience_required") or parsed.get("years_experience") or None
        try:
            years_of_experience_required = float(years_of_experience_required) if years_of_experience_required is not None else None
        except Exception:
            years_of_experience_required = None

        # 5) REQUIRED SKILLS (domain-scoped, strict allowed list) — integrates your TXT logic
        required_skills: List[Dict[str, Any]] = []
        skills_evidence: List[Dict[str, str]] = []
        if config.extract_required_skills and allowed_skills_by_domain:
            allowed_for_domain = allowed_skills_by_domain.get(domain, [])
            rs_list, raw_obj = extract_required_skills_for_job(
                job_md=raw_text, domain=domain, allowed_skills_for_domain=allowed_for_domain, model=config.model
            )
            # Map to JobInfo shape
            required_skills = [{"name": r.name, "min_level": r.min_level} for r in rs_list]
            # Keep evidence for JSONL/audit (ignored by JobInfo)
            skills_evidence = [{"name": r.name, "evidence": r.evidence or ""} for r in rs_list]
        else:
            # Fall back to whatever the generic extractor found
            required_skills = _compact_required_skills(parsed.get("required_skills"))

        candidate = {
            "job_id": job_id,
            "source_key": source_key,
            "title": title,
            "domain": domain,
            "location_city": location_city,
            "work_type": work_type,
            "languages_required": languages_required,
            "education_level_required": education_level_required,
            "years_of_experience_required": years_of_experience_required,
            "required_skills": required_skills,
            "job_role": parsed.get("job_role") or "",
            "domain_confidence": float(domain_conf or 0),
            "rome_title_en": rome_en,
            "rome_confidence": float(rome_conf or 0),
            "source_path": str(path),
            # Debug-only metadata (ignored by Pydantic):
            "skills_evidence": skills_evidence,
        }

        if config.allowed_domains and domain not in config.allowed_domains:
            continue

        # 6) Validate against JobInfo; fallback to minimal if needed
        try:
            validated = validate_job_info(candidate).model_dump()
        except Exception:
            minimal = {
                "job_id": job_id,
                "source_key": source_key,
                "title": title,
                "domain": domain,
                "work_type": work_type,
                "languages_required": languages_required,
                "source_path": str(path),
            }
            validated = validate_job_info(minimal).model_dump()

        append_jsonl(jsonl_path, validated)
        append_csv(csv_path, headers=headers, row=_row_for_csv(validated))
        written += 1

        if config.sleep_seconds:
            time.sleep(config.sleep_seconds)

    # Build submissions bundle {job_id: dict}
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True, parents=True)
    submissions_path = submissions_dir / f"extracted_jobs_{time.strftime('%Y-%m-%d')}.json"

    job_map: Dict[str, Any] = {}
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    jid = obj.get("job_id") or obj.get("id") or obj.get("source_key")
                    if jid:
                        job_map[str(jid)] = obj
                except Exception:
                    continue
    with submissions_path.open("w", encoding="utf-8") as f:
        json.dump(job_map, f, ensure_ascii=False, indent=2)

    return {
        "sources": len(sources),
        "written": written,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "bundle": str(submissions_path),
    }
