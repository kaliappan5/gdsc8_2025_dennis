"""
Pipelines orchestrating the training extraction flows:
1) One-skill extraction from Markdown with LLM (JSONL/CSV).
2) CSV cleaning (checkpoint removal + index extraction).
3) Domain relabeling by index ranges.
4) Triplet-based skill & level labeling.
5) Export tagged CSV -> final dict-of-dicts JSON format (by date).
6) Batch backfill remaining fields (level/language/delivery/city/duration) from Markdown,
   NEVER overwriting skill/domain, preserving existing non-None fields.
"""

from __future__ import annotations
import time
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import pandas as pd

from .llm_client import call_llm, build_extraction_prompt, build_triplet_prompt
from .parsing import extract_title, json_from_llm_text, extract_json
from .skills import canonicalize_skill, load_skill_catalog, save_skill_catalog, map_to_catalog
from .heuristics import (
    detect_level_from_title,
    default_level_by_position,
    candidate_skill_from_titles,
    ensure_domain_unique_skill,
)
from .io import (
    find_markdown_files,
    read_text,
    append_jsonl,
    save_csv,
    load_resume_set,
    derive_allowed_domains_from_jobs_csv,
    write_allowed_domains_txt,
    clean_extracted_csv as _clean_extracted_csv,
    relabel_domains_by_index_ranges as _relabel_domains_by_index_ranges,
    read_json,
    write_json,
)

# Shared canonicalizers
from src.common.canonicalization import (
    to_bia_skill_level,
    canonicalize_domain_or_raise,
)

# Backfill integration
from .backfill import batch_backfill_trainings


# ------------------------------------------------------------------------------
# 1) One-skill extraction (existing)
# ------------------------------------------------------------------------------

def extract_trainings_one_skill(
    *,
    trainings_dir: Path,
    out_dir: Path,
    model: str = "mistral-medium-latest",
    allowed_domains: Iterable[str] | None = None,
    jobs_csv_path: Path | None = None,
    jobs_csv_domain_column: str = "libelle_domaine_professionel_en",
    jsonl_name: str = "trainings_extracted_one_skill.jsonl",
    csv_name: str = "trainings_extracted_one_skill.csv",
    skill_catalog_name: str = "skill_catalog.json",
    max_files: int = 0,
    sleep_seconds: float = 0.3,
    resume: bool = True,
) -> tuple[Path, Path]:
    """
    Run the one-skill extraction on all Markdown files in trainings_dir.
    Returns (jsonl_path, csv_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / jsonl_name
    csv_out_path = out_dir / csv_name
    skill_catalog_path = out_dir / skill_catalog_name

    # Allowed domains
    if allowed_domains is None:
        if jobs_csv_path is None:
            raise ValueError("Either 'allowed_domains' or 'jobs_csv_path' must be provided.")
        domains = derive_allowed_domains_from_jobs_csv(jobs_csv_path, jobs_csv_domain_column)
        write_allowed_domains_txt(out_dir / "allowed_domains.txt", domains)
        allowed_domains = domains

    # Canonicalize allowed domains (tolerate CSV drift)
    allowed_canon_set: set[str] = set()
    for d in allowed_domains:
        try:
            allowed_canon_set.add(canonicalize_domain_or_raise(d))
        except Exception:
            if d:
                allowed_canon_set.add(d.strip())

    # Resume + catalog
    skill_catalog = load_skill_catalog(skill_catalog_path)
    done_set, known_from_prev = load_resume_set(jsonl_path)
    if known_from_prev:
        skill_catalog = sorted(set(skill_catalog).union(known_from_prev))

    files = find_markdown_files(trainings_dir)
    if max_files and max_files > 0:
        files = files[:max_files]

    rows: list[dict] = []
    for i, md_path in enumerate(files, 1):
        if resume and str(md_path) in done_set:
            continue

        md_text = read_text(md_path)
        title = extract_title(md_text) or md_path.stem

        # Prompt
        known_for_prompt = skill_catalog[-300:] if len(skill_catalog) > 300 else skill_catalog
        prompt = build_extraction_prompt(md_text, allowed_canon_set, known_for_prompt)

        # LLM call with retry
        attempts, result = 0, None
        while attempts < 3:
            attempts += 1
            result = call_llm(prompt, model=model)
            if result:
                break
            time.sleep(min(2 ** attempts, 8))
        if not result:
            continue

        data = json_from_llm_text(result.get("content", "") or "")
        if not data:
            data = {
                "course_title": title,
                "domain": "",
                "skill_name": "",
                "skill_level": "Basic",
                "evidence": "",
                "confidence": 0.0,
            }

        # Domain canon
        domain_raw = (data.get("domain") or "").strip()
        domain_canon = ""
        if domain_raw:
            try:
                domain_canon = canonicalize_domain_or_raise(domain_raw)
            except Exception:
                if domain_raw in allowed_canon_set:
                    domain_canon = domain_raw
        if domain_canon and domain_canon not in allowed_canon_set:
            domain_canon = ""

        # Level and skill
        level = to_bia_skill_level((data.get("skill_level") or "").strip())
        raw_skill = (data.get("skill_name") or "").strip()
        canonical = canonicalize_skill(raw_skill)
        final_skill = map_to_catalog(canonical, skill_catalog, cutoff=0.92)
        if final_skill not in skill_catalog:
            skill_catalog.append(final_skill)
            save_skill_catalog(skill_catalog_path, skill_catalog)

        row = {
            "file": str(md_path),
            "course_title": data.get("course_title") or title,
            "domain": domain_canon,
            "skill_name": final_skill,
            "skill_level": level,
            "evidence": (data.get("evidence") or "")[:160],
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "model": result.get("model"),
            "duration": float(result.get("duration") or 0.0),
            "input_tokens": int(result.get("input_tokens") or 0),
            "output_tokens": int(result.get("output_tokens") or 0),
            "total_tokens": int(result.get("total_tokens") or 0),
            "timestamp": time.time(),
        }

        append_jsonl(jsonl_path, row)
        rows.append(row)

        print(
            f"[{i}/{len(files)}] {md_path.name} "
            f"domain='{row['domain']}' skill='{row['skill_name']}' level={row['skill_level']}"
        )
        time.sleep(max(sleep_seconds, 0.0))

    if rows:
        save_csv(csv_out_path, rows)
        print(f"Saved CSV -> {csv_out_path}")
    else:
        # Always create a CSV with headers to avoid FileNotFoundError in notebooks
        import pandas as pd
        cols = [
            "file", "course_title", "domain", "skill_name", "skill_level",
            "evidence", "confidence", "model", "duration",
            "input_tokens", "output_tokens", "total_tokens", "timestamp",
        ]
        pd.DataFrame(columns=cols).to_csv(csv_out_path, index=False)
        print(f"ℹ️ No rows extracted; wrote empty CSV at {csv_out_path}")

    return jsonl_path, csv_out_path


# ------------------------------------------------------------------------------
# 2) Wrapper for CSV cleaning
# ------------------------------------------------------------------------------

def clean_extracted_csv(input_csv_path: Path, output_csv_path: Path) -> int:
    n = _clean_extracted_csv(input_csv_path, output_csv_path)
    print(f"✅ Cleaned CSV saved to {output_csv_path} with {n} rows.")
    return n


# ------------------------------------------------------------------------------
# 3) Relabel domains by index ranges
# ------------------------------------------------------------------------------

def relabel_domains_by_index_ranges(
    in_csv: Path,
    out_csv: Path,
    mapping_ranges: list[Tuple[int, int, str]],
    canonical_map: dict[str, str] | None = None,
) -> None:
    df = pd.read_csv(in_csv)
    df_out = _relabel_domains_by_index_ranges(df, mapping_ranges, canonical_map=canonical_map)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Saved relabelled file -> {out_csv}")


# ------------------------------------------------------------------------------
# 4) Triplet-based skill & level labeling (existing)
# ------------------------------------------------------------------------------

def label_triplets_from_csv(
    in_csv: Path,
    out_csv: Path,
    *,
    cache_path: Path,
    model: str = "mistral-medium-latest",
) -> None:
    df = pd.read_csv(in_csv)
    required_cols = {"course_title", "domain", "file", "index"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.sort_values(["domain", "index"]).reset_index(drop=True)
    df["skill"] = None
    df["skill_level"] = None

    cache: dict = {}
    if cache_path.exists():
        try:
            cache = read_json(cache_path)
        except Exception:
            cache = {}

    def cache_key_for_triplet(rows: pd.DataFrame) -> str:
        titles = [str(r["course_title"]) for _, r in rows.iterrows()]
        return "\n#\n".join(titles)

    def save_cache():
        write_json(cache_path, cache)

    processed_triplets = 0
    errors: list[tuple] = []

    for domain, dgroup in df.groupby("domain", sort=False):
        dgroup = dgroup.sort_values("index").copy()
        dgroup["domain_pos"] = range(len(dgroup))
        dgroup["triplet_in_domain"] = dgroup["domain_pos"] // 3

        used_skills_in_domain: list[str] = []

        for _, rows in dgroup.groupby("triplet_in_domain", sort=True):
            rows = rows.sort_values("index")
            if len(rows) != 3:
                continue
            titles = [str(r["course_title"]) for _, r in rows.iterrows()]
            idxs = [int(r["index"]) for _, r in rows.iterrows()]
            items = [{"index": idxs[i], "title": titles[i]} for i in range(3)]
            key = cache_key_for_triplet(rows)

            llm_skill = None
            level_map: dict[int, str] = {}

            cached = cache.get(key)
            if cached:
                llm_skill = (cached.get("skill") or "").strip() or None
                level_map = {
                    int(m["index"]): m["skill_level"]
                    for m in cached.get("mapping", [])
                    if "index" in m and "skill_level" in m
                }

            if not llm_skill:
                try:
                    prompt = build_triplet_prompt(items, domain_name=str(domain), avoid_skills=used_skills_in_domain)
                    resp = call_llm(prompt, model=model)
                    content = resp.get("content") if resp else None
                    data = extract_json(content) if content else None
                    if data and "skill" in data and "mapping" in data and len(data["mapping"]) == 3:
                        llm_skill = (data["skill"] or "").strip() or None
                        for m in data["mapping"]:
                            try:
                                level_map[int(m["index"])] = m["skill_level"]
                            except Exception:
                                pass
                        cache[key] = data
                        save_cache()
                    else:
                        errors.append((domain, idxs, "LLM invalid JSON/object", content))
                except Exception as e:
                    errors.append((domain, idxs, f"LLM call exception: {e}", None))

            if not llm_skill:
                llm_skill = candidate_skill_from_titles(titles) or "General Skill"

            final_skill = ensure_domain_unique_skill(llm_skill, titles, used_skills_in_domain, jaccard_threshold=0.7)

            if set(level_map.values()) == {"Basic", "Intermediate", "Advanced"} and set(level_map.keys()) == set(idxs):
                pass
            else:
                detected = [detect_level_from_title(t) for t in titles]
                if all(detected):
                    level_map = {idxs[i]: detected[i] for i in range(3)}
                else:
                    level_map = {idxs[i]: default_level_by_position(i) for i in range(3)}

            for _, row in rows.iterrows():
                idx = int(row["index"])
                df.loc[df["index"] == idx, "skill"] = final_skill
                df.loc[df["index"] == idx, "skill_level"] = level_map[idx]

            used_skills_in_domain.append(final_skill)
            processed_triplets += 1

    print(f"Processed triplets: {processed_triplets}; errors: {len(errors)}")
    if errors:
        print("Sample error:", errors[0][:3])

    df_out = df.drop(columns=[c for c in ["triplet_id", "domain_pos", "triplet_in_domain"] if c in df.columns])
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


# ------------------------------------------------------------------------------
# 5) Export final dict-of-dicts JSON from a tagged CSV (integrated from notebook)
# ------------------------------------------------------------------------------

def export_trainings_json_from_tagged_csv(
    *,
    input_csv: Path,
    output_json: Path | None = None,
    index_col: str = "index",
    title_col: str = "course_title",
    domain_col: str = "domain",
    skill_cols: List[str] = ("skill", "skill_name"),
    level_col: str = "skill_level",
    id_prefix: str = "tr",
) -> Path:
    """
    Read a tagged trainings CSV and build the final submissions JSON file:
        {
           "tr<index>": {
              "training_id": "tr<index>",
              "title": ...,
              "domain": ...,
              "skill": ...,
              "level": ...,
              "language": null,
              "delivery_mode": null,
              "city": null,
              "duration": null
           },
           ...
        }

    Mirrors the notebook's 'training_file_creation' export step (module-ized).  # Source
    """
    output_json = output_json or (input_csv.parent / f"extracted_trainings_{date.today().isoformat()}.json")

    df = pd.read_csv(input_csv)

    # Resolve skill column (handle both 'skill' and 'skill_name')
    skill_col = next((c for c in skill_cols if c in df.columns), None)
    if not skill_col:
        raise ValueError(f"CSV missing skill column; tried {skill_cols}")

    out: Dict[str, Dict[str, Any]] = {}
    for i, row in df.iterrows():
        idx = row[index_col] if index_col in row else i
        training_id = f"{id_prefix}{int(idx)}"
        title = row[title_col] if title_col in row else ""
        domain = row[domain_col] if domain_col in row else ""
        skill = row[skill_col] if skill_col in row else ""
        level = row[level_col] if level_col in row else ""

        # Optional passthrough if present in CSV; else None
        language = row.get("language", None)
        delivery_mode = row.get("delivery_mode", None)
        city = row.get("city", None)
        duration = row.get("duration", None)

        out[training_id] = {
            "training_id": training_id,
            "title": title,
            "domain": domain,
            "skill": skill,
            "level": level,
            "language": language if pd.notna(language) else None,
            "delivery_mode": delivery_mode if pd.notna(delivery_mode) else None,
            "city": city if pd.notna(city) else None,
            "duration": duration if pd.notna(duration) else None,
        }

    write_json(output_json, out)
    print(f"✅ Wrote {len(out)} trainings -> {output_json}")
    return output_json


# ------------------------------------------------------------------------------
# 6) Batch backfill remaining fields from Markdown (never overwrite domain/skill)
# ------------------------------------------------------------------------------

def backfill_training_fields(
    *,
    training_md_dir: Path,
    primary_json_path: Path,
    output_json_path: Path,
    model: str = "mistral-medium-latest",
    cache_period: int = 20,
) -> Path:
    """
    For each Markdown file (id = stem), fill only [level, language, delivery_mode, city, duration]
    if they are missing in the existing (primary) JSON. Writes results to 'output_json_path'.

    This is the module-ized integration of the notebook's 'batch_extract_trainings'.  # Source
    """
    paths = find_markdown_files(training_md_dir)
    existing = read_json(primary_json_path) if primary_json_path.exists() else {}
    batch_backfill_trainings(
        paths=paths,
        save_path=output_json_path,
        existing_data=existing,
        cache_period=cache_period,
        model=model,
    )
    print(f"Backfilled trainings saved to: {output_json_path}")
    return output_json_path
