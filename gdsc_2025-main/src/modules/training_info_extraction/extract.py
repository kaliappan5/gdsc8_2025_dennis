"""
Extraction loop and main pipeline for training info extraction.
"""

import json
import time
from pathlib import Path

import pandas as pd

from src.agents import call_mistral
from src.prompts import make_training_extraction_prompt

from .utils import (
    canonicalize_skill,
    extract_title,
    find_markdown_files,
    json_from_llm_text,
    load_skill_catalog,
    map_to_catalog,
    normalize_level_to_bia,
    save_skill_catalog,
)


def main() -> None:
    # Paths (adjust as needed)
    # CSV_PATH = Path('../processed_data/jobs_extracted_labels_en_blocked.csv')
    TRAININGS_DIR = Path("../../GDSC-8/data/trainings")
    OUT_DIR = Path("../processed_data/outputs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ALLOWED_TXT = OUT_DIR / "allowed_domains.txt"
    JSONL_PATH = OUT_DIR / "trainings_extracted_one_skill_21_10_2025_2.jsonl"
    CSV_OUT_PATH = OUT_DIR / "trainings_extracted_one_skill_21_10_2025_2.csv"
    SKILL_CATALOG_PATH = OUT_DIR / "skill_catalog_21_10_2025_2.json"

    # Load allowed domains
    domains = [
        ln.strip()
        for ln in ALLOWED_TXT.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    # Load skill catalog
    skill_catalog = load_skill_catalog(SKILL_CATALOG_PATH)

    # Find markdown files
    files = find_markdown_files(TRAININGS_DIR)

    rows = []
    for i, md_path in enumerate(files, 1):
        md_text = md_path.read_text(encoding="utf-8", errors="ignore")
        title = extract_title(md_text) or md_path.stem
        prompt = make_training_extraction_prompt()
        prompt += f'\n\nTraining description:\n"""\n{md_text}\n"""'
        result = call_mistral(prompt, model="mistral-medium-latest")
        data = json_from_llm_text(result.get("content", ""))
        if not data:
            print(f"⚠️ JSON parse failed for {md_path}; saving placeholder row.")
            data = {
                "course_title": title,
                "domain": "",
                "skill_name": "",
                "skill_level": "Basic",
                "evidence": "",
                "confidence": 0.0,
            }

        # --- Normalize domain to allowed list ---
        domain = (data.get("domain") or "").strip()
        if domain not in domains:
            norm_map = {d.lower(): d for d in domains}
            domain = norm_map.get(domain.lower(), "")

        # --- Normalize skill level to Basic/Intermediate/Advanced ---
        level = normalize_level_to_bia(data.get("skill_level") or "")

        # --- Canonicalize and map skill name for consistency ---
        raw_skill = (data.get("skill_name") or "").strip()
        canonical = canonicalize_skill(raw_skill)
        final_skill = map_to_catalog(canonical, skill_catalog, cutoff=0.92)
        if final_skill not in skill_catalog:
            skill_catalog.append(final_skill)
            save_skill_catalog(SKILL_CATALOG_PATH, skill_catalog)

        row = {
            "file": str(md_path),
            "course_title": data.get("course_title") or title,
            "domain": domain,
            "skill_name": final_skill,
            "skill_level": level,
            "evidence": (data.get("evidence") or "")[:160],
            "confidence": float(data.get("confidence", 0.0)),
            "model": result.get("model"),
            "duration": float(result.get("duration") or 0.0),
            "input_tokens": int(result.get("input_tokens") or 0),
            "output_tokens": int(result.get("output_tokens") or 0),
            "total_tokens": int(result.get("total_tokens") or 0),
            "timestamp": time.time(),
        }

        with JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        rows.append(row)
        print(
            f"[{i}/{len(files)}] {md_path.name} | domain='{row['domain']}' | skill='{row['skill_name']}' | level={row['skill_level']}"
        )
        time.sleep(0.3)

    if rows:
        pd.DataFrame(rows).to_csv(CSV_OUT_PATH, index=False)
        print(f"Saved CSV -> {CSV_OUT_PATH}")
        print(f"Saved JSONL -> {JSONL_PATH}")
    else:
        print("No new rows (resume mode or failures).")


if __name__ == "__main__":
    main()
