"""
I/O helpers: scanning, reading/writing CSV/JSON/JSONL, resume sets, checkpoint-safe discovery,
and simple file content loaders.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

import pandas as pd


# -------- File discovery (skips notebook checkpoints) --------

def find_markdown_files(trainings_dir: Path) -> list[Path]:
    """
    Return all .md files under trainings_dir, excluding Jupyter checkpoint artifacts:
      - any path containing ".ipynb_checkpoints"
      - any filename ending with "-checkpoint.md" (or containing "-checkpoint")
    """
    results: list[Path] = []
    for p in trainings_dir.rglob("*.md"):
        if not p.is_file():
            continue
        parts = set(p.parts)
        name = p.name.lower()
        stem = p.stem.lower()
        if ".ipynb_checkpoints" in parts:
            continue
        if name.endswith("-checkpoint.md") or "-checkpoint" in stem:
            continue
        results.append(p)
    return sorted(results)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_file_content(path: Path) -> str:
    """Alias used by backfill; reads Markdown content as UTF-8."""
    return read_text(path)


# -------- JSON/JSONL & CSV writing --------

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


# -------- Resume set --------

def load_resume_set(jsonl_path: Path) -> tuple[set[str], list[str]]:
    """
    Return (done_set_of_file_paths, known_skills_from_previous_runs).
    """
    done_set: set[str] = set()
    known_skills: list[str] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    file_path = obj.get("file")
                    if file_path:
                        done_set.add(file_path)
                    sk = obj.get("skill_name")
                    if sk:
                        known_skills.append(sk)
                except Exception:
                    pass
    return done_set, known_skills



# -------- Allowed domains --------

def derive_allowed_domains_from_jobs_csv(csv_path: Path, column: str) -> list[str]:
    """
    Build allowed domains list from a jobs CSV column (as notebook does).
    """
    df = pd.read_csv(csv_path)
    assert column in df.columns, f"Column '{column}' not found. Available: {list(df.columns)}"
    series = (
        df[column]
        .astype(str)
        .map(str.strip)
        .replace({"nan": None})
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return series.tolist()


def write_allowed_domains_txt(path: Path, domains: Iterable[str]) -> None:
    path.write_text("\n".join(domains), encoding="utf-8")


# -------- Cleaning extracted CSV --------

def _extract_number_from_path(file_path: str) -> int | None:
    m = re.search(r"(\d+)", file_path)
    return int(m.group(1)) if m else None


def clean_extracted_csv(input_csv_path: Path, output_csv_path: Path) -> int:
    """
    - Remove rows where 'file' contains 'checkpoint'
    - Extract numeric 'index' from the filename
    - Sort by 'index' and save
    Returns number of rows saved.
    """
    df = pd.read_csv(input_csv_path)
    if "file" not in df.columns:
        raise ValueError("CSV must contain a 'file' column.")
    df = df[~df["file"].astype(str).str.contains("checkpoint", case=False)]
    df["index"] = df["file"].astype(str).apply(_extract_number_from_path)
    df = df.dropna(subset=["index"])
    df = df.sort_values(by="index").reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)
    return len(df)


# -------- Relabel domains by index ranges --------

def relabel_domains_by_index_ranges(
    df: pd.DataFrame,
    mapping_ranges: list[Tuple[int, int, str]],
    canonical_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    mapping_ranges: [(start_idx, end_idx, user_label), ...]
    canonical_map: optional user_label -> canonical domain label
    """
    if "index" not in df.columns or "domain" not in df.columns:
        raise ValueError("DataFrame must have 'index' and 'domain' columns.")
    df = df.copy()
    df["index"] = pd.to_numeric(df["index"], errors="coerce").astype("Int64")

    for start, end, user_label in mapping_ranges:
        canon = canonical_map[user_label] if canonical_map else user_label
        mask = df["index"].between(start, end, inclusive="both")
        df.loc[mask, "domain"] = canon
    return df
