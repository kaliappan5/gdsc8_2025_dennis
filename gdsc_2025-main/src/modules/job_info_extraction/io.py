# src/modules/job_info_extraction/io.py
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Set

CHECKPOINT_DIR = ".ipynb_checkpoints"

def discover_sources(input_dir: Path | None = None,
                     inputs: Iterable[Path] | None = None,
                     max_files: int | None = None) -> List[Path]:
    """
    Discover .md/.markdown/.txt/.json sources, ignoring notebooks' checkpoints
    and *-checkpoint.md files. Order is stable for run-determinism.
    """
    paths: List[Path] = []
    if inputs:
        for p in inputs:
            if p.exists() and p.is_file():
                paths.append(p)
    if input_dir:
        for p in sorted(input_dir.rglob("*")):
            if not p.is_file():
                continue
            if CHECKPOINT_DIR in p.parts:
                continue
            if p.name.endswith("-checkpoint.md"):
                continue
            if p.suffix.lower() in {".md", ".markdown", ".txt", ".json"}:
                paths.append(p)
    # de-duplicate while preserving order
    seen = set()
    dedup = []
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    if max_files:
        dedup = dedup[:max_files]
    return dedup

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def load_resume_set(jsonl_path: Path) -> Set[str]:
    """
    Collect processed 'source_key' values from JSONL (resume-safe).
    """
    s: Set[str] = set()
    if not jsonl_path.exists():
        return s
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                key = obj.get("source_key") or obj.get("job_id") or obj.get("id")
                if key:
                    s.add(str(key))
            except Exception:
                continue
    return s

def append_jsonl(jsonl_path: Path, record: Dict[str, Any]) -> None:
    ensure_parent(jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def ensure_empty_csv(csv_path: Path, headers: List[str]) -> None:
    if csv_path.exists():
        return
    ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

def append_csv(csv_path: Path, headers: List[str], row: Dict[str, Any]) -> None:
    ensure_parent(csv_path)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in headers})
