"""
Backfilling remaining training fields from course descriptions (Markdown) using an LLM,
without overwriting critical fields (domain, skill), and preserving any existing non-None
values for other fields.

Integrated from the notebook logic in 'training_file_creation.txt' (module-ized, SoC).  # Source
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import create_model

# Reuse canonicalizers and normalizers from shared modules
from src.common.canonicalization import canonicalize_language
from src.modules.knowledge_graph.normalization import norm_delivery_mode

# Project prompts & models (reuse TrainingInfo for types)
from src.prompts import make_training_extraction_prompt
from src.models import TrainingInfo

# LLM client (plain text -> dict content)
from .llm_client import call_llm
from .parsing import json_from_llm_text
from .io import load_file_content, read_json, write_json

# Fields we are allowed to backfill (never touch 'domain' / 'skill')
TRAINING_FILLABLE_FIELDS: List[str] = ["level", "language", "delivery_mode", "city", "duration"]


def _postprocess_training_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize language & delivery_mode for the fields we handle.
    """
    if "language" in d:
        d["language"] = canonicalize_language(d.get("language"))
    if "delivery_mode" in d:
        d["delivery_mode"] = norm_delivery_mode(d.get("delivery_mode"))
    return d


def _make_partial_model(base_model, fields: List[str]):
    """
    Create a Pydantic (v2) model with only 'fields' from base_model, all Optional with default None.
    """
    field_defs = {}
    for name, f in base_model.model_fields.items():
        if name in fields:
            ann = f.annotation
            field_defs[name] = (Optional[ann], None)  # Optional[...] with default None
    Partial = create_model(
        f"Partial{base_model.__name__}_{abs(hash(tuple(sorted(fields))))}",
        **field_defs,
    )
    return Partial


def extract_training_info_from_file(
    path: Path,
    existing: Optional[Dict[str, Any]] = None,
    model: str = "mistral-medium-latest",
) -> Dict[str, Any]:
    """
    Extract ONLY the remaining fields (excluding domain & skill) using the LLM.
    Preserves existing non-None values and never overwrites domain/skill.
    Returns a plain dict.

    Based on the notebook code you shared (adapted to call_llm + project helpers).  # Source
    """
    text = load_file_content(path)

    # Compute which fields to ask for (only the current None/empty ones in the allowed set)
    if existing is not None:
        missing_fields = [k for k in TRAINING_FILLABLE_FIELDS if existing.get(k) in (None, "")]
        fields_to_extract = missing_fields
    else:
        fields_to_extract = TRAINING_FILLABLE_FIELDS[:]

    # Nothing to request?
    if not fields_to_extract:
        return dict(existing) if existing else {}

    # Prompt limited to needed fields
    prompt = make_training_extraction_prompt(fields_to_extract)
    prompt = f"{prompt}\n\n{text}"

    # Call LLM and parse a dict
    resp = call_llm(prompt, model=model) or {}
    content = resp.get("content") or ""
    data = json_from_llm_text(content) or {}

    # Validate/shape via a partial TrainingInfo model
    PartialModel = _make_partial_model(TrainingInfo, fields_to_extract)
    try:
        validated = PartialModel(**data)
        partial_dict = validated.model_dump()
    except Exception:
        # If validation fails, store only keys we recognize (set to None otherwise)
        partial_dict = {f: data.get(f) for f in fields_to_extract}

    # Merge back: start with existing (if any), overwrite only fields we asked for
    merged = dict(existing) if isinstance(existing, dict) else {}
    for f in fields_to_extract:
        # Keep prior non-None as-is; fill only if prior is None/empty
        if merged.get(f) not in (None, ""):
            continue
        merged[f] = partial_dict.get(f)

    # Never overwrite domain/skill (preserve exactly as they came)
    for critical in ("domain", "skill"):
        if existing and critical in existing:
            merged[critical] = existing[critical]
    # Also preserve id/title if present
    for keep in ("training_id", "title"):
        if existing and keep in existing:
            merged[keep] = existing[keep]

    merged = _postprocess_training_dict(merged)
    return merged


def _coerce_record(value: Any) -> Optional[Dict[str, Any]]:
    """
    Ensure each record is a dict. If it's a JSON-encoded string (double-encoded),
    parse it. Return None if it can't be coerced to a dict.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def batch_backfill_trainings(
    paths: List[Path],
    save_path: Path,
    existing_data: Optional[Dict[str, Any]] = None,
    cache_period: int = 20,
    model: str = "mistral-medium-latest",
) -> Dict[str, Any]:
    """
    Process training Markdown files; backfill only allowed fields (no domain/skill),
    and save as dict-of-dicts (append-safe with periodic cache writes).

    Based directly on the notebook's batch logic (module-ized).  # Source
    """
    # Ensure file exists to allow append-safe periodic writes
    if not save_path.exists():
        write_json(save_path, {})

    extracted = read_json(save_path)
    total = len(paths)
    print(f"Processing {total} files ({len(extracted)} already cached).")

    counter = 0
    new_items = 0

    def _existing_for_id(id_: str) -> Optional[Dict[str, Any]]:
        if not existing_data:
            return None
        return _coerce_record(existing_data.get(id_))

    for path in paths:
        training_id = path.stem
        if training_id in extracted:
            continue

        existing = _existing_for_id(training_id)
        try:
            info_dict = extract_training_info_from_file(path, existing=existing, model=model)
            extracted[training_id] = info_dict  # store as dict
            counter += 1
            new_items += 1
            if counter % cache_period == 0:
                write_json(save_path, extracted)
        except Exception as e:
            print(f"Error processing {training_id}: {e}")

    write_json(save_path, extracted)
    print(f"✅ Completed. New items processed: {new_items}")
    return extracted
