"""
Knowledge Graph utility functions: normalization, key helpers, robust JSON loading.
"""

import html
import json
import os
import re
import unicodedata
from typing import Any, Dict


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = html.unescape(str(s)).strip()
    s = _strip_accents(s)
    s = re.sub(r"\s+", " ", s)
    return str(s)


def _norm_key(s: Any) -> str:
    s = _norm_text(s).lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def _persona_key(pid: str) -> str:
    return f"persona:{pid}"


def _job_key(jid: str) -> str:
    return f"job:{jid}"


def _training_key(tid: str) -> str:
    return f"training:{tid}"


def _domain_key(name: str) -> str:
    return f"domain:{_norm_key(name)}"


def _skill_key(name: str) -> str:
    return f"skill:{_norm_key(name)}"


def _city_key(name: str) -> str:
    return f"city:{_norm_key(name)}"


def _lang_key(code: str) -> str:
    return f"language:{_norm_key(code)}"


def load_json_file(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_to_id_map(
    data: Any, *, id_field: str, fallback_prefix: str
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out
    if isinstance(data, list):
        for i, item in enumerate(data):
            obj = item
            if not isinstance(obj, dict):
                out[f"{fallback_prefix}{i}"] = {"raw": obj}
                continue
            key = obj.get(id_field)
            if not key:
                for alt in ("persona_id", "job_id", "training_id", "id", "key", "name"):
                    if obj.get(alt):
                        key = obj[alt]
                        break
            if not key:
                key = f"{fallback_prefix}{i}"
            out[str(key)] = obj
        return out
    return {f"{fallback_prefix}0": {"raw": data}}


def load_bundle(
    personas_path: str, jobs_path: str, trainings_path: str
) -> tuple[
    Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]
]:
    raw_personas = load_json_file(personas_path)
    raw_jobs = load_json_file(jobs_path)
    raw_trainings = load_json_file(trainings_path)
    personas = coerce_to_id_map(
        raw_personas, id_field="persona_id", fallback_prefix="persona_"
    )
    jobs = coerce_to_id_map(raw_jobs, id_field="job_id", fallback_prefix="j")
    trainings = coerce_to_id_map(
        raw_trainings, id_field="training_id", fallback_prefix="tr"
    )
    print(
        f"Loaded → personas={len(personas)} jobs={len(jobs)} trainings={len(trainings)}"
    )
    return personas, jobs, trainings
