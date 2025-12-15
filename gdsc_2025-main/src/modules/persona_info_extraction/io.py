"""
I/O and persistence functions for persona info extraction.
Extracted from PersonaInfo_Extraction.ipynb notebook.
"""

import csv
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import UsageRecord, estimate_emissions_gco2

# Directory setup constants
SESSIONS_DIR = Path("submissions/sessions")
PERSONAS_DIR = Path("submissions")
USAGE_LOG_PATH = Path("agent_usage_log.csv")

# Ensure directories exist
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
PERSONAS_DIR.mkdir(parents=True, exist_ok=True)


def log_usage(rec: UsageRecord) -> None:
    """Append a single usage record to CSV (creates header on first write)."""
    exists = USAGE_LOG_PATH.exists()
    with USAGE_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow([
                "ts","persona_id","phase","model",
                "prompt_tokens","completion_tokens","total_tokens",
                "latency_s","ok","error","est_gco2"
            ])
        w.writerow([
            rec.ts, rec.persona_id, rec.phase, rec.model,
            rec.prompt_tokens, rec.completion_tokens, rec.total_tokens,
            rec.latency_s, rec.ok, rec.error or "", f"{rec.est_gco2:.6f}"
        ])


def _today_personas_path() -> Path:
    """Generate the file path for today's personas JSON file."""
    return PERSONAS_DIR / f"personas_{datetime.date.today().isoformat()}.json"


def load_personas_map() -> Dict[str, Any]:
    """Load the personas map from today's file, return empty dict if not found."""
    p = _today_personas_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.stat().st_size else {}
    except Exception:
        return {}


def save_personas_map_entry(persona_id: str, profile: Dict[str, Any], session_summary: Dict[str, Any]) -> Path:
    """
    Upsert a single entry with both the final PersonaInfo and a compact session summary.
    """
    p = _today_personas_path()
    data = load_personas_map()
    data[persona_id] = {
        "profile": profile or {},
        "session": session_summary or {}
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def session_jsonl_path(persona_id: str, when: Optional[datetime.date] = None) -> Path:
    """Generate the JSONL file path for a persona's session data."""
    d = (when or datetime.date.today()).isoformat()
    folder = SESSIONS_DIR / d
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{persona_id}.jsonl"


def append_turn(persona_id: str, role: str, text: str, conversation_id: Optional[str], turn_index: int) -> None:
    """Append a conversation turn to the persona's JSONL file."""
    if not conversation_id:
        print(f"[WARN] append_turn: missing conversation_id (role={role}, turn={turn_index}, persona={persona_id})")
    rec = {
        "ts": time.time(),
        "persona_id": persona_id,
        "conversation_id": conversation_id,
        "role": role,
        "turn": turn_index,
        "text": text,
    }
    path = session_jsonl_path(persona_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_full_conversation(result, output_dir="submissions/full_conversations"):
    """Save a complete conversation to a JSON file."""
    # Ensure directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    persona_id = result.get("persona_id")
    conversation_id = result.get("conversation_id") or "unknown"
    # Join all conversation lines into one string
    full_conversation = "\n".join(result.get("conversation", []))

    # Build payload
    payload = {
        "persona_id": persona_id,
        "conversation_id": conversation_id,
        "conversation": full_conversation
    }

    # Save to JSON file
    file_path = Path(output_dir) / f"{persona_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved full conversation for {persona_id} to {file_path}")
    return file_path


def _save_persona_submission_map(profile: dict, persona_id: str) -> Path:
    """
    Persist personas to submissions/personas_{YYYY-MM-DD}.json as a single JSON object.
    """
    submissions_dir = Path("../submissions")
    submissions_dir.mkdir(parents=True, exist_ok=True)
    fname = submissions_dir / f"personas_{datetime.date.today().isoformat()}.json"

    data = {}
    if fname.exists():
        try:
            import json as _json
            with fname.open("r", encoding="utf-8") as f:
                loaded = _json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
        except Exception:
            # Corrupt or unexpected format; start fresh map
            data = {}

    # Update/insert the persona entry (value is the actual JSON object, not a string)
    data[persona_id] = profile

    import json as _json
    with fname.open("w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)

    return fname


def _session_summary_from_state(state) -> dict:
    """Create a session summary from interview state."""
    conv_ids = state.get("conversation_ids", [])
    transcript = state.get("conversation", [])
    return {
        "conversation_ids": conv_ids,
        "message_count": len(transcript),
        "has_transcript_file": os.path.exists(str(session_jsonl_path(state["persona_id"]))),
        "model": state.get("model", ""),
        "phase1_turns": state.get("turn_p1", 0),
        "phase2_turns": state.get("turn_p2", 0),
        "completed_phase1": bool(state.get("phase1_done")),
        "completed_phase2": bool(state.get("phase2_done")),
        "saved_at": time.time(),
    }


def _safe_load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, returning only well-formed objects."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                # Skip malformed lines but keep going.
                print(f"[warn] skipping malformed JSON at line {i} in {path.name}")
    return records


def _load_existing_personas_map() -> dict:
    """Return the current per-day personas map ({} if missing)."""
    p = _today_personas_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}