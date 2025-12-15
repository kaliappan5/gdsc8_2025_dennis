# src/common/env.py
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional: don't crash if not installed

def project_root() -> Path:
    # Adjust if your layout differs; this assumes src/common/env.py depth = 2
    return Path(__file__).resolve().parents[2]

def load_project_dotenv(override: bool = False):
    if not load_dotenv:
        return None
    root = project_root()
    for name in (".env.local", ".env", "env"):
        p = root / name
        if p.is_file():
            load_dotenv(p, override=override)
            return p
    return None
