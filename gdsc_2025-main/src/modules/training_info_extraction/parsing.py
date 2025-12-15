"""
Parsing utilities: title extraction from Markdown and robust JSON extraction from LLM text.
"""

from __future__ import annotations
import json
import re
from typing import Any, Optional


def extract_title(md_text: str) -> Optional[str]:
    """
    Extract a title either from YAML front matter or the first ATX H1 heading.
    """
    # YAML front matter title
    m = re.search(r"^---\n.*?title:\s*\"?(.*?)\"?\s*$.*?\n---", md_text, re.S | re.M)
    if m:
        return m.group(1).strip()
    # First ATX heading
    m = re.search(r"^\s*#\s+(.+)$", md_text, re.M)
    if m:
        return m.group(1).strip()
    return None


def json_from_llm_text(text: str) -> dict[str, Any] | None:
    """
    Extract first {...} JSON object from free-form LLM output.
    Tries to remove trivial trailing commas; also retries after newline stripping.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    # Remove simple trailing commas before ] or }
    candidate = re.sub(r",\s*(?=(\]|\}))", r"", candidate)
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except Exception:
        try:
            result = json.loads(candidate.replace("\n", " ").replace("\r", " "))
            return result if isinstance(result, dict) else None
        except Exception:
            return None


def extract_json(text: str) -> dict[str, Any] | None:
    """
    More generic JSON extractor: attempts full parse, then first {...} block search.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None
