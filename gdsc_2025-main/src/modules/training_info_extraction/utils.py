"""
Compatibility layer combining frequently used helpers by re-exporting from SoC modules.
Use these functions if you need to maintain existing imports.
"""

from __future__ import annotations
from pathlib import Path

# Parsing
from .parsing import extract_title, json_from_llm_text

# Skills
from .skills import (
    canonicalize_skill,
    load_skill_catalog,
    save_skill_catalog,
    map_to_catalog,
)

# I/O
from .io import find_markdown_files

# Level normalization (single source of truth)
from src.common.canonicalization import to_bia_skill_level as normalize_level_to_bia
``
