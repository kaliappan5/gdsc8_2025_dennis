"""
Compatibility wrappers re-exporting heuristics functions for triplet/title logic.
"""

from __future__ import annotations

from .heuristics import (
    detect_level_from_title,
    default_level_by_position,
    normalize_title_for_skill,
    candidate_skill_from_titles,
    ensure_domain_unique_skill,
    find_discriminative_phrase,
)
