# src/modules/knowledge_graph/ranking.py
from typing import Optional, List
from src.config import SKILL_LEVELS, SKILL_RANK
from src.common.canonicalization import (
    to_bia_skill_level, normalize_education_level,
)

def skill_level_rank(level: Optional[str]) -> int:
    if not level:
        return 0
    bia = to_bia_skill_level(level)
    return SKILL_RANK.get(bia, 0)

def level_meets(min_level: Optional[str], have_level: Optional[str]) -> bool:
    min_r = skill_level_rank(min_level or "Basic")
    have_r = skill_level_rank(have_level)
    return have_r >= min_r

def levels_between_exclusive_inclusive(cur: Optional[str], target: Optional[str]) -> List[str]:
    cur_r = skill_level_rank(cur)
    tgt_r = skill_level_rank(target)
    if tgt_r == 0:
        max_rank = max(SKILL_RANK.values())
        tgt_r = 1 if cur_r == 0 else min(cur_r + 1, max_rank)
    ordered = [(lvl, SKILL_RANK[lvl]) for lvl in SKILL_LEVELS]
    return [lvl for (lvl, r) in ordered if r > cur_r and r <= tgt_r]

def edu_rank(level: Optional[str]) -> int:
    if not level:
        return 0
    code, canon = normalize_education_level(level)
    return code or 0

def edu_meets(min_required: Optional[str], have: Optional[str]) -> bool:
    if not min_required:
        return True
    return edu_rank(have) >= edu_rank(min_required)
