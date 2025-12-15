"""
Heuristics for title-based skill & level inference and in-domain de-duplication.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import List, Optional, Set


LEVEL_SYNONYMS = {
    "Basic": [
        "basic", "beginner", "beginners", "intro", "introduction", "fundamentals",
        "foundations", "foundation", "essentials", "getting started", "novice",
        "starter", "level 1", "lvl 1", "101", "fundamental",
    ],
    "Intermediate": [
        "intermediate", "practitioner", "applied", "associate",
        "level 2", "lvl 2", "201", "competent",
    ],
    "Advanced": [
        "advanced", "expert", "professional", "pro", "specialist",
        "master", "guru", "deep dive", "level 3", "lvl 3", "301", "senior",
    ],
}
LEVELS = ["Basic", "Intermediate", "Advanced"]

STOPWORDS = set("""
a an and the of for to in with on by from at into over after before under above about as is are be
your our their this that these those using use via intro introduction fundamentals foundations
""".split())


def detect_level_from_title(title: str) -> Optional[str]:
    t = title.lower()
    for level in LEVELS:
        for syn in LEVEL_SYNONYMS[level]:
            if re.search(rf"(?<!\w){re.escape(syn)}(?!\w)", t):
                return level
    return None


def default_level_by_position(pos: int) -> str:
    # pos ∈ {0,1,2}
    return LEVELS[pos]


def normalize_title_for_skill(title: str) -> str:
    t = title.lower()
    for _, syns in LEVEL_SYNONYMS.items():
        for syn in syns:
            t = re.sub(rf"(?<!\w){re.escape(syn)}(?!\w)", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def ngrams(words: List[str], n: int) -> List[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def candidate_skill_from_titles(titles: List[str]) -> Optional[str]:
    """
    Heuristic: find a repeated trigram/bigram across the titles after removing level words.
    Prefer a phrase present in ≥2 titles; fallback to longest ≥2-token phrase.
    """
    normalized = [normalize_title_for_skill(t) for t in titles]
    token_lists = [[w for w in n.split() if w not in STOPWORDS] for n in normalized]
    counts: dict[str, int] = {}
    for nval in (3, 2):
        counts.clear()
        for toks in token_lists:
            for g in ngrams(toks, nval):
                counts[g] = counts.get(g, 0) + 1
        candidates = [(g, c) for g, c in counts.items() if c >= 2]
        if candidates:
            candidates.sort(key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)
            skill = candidates[0][0]
            skill_tc = " ".join([w.upper() if w.isupper() else w.capitalize() for w in skill.split()])
            return skill_tc
    flat = [" ".join(toks) for toks in token_lists]
    flat.sort(key=len, reverse=True)
    for phrase in flat:
        if len(phrase.split()) >= 2:
            return " ".join([w.capitalize() for w in phrase.split()])
    return None


def normalize_text(s: str) -> str:
    t = s.lower()
    for _, syns in LEVEL_SYNONYMS.items():
        for syn in syns:
            t = re.sub(rf"(?<!\w){re.escape(syn)}(?!\w)", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(s: str) -> List[str]:
    return [w for w in normalize_text(s).split() if w not in STOPWORDS]


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def normalize_skill_for_compare(skill: str) -> Set[str]:
    toks = tokenize(skill)
    generic = {
        "skill", "skills", "techniques", "basics", "advanced", "intermediate",
        "introduction", "intro", "foundations", "fundamentals", "applied",
        "expert", "pro", "101", "201", "301", "level", "lvl",
    }
    return set([t for t in toks if t not in generic])


def find_discriminative_phrase(titles: List[str], used_skills_tokens: List[Set[str]]) -> Optional[str]:
    """
    Pick a 2–3 word n-gram from titles that:
    - appears in ≥2 of the titles (prefer),
    - is NOT already dominated by tokens in used skills.
    """
    normalized = [normalize_title_for_skill(t) for t in titles]
    token_lists = [[w for w in n.split() if w not in STOPWORDS] for n in normalized]

    def rank_candidates(n_val: int) -> List[str]:
        counts = Counter()
        per_title_ngrams: List[Set[str]] = []
        for toks in token_lists:
            grams = ngrams(toks, n_val)
            per_title_ngrams.append(set(grams))
            counts.update(grams)
        title_freq = Counter()
        for grams in per_title_ngrams:
            for g in grams:
                title_freq[g] += 1
        cands: List[tuple[str, int, int, int]] = []  # (g, titles, cnt, len)
        for g, c in counts.items():
            cands.append((g, title_freq[g], c, len(g)))
        cands.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        return [g for g, _, _, _ in cands]

    used_flat = set().union(*used_skills_tokens) if used_skills_tokens else set()
    for n_val in (3, 2):
        for g in rank_candidates(n_val):
            toks = set(g.split())
            if not toks.issubset(used_flat):
                return " ".join([w.upper() if w.isupper() else w.capitalize() for w in g.split()])
    return None


def ensure_domain_unique_skill(
    proposed_skill: str,
    titles: List[str],
    used_skills: List[str],
    jaccard_threshold: float = 0.7,
) -> str:
    """
    Ensure the skill name is distinct from already used skills in the domain.
    If collision or high similarity, append a 1–3 word qualifier derived from titles.
    """
    prop_tokens = normalize_skill_for_compare(proposed_skill)
    used_tokens_list = [normalize_skill_for_compare(s) for s in used_skills]
    for used_skill, used_tokens in zip(used_skills, used_tokens_list):
        if proposed_skill.lower() == used_skill.lower():
            qualifier = find_discriminative_phrase(titles, used_tokens_list) or None
            return f"{proposed_skill} ({qualifier})" if qualifier else f"{proposed_skill} (Variant)"
        sim = jaccard(prop_tokens, used_tokens)
        if sim >= jaccard_threshold:
            qualifier = find_discriminative_phrase(titles, used_tokens_list) or None
            return f"{proposed_skill} ({qualifier})" if qualifier else f"{proposed_skill} (Topic)"
    return proposed_skill
