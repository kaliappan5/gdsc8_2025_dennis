"""
Matching & training recommendation utilities on top of the knowledge graph.

Assumptions:
- Graph is networkx.MultiGraph built by modules.knowledge_graph.graph.build_kg
- Edges carry "rel" attributes, like "has_skill", "requires_skill", "teaches", ...
- Node attributes as per graph.build_kg:
  persona: age, location_city, open_to_relocate, preferred_work_type, education_level,
           years_experience, languages, current_focus, top_domain
  job:     title, domain, location_city, work_type, education_level_required,
           years_of_experience_required, languages_required
  skill:   name
  language: code
  training: title, domain, city, delivery_mode, language, level, skill
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from .utils import _persona_key, _job_key, _norm_key
from src.modules.knowledge_graph.normalization import (
    norm_work_type,
    norm_delivery_mode,
)
from src.modules.knowledge_graph.ranking import (
    level_meets,
    levels_between_exclusive_inclusive,
    edu_meets,
    skill_level_rank,  # used for "next level" logic in standalone training recs
)
from src.modules.knowledge_graph.domain_utils import domains_match


# =============================================================================
# MultiGraph-only edge & node helpers
# =============================================================================

def _edge_datas(G: nx.MultiGraph, u: str, v: str) -> List[Dict[str, Any]]:
    """
    Return a list of edge-attribute dicts between u and v in a MultiGraph.
    """
    if not G.has_edge(u, v):
        return []
    data = G.get_edge_data(u, v)  # {key: {attr...}, ...}
    try:
        return list(data.values())
    except Exception:
        return []


def _has_rel(G: nx.MultiGraph, u: str, v: str, rel: str) -> bool:
    return any(e.get("rel") == rel for e in _edge_datas(G, u, v))


def _first_edge_with_rel(
    G: nx.MultiGraph, u: str, v: str, rel: str
) -> Optional[Dict[str, Any]]:
    for e in _edge_datas(G, u, v):
        if e.get("rel") == rel:
            return e
    return None


def _node_attr(G: nx.MultiGraph, node: str, attr: str, default=None):
    return G.nodes[node].get(attr, default)


# =============================================================================
# Language & skill helpers
# =============================================================================

def _persona_languages(G: nx.MultiGraph, pnode: str) -> Set[str]:
    """
    Collect language codes (node attribute 'code') that the persona speaks.
    """
    langs: Set[str] = set()
    for nbr in G.neighbors(pnode):
        if G.nodes[nbr].get("type") == "language" and _has_rel(G, pnode, nbr, "speaks"):
            code = _node_attr(G, nbr, "code")
            if code:
                langs.add(code)
    return langs


def _job_languages(G: nx.MultiGraph, jnode: str) -> Set[str]:
    """
    Collect language codes (node attribute 'code') that the job requires.
    """
    langs: Set[str] = set()
    for nbr in G.neighbors(jnode):
        if G.nodes[nbr].get("type") == "language" and _has_rel(G, jnode, nbr, "requires_language"):
            code = _node_attr(G, nbr, "code")
            if code:
                langs.add(code)
    return langs


def _persona_skill_level(G: nx.MultiGraph, pnode: str, snode: str) -> Optional[str]:
    """
    Return the level from the 'has_skill' edge between persona and skill.
    """
    e = _first_edge_with_rel(G, pnode, snode, "has_skill")
    return e.get("level") if e else None


def _job_required_skills(G: nx.MultiGraph, jnode: str) -> Dict[str, str]:
    """
    Map of skill-node-id -> min_level for the job.
    """
    req: Dict[str, str] = {}
    for nbr in G.neighbors(jnode):
        if G.nodes[nbr].get("type") == "skill":
            e = _first_edge_with_rel(G, jnode, nbr, "requires_skill")
            if e:
                req[nbr] = e.get("min_level", "Basic")
    return req


# =============================================================================
# Hard filters & job recommendations
# =============================================================================

def hard_filters_pass(G: nx.MultiGraph, persona_id: str, job_id: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Apply strict eligibility filters:
      1) Persona domain == Job domain (canonical domains via DOMAIN_ALIAS)
      2) Location compatible (same city unless persona open_to_relocate or job is remote)
      3) Language overlap (if job requires languages, persona must speak at least one)
      4) Education level: persona >= job requirement (Brazilian ladder)
      5) Years of experience: persona >= job requirement
      + Awareness check: age >= 16

    Returns:
        (ok, reasons_dict)
    """
    pnode = _persona_key(persona_id)
    jnode = _job_key(job_id)
    reasons: Dict[str, Any] = {}

    if not (G.has_node(pnode) and G.has_node(jnode)):
        return False, {"missing_nodes": True}

    # Awareness: minimum age
    age = _node_attr(G, pnode, "age")
    if isinstance(age, (int, float)) and age < 16:
        return False, {"awareness": "too_young"}

    # 1) Domain exact match using canonicalization (tolerant)
    p_dom = _node_attr(G, pnode, "top_domain", "")
    j_dom = _node_attr(G, jnode, "domain", "")
    domain_ok = domains_match(p_dom, j_dom)
    reasons["domain_ok"] = domain_ok
    if not domain_ok:
        return False, reasons

    # 2) Location
    open_to_reloc = bool(_node_attr(G, pnode, "open_to_relocate", False))
    p_city = _node_attr(G, pnode, "location_city", "")
    j_city = _node_attr(G, jnode, "location_city", "")
    j_work_type = norm_work_type(_node_attr(G, jnode, "work_type", ""))

    if open_to_reloc:
        location_ok = True
    else:
        if p_city:
            location_ok = (_norm_key(p_city) == _norm_key(j_city)) or (j_work_type == "remote")
        else:
            location_ok = True  # no city set → don't constrain
    reasons["location_ok"] = location_ok
    if not location_ok:
        return False, reasons

    # 3) Languages overlap
    p_langs = _persona_languages(G, pnode)
    j_langs = _job_languages(G, jnode)
    langs_ok = True if not j_langs else (len(p_langs.intersection(j_langs)) > 0)
    reasons["languages_ok"] = langs_ok
    if not langs_ok:
        return False, reasons

    # 4) Education: persona >= job requirement (Brazilian ladder via src.config)
    p_edu = _node_attr(G, pnode, "education_level")
    j_edu = _node_attr(G, jnode, "education_level_required")
    edu_ok = edu_meets(j_edu, p_edu)
    reasons["education_ok"] = edu_ok
    if not edu_ok:
        return False, reasons

    # 5) Experience: persona >= job requirement
    p_years = float(_node_attr(G, pnode, "years_experience", 0.0) or 0.0)
    j_years = float(_node_attr(G, jnode, "years_of_experience_required", 0.0) or 0.0)
    exp_ok = p_years >= j_years
    reasons["experience_ok"] = exp_ok
    if not exp_ok:
        return False, reasons

    return True, reasons


def recommend_jobs_strict(
    G: nx.MultiGraph,
    persona_id: str,
    top_k: int = 10
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return only jobs that pass ALL hard filters.
    Ranking: (same_city desc, skills_coverage desc)

    If persona prefers 'onsite' and any onsite jobs exist among candidates,
    suppress remote jobs from the final list (keep onsite and hybrid).
    """
    pnode = _persona_key(persona_id)
    if not G.has_node(pnode):
        return []

    age = _node_attr(G, pnode, "age")
    if isinstance(age, (int, float)) and age < 16:
        return []

    # Persona skills map
    have_levels: Dict[str, str] = {}
    for nbr in G.neighbors(pnode):
        if G.nodes[nbr].get("type") == "skill":
            e = _first_edge_with_rel(G, pnode, nbr, "has_skill")
            if e:
                have_levels[nbr] = e.get("level", "Basic")

    p_city = _node_attr(G, pnode, "location_city", "")
    pref_work_norm = norm_work_type(_node_attr(G, pnode, "preferred_work_type", ""))

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for jnode, jd in [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "job"]:
        job_id = jnode.split(":", 1)[1]
        ok, factors = hard_filters_pass(G, persona_id, job_id)
        if not ok:
            continue

        # Skills coverage
        req = _job_required_skills(G, jnode)
        total = max(1, len(req))
        covered = sum(
            1 for snode, min_lv in req.items()
            if snode in have_levels and level_meets(min_lv, have_levels[snode])
        )
        coverage = covered / total

        # Same-city bonus
        same_city = int(bool(p_city) and _norm_key(p_city) == _norm_key(_node_attr(G, jnode, "location_city", "")))

        # Normalize job work type
        j_work_norm = norm_work_type(_node_attr(G, jnode, "work_type", ""))

        candidates.append(
            (job_id, {
                "factors": factors,
                "skills_coverage": coverage,
                "same_city": same_city,
                "work_type_norm": j_work_norm,
                "persona_pref_work_type_norm": pref_work_norm,
            })
        )

    # Optional suppression: prefer onsite -> drop remote if any onsite exists
    if pref_work_norm == "onsite":
        any_onsite = any(info["work_type_norm"] == "onsite" for _, info in candidates)
        if any_onsite:
            candidates = [(jid, info) for jid, info in candidates if info["work_type_norm"] != "remote"]

    # Sort and slice
    candidates.sort(key=lambda x: (x[1]["same_city"], x[1]["skills_coverage"]), reverse=True)
    return candidates[:top_k]


# =============================================================================
# Training recommendations (gap-closing and standalone)
# =============================================================================

def trainings_to_close_gaps_strict(
    G: nx.MultiGraph,
    persona_id: str,
    job_id: str,
    top_per_level: int = 1,
) -> Dict[str, List[Dict[str, Any]]]]:
    """
    For each required job skill that the persona does not yet meet,
    recommend up to `top_per_level` trainings at each required level step
    (strictly above current up to the job's minimum), filtered by:
      - same domain (normalized) as the job
      - training actually 'teaches' that skill
    Scoring favors:
      - language match with persona (+1.0)
      - online/virtual delivery (+0.1)
    """
    pnode = _persona_key(persona_id)
    jnode = _job_key(job_id)
    age = _node_attr(G, pnode, "age")
    if isinstance(age, (int, float)) and age < 16:
        return {}

    req = _job_required_skills(G, jnode)
    results: Dict[str, List[Dict[str, Any]]] = {}
    persona_langs = _persona_languages(G, pnode)
    job_dom_norm = _norm_key(_node_attr(G, jnode, "domain", ""))

    # Persona existing skill levels
    have_levels: Dict[str, str] = {}
    for nbr in G.neighbors(pnode):
        if G.nodes[nbr].get("type") == "skill":
            e = _first_edge_with_rel(G, pnode, nbr, "has_skill")
            if e:
                have_levels[nbr] = e.get("level", "Basic")

    for snode, jreq_level in req.items():
        sname = _node_attr(G, snode, "name", "")
        cur_level = have_levels.get(snode)

        # If already meets requirement → no training for this skill
        if cur_level and level_meets(jreq_level, cur_level):
            continue

        needed_levels = levels_between_exclusive_inclusive(cur_level, jreq_level)

        recs_for_skill: List[Dict[str, Any]] = []
        for needed in needed_levels:
            candidates: List[Tuple[str, float]] = []
            for tnode in G.neighbors(snode):
                if G.nodes[tnode].get("type") != "training":
                    continue
                if not _has_rel(G, tnode, snode, "teaches"):
                    continue

                t_dom_norm = _norm_key(_node_attr(G, tnode, "domain", ""))
                if t_dom_norm != job_dom_norm:
                    continue

                t_level = _node_attr(G, tnode, "level", "")
                if t_level != needed:
                    continue

                lang_bonus = 1.0 if (_node_attr(G, tnode, "language", "") in persona_langs) else 0.0
                online_bonus = 0.1 if norm_delivery_mode(_node_attr(G, tnode, "delivery_mode", "")) == "online" else 0.0
                score = lang_bonus + online_bonus
                candidates.append((tnode, score))

            if candidates:
                candidates.sort(key=lambda x: (x[1], x[0].split(":", 1)[1]), reverse=True)
                for best_tnode, sc in candidates[:max(1, int(top_per_level))]:
                    recs_for_skill.append({
                        "training_id": best_tnode.split(":", 1)[1],
                        "title": _node_attr(G, best_tnode, "title", ""),
                        "level": _node_attr(G, best_tnode, "level", ""),
                        "language": _node_attr(G, best_tnode, "language", ""),
                        "domain": _node_attr(G, best_tnode, "domain", ""),
                        "score": sc
                    })

        results[sname] = recs_for_skill  # may be empty
    return results


def recommend_trainings_standalone(
    G: nx.MultiGraph,
    persona_id: str,
    top_per_skill: int = 1
) -> Dict[str, List[Dict[str, Any]]]]:
    """
    Recommend trainings for skills the persona wants to learn (training_motivation),
    at exactly the next level above their current level for that skill.
    Filters:
      - Same domain as persona top_domain (normalized)
      - Training teaches the skill
      - Training level equals selected target level
    Scoring favors:
      - language match with persona (+1.0)
      - online/virtual delivery (+0.1)
    """
    pnode = _persona_key(persona_id)
    age = _node_attr(G, pnode, "age")
    if isinstance(age, (int, float)) and age < 16:
        return {}

    persona_langs = _persona_languages(G, pnode)
    persona_dom_norm = _norm_key(_node_attr(G, pnode, "top_domain", ""))

    # Desired skills from training_motivation
    desired_nodes: List[str] = []
    for nbr in G.neighbors(pnode):
        if G.nodes[nbr].get("type") == "skill" and _has_rel(G, pnode, nbr, "wants_training_in"):
            desired_nodes.append(nbr)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for snode in desired_nodes:
        sname = _node_attr(G, snode, "name", "")
        cur_level = _persona_skill_level(G, pnode, snode)

        # Determine exactly ONE target level (next level logic)
        cur_r = skill_level_rank(cur_level)
        if cur_r <= 0:
            target_levels = ["Basic"]
        elif cur_r == 1:
            target_levels = ["Intermediate"]
        elif cur_r == 2:
            target_levels = ["Advanced"]
        else:
            target_levels = []

        chosen: List[Dict[str, Any]] = []
        for target in target_levels:
            candidates: List[Tuple[str, float]] = []
            for tnode in G.neighbors(snode):
                if G.nodes[tnode].get("type") != "training":
                    continue
                if not _has_rel(G, tnode, snode, "teaches"):
                    continue

                t_dom_norm = _norm_key(_node_attr(G, tnode, "domain", ""))
                if t_dom_norm != persona_dom_norm:
                    continue
                t_level = _node_attr(G, tnode, "level", "")
                if t_level != target:
                    continue

                lang_bonus = 1.0 if (_node_attr(G, tnode, "language", "") in persona_langs) else 0.0
                online_bonus = 0.1 if norm_delivery_mode(_node_attr(G, tnode, "delivery_mode", "")) == "online" else 0.0
                score = lang_bonus + online_bonus
                candidates.append((tnode, score))

            if candidates:
                candidates.sort(key=lambda x: (x[1], x[0].split(":", 1)[1]), reverse=True)
                for best_tnode, sc in candidates[:max(1, int(top_per_skill))]:
                    chosen.append({
                        "training_id": best_tnode.split(":", 1)[1],
                        "title": _node_attr(G, best_tnode, "title", ""),
                        "level": _node_attr(G, best_tnode, "level", ""),
                        "language": _node_attr(G, best_tnode, "language", ""),
                        "domain": _node_attr(G, best_tnode, "domain", ""),
                        "score": sc
                    })

        out[sname] = chosen  # [] or up to top_per_skill items (usually ≤ 1 by design)
    return out


# =============================================================================
# High-level wrappers & batch runner
# =============================================================================

def recommend_for_persona(G: nx.MultiGraph, persona_id: str, jobs_top_k: int = 10) -> Dict[str, Any]:
    """
    Returns:
      - {'type':'awareness','reason':'too_young'} if under 16
      - {'type':'jobs','jobs':[...]} with strict jobs and per-job gap trainings otherwise
    """
    pnode = _persona_key(persona_id)
    age = _node_attr(G, pnode, "age")
    if isinstance(age, (int, float)) and age < 16:
        return {"type": "awareness", "reason": "too_young"}

    jobs = recommend_jobs_strict(G, persona_id, top_k=jobs_top_k)
    out = {"type": "jobs", "persona_id": persona_id, "jobs": []}
    for job_id, meta in jobs:
        gaps = trainings_to_close_gaps_strict(G, persona_id, job_id, top_per_level=2)
        out["jobs"].append({
            "job_id": job_id,
            "factors": meta["factors"],
            "skills_coverage": meta["skills_coverage"],
            "trainings_for_missing_skills": gaps
        })
    return out


def recommend_trainings_standalone_for_persona(
    G: nx.MultiGraph, persona_id: str, top_per_skill: int = 2
) -> Dict[str, Any]:
    pnode = _persona_key(persona_id)
    age = _node_attr(G, pnode, "age")
    if isinstance(age, (int, float)) and age < 16:
        return {"type": "awareness", "reason": "too_young"}
    return {
        "type": "trainings_standalone",
        "persona_id": persona_id,
        "trainings": recommend_trainings_standalone(G, persona_id, top_per_skill=top_per_skill)
    }


def _flatten_training_ids_from_gaps(gaps_dict: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    seen, out = set(), []
    for _skill, items in gaps_dict.items():
        for it in items:
            tid = it.get("training_id")
            if tid and tid not in seen:
                seen.add(tid)
                out.append(tid)
    return out


def _flatten_training_ids_from_standalone(standalone_dict: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """
    standalone_dict: { skill_name: [ {training_id,...}, ... ], ... }
    Returns a unique, order-preserving list of training_id strings across skills.
    """
    seen, out = set(), []
    for _skill, items in standalone_dict.items():
        for it in items:
            tid = it.get("training_id")
            if tid and tid not in seen:
                seen.add(tid)
                out.append(tid)
    return out


def persona_ids_from_graph(G: nx.MultiGraph) -> List[str]:
    return sorted([n.split(":", 1)[1] for n, d in G.nodes(data=True) if d.get("type") == "persona"])


def get_persona_attr(G: nx.MultiGraph, persona_id: str, key: str, default=None):
    return G.nodes.get(_persona_key(persona_id), {}).get(key, default)


def batch_recommendations(
    G: nx.MultiGraph,
    jobs_top_k: int = 10,
    top_per_level: int = 2,
) -> List[Dict[str, Any]]:
    """
    End-to-end persona routing:
      - If age < 16              -> awareness: too_young
      - If current_focus=awareness -> awareness: info
      - If current_focus=training_only -> trainings_only (flattened training IDs from standalone)
      - Else jobs+trainings: top jobs + flattened gap-training IDs
    """
    results: List[Dict[str, Any]] = []
    for persona_id in persona_ids_from_graph(G):
        age   = get_persona_attr(G, persona_id, "age")
        focus = (get_persona_attr(G, persona_id, "current_focus", "") or "").strip().lower()

        # Case 1: Awareness (too young)
        if isinstance(age, (int, float)) and age < 16:
            results.append({
                "persona_id": persona_id,
                "predicted_type": "awareness",
                "predicted_items": "too_young"
            })
            continue
        # Case 2: Awareness (info mode)
        if focus == "awareness":
            results.append({
                "persona_id": persona_id,
                "predicted_type": "awareness",
                "predicted_items": "info"
            })
            continue

        # Case 3: Trainings-only mode
        if focus == "training_only":
            standalone_map = recommend_trainings_standalone(G, persona_id, top_per_skill=top_per_level)
            training_ids = _flatten_training_ids_from_standalone(standalone_map)
            results.append({
                "persona_id": persona_id,
                "predicted_type": "trainings_only",
                "trainings": training_ids
            })
            continue

        # Case 4: Jobs + trainings
        jobs = recommend_jobs_strict(G, persona_id, top_k=jobs_top_k)
        if not jobs:
            results.append({
                "persona_id": persona_id,
                "predicted_type": "jobs+trainings",
                "jobs": []
            })
            continue

        job_entries = []
        for job_id, _meta in jobs:
            gaps = trainings_to_close_gaps_strict(G, persona_id, job_id, top_per_level=top_per_level)
            suggested_ids = _flatten_training_ids_from_gaps(gaps)
            job_entries.append({
                "job_id": job_id,
                "suggested_trainings": suggested_ids
            })

        results.append({
            "persona_id": persona_id,
            "predicted_type": "jobs+trainings",
            "jobs": job_entries
        })

    return results
