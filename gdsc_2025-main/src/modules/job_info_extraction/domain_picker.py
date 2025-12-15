# src/modules/job_info_extraction/domain_picker.py
from __future__ import annotations
import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.agents import call_mistral  # ← use your Mistral wrapper

# === Config & caches (resume-safe) ===
DOMAIN_CACHE = Path("processed_data/jobs_domain_cache.json")
ROME_CACHE   = Path("processed_data/jobs_rome_cache.json")
BLOCK_CACHE  = Path("processed_data/block_domain_cache.json")

CHUNK_SIZE = 35
SEEN_ONLY_MIN_CONF = 60.0
NEW_DOMAIN_STRONG_CONF = 88.0
CONF_MARGIN = 5.0
MAX_UNIQUE_DOMAINS = 20  # soft cap

@dataclass
class TaxonomyData:
    domain_en_list: List[str]
    domain_to_romes_en: Dict[str, List[str]]
    fr_dom_map: Dict[str, str]
    fr_rome_map: Dict[str, str]
    domain_cache: Dict[str, dict]
    rome_cache: Dict[str, dict]
    block_cache: Dict[str, dict]
    prior_counts: Dict[str, int]

def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def compute_content_key(path: Path, text: str) -> str:
    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{path.as_posix()}::{h}"

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _excerpt(text: str, max_chars: int = 1400) -> str:
    return _normalize_whitespace(text)[:max_chars]

def prepare_taxonomy(taxonomy_csv: Path) -> TaxonomyData:
    df = pd.read_csv(taxonomy_csv, dtype=str, encoding="utf-8").fillna("")
    required = {"libelle_domaine_professionel_en", "libelle_rome_en"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in taxonomy CSV: {missing}")

    dom_list = sorted(df["libelle_domaine_professionel_en"].dropna().unique().tolist())
    d2r = (
        df.dropna(subset=["libelle_domaine_professionel_en", "libelle_rome_en"])
          .groupby("libelle_domaine_professionel_en")["libelle_rome_en"]
          .apply(lambda s: sorted(set([x for x in s.tolist() if x])))
          .to_dict()
    )

    fr_dom_map, fr_rome_map = {}, {}
    if "libelle_domaine_professionel" in df.columns:
        fr_dom_map = (
            df.dropna(subset=["libelle_domaine_professionel_en", "libelle_domaine_professionel"])
              .drop_duplicates(subset=["libelle_domaine_professionel_en"])
              .set_index("libelle_domaine_professionel_en")["libelle_domaine_professionel"]
              .to_dict()
        )
    if "libelle_rome" in df.columns:
        fr_rome_map = (
            df.dropna(subset=["libelle_rome_en", "libelle_rome"])
              .drop_duplicates(subset=["libelle_rome_en"])
              .set_index("libelle_rome_en")["libelle_rome"]
              .to_dict()
        )

    domain_cache = _load_json(DOMAIN_CACHE)
    rome_cache = _load_json(ROME_CACHE)
    block_cache = _load_json(BLOCK_CACHE)

    priors: Dict[str, int] = {}
    for entry in domain_cache.values():
        dom = (entry.get("libelle_domaine_professionel_en") or "").strip()
        if dom:
            priors[dom] = priors.get(dom, 0) + 1

    return TaxonomyData(
        domain_en_list=dom_list,
        domain_to_romes_en=d2r,
        fr_dom_map=fr_dom_map,
        fr_rome_map=fr_rome_map,
        domain_cache=domain_cache,
        rome_cache=rome_cache,
        block_cache=block_cache,
        prior_counts=priors,
    )

def _build_pick_prompt(task_label: str, job_excerpt: str, options: List[str],
                       prior_hint: Optional[List[str]] = None) -> str:
    opt_lines = "\n".join(f"- {o}" for o in options)
    prior_text = ""
    if prior_hint:
        prior_text = (
            "\nPreviously discovered domains (prefer one of these if appropriate):\n" +
            "\n".join(f"- {o}" for o in prior_hint[:20]) + "\n"
        )
    return f"""You are classifying a job description. Task: **Pick exactly ONE** {task_label} from the options list.
Return JSON ONLY with:
{{
  "{task_label}": "<exact option from the list>",
  "confidence": 0-100
}}
{prior_text}
Job Description (excerpt):
\"\"\"{job_excerpt}\"\"\"
Options (choose exactly one by exact text match):
{opt_lines}
JSON only:
"""

def _llm_pick_json(prompt: str, model: str) -> Optional[dict]:
    res = call_mistral(prompt, model=model)  # your agent wrapper
    txt = (res.get("content", "") if isinstance(res, dict) else "").strip()
    if not txt:
        return None
    # strict parse first
    try:
        # find first {...}
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            return json.loads(m.group(0))
        return json.loads(txt)
    except Exception:
        return None

def mistral_pick_one(task_label: str, job_excerpt: str, options: List[str],
                     prior_hint: Optional[List[str]], model: str) -> Optional[Tuple[str, float]]:
    if not options:
        return None
    prompt = _build_pick_prompt(task_label, job_excerpt, options, prior_hint)
    obj = _llm_pick_json(prompt, model=model)
    if obj and isinstance(obj.get(task_label), str):
        choice = obj[task_label]
        conf = float(obj.get("confidence", 0) or 0)
        if choice in options:
            return choice, conf
    # Retry with a stricture
    prompt += "\nReminder: Return EXACTLY one option label from the list, not paraphrased."
    obj = _llm_pick_json(prompt, model=model)
    if obj and isinstance(obj.get(task_label), str):
        choice = obj[task_label]
        conf = float(obj.get("confidence", 0) or 0)
        if choice in options:
            return choice, conf
    return None

def tournament_pick(task_label: str, job_excerpt: str, options: List[str],
                    prior_hint: Optional[List[str]], model: str,
                    chunk_size: int = CHUNK_SIZE) -> Optional[Tuple[str, float]]:
    if not options:
        return None
    if len(options) <= chunk_size:
        return mistral_pick_one(task_label, job_excerpt, options, prior_hint, model)
    winners: List[Tuple[str, float]] = []
    for i in range(0, len(options), chunk_size):
        chunk = options[i:i+chunk_size]
        res = mistral_pick_one(task_label, job_excerpt, chunk, prior_hint, model)
        if res:
            winners.append(res)
    if not winners:
        return None
    final_opts = [w[0] for w in winners]
    return mistral_pick_one(task_label, job_excerpt, final_opts, prior_hint, model)

def tournament_pick_biased_domains(job_excerpt: str,
                                   all_domains: List[str],
                                   seen_domains_sorted: List[str],
                                   model: str) -> Optional[Tuple[str, float]]:
    if not all_domains:
        return None
    seen_set = set(seen_domains_sorted)
    ordered = seen_domains_sorted + [d for d in all_domains if d not in seen_set]
    prior_hint = seen_domains_sorted[:20] if seen_domains_sorted else None
    return tournament_pick("libelle_domaine_professionel_en", job_excerpt, ordered, prior_hint, model, chunk_size=CHUNK_SIZE)

def pick_domain_with_bias(content_key: str,
                          job_excerpt: str,
                          taxonomy: TaxonomyData,
                          model: str) -> Tuple[str, float]:
    dc = taxonomy.domain_cache
    if content_key in dc:
        dom = dc[content_key].get("libelle_domaine_professionel_en", "") or ""
        conf = float(dc[content_key].get("confidence", 0) or 0)
        return dom, conf

    seen_sorted = [d for d, _ in sorted(taxonomy.prior_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    picked = tournament_pick_biased_domains(job_excerpt, taxonomy.domain_en_list, seen_sorted, model=model)
    dom_en, dom_conf = (picked if picked else ("", 0.0))

    if dom_en and dom_en not in taxonomy.prior_counts and len(taxonomy.prior_counts) >= MAX_UNIQUE_DOMAINS:
        seen_only_opts = seen_sorted[:max(5, min(25, len(seen_sorted)))]
        seen_pick = tournament_pick("libelle_domaine_professionel_en", job_excerpt,
                                    seen_only_opts, prior_hint=seen_only_opts, model=model)
        if seen_pick:
            seen_dom, seen_conf = seen_pick
            if (dom_conf < NEW_DOMAIN_STRONG_CONF) and (seen_conf >= SEEN_ONLY_MIN_CONF) and ((dom_conf - seen_conf) <= CONF_MARGIN):
                dom_en, dom_conf = seen_dom, seen_conf

    dc[content_key] = {
        "libelle_domaine_professionel_en": dom_en or "",
        "confidence": float(dom_conf or 0),
    }
    _save_json(DOMAIN_CACHE, dc)
    if dom_en:
        taxonomy.prior_counts[dom_en] = taxonomy.prior_counts.get(dom_en, 0) + 1
    return dom_en, dom_conf

def pick_rome_within_domain(content_key: str,
                            job_excerpt: str,
                            domain_en: str,
                            taxonomy: TaxonomyData,
                            model: str) -> Tuple[str, float]:
    if not domain_en:
        return "", 0.0
    rk = f"{content_key}::domain::{domain_en}"
    rc = taxonomy.rome_cache
    if rk in rc:
        r = rc[rk].get("libelle_rome_en", "") or ""
        c = float(rc[rk].get("confidence", 0) or 0)
        return r, c

    options = taxonomy.domain_to_romes_en.get(domain_en, [])
    chosen, conf = ("", 0.0)
    if options:
        res = tournament_pick("libelle_rome_en", job_excerpt, options, prior_hint=None, model=model, chunk_size=CHUNK_SIZE)
        if res:
            chosen, conf = res

    rc[rk] = {
        "libelle_rome_en": chosen or "",
        "confidence": float(conf or 0),
    }
    _save_json(ROME_CACHE, rc)
    return chosen, conf
