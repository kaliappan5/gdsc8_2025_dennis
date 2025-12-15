# Training Info Extraction Module

LLM‑assisted **one‑skill extraction** from Markdown training descriptions, with **title heuristics**, **canonicalization**, **skill cataloging**, **resume**, and **triplet labeling**.

## Features
- **Prompted extraction** of: `course_title`, `domain`, `skill_name`, `skill_level` (BIA ladder), `evidence`, `confidence`.
- **Robust parsing** (`json_from_llm_text`) and **fallbacks** (title‑based skill/level) when LLM output is missing/invalid.
- **Canonicalization**:
  - Domains via `DOMAIN_ALIAS` (through `canonicalize_domain_or_raise`)
  - Skill names + capitalization + catalog mapping (fuzzy threshold)
  - Levels → `Basic | Intermediate | Advanced`
- **I/O**:
  - Discover `.md` files (skips `.ipynb_checkpoints` and `*-checkpoint.md`)
  - JSONL append (resume‑friendly) and CSV snapshot
- **Post‑processing**:
  - CSV cleaning (drop checkpoint rows, derive numeric index)  
  - Domain relabeling by index ranges  
  - Triplet labeling (group within domain, infer one skill and ordered levels)

## Quick Start

```python
from pathlib import Path
from src.modules.training_info_extraction.pipeline import (
    extract_trainings_one_skill,
    clean_extracted_csv,
    relabel_domains_by_index_ranges,
    label_triplets_from_csv,
)

TRAININGS_DIR = Path("/home/ec2-user/SageMaker/GDSC-8/data/trainings")
OUT_DIR = Path("processed_data/outputs")
JOBS_CSV = Path("processed_data/jobs_extracted_labels_en_blocked.csv")

jsonl_path, csv_path = extract_trainings_one_skill(
    trainings_dir=TRAININGS_DIR,
    out_dir=OUT_DIR,
    jobs_csv_path=JOBS_CSV,
    jobs_csv_domain_column="libelle_domaine_professionel_en",
)
cleaned = OUT_DIR / "trainings_cleaned.csv"
clean_extracted_csv(csv_path, cleaned)

mapping = []  # (210, 239, "Insurance"), ...
relabelled = OUT_DIR / "trainings_relabelled.csv"
relabel_domains_by_index_ranges(cleaned, relabelled, mapping_ranges=mapping)

triplet_out = OUT_DIR / "trainings_triplets.csv"
cache = OUT_DIR / "triplet_skill_cache.json"
label_triplets_from_csv(relabelled, triplet_out, cache_path=cache)
```

## Structure

- `pipeline.py` — Orchestrates extraction, cleaning, relabeling, triplet labeling.
- `llm_client.py` — Prompt builders and LLM call wrapper; loads root `.env`.
- `parsing.py` — Markdown title extraction + robust JSON extraction from LLM text.
- `skills.py` — Skill canonicalization rules and catalog management.
- `heuristics.py` — Title‑based skill/level detection, uniqueness disambiguation within a domain.
- `io.py` — File discovery (checkpoint‑safe), CSV/JSONL readers/writers, resume set helpers.
- `utils.py` / `triplet.py` — Thin compatibility re‑exports.
