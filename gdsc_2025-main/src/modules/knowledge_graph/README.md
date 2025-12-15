# Knowledge Graph Module

Constructs and queries a **typed multi‑relational graph** of personas, jobs, trainings, skills, domains, cities, and languages to enable **symbolic matching** and **gap‑closing** recommendations.

## Features

- **Graph construction** (`nx.MultiGraph`)
  - Node types: `persona`, `job`, `training`, `skill`, `domain`, `city`, `language`
  - Edge relations: `has_skill`, `wants_training_in`, `requires_skill`, `teaches`, `speaks`, `in_domain`, `located_in`, `in_language`, `delivered_in`
  - Careful normalization for node IDs & attributes

- **Matching**
  - Hard filters: **domain**, **location/relocation**, **language overlap**, **education ladder**, **experience**
  - Ranking: same‑city bonus + skills coverage (meets/exceeds required min level)
  - Optionally suppress remote jobs if persona prefers onsite

- **Gap‑Closing**
  - Compute missing skill levels to reach job requirements
  - Recommend trainings at each intermediate level (scored by persona language + online delivery)

## Quick Start

```python
from src.modules.knowledge_graph.graph import build_kg, save_graph
from src.modules.knowledge_graph.utils import load_bundle
from src.modules.knowledge_graph.matching import (
    recommend_jobs_strict,
    trainings_to_close_gaps_strict,
    recommend_trainings_standalone,
)

personas, jobs, trainings = load_bundle(
    "submissions/personas.json",
    "submissions/jobs.json",
    "submissions/trainings.json",
)
G = build_kg(jobs, trainings, personas)
save_graph(G, "processed_data/outputs/kg.gpickle")

persona_id = next(iter(personas.keys()))
jobs = recommend_jobs_strict(G, persona_id, top_k=10)
if jobs:
    job_id, meta = jobs[0]
    gaps = trainings_to_close_gaps_strict(G, persona_id, job_id, top_per_level=1)
    standalone = recommend_trainings_standalone(G, persona_id)
    print(job_id, meta, gaps, standalone)
```

## Files

- `graph.py` — Node/edge creation, save/load.
- `matching.py` — Hard filters, ranking, gap‑closing, and batch routing.
- `utils.py` — Normalization helpers, bundle loaders.
