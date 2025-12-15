"""
Knowledge Graph construction and manipulation logic.
Separate from LLM output models (see models.py).
"""

from typing import Any, Dict

import networkx as nx

from .utils import (
    _city_key,
    _domain_key,
    _job_key,
    _lang_key,
    _norm_text,
    _persona_key,
    _skill_key,
    _training_key,
    load_bundle,
)


def add_node_once(G: nx.Graph, node_id: str, **attrs: Any) -> None:
    if not G.has_node(node_id):
        G.add_node(node_id, **attrs)
    else:
        G.nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})


def build_kg(
    jobs: Dict[str, Any], trainings: Dict[str, Any], personas: Dict[str, Any]
) -> nx.Graph:
    G = nx.MultiGraph()
    # Personas
    for pid, pobj in personas.items():
        pnode = _persona_key(pid)
        add_node_once(
            G,
            pnode,
            type="persona",
            full_name=pobj.get("full_name"),
            age=pobj.get("age"),
            location_city=pobj.get("location_city"),
            open_to_relocate=pobj.get("open_to_relocate"),
            preferred_work_type=pobj.get("preferred_work_type"),
            education_level=pobj.get("education_level"),
            years_experience=pobj.get("years_experience"),
            languages=pobj.get("languages", []),
            current_focus=pobj.get("current_focus"),
            top_domain=pobj.get("top_domain"),
        )
        # Persona → Domain
        if pobj.get("top_domain"):
            dnode = _domain_key(pobj["top_domain"])
            add_node_once(G, dnode, type="domain", name=_norm_text(pobj["top_domain"]))
            G.add_edge(pnode, dnode, rel="has_domain")
        # Persona → City
        if pobj.get("location_city"):
            cnode = _city_key(pobj["location_city"])
            add_node_once(G, cnode, type="city", name=_norm_text(pobj["location_city"]))
            G.add_edge(pnode, cnode, rel="located_in")
        # Persona → Languages
        for lang in pobj.get("languages", []):
            lnode = _lang_key(lang)
            add_node_once(G, lnode, type="language", code=_norm_text(lang))
            G.add_edge(pnode, lnode, rel="speaks")
        # Persona → Skills (has_skill)
        for s in pobj.get("technical_skills", []):
            if not isinstance(s, dict):
                continue
            sname, level = s.get("name"), s.get("level")
            if not sname:
                continue
            snode = _skill_key(sname)
            add_node_once(G, snode, type="skill", name=_norm_text(sname))
            G.add_edge(pnode, snode, rel="has_skill", level=level)
        # Persona → training_motivation (wants_training_in)
        for s in pobj.get("training_motivation", []):
            if not s:
                continue
            snode = _skill_key(s)
            add_node_once(G, snode, type="skill", name=_norm_text(s))
            G.add_edge(pnode, snode, rel="wants_training_in")

    # Jobs
    for jid, job in jobs.items():
        jnode = _job_key(job.get("job_id", jid))
        add_node_once(
            G,
            jnode,
            type="job",
            title=_norm_text(job.get("title")),
            domain=_norm_text(job.get("domain")),
            location_city=job.get("location_city"),
            work_type=job.get("work_type"),
            education_level_required=job.get("education_level_required"),
            years_of_experience_required=job.get("years_of_experience_required"),
            languages_required=job.get("languages_required", []),
        )
        if job.get("domain"):
            dnode = _domain_key(job["domain"])
            add_node_once(G, dnode, type="domain", name=_norm_text(job["domain"]))
            G.add_edge(jnode, dnode, rel="in_domain")
        if job.get("location_city"):
            cnode = _city_key(job["location_city"])
            add_node_once(G, cnode, type="city", name=_norm_text(job["location_city"]))
            G.add_edge(jnode, cnode, rel="located_in")
        for lang in job.get("languages_required", []):
            lnode = _lang_key(lang)
            add_node_once(G, lnode, type="language", code=_norm_text(lang))
            G.add_edge(jnode, lnode, rel="requires_language")
        for s in job.get("required_skills", []):
            if not isinstance(s, dict):
                continue
            sname, min_level = s.get("name"), s.get("min_level", "Basic")
            if not sname:
                continue
            snode = _skill_key(sname)
            add_node_once(G, snode, type="skill", name=_norm_text(sname))
            G.add_edge(jnode, snode, rel="requires_skill", min_level=min_level)

    # Trainings
    for tid, t in trainings.items():
        tnode = _training_key(t.get("training_id", tid))
        add_node_once(
            G,
            tnode,
            type="training",
            title=_norm_text(t.get("title")),
            domain=_norm_text(t.get("domain")),
            city=t.get("city"),
            delivery_mode=t.get("delivery_mode"),
            language=_norm_text(t.get("language")),
            level=t.get("level"),
            skill=_norm_text(t.get("skill")),
        )
        if t.get("domain"):
            dnode = _domain_key(t["domain"])
            add_node_once(G, dnode, type="domain", name=_norm_text(t["domain"]))
            G.add_edge(tnode, dnode, rel="in_domain")
        if t.get("city"):
            cnode = _city_key(t["city"])
            add_node_once(G, cnode, type="city", name=_norm_text(t["city"]))
            G.add_edge(tnode, cnode, rel="delivered_in")
        if t.get("language"):
            lnode = _lang_key(t["language"])
            add_node_once(G, lnode, type="language", code=_norm_text(t["language"]))
            G.add_edge(tnode, lnode, rel="in_language")
        if t.get("skill"):
            snode = _skill_key(t["skill"])
            add_node_once(G, snode, type="skill", name=_norm_text(t["skill"]))
            G.add_edge(tnode, snode, rel="teaches", level=t.get("level"))

    return G


def save_graph(G: nx.Graph, path: str = "kg.gpickle") -> None:
    import os
    import pickle

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        nx.write_gpickle(G, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(G, f)


def load_graph(path: str = "kg.gpickle") -> nx.Graph:
    import os
    import pickle

    if not os.path.exists(path):
        raise FileNotFoundError(os.path.abspath(path))
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


# When run as a script, build the graph from example files and print summary
if __name__ == "__main__":
    # Example file paths (update as needed)
    personas_path = (
        "../submissions/personas_merged_reassigned_domains_21_10_2025_2.json"
    )
    jobs_path = "../submissions/extracted_jobs_merged_2025-10-21.json"
    trainings_path = "../submissions/extracted_trainings_backfilled_2025-10-21_2.json"
    print("Loading data and building knowledge graph...")
    personas, jobs, trainings = load_bundle(personas_path, jobs_path, trainings_path)
    G = build_kg(jobs, trainings, personas)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    save_graph(G, "kg.gpickle")
    print("Graph saved to kg.gpickle.")
