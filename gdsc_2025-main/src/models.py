
from __future__ import annotations
import csv
import os
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from src.config import (
    EDU_LEVELS,
    EDU_RANK, 
    RANK_TO_EDU,
    KNOWN_JOB_DOMAINS,
    DOMAIN_ALIAS,
    WorkType,
    SKILL_LEVELS,
)
from src.common.canonicalization import (
    canonicalize_language, canonicalize_language_list,
    canonicalize_work_type,
    to_bia_skill_level as normalize_level_to_bia,   # same behavior
    normalize_education_level,
    canonicalize_edu_level,
    canonicalize_domain_or_raise,
)

# -------------------------
# Helpers / canonicalizers
# -------------------------

def _norm_lite(s: str) -> str:
    """Lowercase, strip, replace underscores with hyphens (for language + work-type)."""
    return s.strip().lower().replace("_", "-")


def _normalize_domain_key(s: str) -> str:
    """
    Match the normalization used to build DOMAIN_ALIAS in src.config:
    - lowercase/strip
    - replace ()&,- with spaces
    - collapse whitespace
    """
    s = s.lower().strip()
    s = re.sub(r"[()&,-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


LANGUAGE_MAP = {
    "pt": "pt-br", "pt-br": "pt-br", "pt br": "pt-br",
    "portuguese (br)": "pt-br", "portugues (br)": "pt-br",
    "português (br)": "pt-br", "portugues-br": "pt-br",
    "en": "en", "english": "en", "inglês": "en", "ingles": "en",
    "es": "es", "spanish": "es", "espanol": "es", "español": "es",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

# Build a synonym map -> code (1-based), tuned to your EDU_LEVELS
# NOTE: Keep keys lowercased and accent-stripped.
_EDU_SYNONYMS = {
    # Fundamental / Médio
    "ensino fundamental": EDU_RANK["Ensino Fundamental"],
    "fundamental completo": EDU_RANK["Ensino Fundamental"],
    "ensino medio": EDU_RANK["Ensino Médio"],
    "colegio": EDU_RANK["Ensino Médio"],
    "segundo grau": EDU_RANK["Ensino Médio"],

    # Técnico
    "tecnico": EDU_RANK["Técnico"],
    "curso tecnico": EDU_RANK["Técnico"],

    # Tecnólogo
    "tecnologo": EDU_RANK["Tecnólogo"],

    # Graduação / Bacharelado / Licenciatura
    "graduacao": EDU_RANK["Graduação"],  # generic graduation (pt-br)
    "graduacao completa": EDU_RANK["Graduação"],
    "superior completo": EDU_RANK["Graduação"],  # often used generically
    "bacharel": EDU_RANK["Bacharelado"],
    "bacharelado": EDU_RANK["Bacharelado"],
    "licenciatura": EDU_RANK["Licenciatura"],

    # Pós / Especialização / MBA / Mestrado / Doutorado
    "pos graduacao": EDU_RANK["Pós-graduação"],
    "pos-graduacao": EDU_RANK["Pós-graduação"],
    "lato sensu": EDU_RANK["Pós-graduação"],  # often synonym with especialização
    "especializacao": EDU_RANK["Especialização"],
    "mba": EDU_RANK["MBA"],
    "mestrado": EDU_RANK["Mestrado"],
    "stricto sensu": EDU_RANK["Mestrado"],  # generic stricto can be mestrado/doutorado; prefer conservative
    "doutorado": EDU_RANK["Doutorado"],
    "phd": EDU_RANK["Doutorado"],
}


_ENGLISH_SYNONYMS = {
    "elementary school": EDU_RANK["Ensino Fundamental"],
    "high school": EDU_RANK["Ensino Médio"],
    "technical": EDU_RANK["Técnico"],
    "technologist": EDU_RANK["Tecnólogo"],
    "undergraduate": EDU_RANK["Graduação"],
    "bachelor": EDU_RANK["Bacharelado"],
    "licentiate": EDU_RANK["Licenciatura"],
    "postgraduate": EDU_RANK["Pós-graduação"],
    "specialization": EDU_RANK["Especialização"],
    "master": EDU_RANK["Mestrado"],
    "mba": EDU_RANK["MBA"],
    "doctorate": EDU_RANK["Doutorado"],
    "phd": EDU_RANK["Doutorado"],
}


# -------------------------
# Load domain -> allowed skills mapping (lazy)
# -------------------------

_DOMAIN_SKILLS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "processed_data", "domain_allowed_skills_2025-10-21_adjusted.csv"
)

# Cache
_DOMAIN_ALLOWED_SKILLS: Optional[Dict[str, Set[str]]] = None
_DOMAIN_ALLOWED_SKILLS_LOWER_INDEX: Optional[Dict[str, Dict[str, str]]] = None  # domain -> {lower_skill: canonical_skill}


def _load_domain_allowed_skills() -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, str]]]:
    """
    Load CSV: columns = domain,skill_name
    Returns:
      (domain->set(canonical_skill_names), domain->lower_index(lower_name->canonical_name))
    Domains are resolved to canonical domain labels to match KNOWN_JOB_DOMAINS / DOMAIN_ALIAS.
    """
    global _DOMAIN_ALLOWED_SKILLS, _DOMAIN_ALLOWED_SKILLS_LOWER_INDEX
    if _DOMAIN_ALLOWED_SKILLS is not None and _DOMAIN_ALLOWED_SKILLS_LOWER_INDEX is not None:
        return _DOMAIN_ALLOWED_SKILLS, _DOMAIN_ALLOWED_SKILLS_LOWER_INDEX

    mapping: Dict[str, Set[str]] = {}
    lower_index: Dict[str, Dict[str, str]] = {}

    if not os.path.exists(_DOMAIN_SKILLS_PATH):
        raise FileNotFoundError(
            f"Domain allowed skills CSV not found at '{_DOMAIN_SKILLS_PATH}'. "
            "Expected file with columns: domain,skill_name"
        )

    with open(_DOMAIN_SKILLS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"domain", "skill_name"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV '{_DOMAIN_SKILLS_PATH}' must contain columns: {required_cols}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            raw_dom = (row.get("domain") or "").strip()
            skill = (row.get("skill_name") or "").strip()
            if not raw_dom or not skill:
                continue

            # Canonicalize domain via alias
            try:
                dom = canonicalize_domain_or_raise(raw_dom)
            except ValueError:
                # If CSV has domains not in the allowed list, this ensures we fail fast.
                raise ValueError(
                    f"CSV domain '{raw_dom}' is not recognized as a known domain."
                )

            mapping.setdefault(dom, set()).add(skill)
            lower_index.setdefault(dom, {})[skill.lower()] = skill

    _DOMAIN_ALLOWED_SKILLS = mapping
    _DOMAIN_ALLOWED_SKILLS_LOWER_INDEX = lower_index
    return mapping, lower_index


def _get_allowed_skills_for_domain(dom: str) -> Tuple[Set[str], Dict[str, str]]:
    mapping, lower_index = _load_domain_allowed_skills()
    return mapping.get(dom, set()), lower_index.get(dom, {})


# -------------------------
# Models (strict)
# -------------------------

class TechnicalSkill(BaseModel):
    """
    A skill with a normalized level mapped to the BIA ladder.
    """
    model_config = ConfigDict(extra="ignore")

    name: str
    level: str

    @field_validator("name")
    @classmethod
    def _name_strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("level")
    @classmethod
    def _level_normalize(cls, v: str) -> str:
        bia = normalize_level_to_bia(v)
        # Ensure membership in the known ladder
        return bia if bia in SKILL_LEVELS else "Basic"


class PersonaInfo(BaseModel):
    """
    Enhanced structured profile of a job seeker with strict domain/education/skills validation.
    """
    model_config = ConfigDict(extra="ignore")

    full_name: str = Field(default="", description="Persona's full name")
    age: Optional[int] = Field(default=None, description="Age of the persona")
    location_city: str = Field(default="unknown", description="City of residence")
    open_to_relocate: bool = Field(default=False, description="Willingness to relocate")
    preferred_work_type: WorkType = Field(
        default="unspecified",
        description="Preferred work type - one of: 'onsite', 'remote', 'hybrid', 'either', 'unspecified'",
    )
    education_level: Optional[str] = None
    education_level_code: Optional[int] = None
    years_experience: Optional[float] = Field(default=None, description="Years of professional experience")
    languages: List[str] = Field(default_factory=list, description="Languages spoken (canonicalized)")
    top_domain: str = Field(default="unspecified", description="Primary domain of interest or expertise")
    current_focus: str = Field(default="unspecified", description="Current career or training focus")
    training_motivation: List[str] = Field(default_factory=list, description="Motivations or goals for training")
    # STRICT: skills must be allowed for the specified top_domain
    technical_skills: List[TechnicalSkill] = Field(default_factory=list, description="List of technical skills with levels (BIA)")
    desired_job_roles: List[str] = Field(default_factory=list, description="Target job roles")

    # -------- Field validators --------

    @field_validator("full_name", "location_city", "top_domain", "education_level", "current_focus", mode="before")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("preferred_work_type", mode="before")
    @classmethod
    def _canon_work_type(cls, v) -> WorkType:
        if v in {"onsite", "remote", "hybrid", "either", "unspecified", None, ""}:
            return v or "unspecified"
        return canonicalize_work_type(str(v))

    @field_validator("languages", mode="before")
    @classmethod
    def _canon_languages(cls, v):
        return canonicalize_language_list(v)

    @field_validator("education_level", mode="before")
    @classmethod
    def _edu_only_allowed(cls, v: Optional[str]) -> Optional[str]:
        return canonicalize_edu_level(v)

    @field_validator("top_domain", mode="before")
    @classmethod
    def _domain_only_known(cls, v: Optional[str]) -> str:
        # Allow unspecified as a deliberate "no domain yet"
        if v is None or (isinstance(v, str) and v.strip().lower() in {"", "unspecified"}):
            return "unspecified"
        return canonicalize_domain_or_raise(v)

    
    @model_validator(mode="after")
    def _validate_training_motivation(self) -> "PersonaInfo":
        # If no domain or no motivations, nothing to check
        if not self.top_domain or self.top_domain == "unspecified" or not self.training_motivation:
            return self

        allowed, lower_index = _get_allowed_skills_for_domain(self.top_domain)
        if not allowed:
            # Data issue in CSV
            raise ValueError(
                f"No allowed skills found for domain '{self.top_domain}' when validating training_motivation."
            )

        invalid = [m for m in self.training_motivation if m not in allowed]
        if invalid:
            # Keep strict: fail fast with details
            sample = ", ".join(sorted(list(allowed))[:10])
            raise ValueError(
                "training_motivation contains skills not allowed for domain "
                f"'{self.top_domain}': {invalid}. Sample allowed skills: {sample} ..."
            )
        return self

    
    # -------- Model-wide validation (needs multiple fields) --------

    @model_validator(mode="after")
    def _validate_skills_against_domain(self) -> "PersonaInfo":
        """
        Ensure every technical skill is allowed for the chosen top_domain.
        If top_domain is 'unspecified', but skills are provided, raise.
        """
        if not self.technical_skills:
            return self

        if self.top_domain == "unspecified":
            raise ValueError(
                "technical_skills provided but top_domain is 'unspecified'. "
                "Please set a known domain before adding skills."
            )

        allowed, lower_index = _get_allowed_skills_for_domain(self.top_domain)
        if not allowed:
            # If CSV has no entry for the domain, consider that a data issue.
            raise ValueError(
                f"No allowed skills found for domain '{self.top_domain}' in CSV. "
                f"Please ensure the mapping file is populated."
            )

        invalid: List[str] = []
        # Also upgrade skill names to canonical casing from CSV when matched case-insensitively
        for ts in self.technical_skills:
            key = ts.name.strip().lower()
            if key not in lower_index:
                invalid.append(ts.name)
            else:
                # Replace with canonical label from CSV (preserve official casing)
                ts.name = lower_index[key]

        if invalid:
            sample = ", ".join(sorted(list(allowed))[:10])
            raise ValueError(
                "The following skills are not allowed for domain "
                f"'{self.top_domain}': {invalid}. "
                f"Make sure they exist in processed_data/domain_allowed_skills.csv. "
                f"Sample allowed skills for this domain: {sample} ..."
            )

        return self

    # -------- Presentation --------

    def describe(self) -> str:
        skills = ", ".join([f"{s.name}: {s.level}" for s in self.technical_skills]) or "—"
        langs = ", ".join(self.languages) or "—"
        motives = ", ".join(self.training_motivation) or "—"
        roles = ", ".join(self.desired_job_roles) or "—"
        return (
            f"Name: {self.full_name or '—'}\n"
            f"Age: {self.age if self.age is not None else '—'}\n"
            f"Location: {self.location_city or '—'}\n"
            f"Open to relocate: {self.open_to_relocate}\n"
            f"Preferred work type: {self.preferred_work_type}\n"
            f"Education level: {self.education_level or '—'}\n"
            f"Experience: {self.years_experience if self.years_experience is not None else '—'} years\n"
            f"Languages: {langs}\n"
            f"Current focus: {self.current_focus}\n"
            f"Top domain: {self.top_domain}\n"
            f"Training motivation: {motives}\n"
            f"Technical skills: {skills}\n"
            f"Desired job roles: {roles}"
        )


def normalize_text(s: str) -> str:
    import re
    import unicodedata
    if not s:
        return ""
    t = unicodedata.normalize("NFKD", s)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

class Canonicalizer(BaseModel):
    skill_alias: Dict[str, str] = Field(default_factory=dict)  # normalized -> canonical
    domain_alias: Dict[str, str] = Field(default_factory=dict) # normalized -> canonical

    class Config:
        extra = "ignore"

    def skill(self, raw: str) -> str:
        key = normalize_text(raw)
        return self.skill_alias.get(key, raw.strip())

    def domain(self, raw: str) -> str:
        key = normalize_text(raw)
        return self.domain_alias.get(key, raw.strip())

class JobSkillRequirement(BaseModel):
    name: str
    min_level: Optional[str] = None

    class Config:
        extra = "ignore"

    @property
    def required_rank(self) -> Optional[int]:
        return SKILL_RANK.get(self.min_level) if self.min_level else None

class JobInfo(BaseModel):
    job_id: Optional[str] = ""
    title: Optional[str] = ""
    domain: Optional[str] = None
    location_city: Optional[str] = None
    work_type: Optional[WorkType] = None
    languages_required: Optional[List[str]] = None
    education_level_required: Optional[str] = None
    years_of_experience_required: Optional[float] = None
    required_skills: Optional[List[JobSkillRequirement]] = None
    job_role: Optional[str] = None
    canonicalizer: Optional[Canonicalizer] = Field(default=None, exclude=True)

    class Config:
        extra = "ignore"

    @property
    def edu_rank_required(self) -> Optional[int]:
        return (
            EDU_RANK.get(self.education_level_required)
            if self.education_level_required
            else None
        )

    def required_skills_map(self) -> Dict[str, Optional[int]]:
        can = self.canonicalizer or Canonicalizer()
        out: Dict[str, Optional[int]] = {}
        for r in self.required_skills or []:
            k = can.skill(r.name)
            rr = r.required_rank
            if k in out:
                if rr is not None and (out[k] is None or rr > out[k]):
                    out[k] = rr
            else:
                out[k] = rr
        return out

    def flatten_for_llm(self) -> Dict[str, str]:
        skills_csv = "; ".join(
            f"{r.name}:{r.min_level if r.min_level else 'unspecified'}"
            for r in (self.required_skills or [])
        )
        langs_csv = ", ".join(self.languages_required or [])
        can = self.canonicalizer or Canonicalizer(domain_alias=DOMAIN_ALIAS)
        return {
            "title": self.title or "",
            "domain": can.domain(self.domain or "") if self.domain else "",
            "city": self.location_city or "",
            "work_type": self.work_type or "",
            "languages_required": langs_csv,
            "education_level_required": self.education_level_required or "",
            "years_of_experience_required": (
                "" if self.years_of_experience_required is None
                else str(self.years_of_experience_required)
            ),
            "required_skills": skills_csv,
        }

    def describe(self) -> str:
        skills = (
            ", ".join(
                f"{r.name}:{r.min_level or 'unspecified'}"
                for r in (self.required_skills or [])
            )
            or "—"
        )
        langs = ", ".join(self.languages_required or []) or "—"
        return (
            f"Title: {self.title or '—'}\n"
            f"Domain: {self.domain or '—'}\n"
            f"City: {self.location_city or '—'}\n"
            f"Work type: {self.work_type or '—'}\n"
            f"Education required: {self.education_level_required or '—'}\n"
            f"Experience required: {self.years_of_experience_required or '—'}\n"
            f"Languages: {langs}\n"
            f"Required skills: {skills}\n"
        )

class TrainingInfo(BaseModel):
    training_id: Optional[str] = ""
    title: Optional[str] = ""
    domain: Optional[str] = None
    skill: Optional[str] = ""
    level: Optional[str] = None
    language: Optional[str] = None
    delivery_mode: Optional[str] = None
    city: Optional[str] = None
    duration: Optional[str] = None
    canonicalizer: Optional[Canonicalizer] = Field(default=None, exclude=True)

    class Config:
        extra = "ignore"

    @property
    def skill_canonical(self) -> str:
        can = self.canonicalizer or Canonicalizer()
        return can.skill(self.skill or "")

    @property
    def level_rank(self) -> Optional[int]:
        return SKILL_RANK.get(self.level) if self.level else None

    def flatten_for_llm(self) -> Dict[str, str]:
        can = self.canonicalizer or Canonicalizer(domain_alias=DOMAIN_ALIAS)
        return {
            "training_id": self.training_id or "",
            "title": self.title or "",
            "domain": can.domain(self.domain or "") if self.domain else "",
            "skill": self.skill or "",
            "level": self.level or "",
            "language": self.language or "",
            "delivery_mode": self.delivery_mode or "",
            "city": self.city or "",
            "duration": self.duration or "",
        }

    def describe(self) -> str:
        return (
            f"Title: {self.title or '—'}\n"
            f"Domain: {self.domain or '—'}\n"
            f"Teaches: {self.skill or '—'} (Level: {self.level or '—'})\n"
            f"Duration: {self.duration or '—'}\n"
            f"Language: {self.language or '—'} | "
            f"Mode: {self.delivery_mode or '—'} | City: {self.city or '—'}"
        )
