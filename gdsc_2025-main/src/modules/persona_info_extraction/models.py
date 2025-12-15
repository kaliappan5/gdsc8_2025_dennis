"""
Data models for the interview system.
Extracted from PersonaInfo_Extraction.ipynb notebook.
"""

from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from pydantic.functional_validators import field_validator

# Constants
G_CO2_PER_1K_TOKENS = 2.85

class PersonaInfo(BaseModel):
    """Pydantic model for storing persona information from interviews."""
    
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        use_enum_values=True,
        validate_assignment=False,
    )

    # Core identification & context
    full_name: str = Field(default="")
    age: Optional[int] = Field(default=None)
    location_city: str = Field(default="unknown")
    open_to_relocate: bool = Field(default=False)
    preferred_work_type: str = Field(default="unspecified")  # onsite/remote/hybrid/either/unspecified
    education_level: Optional[str] = Field(default=None)
    years_experience: Optional[float] = Field(default=None)

    # Skills & goals
    languages: List[str] = Field(default_factory=list)
    top_domain: str = Field(default="unspecified")
    current_focus: str = Field(default="unspecified")  # "job" | "training" | "awareness" (string)

    training_motivation: List[str] = Field(default_factory=list)  # skills to pursue
    technical_skills: List[dict] = Field(default_factory=list)    # e.g., [{"name":"Fraud Detection","level":"Intermediate"}]
    desired_job_roles: List[str] = Field(default_factory=list)

    # Free‑text evolving summary of the user's goal
    intent: Optional[str] = Field(default=None)

    @field_validator("preferred_work_type", mode="before")
    @classmethod
    def _canon_work_type(cls, v):
        if v is None or v == "":
            return "unspecified"
        v = str(v).strip().lower()
        v = {"on-site": "onsite", "on site": "onsite"}.get(v, v)
        return v if v in {"onsite", "remote", "hybrid", "either", "unspecified"} else "unspecified"

    @field_validator("languages", mode="before")
    @classmethod
    def _canon_languages(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return []


class InterviewState(TypedDict, total=False):
    """TypedDict for managing interview state throughout the conversation."""
    
    # Inputs
    persona_id: str

    # Threading
    conversation_id: Optional[str]

    # Runtime
    model: str
    max_turns_p1: int
    max_turns_p2: int

    # Loop bookkeeping
    turn_p1: int
    turn_p2: int
    phase1_done: bool
    phase2_done: bool

    # Agent "next message" buffers
    agent_message_p1: Optional[str]
    agent_message_p2: Optional[str]

    # Conversation transcript
    conversation: List[str]
    conversation_ids: List[str]

    # Key signals (free‑text/values)
    intent: Optional[str]            # evolving free‑text goal
    current_focus: Optional[str]     # "job" | "training" | "awareness" (string)
    domain: str                      # e.g., "Insurance"
    age: Optional[int]               # for routing < 16 -> awareness

    # Incremental profile (merged after each phase and finalized)
    profile_dict: Dict[str, Any]

    # Diagnostics
    meta: Dict[str, Any]


@dataclass
class UsageRecord:
    """Dataclass for tracking API usage and emissions."""
    
    ts: float
    persona_id: str
    phase: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    ok: bool
    error: str | None = None
    # NEW: ecological
    est_gco2: float = 0.0


def estimate_emissions_gco2(total_tokens: int, factor_per_1k: float = G_CO2_PER_1K_TOKENS) -> float:
    """Estimate grams of CO₂e for total token usage."""
    return (float(total_tokens) / 1000.0) * float(factor_per_1k)