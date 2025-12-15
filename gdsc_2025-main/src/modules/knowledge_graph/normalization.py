# src/modules/knowledge_graph/normalization.py
from typing import Optional
from src.config import WorkType
from .utils import _norm_key
# Map common aliases to canonical WorkType values
_WORK_TYPE_NORM = {
    "onsite": "onsite",
    "on site": "onsite",
    "on-site": "onsite",
    "office": "onsite",
    "hybrid": "hybrid",
    "remote": "remote",
    "wfh": "remote",
    "unspecified": "unspecified",
}

def norm_work_type(value: Optional[str]) -> WorkType:
    if not value:
        return "unspecified"
    return _WORK_TYPE_NORM.get(_norm_key(value), "unspecified")  # type: ignore

# Delivery mode normalization: merge virtual/remote-understood into online
_DELIVERY_MODE_NORM = {
    "online": "online",
    "virtual": "online",
    "remote": "online",
    "in-person": "in-person",
    "in person": "in-person",
    "classroom": "in-person",
    "unspecified": "unspecified",
}

def norm_delivery_mode(value: Optional[str]) -> str:
    if not value:
        return "unspecified"
    return _DELIVERY_MODE_NORM.get(_norm_key(value), "unspecified")
