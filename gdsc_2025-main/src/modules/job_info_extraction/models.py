# src/modules/job_info_extraction/models.py
from typing import Any, Dict
from src.models import JobInfo  # Provided by your repo (Pydantic model)

def validate_job_info(payload: Dict[str, Any]) -> JobInfo:
    """
    Validate against the canonical JobInfo Pydantic model.
    Raises pydantic.ValidationError on failure.
    """
    return JobInfo.model_validate(payload)  # pydantic v2 style
