
# src/modules/job_info_extraction/__init__.py
from .pipeline import extract_jobs
from .pipeline import ExtractJobsConfig

__all__ = ["extract_jobs", "ExtractJobsConfig"]
