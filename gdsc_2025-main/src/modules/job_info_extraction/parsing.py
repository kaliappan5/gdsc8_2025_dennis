# src/modules/job_info_extraction/parsing.py
# Re-export the robust JSON extraction used by training extractor.
from src.modules.training_info_extraction.parsing import json_from_llm_text

__all__ = ["json_from_llm_text"]
