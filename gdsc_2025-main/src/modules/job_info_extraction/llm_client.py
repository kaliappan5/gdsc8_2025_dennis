# src/modules/job_info_extraction/llm_client.py
from __future__ import annotations
from typing import List, Optional

from src.common.env import load_project_dotenv
from src.agents import call_mistral
from src.prompts import make_job_extraction_prompt  # aligned with your prompts.py

# Ensure env is loaded (e.g., MISTRAL_API_KEY)
load_project_dotenv()

def build_job_prompt(source_text: str, fields_to_extract: Optional[List[str]] = None) -> str:
    """
    Compose the full prompt expected by the job extractor.
    Your make_job_extraction_prompt already ends with 'Job description:'.
    We simply append the raw description.
    """
    header = make_job_extraction_prompt(fields_to_extract)
    # Keep the body unwrapped (no Markdown fences), as your agents stack doesn't need them.
    return f"{header}\n{source_text.strip()}"

def call_llm_job(source_text: str,
                 model: str = "mistral-small-latest",
                 fields_to_extract: Optional[List[str]] = None) -> str:
    """
    Call the Mistral API via src.agents and return TEXT content.
    """
    prompt = build_job_prompt(source_text=source_text, fields_to_extract=fields_to_extract)
    res = call_mistral(prompt, model=model)
    return res.get("content", "") if isinstance(res, dict) else ""


def call_llm_job_fields(source_text: str,
                        fields: List[str],
                        model: str = "mistral-small-latest") -> str:
    return call_llm_job(source_text=source_text, model=model, fields_to_extract=fields)
