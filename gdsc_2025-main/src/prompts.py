from string import Template
from typing import List, Optional

from src.config import KNOWN_JOB_DOMAINS

domains_bulleted = "\n  - " + "\n  - ".join(KNOWN_JOB_DOMAINS)


def make_job_extraction_prompt(fields_to_extract: Optional[List[str]] = None) -> str:
    """
    Returns a job extraction prompt, including only the fields in fields_to_extract (if provided).
    If fields_to_extract is None, includes all fields.
    """
    all_fields = [
        ("job_id", "if present in text; else empty"),
        ("title", None),
        ("company", "if present; else empty"),
        (
            "domain",
            "choose ONE from:\n$domains\n  * Guidance:\n    - Port/terminal/vessel/scheduling/dispatch → Maritime & Port Operations\n    - HSE/NR/OSHA/safety → Safety, Health & Environment (HSE)\n    - Statutory/legal/interpretation → Legal & Statutory\n    - Banking ops/regulatory reports → Finance & Banking Operations\n    - Bookkeeping/tax/ERP/accounting → Accounting & Tax\n    - Research/studies/prototyping → Design Research Studies and Development\n    - Studio/arts/design/media → Creative Studio & Visual Arts",
        ),
        ("location_city", "city string if stated; else null"),
        (
            "work_type",
            'one of ["remote","onsite","hybrid","unspecified"]\n  * Map “in-person/on-site/field/terminal presence required” → onsite; “home office/remote” → remote; “hybrid” → hybrid.',
        ),
        (
            "languages_required",
            'list of codes (use "pt-br","en" etc.). If proficiency not stated, omit levels.',
        ),
        (
            "education_level_required",
            'one of the Brazilian ladder (or null if unspecified)\n  ["Ensino Fundamental","Ensino Médio","Técnico","Tecnólogo","Graduação","Bacharelado","Licenciatura","Pós-graduação","Especialização","Mestrado","MBA","Doutorado"]',
        ),
        (
            "years_of_experience_required",
            "float (if a range like “1–3”, set the minimum, e.g., 1.0)",
        ),
        (
            "required_skills",
            'list of objects → { "name": str, "min_level": one of ["Basic","Intermediate","Advanced"] | null }\n  * If level unstated, leave null (do NOT guess).',
        ),
    ]
    if fields_to_extract is not None:
        fields = [f for f in all_fields if f[0] in fields_to_extract]
    else:
        fields = all_fields
    field_lines = []
    for name, desc in fields:
        if desc:
            if name == "domain":
                field_lines.append(f"- {name}: {desc}")
            else:
                field_lines.append(f"- {name}: {desc}")
        else:
            field_lines.append(f"- {name}")
    fields_block = "\n".join(field_lines)
    prompt = Template(f"""
Extract structured job requirements from the job description. Populate the target schema fields precisely; leave unknowns null/empty.

Target fields (map to JobInfo):
{fields_block}

Rules:
- Use semantic understanding for category mapping (e.g., sustainability ≈ environmental/HSE; analytics ≈ data analysis).
- Prefer explicit statements over general ones where contradictions exist.
- Output strictly the fields above; no explanations or extra keys.

Job description:
""").substitute(domains=domains_bulleted)
    return prompt


def make_training_extraction_prompt(
    fields_to_extract: Optional[List[str]] = None,
) -> str:
    """
    Returns a training extraction prompt, including only the fields in fields_to_extract (if provided).
    If fields_to_extract is None, includes all fields.
    """
    all_fields = [
        ("training_id", "if stated; else empty"),
        ("title", "if stated; else empty"),
        (
            "domain",
            "choose ONE from the known domains list ($domains) if evident; else null",
        ),
        (
            "skill",
            "main skill taught (e.g., 'Maritime Safety', 'Tax Compliance', 'Project Coordination')",
        ),
        (
            "level",
            'one of ["Basic","Intermediate","Advanced"]\n  * If the level is implied by words like “introduction/fundamentals” → Basic; “intermediate/advanced” → map accordingly.',
        ),
        ("language", "language code like 'pt-br', 'en' if stated; else null"),
        ("delivery_mode", "'online', 'in-person' (or null if not specified)"),
        ("city", "if stated; else null"),
    ]
    if fields_to_extract is not None:
        fields = [f for f in all_fields if f[0] in fields_to_extract]
    else:
        fields = all_fields
    field_lines = []
    for name, desc in fields:
        if desc:
            field_lines.append(f"- {name}: {desc}")
        else:
            field_lines.append(f"- {name}")
    fields_block = "\n".join(field_lines)
    prompt = Template(f"""
Extract structured training info. Populate the target schema fields; leave unknowns null/empty.

Target fields (map to TrainingInfo):
{fields_block}

Rules:
- Use semantic inference to classify domain (e.g., SOLAS/MARPOL → Maritime & Port Operations; NR/OSHA → Safety, Health & Environment (HSE)).
- Do not invent details not present in the text.
- Output strictly the fields above; no explanations or extra keys.

Training description:
""").substitute(domains=domains_bulleted)
    return prompt
