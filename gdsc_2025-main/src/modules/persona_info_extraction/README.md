# Persona Info Extraction Module

This module provides a complete system for conducting persona interviews and extracting structured persona information using a multi-phase conversational flow with LangGraph state machines.

## Architecture

The module implements a **neuro-symbolic interview system** with the following components:

- **Multi-phase conversational flow** using LangGraph state machines
- **Structured information extraction** using LLM agents with Pydantic models
- **Domain-aware skill filtering** and validation
- **Incremental profile building** and merging across interview phases
- **AWS persona API integration** for conversation management
- **Comprehensive logging and persistence** for conversation tracking

## Key Components

### Core Modules

- **`models.py`** - Data models (PersonaInfo, InterviewState, UsageRecord)
- **`interview.py`** - LangGraph state machine and interview flow logic
- **`llm_client.py`** - LLM communication and AWS API integration
- **`validation.py`** - Data validation, normalization, and skill filtering
- **`extraction.py`** - Profile extraction and merging functions  
- **`io.py`** - Persistence, logging, and file I/O operations
- **`utils.py`** - Utility functions for text processing and formatting
- **`pipeline.py`** - High-level orchestration and main entry points

### Interview Flow

The system implements a **two-phase interview process**:

#### Phase 1 - Intent and Context Collection
- Collects user's career/training goals in their own words
- Determines focus type: jobs+trainings, training_only, or awareness
- Identifies professional domain from KNOWN_JOB_DOMAINS
- Captures basic information: age, location, languages

#### Phase 2 - Detailed Information (Path-dependent)
- **Job Path**: Education, work preferences, experience, job roles
- **Training Path**: Current skills, learning motivations  
- **Awareness Path**: Brief domain orientation and overview

#### Profile Building
- Incremental extraction and merging across phases
- Domain-specific skill filtering using ALLOWED_SKILLS
- Final validation and normalization
- Optional recommendations using knowledge graph

### Data Models

#### PersonaInfo (Pydantic Model)
```python
class PersonaInfo(BaseModel):
    # Core identification
    full_name: str
    age: Optional[int] 
    location_city: str
    
    # Professional context
    top_domain: str
    current_focus: str  # "jobs+trainings" | "training_only" | "awareness"
    intent: Optional[str]  # Free-text career goal
    
    # Skills and experience
    technical_skills: List[dict]  # [{"name": "skill", "level": "Basic|Intermediate|Advanced"}]
    training_motivation: List[str]  # Skills to pursue
    languages: List[str]
    
    # Job-specific (when applicable)
    education_level: Optional[str]
    years_experience: Optional[float]
    preferred_work_type: str
    open_to_relocate: bool
    desired_job_roles: List[str]
```

#### InterviewState (TypedDict)
```python
class InterviewState(TypedDict):
    persona_id: str
    conversation: List[str]
    profile_dict: Dict[str, Any]
    
    # Phase management
    phase1_done: bool
    phase2_done: bool
    turn_p1: int 
    turn_p2: int
    
    # Routing signals
    domain: str
    age: Optional[int]
    current_focus: Optional[str]
    intent: Optional[str]
    
    # API integration
    conversation_id: Optional[str]
    conversation_ids: List[str]
```

## Usage

### Basic Interview

```python
from src.modules.persona_info_extraction.pipeline import main_interview_pipeline
from src.agents import get_agent

result = main_interview_pipeline(
    "persona_001",
    get_agent=get_agent,
    model_id="mistral-small-latest",
    max_turns_p1=5,
    max_turns_p2=5
)

if "error" not in result:
    print(f"Interview completed: {result['profile']['full_name']}")
    print(f"Domain: {result['profile']['top_domain']}")
    print(f"Focus: {result['profile']['current_focus']}")
```

### Direct Profile Extraction

```python
from src.modules.persona_info_extraction.extraction import extract_and_merge_profile

profile = extract_and_merge_profile(
    extractor_agent=get_agent("extractor"),
    intent_agent=get_agent("intent"),
    transcript="User: I want to learn data science...\nAssistant: What's your background?",
    persona_id="persona_001",
    model_id="mistral-small-latest"
)
```

### Custom Interview Flow

```python
from src.modules.persona_info_extraction.interview import build_interview_app_with_profile
from src.config import KNOWN_JOB_DOMAINS, ALLOWED_SKILLS

app = build_interview_app_with_profile(
    get_agent=get_agent,
    model_id="mistral-small-latest", 
    max_turns_p1_default=3,
    max_turns_p2_default=4,
    KNOWN_JOB_DOMAINS=KNOWN_JOB_DOMAINS,
    ALLOWED_SKILLS=ALLOWED_SKILLS
)

initial_state = {
    "persona_id": "test_001",
    "conversation": [],
    "profile_dict": {}
}

result = app.invoke(initial_state)
final_profile = result["profile_dict"]
```

## Configuration

The module integrates with the main project configuration:

- **Domains**: `src.config.KNOWN_JOB_DOMAINS` - List of valid professional domains
- **Skills**: `src.config.ALLOWED_SKILLS` - Domain→skills mapping for filtering
- **Education**: `src.config.EDU_LEVELS` - Valid education level progression
- **Agents**: `src.agents.get_agent()` - LLM agent factory function

## Dependencies

### Core Dependencies
- **LangGraph** - State machine orchestration
- **Pydantic** - Data validation and models
- **boto3** - AWS API integration
- **requests** - HTTP client for persona API

### Project Dependencies
- **src.config** - Domain and skill configurations
- **src.agents** - LLM agent management
- **src.logs** - Interview event logging
- **src.reco** - Knowledge graph recommendations (optional)

## File Structure

```
src/modules/persona_info_extraction/
├── __init__.py              # Module initialization
├── models.py                # Data models and types
├── interview.py             # LangGraph state machine
├── llm_client.py            # LLM and AWS API client
├── validation.py            # Data validation and filtering
├── extraction.py            # Profile extraction and merging
├── io.py                    # Persistence and I/O operations  
├── utils.py                 # Utility functions
├── pipeline.py              # High-level orchestration
└── README.md               # This documentation
```
