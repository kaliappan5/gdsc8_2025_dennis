"""
Pipeline orchestration for persona info extraction.

This module provides the main entry points for running persona interviews
and extracting structured information.
"""

from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import datetime

from .models import InterviewState, PersonaInfo
from .interview import build_interview_app_with_profile
from .io import save_personas_map_entry, _session_summary_from_state, load_personas_map
from .validation import _postprocess_persona
from src.config import KNOWN_JOB_DOMAINS, ALLOWED_SKILLS, EDU_LEVELS
from src.logs import set_log_dir


def run_interview(
    persona_id: str,
    *,
    get_agent: Callable[..., Any],
    model_id: str = "mistral-small-latest",
    max_turns_p1: int = 5,
    max_turns_p2: int = 5,
    conversation_id: Optional[str] = None,
    initial_conversation: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a complete persona interview and return the final state and profile.
    
    Args:
        persona_id: Unique identifier for the persona
        get_agent: Function to create LLM agents
        model_id: Model identifier for the LLM
        max_turns_p1: Maximum turns for Phase 1
        max_turns_p2: Maximum turns for Phase 2
        conversation_id: Optional conversation ID for API
        initial_conversation: Optional existing conversation
        log_dir: Directory for logging
        **kwargs: Additional parameters
        
    Returns:
        Dict containing final state, profile, and session summary
    """
    # Set up logging
    if log_dir:
        set_log_dir(log_dir)
    
    # Build interview app
    app = build_interview_app_with_profile(
        get_agent=get_agent,
        model_id=model_id,
        max_turns_p1_default=max_turns_p1,
        max_turns_p2_default=max_turns_p2,
        KNOWN_JOB_DOMAINS=KNOWN_JOB_DOMAINS,
        ALLOWED_SKILLS=ALLOWED_SKILLS,
        **kwargs
    )
    
    # Initial state
    initial_state: InterviewState = {
        "persona_id": persona_id,
        "model": model_id,
        "max_turns_p1": max_turns_p1,
        "max_turns_p2": max_turns_p2,
        "conversation": initial_conversation or [],
        "conversation_id": conversation_id,
        "profile_dict": {},
        "meta": {}
    }
    
    # Run interview
    result = app.invoke(initial_state)
    
    # Extract final profile
    profile = result.get("profile_dict", {})
    
    # Create session summary
    session_summary = _session_summary_from_state(result)
    
    # Save to personas map
    save_path = save_personas_map_entry(persona_id, profile, session_summary)
    
    return {
        "final_state": result,
        "profile": profile,
        "session_summary": session_summary,
        "save_path": str(save_path),
        "conversation": result.get("conversation", []),
        "conversation_ids": result.get("conversation_ids", [])
    }


def extract_profile_from_transcript(
    transcript: str,
    extractor_agent,
    domain: str = "unspecified",
    intent: Optional[str] = None,
    focus: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract and validate a persona profile from a conversation transcript.
    
    Args:
        transcript: Conversation transcript
        extractor_agent: LLM agent for extraction
        domain: Professional domain
        intent: User's stated intent
        focus: User's focus (job/training/awareness)
        
    Returns:
        Processed PersonaInfo dictionary
    """
    from .extraction import _extract_profile
    
    # Extract raw profile
    raw_profile = _extract_profile(extractor_agent, transcript)
    
    # Post-process with validation
    final_profile = _postprocess_persona(
        raw_profile,
        domain,
        ALLOWED_SKILLS,
        EDU_LEVELS,
        intent_in_state=intent,
        focus_in_state=focus
    )
    
    return final_profile


def load_all_personas(date: Optional[datetime.date] = None) -> Dict[str, Any]:
    """
    Load all personas for a given date.
    
    Args:
        date: Date to load personas for (defaults to today)
        
    Returns:
        Dictionary mapping persona_id to persona data
    """
    return load_personas_map()


def create_persona_summary(profile: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of a persona profile.
    
    Args:
        profile: PersonaInfo dictionary
        
    Returns:
        Human-readable summary string
    """
    name = profile.get("full_name", "Unknown")
    age = profile.get("age")
    domain = profile.get("top_domain", "unspecified")
    focus = profile.get("current_focus", "unspecified")
    location = profile.get("location_city", "unknown")
    
    # Technical skills summary
    tech_skills = profile.get("technical_skills", [])
    skills_summary = ""
    if tech_skills:
        skills_list = [f"{skill.get('name')} ({skill.get('level', 'Basic')})" 
                      for skill in tech_skills if isinstance(skill, dict)]
        if skills_list:
            skills_summary = f"\nTechnical Skills: {', '.join(skills_list)}"
    
    # Training motivation
    training = profile.get("training_motivation", [])
    training_summary = ""
    if training:
        training_summary = f"\nWants to learn: {', '.join(training)}"
    
    age_str = f", {age} years old" if age else ""
    
    summary = (f"{name}{age_str}\n"
               f"Location: {location}\n" 
               f"Domain: {domain}\n"
               f"Focus: {focus}")
    
    if skills_summary:
        summary += skills_summary
    if training_summary:
        summary += training_summary
        
    return summary


def validate_persona_profile(profile: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate a persona profile and return validation results.
    
    Args:
        profile: PersonaInfo dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Try to create PersonaInfo instance
        persona_info = PersonaInfo(**profile)
        
        # Check required fields
        if not persona_info.full_name.strip():
            errors.append("Full name is required")
            
        if persona_info.top_domain == "unspecified":
            errors.append("Domain should be specified")
            
        if persona_info.current_focus == "unspecified":
            errors.append("Focus should be specified")
            
        # Validate technical skills format
        for skill in persona_info.technical_skills:
            if not isinstance(skill, dict):
                errors.append("Technical skills must be dictionaries")
                continue
            if "name" not in skill or "level" not in skill:
                errors.append("Technical skills must have 'name' and 'level' fields")
                
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Invalid profile structure: {str(e)}")
        return False, errors


# Main entry point
def main_interview_pipeline(
    persona_id: str,
    get_agent: Callable[..., Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for running a complete interview pipeline.
    
    This is a convenience function that wraps run_interview with
    sensible defaults and error handling.
    """
    try:
        result = run_interview(persona_id, get_agent=get_agent, **kwargs)
        
        # Validate final profile
        is_valid, errors = validate_persona_profile(result["profile"])
        
        result["validation"] = {
            "is_valid": is_valid,
            "errors": errors
        }
        
        if is_valid:
            result["summary"] = create_persona_summary(result["profile"])
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "persona_id": persona_id,
            "success": False
        }


if __name__ == "__main__":
    # Example usage
    from src.agents import get_agent
    
    result = main_interview_pipeline(
        "test_persona_001",
        get_agent=get_agent,
        model_id="mistral-small-latest",
        max_turns_p1=3,
        max_turns_p2=3
    )
    
    if "error" in result:
        print(f"Interview failed: {result['error']}")
    else:
        print(f"Interview completed for {result['profile']['full_name']}")
        if "summary" in result:
            print("\nProfile Summary:")
            print(result["summary"])