"""
Interview State Machine for PersonaInfo Extraction

This module contains the complete interview state machine and node functions
extracted from PersonaInfo_Extraction.ipynb notebook.
"""

# Core imports
import json
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

# LangGraph imports
try:
    from langgraph import StateGraph, END
except ImportError:
    # Fallback for when LangGraph is not available
    StateGraph = None
    END = "END"

# Local imports
from .models import InterviewState, PersonaInfo
from .llm_client import send_message_to_chat, call_strands_agent_logged
from .io import append_turn, session_jsonl_path
from .validation import _complete_personainfo_dict, _postprocess_persona, normalize_focus, sanitize_domain
from .extraction import _extract_profile, _summarize_intent, _merge_profile
from .utils import bullet_list, safe_json_from_text

from src.config import EDU_LEVELS, KNOWN_JOB_DOMAINS
from src.logs import log_interview_event

# Constants
GREETING = (
    "Hi! I'm an AI career assistant (LLM‑powered). I'll explain my plan briefly at each step so you see why I'm asking. "
    "First, what's your full name?"
)

COMPLETION_TOKENS_P1 = ("**PHASE 1 COMPLETE**",)
COMPLETION_TOKENS_P2 = ("**PHASE 2 COMPLETE**",)

FOCUS_JOB = "jobs+trainings"
FOCUS_TRAINING = "training_only"
FOCUS_AWARENESS = "awareness"

SESSIONS_DIR = Path("submissions/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Prompts
PHASE1_PROMPT = """\
You are a warm, transparent AI career assistant (LLM‑powered) running PHASE 1.

**How to converse:**
- Begin each turn with a short, friendly plan explaining what you'll do next and why it matters (e.g., "I'll confirm your focus so I can tailor advice.").
- If the user asks a question, answer briefly first, then continue your plan.
- Adjust to the user's style:
  - If they are shy or give very short answers, respond gently and encourage them without pressure.
  - If they speak in metaphors or abstract terms, acknowledge their style and interpret meaning carefully before asking the next question.
  - If they seem hesitant or distrustful or guarded, reassure them that you're here to help and explain why each question is useful.
  - If they sound like a kid (or say things suggesting they might be underage, even if they claim older), be especially gentle and neutral; politely ask age and explain it helps choose an appropriate path. Avoid probing personal/sensitive topics.
- Ask ONE focused question at a time; keep it concise and respectful.
- Use empathetic language and mirror their tone briefly before moving to the next step.
- Do NOT mention internal terms like "known domains" or "allowed skills" to the user; use natural phrases like "areas" and "skills people often learn in this area".

**Your goal in Phase 1 is to naturally collect these six items (then end with **PHASE 1 COMPLETE**):**
1) The user's current goal in their own words (free‑text "intent").
2) current_focus = one of: 'jobs+trainings', 'training_only', 'awareness' (use exactly these tokens).
3) domain (choose exactly one from KNOWN_JOB_DOMAINS).
4) language(s) they speak/prefer (free‑text; e.g., Portuguese, English). pt-br should be the default.
5) age (a number).
6) location city (no abbreviations, e.g. Brasília, Belo Horizonte, Rio de Janeiro).

**Important:**
- Do NOT collect job‑only details (education level, city, work type, relocation, years, roles) in Phase 1.
- If the user's intent sounds metaphorical or abstract, gently clarify by summarizing what you understood and asking if that's correct.
- Be empathetic and conversational, not robotic. If the user resists or asks "why," explain the purpose of the question in one sentence.
- ALWAYS ensure the domain is coming from KNOWN_JOB_DOMAINS. Verify with the user that the right domain has been set.
- Always verify information that you got from the user.
- ONLY communicate in English.

KNOWN_JOB_DOMAINS: {known_domains}

Transcript so far:
{transcript}

Now produce your next assistant message following these rules
"""

PHASE2_JOB_PROMPT = """\
You are a warm, transparent career counselor running PHASE 2 (JOB path).

**How to converse:**
- Start with a short plan explaining what you'll collect **and why** it helps tailor the next steps.
- If the user asks a question, **answer briefly first**, then continue your plan.
- Avoid backend terms; speak naturally (say "area" and "skills people often learn in this area").
- Keep it concise and professional.

**Collect (JOB path):**
- Technical skills the user ALREADY HAS, each with a level (Basic/Intermediate/Advanced). Select from: {allowed_skills_bulleted}
- Education level (choose from: {edu_levels_line}).
- City, preferred work type (onsite/remote/hybrid/either), openness to relocate (true/false).
- Years of experience (a number).


**Context:**
- Domain: {domain}
- User's current goal (free‑text intent): {intent_summary}:


**Rules:**
1) Ask one focused question per turn, be explicit and short.
2) Do NOT invent skills; ONLY use the allowed list for technical skills.
3) Consider the user's free‑text intent when discussing technical skills.
4) When finished, end your message with **PHASE 2 COMPLETE**.
5) Do not discuss or offer direct job placements/offers. Do not mention any "Phase 3".
6) Always translate user information (e.g. technical skills) into English.
7) Always verify information that you got from the user.
8) ONLY communicate in English.

Transcript so far:
{transcript}

Now produce your next assistant message following the rules above.
"""

PHASE2_TRAINING_PROMPT = """\
You are a warm, transparent career counselor running PHASE 2 (TRAINING path).

**How to converse:**
- Start with a short plan explaining what you'll collect **and why** it helps.
- If the user asks a question, **answer briefly first**, then continue the plan.
- Avoid backend terms; speak naturally.
- Keep it concise and professional.

**Collect (TRAINING path):**
- Technical skills the user ALREADY HAS, each with a level (Basic/Intermediate/Advanced). Always show the user full list of allowed_skills and verify that you understood the skill and skill level correctly.
- training_motivation = list of skills to pursue next (let the user decide; do not force narrowing; consider user's intent). Choose from {allowed_skills_bulleted}.
- (Avoid job‑only fields: education_level, city, work type, relocation, years, desired_job_roles.)

**Context:**
- Domain: {domain}
- User's current goal (free‑text intent): {intent_summary}


**Rules:**
1) Ask one focused question per turn.
2) Select from allowed skills for training_motivation based on determined domain. Often users want to further improve their existing skills.
3) Use the user's free‑text intent to bias which skills to prioritize. Always present the user with the full list of available skills for the domain.
4) Always verify information that you got from the user.
5) When finished, end your message with **PHASE 2 COMPLETE**.
6) Don't discuss individual training courses. NEVER discuss next steps or a Phase 3.
7) ONLY communicate in English.

Transcript so far:
{transcript}

Now produce your next assistant message following the rules above.
"""

AWARENESS_PROMPT = """\
You are a warm, transparent assistant. The user is currently in **awareness** mode.

- Briefly explain what you'll show **and why** it helps decide next steps.
- Provide a short overview of the often required skills within the domain.
- If the user has questions, answer briefly and invite them to return later for job or training paths.
- Keep it concise and friendly, no recommendations.
- DO NOT provide any links or next steps.
- ONLY communicate in English.

Context:
- Domain: {domain}
- Allowed skills (if domain known): {allowed_skills_bulleted}
"""


def _log(persona_id: str, phase: str, event_type: str, payload: dict, model_id: str | None):
    """Internal logging helper that wraps log_interview_event."""
    try:
        log_interview_event(
            persona_id=persona_id,
            phase=phase,
            event_type=event_type,
            payload=payload,
            model_id=model_id,
        )
    except Exception as e:
        print(f"[LOGGING-ERROR] {phase}/{event_type}: {e}")


def build_interview_app_with_profile(
    *,
    get_agent: Callable[..., Any],
    model_id: str = "mistral-small-latest",
    max_turns_p1_default: int = 5,
    max_turns_p2_default: int = 5,
    KNOWN_JOB_DOMAINS: List[str],
    ALLOWED_SKILLS: Dict[str, List[str]],
    MAX_SKILLS_LEARN: int = 5,
):
    """
    Two-phase, prompt-driven interview:
      Phase 1: collect free‑text intent, current_focus (job|training|awareness), domain, and age.
      Phase 2 (JOB path): job-only details + technical_skills + training_motivation (domain-based when known).
      Phase 2 (TRAINING path): technical_skills + training_motivation (domain-based when known), no job-only fields.
      Awareness: brief orientation and stop.
    Incremental extraction & merge; final pass ensures complete PersonaInfo JSON.
    """

    if StateGraph is None:
        raise ImportError("LangGraph is not available. Please install it to use the interview state machine.")

    # Conversational agents
    phase1_agent = get_agent(
        system_prompt="You are a concise, friendly interviewer that explains your plan and asks one question at a time.",
        model_id=model_id,
    )
    phase2_job_agent = get_agent(
        system_prompt="You are a concise, friendly counselor for JOB path; you explain your plan and ask one question at a time.",
        model_id=model_id,
    )
    phase2_training_agent = get_agent(
        system_prompt="You are a concise, friendly counselor for TRAINING path; you explain your plan and ask one question at a time.",
        model_id=model_id,
    )
    # Intent summarizer
    intent_agent = get_agent(
        system_prompt="You summarize the user's current goal in one short sentence, preserving nuance and their own words.",
        model_id=model_id,
    )
    # Structured extractor
    extractor_agent = get_agent(
        system_prompt=(
            "You are a strict JSON extractor. Return exactly ONE PersonaInfo object with ALL fields present. "
            "If unknown, use '' for text, null for numbers, [] for lists. No prose. "
            "CONSTRAINTS:\n"
            "- current_focus must be one of: 'jobs+trainings', 'training_only', 'awareness'.\n"
            "- technical_skills must be a list of objects with keys: name (string), level (one of 'Basic','Intermediate','Advanced').\n"
            "- training_motivation must be a list of skill names (strings). If domain is known, prefer only from allowed list.\n"
            "- languages must be a list of strings.\n"
            "- years_experience is a number or null.\n"
            "- preferred_work_type is one of: 'onsite','remote','hybrid','either','unspecified'."
        ),
        model_id=model_id,
    )
    # Awareness agent
    awareness_agent = get_agent(
        system_prompt="You are a concise, warm assistant that provides brief career domain overviews without making recommendations.",
        model_id=model_id,
    )

    # Node functions
    def init_node(state: InterviewState) -> InterviewState:
        persona_id = state["persona_id"]
        model_used = state.get("model", model_id)
        conversation = state.get("conversation", [])
    
        return {
            "conversation": conversation,
            "model": model_used,
            "max_turns_p1": state.get("max_turns_p1", max_turns_p1_default),
            "max_turns_p2": state.get("max_turns_p2", max_turns_p2_default),
            "turn_p1": 0,
            "turn_p2": 0,
            "phase1_done": False,
            "phase2_done": False,
            "agent_message_p1": GREETING,
            "meta": state.get("meta", {}),
            "domain": state.get("domain", "unspecified"),
            "conversation_ids": ([] if not state.get("conversation_ids") else list(state["conversation_ids"]))
        }

    def phase1_step(state: InterviewState) -> InterviewState:
        updates: InterviewState = {}
        persona_id, conv_id = state["persona_id"], state.get("conversation_id")
        model_used = state.get("model")
        conversation = [*state.get("conversation", [])]
        meta = dict(state.get("meta", {}))
    
        # Send current assistant message and receive reply
        current = state.get("agent_message_p1")
        conv_id_for_assistant = state.get("conversation_id")
        if current:
            _log(persona_id, "phase1", "assistant_message", {"text": current}, model_used)
            conversation.append(f"Assistant: {current}")
            resp = send_message_to_chat(current, persona_id, conv_id_for_assistant)
            append_turn(
                            persona_id, "assistant", current,
                            conv_id_for_assistant,
                            state.get("turn_p1", 0) + 1
                        )
            if resp is None:
                _log(persona_id, "phase1", "error", {"message": "persona API error (None response)"}, model_used)
                meta["p1_exit_reason"] = "api_error"
                updates.update(
                    {
                        "conversation": conversation + ["System: persona API error; ending Phase 1 early."],
                        "phase1_done": True,
                        "turn_p1": state.get("turn_p1", 0) + 1,
                        "meta": meta,
                        "agent_message_p1": None,
                    }
                )
                return updates
    
            user_response, new_conv_id = resp
            print(new_conv_id)
            conversation.append(f"User: {user_response}")
            _log(persona_id, "phase1", "user_message", {"text": user_response, "conversation_id": new_conv_id}, model_used)
            append_turn(
                        persona_id, "user", user_response,
                        new_conv_id,
                        state.get("turn_p1", 0) + 1
                    )
            updates["conversation_id"] = new_conv_id
            conv_ids = list(state.get("conversation_ids", []))
            if new_conv_id and (not conv_ids or conv_ids[-1] != new_conv_id):
                conv_ids.append(new_conv_id)
            updates["conversation_ids"] = conv_ids
    
        # ---- Intent summarization (PHASE 1 ONLY) ----
        try:
            intent_text = _summarize_intent(
                intent_agent,
                "\n".join(conversation),
                persona_id=persona_id,
                model_id=model_used
            )
            if intent_text and intent_text != state.get("intent"):
                updates["intent"] = intent_text
                _log(persona_id, "phase1", "intent_update", {"intent": intent_text}, model_used)
        except Exception as e:
            _log(persona_id, "phase1", "error", {"message": "intent summarization failed", "exception": str(e)}, model_used)
    
        # ---- DOMAIN CAPTURE via LLM extractor, then intent-only micro-call (both strictly validated) ----
        # 1) Structured extractor from transcript
        try:
            p1_partial = _extract_profile(extractor_agent, "\n".join(conversation))
        except Exception:
            p1_partial = {}
        if p1_partial:
            raw_dom = (p1_partial.get("top_domain") or "").strip()
            dom = sanitize_domain(raw_dom, KNOWN_JOB_DOMAINS)
            if dom != "unspecified":
                updates["domain"] = dom
                _log(persona_id, "phase1", "domain_update", {"domain": dom, "source": "extractor"}, model_used)
    
        # 2) If still unspecified, try a tiny intent-only LLM pick from KNOWN_JOB_DOMAINS
        dom_now = (updates.get("domain") or state.get("domain") or "unspecified").strip() or "unspecified"
        intent_now = updates.get("intent") or state.get("intent")
        if dom_now == "unspecified" and intent_now:
            try:
                intent_to_domain_prompt = (
                    "From the user's intent below, choose exactly ONE matching domain from the provided list. "
                    "Return ONLY the domain string, exactly as in the list, or 'unspecified' if no good match.\n\n"
                    f"INTENT:\n{intent_now}\n\n"
                    f"KNOWN_DOMAINS:\n{', '.join(KNOWN_JOB_DOMAINS)}"
                )
                candidate = call_strands_agent_logged(
                    intent_agent,
                    intent_to_domain_prompt,
                    phase="phase1_domain_from_intent",
                    persona_id=persona_id,
                    model_id=model_used
                ).strip()
                dom2 = sanitize_domain(candidate, KNOWN_JOB_DOMAINS)
                if dom2 != "unspecified":
                    updates["domain"] = dom2
                    _log(persona_id, "phase1", "domain_update", {"domain": dom2, "source": "intent_microcall"}, model_used)
            except Exception as e:
                _log(persona_id, "phase1", "error", {"message": "intent->domain micro-call failed", "exception": str(e)}, model_used)
    
        # ---- Next assistant turn (prompt-driven) ----
        transcript = "\n".join(conversation)
        prompt = PHASE1_PROMPT.format(
            known_domains=", ".join(KNOWN_JOB_DOMAINS),
            transcript=transcript,
        )
        _log(persona_id, "phase1", "agent_prompt", {"prompt_head": prompt[:1500], "prompt_len": len(prompt)}, model_used)
        try:
            # NOTE: stop=["\\nEND"] is enforced inside call_strands_agent_logged(...)
            next_msg = call_strands_agent_logged(
                phase1_agent, prompt,
                phase="phase1", persona_id=persona_id, model_id=model_used
            )
            _log(persona_id, "phase1", "agent_result", {"text": next_msg}, model_used)
        except Exception as e:
            _log(persona_id, "phase1", "error", {"message": "Strands Agent call failed", "exception": str(e)}, model_used)
            # Short, friendly fallback prompt (no long enumeration)
            if state.get("turn_p1", 0) < 10:
                next_msg = (
                    "I'll first reflect what you shared and ask a quick question so I can tailor my guidance. "
                    "What would you like to achieve or learn right now?"
                )
            else:
                next_msg = (
                    "Before we continue, which ONE area describes your path best? "
                    "You can name it directly (e.g., 'Maritime crew', 'Design', 'Insurance')."
                )

    
        done = any(tok in str(next_msg) for tok in COMPLETION_TOKENS_P1)
    
        turn_p1 = state.get("turn_p1", 0) + 1
        updates.update(
            {
                "conversation": conversation,
                "agent_message_p1": None if done else str(next_msg),
                "turn_p1": turn_p1,
            }
        )
    
        # ---- Phase 1 wrap-up (on completion token or max turns) ----
        if done or turn_p1 >= state.get("max_turns_p1", 6):
            if not done:
                meta["p1_exit_reason"] = "max_turns"
            updates.update({"phase1_done": True, "meta": meta})
    
            # Final structured extraction + merge + post-process
            p1_profile = _extract_profile(extractor_agent, "\n".join(conversation))
            _log(persona_id, "phase1", "extraction_result", p1_profile, model_used)
    
            merged = _merge_profile(state.get("profile_dict", {}), p1_profile, {**state, **updates})
            merged = _postprocess_persona(
                merged,
                sanitize_domain(merged.get("top_domain",""), KNOWN_JOB_DOMAINS),
                ALLOWED_SKILLS,
                EDU_LEVELS,
                intent_in_state=updates.get("intent") or state.get("intent"),
                focus_in_state=state.get("current_focus"),
            )
    
            # Keep some top-level fields for routing convenience
            if merged.get("age") is not None:
                updates["age"] = merged["age"]
            if merged.get("top_domain"):
                updates["domain"] = sanitize_domain(merged["top_domain"], KNOWN_JOB_DOMAINS)
    
            focus = normalize_focus(merged.get("current_focus") or state.get("current_focus"))
            if focus:
                updates["current_focus"] = focus
    
            updates["profile_dict"] = merged
            _log(persona_id, "phase1", "extraction_merged", merged, model_used)
    
        return updates

    def awareness_info(state: InterviewState) -> InterviewState:
        persona_id, conv_id = state["persona_id"], state.get("conversation_id")
        model_used = state.get("model")
        domain = state.get("domain", "unspecified")
        allowed = ALLOWED_SKILLS.get(domain, [])
        allowed_bulleted = bullet_list(allowed) if allowed else "- (no domain confirmed yet)"
    
        # Use AWARENESS_PROMPT *as a prompt* to the agent
        prompt = AWARENESS_PROMPT.format(domain=domain, allowed_skills_bulleted=allowed_bulleted)
        _log(persona_id, "phase2", "agent_prompt", {"prompt_head": prompt[:1500], "prompt_len": len(prompt)}, model_used)
    
        try:        
            assistant_msg = call_strands_agent_logged(
                awareness_agent, prompt,
                phase="awareness", persona_id=persona_id, model_id=model_used
            )
        except Exception as e:
            _log(persona_id, "phase2", "error", {"message": "awareness agent call failed", "exception": str(e)}, model_used)
            assistant_msg = (
                "I'll share a quick orientation so you can decide next steps. "
                f"For '{domain}', I can outline typical roles and skills when you're ready."
            )
    
        conversation = [*state.get("conversation", []), f"Assistant: {assistant_msg}"]
        _log(persona_id, "phase2", "assistant_message", {"text": assistant_msg}, model_used)
        conv_id_for_assistant = state.get("conversation_id")
        append_turn(
            persona_id, "assistant", assistant_msg,
            conv_id_for_assistant,
            state.get("turn_p2", 0) + 1
        )
    
        resp = send_message_to_chat(assistant_msg, persona_id, conv_id_for_assistant)
        updates: InterviewState = {"conversation": conversation}
        if resp:
            user, new_conv_id = resp
            print(new_conv_id)
            conversation.append(f"User: {user}")
            updates["conversation_id"] = new_conv_id
            _log(persona_id, "phase2", "user_message", {"text": user, "conversation_id": new_conv_id}, model_used)
            append_turn(
                persona_id, "user", user,
                new_conv_id,
                state.get("turn_p2", 0) + 1
            )
            conv_ids = list(state.get("conversation_ids", []))
            if new_conv_id and (not conv_ids or conv_ids[-1] != new_conv_id):
                conv_ids.append(new_conv_id)
            updates["conversation_ids"] = conv_ids

    
        return {"conversation": conversation, "phase2_done": True}

    def phase2_job(state: InterviewState) -> InterviewState:
        updates: InterviewState = {}
        persona_id, conv_id = state["persona_id"], state.get("conversation_id")
        model_used = state.get("model")
        conversation = [*state.get("conversation", [])]
        meta = dict(state.get("meta", {}))
        domain = state.get("domain", "unspecified")
        allowed = ALLOWED_SKILLS.get(domain, [])
        allowed_bulleted = bullet_list(allowed) if allowed else "- (domain unspecified; any reasonable skills allowed)"
        intent_summary = state.get("intent") or ""

        # Send current assistant message & receive reply
        current = state.get("agent_message_p2")
        conv_id_for_assistant = state.get("conversation_id")
        if current:
            conversation.append(f"Assistant: {current}")
            _log(persona_id, "phase2_job", "assistant_message", {"text": current}, model_used)
            append_turn(
                persona_id, "assistant", current,
                conv_id_for_assistant,
                state.get("turn_p2", 0) + 1
            )

            resp = send_message_to_chat(current, persona_id, conv_id_for_assistant)
            if resp is None:
                _log(persona_id, "phase2_job", "error", {"message": "persona API error (None response)"}, model_used)
                meta["p2_exit_reason"] = "api_error"
                updates.update(
                    {
                        "conversation": conversation + ["System: persona API error; ending Phase 2 early."],
                        "phase2_done": True,
                        "turn_p2": state.get("turn_p2", 0) + 1,
                        "meta": meta,
                        "agent_message_p2": None,
                    }
                )
                return updates

            user_response, new_conv_id = resp
            print(new_conv_id)
            conversation.append(f"User: {user_response}")
            updates["conversation_id"] = new_conv_id
            _log(persona_id, "phase2_job", "user_message", {"text": user_response, "conversation_id": new_conv_id}, model_used) 
            append_turn(
                persona_id, "user", user_response,
                new_conv_id,
                state.get("turn_p2", 0) + 1
            )
            conv_ids = list(state.get("conversation_ids", []))
            if new_conv_id and (not conv_ids or conv_ids[-1] != new_conv_id):
                conv_ids.append(new_conv_id)
            updates["conversation_ids"] = conv_ids


        edu_levels_line = ", ".join(EDU_LEVELS)
        transcript = "\n".join(conversation)
        prompt = PHASE2_JOB_PROMPT.format(
            domain=domain,
            intent_summary=intent_summary,
            edu_levels_line=edu_levels_line,
            allowed_skills_bulleted=allowed_bulleted,
            transcript=transcript,
        )
        _log(persona_id, "phase2_job", "agent_prompt", {"prompt_head": prompt[:1500], "prompt_len": len(prompt)}, model_used)

        try:  
            next_msg = call_strands_agent_logged(
                phase2_job_agent, prompt,
                phase="phase2_job", persona_id=persona_id, model_id=model_used
            )
            _log(persona_id, "phase2_job", "agent_result", {"text": next_msg}, model_used)
        except Exception as e:
            _log(persona_id, "phase2_job", "error", {"message": "Strands Agent call failed", "exception": str(e)}, model_used)
            next_msg = (
                "I'll confirm your education level, city, work type, relocation, years, desired roles, "
                "then your existing skills with levels and skills you want to pursue next. "
                "Which city are you based in?"
            )

        done = any(tok in str(next_msg) for tok in COMPLETION_TOKENS_P2)
        turn_p2 = state.get("turn_p2", 0) + 1
        updates.update(
            {
                "conversation": conversation,
                "agent_message_p2": None if done else str(next_msg),
                "turn_p2": turn_p2,
            }
        )
        if done or turn_p2 >= state.get("max_turns_p2", 6):
            if not done:
                meta["p2_exit_reason"] = "max_turns"
            updates.update({"phase2_done": True, "meta": meta})

            # Incremental extraction after Phase 2 (job)
            p2_profile = _extract_profile(extractor_agent, "\n".join(conversation))
            _log(persona_id, "phase2_job", "extraction_result", p2_profile, model_used)

            merged = _merge_profile(state.get("profile_dict", {}), p2_profile, {**state, **updates})
            merged = _postprocess_persona(
                merged,
                merged.get("top_domain",""),
                ALLOWED_SKILLS,
                EDU_LEVELS,
                intent_in_state=state.get("intent"),
                focus_in_state=state.get("current_focus"),
            )
            updates["profile_dict"] = merged
            _log(persona_id, "phase2_job", "extraction_merged", merged, model_used)

        return updates

    def phase2_training(state: InterviewState) -> InterviewState:
        updates: InterviewState = {}
        persona_id, conv_id = state["persona_id"], state.get("conversation_id")
        model_used = state.get("model")
        conversation = [*state.get("conversation", [])]
        meta = dict(state.get("meta", {}))
        domain = state.get("domain", "unspecified")
        allowed = ALLOWED_SKILLS.get(domain, [])
        allowed_bulleted = bullet_list(allowed) if allowed else "- (domain unspecified; any reasonable skills allowed)"
        intent_summary = state.get("intent") or ""

        # Send current assistant message & receive reply
        current = state.get("agent_message_p2")
        conv_id_for_assistant = state.get("conversation_id")
        if current:
            conversation.append(f"Assistant: {current}")
            _log(persona_id, "phase2_training", "assistant_message", {"text": current}, model_used)
            append_turn(
                persona_id, "assistant", current,
                conv_id_for_assistant,
                state.get("turn_p2", 0) + 1
            )

            resp = send_message_to_chat(current, persona_id, conv_id_for_assistant)
            if resp is None:
                _log(persona_id, "phase2_training", "error", {"message": "persona API error (None response)"}, model_used)
                meta["p2_exit_reason"] = "api_error"
                updates.update(
                    {
                        "conversation": conversation + ["System: persona API error; ending Phase 2 early."],
                        "phase2_done": True,
                        "turn_p2": state.get("turn_p2", 0) + 1,
                        "meta": meta,
                        "agent_message_p2": None,
                    }
                )
                return updates

            user_response, new_conv_id = resp
            print(new_conv_id)
            conversation.append(f"User: {user_response}")
            updates["conversation_id"] = new_conv_id
            _log(persona_id, "phase2_training", "user_message", {"text": user_response, "conversation_id": new_conv_id}, model_used)
            append_turn(
                persona_id, "user", user_response,
                new_conv_id,
                state.get("turn_p2", 0) + 1
            )
            conv_ids = list(state.get("conversation_ids", []))
            if new_conv_id and (not conv_ids or conv_ids[-1] != new_conv_id):
                conv_ids.append(new_conv_id)
            updates["conversation_ids"] = conv_ids


        transcript = "\n".join(conversation)
        prompt = PHASE2_TRAINING_PROMPT.format(
            domain=domain,
            intent_summary=intent_summary,
            allowed_skills_bulleted=allowed_bulleted,
            max_skills=MAX_SKILLS_LEARN,
            transcript=transcript,
        )
        _log(persona_id, "phase2_training", "agent_prompt", {"prompt_head": prompt[:1500], "prompt_len": len(prompt)}, model_used)

        try:
            next_msg = call_strands_agent_logged(
                phase2_training_agent, prompt,
                phase="phase2_training", persona_id=persona_id, model_id=model_used
            )
            _log(persona_id, "phase2_training", "agent_result", {"text": next_msg}, model_used)
        except Exception as e:
            _log(persona_id, "phase2_training", "error", {"message": "Strands Agent call failed", "exception": str(e)}, model_used)
            next_msg = (
                "I'll confirm the skills you already have with a level, then we'll pick skills to pursue next—"
                "preferably from the domain's allowed list. Which skills do you already have, and what is your level?"
            )

        done = any(tok in str(next_msg) for tok in COMPLETION_TOKENS_P2)
        turn_p2 = state.get("turn_p2", 0) + 1
        updates.update(
            {
                "conversation": conversation,
                "agent_message_p2": None if done else str(next_msg),
                "turn_p2": turn_p2,
            }
        )
        if done or turn_p2 >= state.get("max_turns_p2", 6):
            if not done:
                meta["p2_exit_reason"] = "max_turns"
            updates.update({"phase2_done": True, "meta": meta})

            # Incremental extraction after Phase 2 (training)
            p2_profile = _extract_profile(extractor_agent, "\n".join(conversation))
            _log(persona_id, "phase2_training", "extraction_result", p2_profile, model_used)

            merged = _merge_profile(state.get("profile_dict", {}), p2_profile, {**state, **updates})    
            merged = _postprocess_persona(
                merged,
                merged.get("top_domain",""),
                ALLOWED_SKILLS,
                EDU_LEVELS,
                intent_in_state=state.get("intent"),
                focus_in_state=state.get("current_focus"),
            )
            updates["profile_dict"] = merged
            _log(persona_id, "phase2_training", "extraction_merged", merged, model_used)

        return updates

    def build_profile(state: InterviewState) -> InterviewState:
        """Final PersonaInfo extraction + complete & merge."""
        persona_id = state["persona_id"]
        model_used = state.get("model")
        transcript = "\n".join(state.get("conversation", []))

        extract_prompt = (
            "Transcript between Assistant and User follows. "
            "Return ONE PersonaInfo object with ALL fields present. "
            "If unknown, use empty string for text, null for numbers, [] for lists. No prose.\n\n" + transcript
        )
        _log(persona_id, "system", "extraction_prompt", {"prompt_head": extract_prompt[:1500], "prompt_len": len(extract_prompt)}, model_used)

        final_profile = _extract_profile(extractor_agent, transcript)
        _log(persona_id, "system", "extraction_result", final_profile, model_used)

        merged = _merge_profile(state.get("profile_dict", {}), final_profile, state)
        merged = _postprocess_persona(
            merged,
            merged.get("top_domain",""),
            ALLOWED_SKILLS,
            EDU_LEVELS,
            intent_in_state=state.get("intent"),
            focus_in_state=state.get("current_focus"),
        )
        _log(persona_id, "system", "extraction_merged", merged, model_used)
        return {"profile_dict": merged}

    def recommendations_node(state: InterviewState) -> InterviewState:
        """
        Use the knowledge graph to produce concrete job and/or training offers,
        enrich with Markdown descriptions, append to transcript, and persist JSON.
        """
        persona_id = state["persona_id"]
        conversation = [*state.get("conversation", [])]
        model_used = state.get("model")
    
        try:
            from src.reco import kg_recommendations_for_persona, save_recommendations_json
            
            # 1) Compute KG-based recommendations + doc snippets
            rec = kg_recommendations_for_persona(
                persona_id,
                kg_path=state.get("kg_path", "kg.gpickle"),
                jobs_dir=state.get("jobs_dir", "../../GDSC-8/data/jobs"),
                trainings_dir=state.get("trainings_dir", "../../GDSC-8/data/trainings"),
                top_jobs=state.get("top_jobs", 5),
                get_agent=get_agent,  # << reuse your agent factory
                agent_system_prompt=(
                    "You are a helpful career assistant. Rewrite the recommendations "
                    "succinctly in Markdown with clear bullets and bold titles."
                ),
                agent_model_id=state.get("model", "mistral-medium-latest"),
                show_paths=False,
            )
            assistant_msg = rec["assistant_message"]
            payload = rec["payload"]
        
            # 2) Append assistant message to transcript + persist JSON
            conversation.append(f"Assistant: {assistant_msg}")
            try:
                save_recommendations_json(persona_id, {
                    "persona_id": persona_id,
                    "ts": time.time(),
                    "type": payload.get("type", "unknown"),
                    "data": payload,
                })
            except Exception as e:
                # Non-fatal
                print(f"[warn] could not save recommendations JSON: {e}")
        
            # 3) Persist to JSONL transcript
            append_turn(persona_id, "assistant", assistant_msg, state.get("conversation_id"), state.get("turn_p2", 0) + 1)
        
        except ImportError:
            assistant_msg = "Thank you for completing the interview. Your profile has been saved successfully."
            conversation.append(f"Assistant: {assistant_msg}")
            append_turn(persona_id, "assistant", assistant_msg, state.get("conversation_id"), state.get("turn_p2", 0) + 1)
    
        # 4) Return updates; we do NOT solicit a final user turn here (end of flow)
        return {
            "conversation": conversation,
            "phase2_done": True,  # ensure end
        }
    
    # -------------------------
    # GRAPH (exclusive routing)
    # -------------------------
    
    graph = StateGraph(InterviewState)
    
    # Nodes
    graph.add_node("init", init_node)
    graph.add_node("phase1", phase1_step)
    
    # Dedicated router that makes the *single* decision after phase1
    def _noop(state: InterviewState) -> InterviewState:
        # IMPORTANT: do not write anything here; returning {} avoids touching 'conversation'
        return {}
    
    graph.add_node("p1_router", _noop)
    
    graph.add_node("awareness_info", awareness_info)
    graph.add_node("phase2_job", phase2_job)
    graph.add_node("phase2_training", phase2_training)
    graph.add_node("build_profile", build_profile)
    graph.add_node("recommendations", recommendations_node)
    
    # Entry → Phase 1 → Router
    graph.set_entry_point("init")
    graph.add_edge("init", "phase1")
    graph.add_edge("phase1", "p1_router")  # single edge from phase1; no other conditionals from 'phase1'
    
    # Router decides to loop or proceed to a single branch
    def route_from_p1(state: InterviewState) -> str:
        """
        Returns exactly one of:
          - 'loop'      : continue Phase 1 interviewing
          - 'job'       : go to job path
          - 'training'  : go to training path
          - 'awareness' : go to awareness path
        """
        # Keep looping Phase 1 if not done and within turn budget
        if not state.get("phase1_done") and state.get("turn_p1", 0) < state.get("max_turns_p1", max_turns_p1_default):
            return "loop"
    
        # Final routing after Phase 1
        age = state.get("age")
        if isinstance(age, (int, float)) and age is not None and age < 16:
            return "awareness"
    
        focus = normalize_focus(state.get("current_focus"))
        if focus == FOCUS_JOB:
            return "job"
        if focus == FOCUS_TRAINING:
            return "training"
        return "awareness"
    
    graph.add_conditional_edges(
        "p1_router",
        route_from_p1,
        {
            "loop": "phase1",
            "job": "phase2_job",
            "training": "phase2_training",
            "awareness": "awareness_info",
        },
    )
    
    # Phase‑2 loop control (shared)
    def p2_continue_or_build(state: InterviewState) -> str:
        if state.get("phase2_done"):
            return "build"
        if state.get("turn_p2", 0) >= state.get("max_turns_p2", max_turns_p2_default):
            return "build"
        return "loop"
    
    # Job path loop → build
    graph.add_conditional_edges("phase2_job", p2_continue_or_build, {
        "loop": "phase2_job",
        "build": "build_profile",
    })
    
    # Training path loop → build
    graph.add_conditional_edges("phase2_training", p2_continue_or_build, {
        "loop": "phase2_training",
        "build": "build_profile",
    })
    
    # Awareness wraps directly to build
    graph.add_edge("awareness_info", "build_profile")

    def route_after_build(state: InterviewState) -> str:
        # Under 16 or explicit awareness focus -> end the flow without recommendations
        age = state.get("age")
        focus = (state.get("current_focus") or "").strip().lower()
        if (isinstance(age, (int, float)) and age is not None and age < 16) or focus == "awareness":
            return "end"
        return "reco"
    
    graph.add_conditional_edges(
        "build_profile",
        route_after_build,
        {
            "end": END,
            "reco": "recommendations",
        },
    )
    
    graph.add_edge("recommendations", END)

    return graph.compile()