"""
LLM client and agent interaction utilities for persona info extraction.

This module provides communication functions for AWS chat API integration,
Strands agent calls with usage logging, and text extraction utilities.
"""

from __future__ import annotations
import json
import time
import csv
import os
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from src.common.env import load_project_dotenv
from src.logs import log_interview_event

# Ensure env is loaded
load_project_dotenv()

# ==========================
# Usage logging + emissions
# ==========================

# Where to write CSV usage logs (adjust as you prefer)
USAGE_LOG_PATH = Path("agent_usage_log.csv")

# Emissions factor (grams CO₂e per 1,000 tokens). Feel free to tune per model/provider.
G_CO2_PER_1K_TOKENS = 2.85


def estimate_emissions_gco2(total_tokens: int, factor_per_1k: float = G_CO2_PER_1K_TOKENS) -> float:
    """Estimate grams of CO₂e for total token usage."""
    return (float(total_tokens) / 1000.0) * float(factor_per_1k)


@dataclass
class UsageRecord:
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
    # Ecological impact
    est_gco2: float = 0.0


def log_usage(rec: UsageRecord) -> None:
    """Append a single usage record to CSV (creates header on first write)."""
    exists = USAGE_LOG_PATH.exists()
    with USAGE_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow([
                "ts", "persona_id", "phase", "model",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "latency_s", "ok", "error", "est_gco2"
            ])
        w.writerow([
            rec.ts, rec.persona_id, rec.phase, rec.model,
            rec.prompt_tokens, rec.completion_tokens, rec.total_tokens,
            rec.latency_s, rec.ok, rec.error or "", f"{rec.est_gco2:.6f}"
        ])


# ==========================
# AWS Chat API Integration
# ==========================

def send_message_to_chat(
    message: str,
    persona_id: str,
    conversation_id: str = None
) -> Optional[Tuple[str, str]]:
    """Send message to persona API and get response (response_text, conversation_id)."""

    url = "https://cygeoykm2i.execute-api.us-east-1.amazonaws.com/main/chat"

    # Build payload (omit conversation_id if None)
    payload = {"persona_id": persona_id, "message": message}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id

    # AWS SigV4 signing
    session = boto3.Session(region_name='us-east-1')
    credentials = session.get_credentials()
    if credentials is None:
        return None
    frozen = credentials.get_frozen_credentials()

    body = json.dumps(payload)
    request = AWSRequest(
        method='POST',
        url=url,
        data=body,
        headers={'Content-Type': 'application/json'}
    )
    SigV4Auth(frozen, 'execute-api', 'us-east-1').add_auth(request)

    # Send request
    try:
        response = requests.request(
            method=request.method,
            url=request.url,
            headers=dict(request.headers),
            data=request.body,
            timeout=15
        )
    except Exception:
        return None

    if response.status_code != 200:
        return None

    response_json = response.json()
    return response_json.get('response'), response_json.get('conversation_id')


# ==========================
# Agent Communication
# ==========================

def call_strands_agent(agent, prompt: str) -> str:
    """
    Returns a string response; raises exceptions so caller can log and fallback.
    Docs:
      - Quickstart shows `response = agent("...")`
      - Prompts doc shows `agent = Agent(system_prompt=...); response = agent("...")`
    """
    # Debug: optional short preview to diagnose prompt shaping
    # print("[DEBUG] Phase agent prompt preview:", prompt[:400])

    result = agent(prompt)          # <-- Strands Agent sync entrypoint
    # Coerce to plain string; covers any response wrapper Strands may return
    return str(getattr(result, "content", result))


def _extract_text_from_agent_result(result) -> str:
    """
    Normalize diverse agent result shapes to plain text.
    Handles:
      - str
      - objects with .content (str | list[dict|str] | dict)
      - {'role': 'assistant', 'content': [{'type':'text','text':'...'}]}
    """
    def _from_content(c):
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            parts = []
            for b in c:
                if isinstance(b, dict):
                    t = b.get("text") or b.get("content")
                    if isinstance(t, str):
                        parts.append(t)
                elif isinstance(b, str):
                    parts.append(b)
            return "\n".join(p for p in parts if p).strip()
        if isinstance(c, dict):
            # OpenAI/Blocks-like
            if "content" in c and isinstance(c["content"], list):
                parts = []
                for b in c["content"]:
                    if isinstance(b, dict) and isinstance(b.get("text"), str):
                        parts.append(b["text"])
                if parts:
                    return "\n".join(parts).strip()
            if isinstance(c.get("text"), str):
                return c["text"].strip()
        return ""

    c = getattr(result, "content", None)
    if c is not None:
        return _from_content(c)
    c = getattr(result, "message", None)
    if c is not None:
        return _from_content(c)
    return str(result).strip()


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


def call_strands_agent_logged(
    agent, 
    prompt: str, 
    *, 
    phase: str, 
    persona_id: str, 
    model_id: str | None
) -> str:
    """
    Strands Agent sync call with usage logging (tokens, latency, cycles, tools),
    estimating ecological impact (gCO2e), and returning plain text only.
    Also enforces stop=["\nEND"] if supported and trims at that token.
    """
    t0 = time.time()
    try:
        # Try passing a stop sequence. Fall back if the agent doesn't accept it.
        try:
            result = agent(prompt, stop=["\nEND"])
        except TypeError:
            result = agent(prompt)

        # Prefer consolidated view from Strands
        summary = None
        try:
            if hasattr(result, "metrics") and hasattr(result.metrics, "get_summary"):
                summary = result.metrics.get_summary()
        except Exception:
            summary = None

        # Defaults
        prompt_tokens = completion_tokens = total_tokens = 0
        total_duration = time.time() - t0
        total_cycles = 0
        tools_used = []

        if summary:
            usage = summary.get("accumulated_usage", {}) or {}
            prompt_tokens = int(usage.get("inputTokens", usage.get("prompt_tokens", 0)) or 0)
            completion_tokens = int(usage.get("outputTokens", usage.get("completion_tokens", 0)) or 0)
            total_tokens = int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0)
            total_duration = float(summary.get("total_duration", total_duration))
            total_cycles = int(summary.get("total_cycles", 0) or 0)
            tool_usage = summary.get("tool_usage", {}) or {}
            tools_used = list(tool_usage.keys())

        # Ecological metrics
        est_gco2 = estimate_emissions_gco2(total_tokens)

        # CSV usage log
        log_usage(UsageRecord(
            ts=time.time(), persona_id=persona_id, phase=phase,
            model=(model_id or ""), prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, total_tokens=total_tokens,
            latency_s=total_duration, ok=True, est_gco2=est_gco2
        ))

        # Structured JSONL event (your existing event logger)
        _log(persona_id, phase, "usage_metrics", {
            "model": model_id,
            "summary": summary or {},
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "timing": {
                "duration_s": total_duration,
                "total_cycles": total_cycles,
            },
            "tools_used": tools_used,
            "ecology": {
                "gco2_per_1k_tokens": G_CO2_PER_1K_TOKENS,
                "estimated_gco2": est_gco2,
            },
        }, model_id)

        # Return assistant text (plain text only) and trim at stop token if present
        text = _extract_text_from_agent_result(result)
        if "\nEND" in text:
            text = text.split("\nEND", 1)[0].rstrip()
        return text

    except Exception as e:
        latency = time.time() - t0
        # CSV error record
        log_usage(UsageRecord(
            ts=time.time(), persona_id=persona_id, phase=phase,
            model=(model_id or ""), prompt_tokens=0, completion_tokens=0,
            total_tokens=0, latency_s=latency, ok=False, error=str(e), est_gco2=0.0
        ))
        # JSONL error
        _log(persona_id, phase, "usage_error", {"error": str(e), "latency_s": latency}, model_id)
        raise


# ==========================
# Utility Functions
# ==========================

def get_usage_summary() -> dict:
    """Get summary statistics from usage log if it exists."""
    if not USAGE_LOG_PATH.exists():
        return {"status": "no_log_file", "path": str(USAGE_LOG_PATH.resolve())}
    
    try:
        import pandas as pd
        df = pd.read_csv(USAGE_LOG_PATH)
        
        return {
            "status": "success",
            "path": str(USAGE_LOG_PATH.resolve()),
            "total_calls": len(df),
            "total_tokens": df["total_tokens"].sum(),
            "total_emissions_gco2": df["est_gco2"].sum(),
            "avg_latency_s": df["latency_s"].mean(),
            "error_rate": (df["ok"] == False).mean(),
            "phases": df["phase"].value_counts().to_dict(),
            "models": df["model"].value_counts().to_dict()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}