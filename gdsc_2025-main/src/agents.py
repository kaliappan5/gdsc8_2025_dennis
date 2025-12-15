import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from strands.agent import Agent
from strands.models.mistral import MistralModel

try:
    from .llm_ops import LLMPolicy, build_system_prompt

    DEFAULT_POLICY = LLMPolicy()
except Exception:
    # Fallbacks so this file still works if llm_ops isn't in place yet
    class _FallbackPolicy:
        model_id = "mistral-medium-latest"
        temperature = 0.1
        top_p = 0.9
        max_tokens = 16000
        per_call_token_cap = 32000
        max_tokens_ceiling = 64000

    DEFAULT_POLICY = _FallbackPolicy()

    def build_system_prompt() -> str:
        return (
            "You are a concise assistant.\n"
            "- Output MUST be a single JSON object (no prose, no markdown).\n"
            "- Keep answers brief and factual.\n"
            "- Use only the fields requested.\n"
            "- If uncertain, use null.\n"
            "- Do not repeat the question.\n"
            "- Stop when JSON is complete.\n"
        )


def get_agent(
    system_prompt: str = "",
    model_id: Optional[str] = None,
    policy: Any = DEFAULT_POLICY,
) -> Agent:
    """
    Build an Agent with deterministic-ish defaults and conservative token bounds.
    NOTE: The MistralModel constructor supports only: max_tokens, model_id, stream, temperature, top_p.
          Pass stop sequences at CALL TIME instead (if needed).
    """
    sys_prompt = system_prompt or build_system_prompt()
    model_id = model_id or policy.model_id

    model = MistralModel(
        api_key=os.environ["MISTRAL_API_KEY"],
        model_id=model_id,
        stream=False,
        temperature=policy.temperature,
        top_p=policy.top_p,
        max_tokens=min(
            policy.max_tokens,
            getattr(policy, "per_call_token_cap", policy.max_tokens),
            getattr(policy, "max_tokens_ceiling", policy.max_tokens),
        ),
        # DO NOT pass stop or random_seed here (constructor doesn't support them)
    )

    # Defensive: ensure attributes are set if the SDK reflects them as properties
    # Do not assign attributes directly; handled by constructor above
    return Agent(model=model, system_prompt=sys_prompt, callback_handler=None)


def send_message_to_chat(
    message: str, persona_id: str, conversation_id: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """Send message to persona API and get response"""
    url = "https://cygeoykm2i.execute-api.us-east-1.amazonaws.com/main/chat"

    session = boto3.Session(region_name="us-east-1")
    credentials = session.get_credentials()

    payload = {
        "persona_id": persona_id,
        "conversation_id": conversation_id,
        "message": message,
    }

    request = AWSRequest(
        method="POST",
        url=url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(credentials, "execute-api", "us-east-1").add_auth(request)

    response = requests.request(
        method=request.method,
        url=request.url,
        headers=dict(request.headers),
        data=request.body,
    )

    if response.status_code != 200:
        return None

    response_json = response.json()
    return response_json["response"], response_json["conversation_id"]


def call_mistral(prompt: str, model: str = "mistral-small-latest") -> Dict[str, Any]:
    """Call Mistral API and track what it costs us"""
    mistral_model = MistralModel(
        api_key=os.environ["MISTRAL_API_KEY"], model_id=model, stream=False
    )
    agent = Agent(model=mistral_model, callback_handler=None)
    start_time = time.time()

    try:
        response = agent(prompt)
        end_time = time.time()

        # Extract useful info
        result = {
            "content": response.message["content"][0]["text"],
            "model": model,
            "duration": end_time - start_time,
            "input_tokens": response.metrics.accumulated_usage["inputTokens"],
            "output_tokens": response.metrics.accumulated_usage["outputTokens"],
            "total_tokens": response.metrics.accumulated_usage["totalTokens"],
        }

        return result
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return {}


def _call_agent_with_retry(
    agent: Any,
    prompt: str,
    max_tokens_first: int = 16000,
    max_tokens_second: int = 32000,
) -> Any:
    try:
        # Call-time kwargs supported by your SDK: temperature, top_p, max_tokens
        return agent(prompt, temperature=0.1, top_p=0.9, max_tokens=max_tokens_first)
    except Exception as e:
        # Defensive retry if the SDK surfaces this error type or message
        if "MaxTokensReachedException" in str(e) or "max_tokens" in str(e).lower():
            return agent(
                prompt, temperature=0.1, top_p=0.9, max_tokens=max_tokens_second
            )
        raise
