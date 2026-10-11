"""
glassbox/gemini_client.py — Gemini API client for trace explanation & prompt diagnosis.

Uses the Gemini REST API directly via urllib (zero extra dependencies).
Requires GEMINI_API_KEY in environment or .env file.

Made with 🔮 by Karan Ray & contributors
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_MODEL = "gemini-2.0-flash"  # Gemma 4 model via Gemini API
_TIMEOUT_SECONDS = 60


def _get_api_key() -> str | None:
    """Return the GEMINI_API_KEY from env, or None if missing."""
    return os.environ.get("GEMINI_API_KEY")


class GeminiError(Exception):
    """Raised when the Gemini API returns an error or is unreachable."""
    pass


class GeminiKeyMissing(GeminiError):
    """Raised when GEMINI_API_KEY is not set."""
    pass


class GeminiRateLimited(GeminiError):
    """Raised on 429 rate-limit responses."""
    pass


# ---------------------------------------------------------------------------
# Low-level API call
# ---------------------------------------------------------------------------

def _call_gemini(
    parts: list[dict[str, Any]],
    system_instruction: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 2048,
    response_mime_type: str = "application/json",
) -> str:
    """Send a generateContent request to the Gemini API.

    Parameters
    ----------
    parts : list of content parts (text and/or inline_data for images)
    system_instruction : optional system-level instruction
    temperature : sampling temperature (low = deterministic analysis)
    max_output_tokens : max output tokens
    response_mime_type : mime type ("application/json" or "text/plain")

    Returns
    -------
    The model's text response.

    Raises
    ------
    GeminiKeyMissing, GeminiRateLimited, GeminiError
    """
    api_key = _get_api_key()
    if not api_key:
        raise GeminiKeyMissing(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com and add it to your .env file."
        )

    url = f"{_API_BASE}/models/{_MODEL}:generateContent?key={api_key}"

    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if response_mime_type == "application/json":
        generation_config["responseMimeType"] = "application/json"

    body: dict[str, Any] = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }
    if system_instruction:
        body["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
            err_msg = err_body.get("error", {}).get("message", str(exc))
        except Exception:
            err_msg = str(exc)

        if status == 429:
            raise GeminiRateLimited(
                f"Gemini API rate limit reached. Please wait a moment and try again. ({err_msg})"
            ) from exc
        elif status == 403:
            raise GeminiError(
                f"Gemini API key is invalid or lacks permission. Check your GEMINI_API_KEY. ({err_msg})"
            ) from exc
        else:
            raise GeminiError(f"Gemini API error (HTTP {status}): {err_msg}") from exc
    except urllib.error.URLError as exc:
        raise GeminiError(f"Could not reach Gemini API: {exc.reason}") from exc
    except TimeoutError:
        raise GeminiError(
            f"Gemini API request timed out after {_TIMEOUT_SECONDS}s. Try again."
        )

    # Extract text from response
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected Gemini response structure: %s", result)
        raise GeminiError("Unexpected response from Gemini API.") from exc


# ---------------------------------------------------------------------------
# High-level functions
# ---------------------------------------------------------------------------

def explain_trace(
    prompt: str,
    generated_tokens: list[dict],
    top_k_per_step: list[list[dict]],
    entropy_per_step: list[float],
    heatmap_b64_png: str | None = None,
) -> dict:
    """Call Gemma 4 to explain a generation trace.

    Parameters
    ----------
    prompt : the user's original prompt
    generated_tokens : list of {step, token_str, token_id}
    top_k_per_step : list of lists of {token_str, prob}
    entropy_per_step : entropy in bits at each step
    heatmap_b64_png : base64-encoded PNG of the attention heatmap (optional)

    Returns
    -------
    dict with keys: summary, key_observations, suspicious_steps
    """
    system = (
        "You are an expert LLM interpretability analyst. You are given a trace from "
        "a small language model (SmolLM2-135M) running locally. Analyze the trace data "
        "and the attention heatmap image (if provided) to explain what the model did.\n\n"
        "Return a JSON object with exactly these keys:\n"
        '- "summary": a 2-3 sentence overview of the generation behavior\n'
        '- "key_observations": an array of 3-5 strings, each a notable observation\n'
        '- "suspicious_steps": an array of objects {step_index: int, reason: string} '
        "for steps that show signs of confusion, hallucination, or unexpected behavior\n\n"
        "Be specific, cite step numbers, token names, and entropy values."
    )

    trace_text = (
        f"## Prompt\n{prompt}\n\n"
        f"## Generated Tokens ({len(generated_tokens)} steps)\n"
    )
    for i, tok in enumerate(generated_tokens):
        ent = entropy_per_step[i] if i < len(entropy_per_step) else 0
        candidates = top_k_per_step[i] if i < len(top_k_per_step) else []
        top3 = ", ".join(
            f'"{c.get("token_str", "?")}"({c.get("prob", 0):.2f})'
            for c in candidates[:3]
        )
        trace_text += (
            f"  Step {i}: \"{tok.get('token_str', '')}\" "
            f"(entropy={ent:.2f} bits, top-3=[{top3}])\n"
        )

    parts: list[dict] = [{"text": trace_text}]

    if heatmap_b64_png:
        # Strip data URL prefix if present
        img_data = heatmap_b64_png
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        parts.insert(0, {
            "inline_data": {
                "mime_type": "image/png",
                "data": img_data,
            }
        })

    raw = _call_gemini(parts, system_instruction=system)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {
                "summary": raw,
                "key_observations": [],
                "suspicious_steps": [],
            }

    # Ensure required keys
    result.setdefault("summary", "")
    result.setdefault("key_observations", [])
    result.setdefault("suspicious_steps", [])
    return result


def diagnose_prompt(
    prompt: str,
    generated_tokens: list[dict] | None = None,
    entropy_per_step: list[float] | None = None,
    top_k_per_step: list[list[dict]] | None = None,
) -> dict:
    """Analyze a prompt and generation trace for issues, and suggest improvements.

    Returns
    -------
    dict with keys:
        ambiguity: str
        likely_points_of_failure: list[str]
        suggested_rephrasing: str
        confidence_score: int (0-100)
        suggested_prompt: str
        explanation: str
        issues: list[dict]
    """
    generated_tokens = generated_tokens or []
    entropy_per_step = entropy_per_step or []
    top_k_per_step = top_k_per_step or []

    system = (
        "You are an expert prompt engineer and LLM interpretability analyst. "
        "Given a user prompt and internal trace data from a local language model (SmolLM2-135M), "
        "critique the prompt and diagnose points of vulnerability.\n\n"
        "Return a JSON object with exactly these keys:\n"
        '- "ambiguity": string critique of vagueness, polysemy, or under-specification in the prompt\n'
        '- "likely_points_of_failure": array of strings explaining where and why small models often fail on this prompt\n'
        '- "suggested_rephrasing": a rewritten version of the prompt optimized for clarity, precision, and reliable reasoning\n'
        '- "suggested_prompt": the same string as suggested_rephrasing\n'
        '- "confidence_score": integer (0 to 100) indicating estimated model success likelihood\n'
        '- "explanation": 2-3 sentences explaining why the suggested prompt yields superior output\n'
        '- "issues": array of objects {type: string, description: string, step_indices: [int]} '
        "where type is one of: 'ambiguity', 'attention_sink_dominance', 'high_entropy', "
        "'likely_hallucination', 'repetition', 'premature_eos'\n\n"
        "Be practical, incisive, and rigorous."
    )

    trace_text = f"## Original Prompt\n{prompt}\n\n## Generation Trace\n"
    if generated_tokens:
        for i, tok in enumerate(generated_tokens):
            ent = entropy_per_step[i] if i < len(entropy_per_step) else 0
            candidates = top_k_per_step[i] if i < len(top_k_per_step) else []
            top3 = ", ".join(
                f'"{c.get("token_str", "?")}"({c.get("prob", 0):.2f})'
                for c in candidates[:3]
            )
            trace_text += (
                f"  Step {i}: \"{tok.get('token_str', '')}\" "
                f"(entropy={ent:.2f} bits, top-3=[{top3}])\n"
            )

        avg_entropy = sum(entropy_per_step) / max(len(entropy_per_step), 1)
        max_entropy = max(entropy_per_step) if entropy_per_step else 0
        high_entropy_count = sum(1 for e in entropy_per_step if e > 8.0)
        trace_text += (
            f"\n## Summary Statistics\n"
            f"  Average entropy: {avg_entropy:.2f} bits\n"
            f"  Max entropy: {max_entropy:.2f} bits\n"
            f"  High-entropy steps (>8 bits): {high_entropy_count}/{len(entropy_per_step)}\n"
        )
    else:
        trace_text += "  (No local generation trace provided; critique based on prompt structure.)\n"

    parts: list[dict] = [{"text": trace_text}]
    raw = _call_gemini(parts, system_instruction=system)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {
                "ambiguity": "Could not parse JSON response from analyzer.",
                "likely_points_of_failure": [],
                "suggested_rephrasing": prompt,
                "suggested_prompt": prompt,
                "confidence_score": 50,
                "issues": [],
                "explanation": raw,
            }

    # Ensure required schema defaults
    result.setdefault("ambiguity", "Prompt is relatively clear but may benefit from additional context.")
    result.setdefault("likely_points_of_failure", [])
    suggested = result.get("suggested_rephrasing") or result.get("suggested_prompt") or prompt
    result["suggested_rephrasing"] = suggested
    result["suggested_prompt"] = suggested
    result.setdefault("confidence_score", 70)
    result.setdefault("issues", [])
    result.setdefault("explanation", "")
    return result


def gemma_generate(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate completion via Gemma 4 (Gemini API) for model comparison.

    Parameters
    ----------
    prompt : Prompt string
    max_tokens : Maximum new tokens to generate
    temperature : Sampling temperature

    Returns
    -------
    dict with completion, tokens, latency_ms, model
    """
    import time
    t0 = time.perf_counter()

    # Call Gemini with text/plain mime type for natural generation
    raw_text = _call_gemini(
        parts=[{"text": prompt}],
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Simple whitespace/word splitting for token comparison in UI
    tokens = raw_text.split()

    return {
        "completion": raw_text.strip(),
        "tokens": tokens,
        "latency_ms": latency_ms,
        "model": "Gemma 4 (Gemini API)",
    }
