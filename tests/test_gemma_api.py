"""
Smoke tests for Gemma 4 interpretability endpoints:
  - POST /api/explain
  - POST /api/diagnose
  - POST /api/compare

Mocks external Gemini API calls so tests run fast, reliably, and offline.
Made with 🔮 by Karan Ray
"""
import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from glassbox.server.app import app
from glassbox.gemini_client import GeminiKeyMissing, GeminiRateLimited, GeminiError

client = TestClient(app)


# ─── Mock Fixtures ──────────────────────────────────────────────

MOCK_EXPLAIN_RESPONSE = {
    "summary": "The model displays high confidence in predicting common sentence completions.",
    "key_observations": [
        "First token shows dominant attention sink.",
        "Step 4 has high entropy due to multiple candidate nouns."
    ],
    "suspicious_steps": [
        {"step_index": 4, "reason": "Entropy spiked above 8.5 bits with uniform top-k spread"}
    ]
}

MOCK_DIAGNOSE_RESPONSE = {
    "ambiguity": "Prompt is slightly broad but well-formed.",
    "likely_points_of_failure": ["Knowledge cutoff beyond recent events"],
    "suggested_rephrasing": "What is the official capital city of India?",
    "suggested_prompt": "What is the official capital city of India?",
    "confidence_score": 88,
    "issues": [
        {"type": "ambiguity", "description": "Could be clarified with specific domain constraint", "step_indices": []}
    ],
    "explanation": "Adding explicit phrasing guides attention heads to state entity representations."
}

MOCK_COMPARE_RESPONSE = {
    "completion": "New Delhi is the capital of India.",
    "tokens": ["New", "Delhi", "is", "the", "capital", "of", "India."],
    "latency_ms": 142.5,
    "model": "Gemma 4 (Gemini API)"
}


# ─── Tests ──────────────────────────────────────────────────────

def test_explain_endpoint_success():
    """Verify /api/explain returns valid schema when Gemini responds."""
    payload = {
        "prompt": "What is the capital of India?",
        "generated_tokens": [
            {"step": 0, "token_str": "The", "token_id": 456},
            {"step": 1, "token_str": " capital", "token_id": 789},
        ],
        "top_k_per_step": [
            [{"token_str": "The", "prob": 0.8}, {"token_str": "It", "prob": 0.1}],
            [{"token_str": " capital", "prob": 0.9}],
        ],
        "entropy_per_step": [1.2, 0.4],
        "heatmap_b64_png": None,
    }

    with patch("glassbox.server.app.explain_trace", return_value=MOCK_EXPLAIN_RESPONSE):
        resp = client.post("/api/explain", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "key_observations" in data
        assert "suspicious_steps" in data
        assert len(data["suspicious_steps"]) == 1


def test_diagnose_endpoint_success():
    """Verify /api/diagnose returns ambiguity, likely points of failure, and confidence."""
    payload = {
        "prompt": "Capital of France",
        "generated_tokens": [],
        "entropy_per_step": [],
        "top_k_per_step": [],
    }

    with patch("glassbox.server.app.diagnose_prompt", return_value=MOCK_DIAGNOSE_RESPONSE):
        resp = client.post("/api/diagnose", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence_score"] == 88
        assert "ambiguity" in data
        assert "likely_points_of_failure" in data
        assert "suggested_rephrasing" in data


def test_compare_endpoint_success():
    """Verify /api/compare returns completion and latency."""
    payload = {
        "prompt": "What is the capital of India?",
        "temperature": 0.7,
        "max_tokens": 50,
    }

    with patch("glassbox.server.app.gemma_generate", return_value=MOCK_COMPARE_RESPONSE):
        resp = client.post("/api/compare", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "completion" in data
        assert "latency_ms" in data
        assert "tokens" in data


def test_missing_api_key_graceful_error():
    """Verify missing GEMINI_API_KEY returns a clear 503 response with error_type."""
    with patch("glassbox.server.app.explain_trace", side_effect=GeminiKeyMissing("GEMINI_API_KEY is not set")):
        resp = client.post("/api/explain", json={"prompt": "test"})
        assert resp.status_code == 503
        data = resp.json()
        assert data["error_type"] == "missing_key"
        assert "GEMINI_API_KEY" in data["error"]


def test_rate_limit_graceful_error():
    """Verify 429 rate limit from Gemini returns 429 to client with rate_limit error_type."""
    with patch("glassbox.server.app.diagnose_prompt", side_effect=GeminiRateLimited("Rate limit reached")):
        resp = client.post("/api/diagnose", json={"prompt": "test"})
        assert resp.status_code == 429
        data = resp.json()
        assert data["error_type"] == "rate_limit"


def test_invalid_json_body():
    """Verify endpoints reject malformed requests with 400."""
    resp = client.post(
        "/api/explain",
        content="not valid json",
        headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400
