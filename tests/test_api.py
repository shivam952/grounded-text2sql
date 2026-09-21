"""Tests for FastAPI demo endpoints in groundedsql.api."""
from __future__ import annotations

import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from groundedsql import api
from groundedsql.api import app, DB_ALLOWLIST
from groundedsql.models import GroundingResult, ReActTrace


@pytest.fixture(autouse=True)
def reset_state():
    # Reset budget state before each test
    with api._budget_lock:
        api._budget_state = {"count": 0, "date": ""}
    # Reset slowapi limiter storage
    api.limiter.reset()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GroundedSQL" in response.text
    assert "student_club" in response.text
    assert "superhero" in response.text


def test_ask_validation_errors(client):
    # Unknown DB
    res = client.post("/ask", json={"question": "What is here?", "db": "non_existent_db"})
    assert res.status_code == 400
    assert "Unknown db" in res.json()["detail"]

    # Empty question
    res = client.post("/ask", json={"question": "   ", "db": "student_club"})
    assert res.status_code == 400
    assert "question must not be empty" in res.json()["detail"]


def test_ask_success_mocked(client):
    mock_trace = ReActTrace(
        question="How many superheroes are there?",
        answer="There are 734 superheroes.",
        sql_used="SELECT COUNT(*) FROM superhero;",
        confidence=0.95,
        iterations_used=2,
        total_tool_calls=2,
        grounding_interventions=0,
        grounding=GroundingResult(passed=True),
    )

    with patch("groundedsql.api.ReActSqlAgent.answer", return_value=mock_trace):
        res = client.post("/ask", json={"question": "How many superheroes are there?", "db": "superhero"})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "There are 734 superheroes."
        assert data["sql_used"] == "SELECT COUNT(*) FROM superhero;"
        assert data["confidence"] == 0.95
        assert data["iterations"] == 2
        assert data["tool_calls"] == 2
        assert data["grounding_interventions"] == 0
        assert data["grounding_passed"] is True
        assert "elapsed_s" in data


def test_rate_limiting(client):
    mock_trace = ReActTrace(
        question="Test",
        answer="Test answer",
        sql_used="SELECT 1;",
        confidence=1.0,
        iterations_used=1,
        total_tool_calls=1,
        grounding_interventions=0,
        grounding=GroundingResult(passed=True),
    )

    with patch("groundedsql.api.ReActSqlAgent.answer", return_value=mock_trace):
        # Default RATE_LIMIT is 5/minute
        responses = [
            client.post("/ask", json={"question": f"Q{i}", "db": "superhero"})
            for i in range(6)
        ]
        status_codes = [r.status_code for r in responses]
        assert status_codes[:5] == [200, 200, 200, 200, 200]
        assert status_codes[5] == 429


def test_daily_budget_circuit_breaker(client, monkeypatch):
    monkeypatch.setattr(api, "DAILY_REQUEST_CAP", 2)
    
    mock_trace = ReActTrace(
        question="Test",
        answer="Test answer",
        sql_used="SELECT 1;",
        confidence=1.0,
        iterations_used=1,
        total_tool_calls=1,
        grounding_interventions=0,
        grounding=GroundingResult(passed=True),
    )

    with patch("groundedsql.api.ReActSqlAgent.answer", return_value=mock_trace):
        # 1st request -> ok
        r1 = client.post("/ask", json={"question": "Q1", "db": "superhero"})
        assert r1.status_code == 200
        
        # 2nd request -> ok
        r2 = client.post("/ask", json={"question": "Q2", "db": "superhero"})
        assert r2.status_code == 200
        
        # 3rd request -> 503 circuit breaker
        r3 = client.post("/ask", json={"question": "Q3", "db": "superhero"})
        assert r3.status_code == 503
        assert "Daily request cap" in r3.json()["detail"]


def test_ask_stream_success(client):
    mock_trace = ReActTrace(
        question="How many superheroes are there?",
        answer="There are 734 superheroes.",
        sql_used="SELECT COUNT(*) FROM superhero;",
        confidence=0.95,
        iterations_used=2,
        total_tool_calls=2,
        grounding_interventions=0,
        grounding=GroundingResult(passed=True),
    )

    def mock_answer(question, on_step=None):
        if on_step:
            on_step(mock_trace, "Inspecting tables")
            on_step(mock_trace, "Counting rows")
        return mock_trace

    with patch("groundedsql.api.ReActSqlAgent.answer", side_effect=mock_answer):
        res = client.post("/ask/stream", json={"question": "How many superheroes are there?", "db": "superhero"})
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        events = [line for line in res.text.split("\n\n") if line.startswith("data: ")]
        assert len(events) >= 3
        # First 2 are step events
        step1 = json.loads(events[0][6:])
        assert step1["type"] == "step"
        assert step1["text"] == "Inspecting tables"
        # Last is final event
        final = json.loads(events[-1][6:])
        assert final["type"] == "final"
        assert final["answer"] == "There are 734 superheroes."

