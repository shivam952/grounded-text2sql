"""End-to-end golden scenario and retry tests for ReActSqlAgent."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from groundedsql.agent import ReActSqlAgent, _call_llm_with_retry
from groundedsql.models import ReActTrace
from groundedsql.observability import Tracer

DEMO_DB = Path(__file__).parent.parent / "src" / "groundedsql" / "demo_data" / "superhero.sqlite"


def _make_tool_call_msg(tool_name: str, arguments: dict, tc_id: str = "call_1"):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(arguments)
    msg = MagicMock()
    msg.content = f"STEP: executing {tool_name}"
    msg.tool_calls = [tc]
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return resp


def test_retry_helper_success():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = "success"

    res = _call_llm_with_retry(mock_client, max_retries=3, initial_delay=0.01, model="test")
    assert res == "success"
    assert mock_client.chat.completions.create.call_count == 1


def test_retry_helper_retries_on_failure():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        RuntimeError("Rate limit 429"),
        RuntimeError("Server timeout 503"),
        "success_after_retries",
    ]

    res = _call_llm_with_retry(mock_client, max_retries=3, initial_delay=0.01, model="test")
    assert res == "success_after_retries"
    assert mock_client.chat.completions.create.call_count == 3


def test_retry_helper_raises_after_max_retries():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Persistent error")

    with pytest.raises(RuntimeError, match="Persistent error"):
        _call_llm_with_retry(mock_client, max_retries=2, initial_delay=0.01, model="test")
    assert mock_client.chat.completions.create.call_count == 2


def test_agent_golden_react_loop():
    """Test full multi-step ReAct loop: run_sql -> submit_result."""
    resp_step1 = _make_tool_call_msg(
        "run_sql",
        {"sql": "SELECT COUNT(*) AS total FROM superhero;"},
        tc_id="call_1",
    )
    resp_step2 = _make_tool_call_msg(
        "submit_result",
        {
            "answer": "There are 750 superheroes in the database.",
            "sql_used": "SELECT COUNT(*) AS total FROM superhero;",
            "confidence": 1.0,
            "grounding_summary": "750 matches count in total column",
        },
        tc_id="call_2",
    )

    agent = ReActSqlAgent(
        db_path=DEMO_DB,
        model="test-model",
        max_iterations=5,
        tracer=Tracer(),
    )

    with patch.object(agent.client.chat.completions, "create", side_effect=[resp_step1, resp_step2]):
        trace = agent.answer("How many superheroes are there?")

        assert trace.answer == "There are 750 superheroes in the database."
        assert trace.sql_used == "SELECT COUNT(*) AS total FROM superhero;"
        assert trace.confidence == 1.0
        assert trace.iterations_used == 2
        assert trace.total_tool_calls == 2
        assert trace.grounding_interventions == 0
        assert trace.grounding.passed is True
        assert len(trace.steps) == 1
        assert trace.steps[0].row_count == 1


def test_agent_grounding_rejection_and_self_correction():
    """Test grounding intervention triggering self-correction in ReAct loop."""
    # Step 1: run query
    resp_step1 = _make_tool_call_msg(
        "run_sql",
        {"sql": "SELECT COUNT(*) AS total FROM superhero;"},
        tc_id="call_1",
    )
    # Step 2: submit ungrounded number (999 is not in rows returned)
    resp_step2 = _make_tool_call_msg(
        "submit_result",
        {
            "answer": "There are 999 superheroes in total.",
            "sql_used": "SELECT COUNT(*) AS total FROM superhero;",
            "confidence": 0.9,
        },
        tc_id="call_2",
    )
    # Step 2b: Stage 2 LLM-as-judge response for claim "999" -> "NO"
    judge_msg = MagicMock(content="NO", tool_calls=None)
    resp_judge = MagicMock(choices=[MagicMock(message=judge_msg)], usage=None)

    # Step 3: self-correct and submit grounded number (750 is in rows)
    resp_step3 = _make_tool_call_msg(
        "submit_result",
        {
            "answer": "There are 750 superheroes in total.",
            "sql_used": "SELECT COUNT(*) AS total FROM superhero;",
            "confidence": 1.0,
        },
        tc_id="call_3",
    )

    agent = ReActSqlAgent(
        db_path=DEMO_DB,
        model="test-model",
        max_iterations=5,
        tracer=Tracer(),
    )

    with patch.object(
        agent.client.chat.completions,
        "create",
        side_effect=[resp_step1, resp_step2, resp_judge, resp_step3],
    ):
        trace = agent.answer("How many superheroes are there?")

        assert trace.answer == "There are 750 superheroes in total."
        assert trace.iterations_used == 3
        assert trace.total_tool_calls == 3
        assert trace.grounding_interventions == 1  # Fired once and self-corrected!
        assert trace.grounding.passed is True
