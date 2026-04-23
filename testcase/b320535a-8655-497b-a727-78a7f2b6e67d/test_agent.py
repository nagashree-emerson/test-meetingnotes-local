
import pytest
import json
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient
import agent

@pytest.fixture
def client():
    """Fixture to provide a FastAPI test client for the agent app."""
    return TestClient(agent.app)

def test_security_content_safety_decorator_enforcement_on_query_validation_error(client):
    """
    Security test: Verifies that the /query endpoint is wrapped with with_content_safety
    and returns a proper error response (not an unhandled exception) when a validation error occurs.
    """
    # The /query endpoint expects POST with no body, so send invalid JSON to trigger JSONDecodeError
    with patch("agent.with_content_safety", wraps=agent.with_content_safety) as mock_guardrails:
        response = client.post("/query", data="{invalid_json: true")  # malformed JSON
        assert response.status_code in (400, 422)
        data = response.json()
        assert isinstance(data, dict)
        assert data.get("success") is False
        assert "error" in data
        # Ensure no stack trace or file path is leaked
        assert "Traceback" not in str(data)
        assert "File \"" not in str(data)
        # Ensure the content safety decorator was applied (called)
        assert mock_guardrails.called

def test_security_content_safety_decorator_enforcement_on_query_generic_exception(client):
    """
    Security test: Verifies that the /query endpoint returns a safe error response
    if an unexpected exception occurs (simulating a bug in process_query) 
    """
    # Patch MeetingNotesSummarizerAgent.process_query to raise an Exception
    # AUTO-FIXED invalid syntax: with patch("agent.MeetingNotesSummarizerAgent.process_query", side_effect=Exception("Simulated failure"):
    response = client.post("/query")
    assert response.status_code == 500
    data = response.json()
    assert isinstance(data, dict)
    assert data.get("success") is False
    assert "error" in data
    assert "Simulated failure" in str(data.get("details", "")) or "Simulated failure" in str(data.get("error", ""))
    # Ensure no stack trace or file path is leaked
    assert "Traceback" not in str(data)
    assert "File \"" not in str(data)

def test_security_content_safety_decorator_enforcement_on_health_check(client):
    """
    Security test: Verifies that the /health endpoint returns a valid response and does not raise unhandled exceptions.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"