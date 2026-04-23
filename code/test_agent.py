# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")

# NOTE: If you see "Unknown pytest.mark.X" warnings, create a conftest.py file with:
# import pytest
# def pytest_configure(config):
#     config.addinivalue_line("markers", "performance: mark test as performance test")
#     config.addinivalue_line("markers", "security: mark test as security test")
#     config.addinivalue_line("markers", "integration: mark test as integration test")


import pytest
import asyncio
import time
import json
from unittest.mock import patch, MagicMock, AsyncMock

import agent

from fastapi.testclient import TestClient
import httpx

FALLBACK_RESPONSE = (
    "No relevant meeting notes were found in the knowledge base to generate a summary."
)

@pytest.fixture(scope="module")
def test_app():
    # Use the FastAPI app from agent.py
    return agent.app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

@pytest.mark.functional
def test_health_check_endpoint_returns_ok(client):
    """Validates that the /health endpoint returns a 200 status and correct response structure."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"

@pytest.mark.functional
def test_query_endpoint_returns_summary_on_success(monkeypatch, client):
    """Checks that the /query endpoint returns a successful summary when meeting notes are present."""
    # Patch MeetingNotesSummarizerAgent.process_query to return a valid summary
    summary_text = "Summary of meeting notes. Action items:\n- Do X\n- Do Y"
    fake_result = {
        "success": True,
        "summary": summary_text,
        "error": None,
        "details": None
    }
    # AUTO-FIXED invalid syntax: with patch("agent.MeetingNotesSummarizerAgent.process_query", new=AsyncMock(return_value=fake_result):
    resp = client.post("/query")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("success") is True
    assert isinstance(data.get("summary"), str)
    assert data.get("summary")
    assert data.get("error") is None

@pytest.mark.functional
def test_query_endpoint_returns_fallback_when_no_notes(monkeypatch, client):
    """Ensures /query returns fallback response when no meeting notes are found."""
    fake_result = {
        "success": False,
        "summary": None,
        "error": FALLBACK_RESPONSE,
        "details": None
    }
    # AUTO-FIXED invalid syntax: with patch("agent.MeetingNotesSummarizerAgent.process_query", new=AsyncMock(return_value=fake_result):
    resp = client.post("/query")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("success") is False
    assert data.get("summary") is None
    assert data.get("error") == FALLBACK_RESPONSE

@pytest.mark.unit
def test_azure_ai_search_client_raises_on_missing_credentials(monkeypatch):
    """Unit test to ensure AzureAISearchClient.get_client() raises ValueError if credentials are missing."""
    from agent import AzureAISearchClient
    # Patch Config to remove credentials
    monkeypatch.setattr(agent.Config, "AZURE_SEARCH_ENDPOINT", "")
    monkeypatch.setattr(agent.Config, "AZURE_SEARCH_INDEX_NAME", "")
    monkeypatch.setattr(agent.Config, "AZURE_SEARCH_API_KEY", "")
    client_instance = AzureAISearchClient()
    with pytest.raises(ValueError) as excinfo:
        client_instance.get_client()
    assert "Azure AI Search credentials are not fully configured." in str(excinfo.value)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_llmservice_generate_summary_returns_cleaned_output(monkeypatch):
    """Unit test to verify LLMService.generate_summary returns sanitized summary text."""
    from agent import LLMService, sanitize_llm_output
    # Patch get_llm_client to return a mock client
    mock_client = MagicMock()
    # Simulate LLM response with markdown/code wrappers
    raw_llm_output = "```markdown\nHere is the summary:\n- Action 1\n- Action 2\n```\nSure, here is your answer."
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = raw_llm_output
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(agent, "get_llm_client", lambda: mock_client)
    # Patch Config.LLM_MODEL and Config.get_llm_kwargs
    monkeypatch.setattr(agent.Config, "LLM_MODEL", "gpt-4o")
    monkeypatch.setattr(agent.Config, "get_llm_kwargs", lambda: {})
    service = LLMService()
    context_chunks = ["Meeting notes chunk 1.", "Meeting notes chunk 2."]
    user_query = "Summarize the meeting notes."
    summary = await service.generate_summary(context_chunks, user_query)
    assert isinstance(summary, str)
    # Should not contain markdown/code wrappers
    assert "```" not in summary
    assert "Here is the summary:" not in summary or summary.strip() != raw_llm_output
    # Should be cleaned by sanitize_llm_output
    cleaned = sanitize_llm_output(raw_llm_output, content_type="text")
    assert summary == cleaned

@pytest.mark.unit
@pytest.mark.asyncio
async def test_meeting_notes_summarizer_agent_process_query_returns_fallback_on_empty_chunks(monkeypatch):
    """Unit test to ensure process_query returns fallback response if no context chunks are found."""
    from agent import MeetingNotesSummarizerAgent
    # Patch ChunkRetriever.get_context_chunks to return empty list
    # AUTO-FIXED invalid syntax: with patch.object(agent.ChunkRetriever, "get_context_chunks", new=AsyncMock(return_value=[]):
    agent_instance = MeetingNotesSummarizerAgent()
    result = await agent_instance.process_query()
    assert isinstance(result, dict)
    assert result.get("success") is False
    assert result.get("summary") is None
    assert result.get("error") == FALLBACK_RESPONSE

@pytest.mark.integration
def test_validation_exception_handler_returns_422(client):
    """Integration test to ensure malformed JSON triggers validation_exception_handler and returns 422."""
    # Send malformed JSON (missing closing brace)
    resp = client.post("/query", data="{invalid_json: true", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    data = resp.json()
    assert data.get("success") is False
    assert "error" in data
    assert "Malformed JSON" in data.get("error") or "Input validation failed" in data.get("error")

@pytest.mark.integration
def test_generic_exception_handler_returns_500(monkeypatch, client):
    """Integration test to ensure unhandled exceptions trigger generic_exception_handler and return 500."""
    # Patch MeetingNotesSummarizerAgent.process_query to raise Exception
    # AUTO-FIXED invalid syntax: with patch("agent.MeetingNotesSummarizerAgent.process_query", new=AsyncMock(side_effect=Exception("Simulated error"):
    resp = client.post("/query")
    assert resp.status_code == 500
    data = resp.json()
    assert data.get("success") is False
    assert "error" in data
    # Should mention internal server error
    assert "Internal server error." in data.get("error") or data.get("error") == "Internal server error."

@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_of_query_endpoint_under_load(monkeypatch):
    """Performance test to measure response time of /query endpoint under concurrent requests."""
    from agent import app
    # Patch MeetingNotesSummarizerAgent.process_query to simulate a fast response
    fake_result = {
        "success": True,
        "summary": "Summary.",
        "error": None,
        "details": None
    }
    # AUTO-FIXED invalid syntax: with patch("agent.MeetingNotesSummarizerAgent.process_query", new=AsyncMock(return_value=fake_result):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
        n_requests = 50
        start = time.time()
        tasks = [async_client.post("/query") for _ in range(n_requests)]
        responses = await asyncio.gather(*tasks)
        duration = time.time() - start
        # All responses should be 200 and valid
        for resp in responses:
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("success") is True
        avg_time = duration / n_requests
        # Acceptable threshold: <2s average, <30s total
        assert avg_time < 2.0
        assert duration < 30.0

@pytest.mark.edge_case
@pytest.mark.asyncio
async def test_llmservice_generate_summary_handles_empty_context(monkeypatch):
    """Edge case test to ensure generate_summary returns empty string or fallback when context_chunks is empty."""
    from agent import LLMService
    # Patch get_llm_client to return a mock client that returns empty content
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 0
    mock_response.usage.completion_tokens = 0
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(agent, "get_llm_client", lambda: mock_client)
    monkeypatch.setattr(agent.Config, "LLM_MODEL", "gpt-4o")
    monkeypatch.setattr(agent.Config, "get_llm_kwargs", lambda: {})
    service = LLMService()
    summary = await service.generate_summary([], "Summarize the meeting notes.")
    assert isinstance(summary, str)
    assert summary == ""

@pytest.mark.edge_case
def test_sanitize_llm_output_handles_none_input():
    """Edge case test to ensure sanitize_llm_output returns empty string when input is None."""
    from agent import sanitize_llm_output
    result = sanitize_llm_output(None)
    assert result == ""