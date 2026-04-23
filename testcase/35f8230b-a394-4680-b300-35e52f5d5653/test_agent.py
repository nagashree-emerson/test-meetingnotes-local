
import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.testclient import TestClient

import agent

@pytest.fixture
def client():
    """Fixture to provide a FastAPI test client for the agent app."""
    return TestClient(agent.app)

@pytest.fixture
def mock_retrieve_chunks():
    """Fixture to mock AzureAISearchClient.retrieve_chunks to return dummy meeting notes."""
    return AsyncMock(return_value=[
        "Meeting started at 10am. Discussed Q2 targets.",
        "Action item: Alice to send follow-up email.",
        "Decision: Move project deadline to May 15."
    ])

@pytest.fixture
def mock_generate_summary():
    """Fixture to mock LLMService.generate_summary to return a dummy summary."""
    return AsyncMock(return_value="Summary: Q2 targets discussed. Action: Alice to follow up. Deadline: May 15.")

def test_functional_successful_meeting_notes_summarization(client, mock_retrieve_chunks, mock_generate_summary):
    """
    Validates that the /query endpoint returns a successful summary when relevant meeting notes are present.
    """
    # Patch the retrieval and LLM layers to simulate a successful summarization
    with patch("agent.ChunkRetriever.get_context_chunks", mock_retrieve_chunks), \
         patch("agent.LLMService.generate_summary", mock_generate_summary):
        response = client.post("/query")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["summary"], str)
        assert data["summary"].strip() != ""
        assert data["error"] is None