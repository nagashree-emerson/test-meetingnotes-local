# AUTO-FIX runtime fallbacks for unresolved names
SearchClient = None

try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
    from observability.instrumentation import initialize_tracer
except ImportError:
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass
    def trace_tool_call(*_a, **_kw): pass
    def initialize_tracer(*_a, **_kw): pass

import asyncio as _asyncio

import time as _time
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Optional, Dict, List
from pathlib import Path

from config import Config

import openai
try:
    from azure.search.documents import SearchClient
except ImportError:
    pass
try:
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    pass
try:
    from azure.search.documents.models import VectorizedQuery
except ImportError:
    pass
# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT, OUTPUT FORMAT, FALLBACK RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a professional assistant that summarizes meeting notes. "
    "Given a set of meeting notes, generate a concise, clear, and actionable summary. "
    "Highlight key decisions, action items, and important discussion points. "
    "Be objective and avoid unnecessary details. "
    "If no meeting notes are available, respond with the fallback message."
)
OUTPUT_FORMAT = (
    "Return the summary as a well-structured paragraph in plain text. "
    "If there are action items, list them as bullet points at the end. "
    "Do not include any introductory or closing remarks unless they are part of the meeting notes."
)
FALLBACK_RESPONSE = (
    "No relevant meeting notes were found in the knowledge base to generate a summary."
)

# No document filtering section provided, so search all documents.
SELECTED_DOCUMENT_TITLES: List[str] = []

# Validation config path for input validation
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY LIFESPAN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Meeting Notes Summarizer Agent",
    description="Summarizes meeting notes into concise, actionable summaries with key decisions and action items.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is properly formatted (quotes, commas, brackets).",
                "Check that all required fields are present and correctly typed.",
                "If sending large text, keep it under 50,000 characters."
            ]
        },
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation failed.",
            "details": exc.errors(),
            "tips": [
                "Check that all required fields are present and correctly typed.",
                "If sending large text, keep it under 50,000 characters."
            ]
        },
    )

@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": "Malformed JSON in request body.",
            "details": str(exc),
            "tips": [
                "Ensure your JSON is properly formatted (quotes, commas, brackets).",
                "If sending large text, keep it under 50,000 characters."
            ]
        },
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error.",
            "details": str(exc),
            "tips": [
                "If you believe this is a bug, please contact support.",
                "Check your input and try again."
            ]
        },
    )

# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was successful")
    summary: Optional[str] = Field(None, description="The generated meeting summary")
    error: Optional[str] = Field(None, description="Error message if any")
    details: Optional[Any] = Field(None, description="Additional error details")

# ═══════════════════════════════════════════════════════════════════════════════
# SANITIZER UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL AND LLM SERVICE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class AzureAISearchClient:
    """Handles retrieval of relevant chunks from Azure AI Search."""

    def __init__(self):
        self._search_client = None

    def get_client(self):
        if self._search_client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            api_key = Config.AZURE_SEARCH_API_KEY
            if not endpoint or not index_name or not api_key:
                raise ValueError("Azure AI Search credentials are not fully configured.")
            self._search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self._search_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, k: int = 5, selected_titles: Optional[List[str]] = None) -> List[str]:
        """
        Retrieve top-k relevant chunks from Azure AI Search using vector + keyword search.
        """
        search_client = self.get_client()
        # Get embedding for the query
        openai_client = get_llm_client()
        _t0 = _time.time()
        embedding_resp = await openai_client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(embedding_resp)[:200] if embedding_resp is not None else None,
                status="success",
            )
        except Exception:
            pass

        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=k,
            fields="vector"
        )
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": k,
            "select": ["chunk", "title"],
        }
        if selected_titles:
            odata_parts = [f"title eq '{t}'" for t in selected_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)
        _t1 = _time.time()
        results = search_client.search(**search_kwargs)
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t1) * 1000),
                output="retrieved",
                status="success",
            )
        except Exception:
            pass
        context_chunks = [r["chunk"] for r in results if r.get("chunk")]
        return context_chunks

class ChunkRetriever:
    """Orchestrates chunk retrieval from Azure AI Search."""

    def __init__(self):
        self._search_client = AzureAISearchClient()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_context_chunks(self, query: str, k: int = 5) -> List[str]:
        return await self._search_client.retrieve_chunks(query, k, selected_titles=SELECTED_DOCUMENT_TITLES)

class LLMService:
    """Handles LLM calls for summarization."""

    def __init__(self):
        self._client = None

    def get_client(self):
        return get_llm_client()

    async def generate_summary(self, context_chunks: List[str], user_query: str) -> str:
        """
        Calls the LLM with the context and user query to generate the summary.
        """
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
        context_text = "\n\n".join(context_chunks)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query},
            {"role": "user", "content": context_text}
        ]
        client = self.get_client()
        _t0 = _time.time()
        response = await client.chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4o",
            messages=messages,
            **Config.get_llm_kwargs()
        )
        content = response.choices[0].message.content if response.choices else ""
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4o",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return sanitize_llm_output(content, content_type="text")

@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    api_key = Config.AZURE_OPENAI_API_KEY
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not configured")
    return openai.AsyncAzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MeetingNotesSummarizerAgent:
    """Agent that summarizes meeting notes from a knowledge base."""

    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.guardrails_config = GUARDRAILS_CONFIG

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self) -> Dict[str, Any]:
        """
        Main agent entry point. Retrieves relevant meeting notes and generates a summary.
        Returns:
            dict: {success, summary, error, details}
        """
        async with trace_step(
            "retrieve_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant meeting notes from Azure AI Search",
            output_fn=lambda r: f"{len(r)} chunks" if isinstance(r, list) else "0 chunks",
        ) as step:
            context_chunks = await self.chunk_retriever.get_context_chunks(SYSTEM_PROMPT, k=5)
            step.capture(context_chunks)

        if not context_chunks:
            return {
                "success": False,
                "summary": None,
                "error": FALLBACK_RESPONSE,
                "details": None
            }

        async with trace_step(
            "llm_generate_summary",
            step_type="llm_call",
            decision_summary="Generate meeting summary from retrieved chunks",
            output_fn=lambda r: f"{len(r)} chars" if isinstance(r, str) else "0 chars",
        ) as step:
            summary = await self.llm_service.generate_summary(context_chunks, SYSTEM_PROMPT)
            step.capture(summary)

        if not summary or not summary.strip():
            return {
                "success": False,
                "summary": None,
                "error": FALLBACK_RESPONSE,
                "details": None
            }

        return {
            "success": True,
            "summary": summary,
            "error": None,
            "details": None
        }

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint():
    """
    Summarize meeting notes from the knowledge base.
    No user input is required; the agent uses its internal system prompt.
    """
    agent = MeetingNotesSummarizerAgent()
    result = await agent.process_query()
    return QueryResponse(**result)

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    try:
        import uvicorn
    except ImportError:
        pass
    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())
# __agent_sanitized_for_testing__
