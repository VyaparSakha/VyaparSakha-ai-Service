import os
import time

from contextlib import asynccontextmanager

import structlog

from fastapi import FastAPI, Request, Response

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from app.config.logging import configure_logging
from app.config.database import close_connection, get_database
from app.config.redis import init_redis, close_redis
from app.workers.ai_worker import start_python_worker, stop_python_worker
from app.config.settings import get_settings

from app.routes.index import api_router

from app.services.forecast_service import _lazy_load_artifacts

from app.agent.checkpointer import (
    get_checkpointer,
    close_checkpointer,
)


# ---------------------------------------------------------------------------
# LangSmith Configuration
# ---------------------------------------------------------------------------

def _configure_langsmith() -> None:
    """
    Configure LangSmith environment variables.

    Environment variables are configured during application startup
    before LangChain / LangGraph components are initialized.
    """

    settings = get_settings()

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = (
            settings.langchain_api_key
        )

    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = (
            settings.langchain_project
        )

    os.environ["LANGCHAIN_VERBOSE"] = str(
        settings.langchain_verbose
    ).lower()

    os.environ["LANGCHAIN_TRACING_V2"] = str(
        settings.langchain_tracing_v2
    ).lower()


# ---------------------------------------------------------------------------
# FastAPI Application Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Manage application startup and shutdown resources.

    Startup:
        1. Configure structured logging.
        2. Configure LangSmith.
        3. Initialize the MongoDB database.
        4. Load forecasting model artifacts.
        5. Initialize the LangGraph MongoDB checkpointer.

    Shutdown:
        1. Close the main MongoDB database connection.
        2. Close the LangGraph checkpointer MongoDB client.
    """

    # -----------------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------------

    configure_logging()

    logger = structlog.get_logger(
        "vyaparsathi.ai.lifecycle"
    )

    logger.info(
        "application_startup"
    )

    # Configure LangSmith before agent initialization
    _configure_langsmith()

    # -----------------------------------------------------------------------
    # Main application database
    # -----------------------------------------------------------------------

    application.state.db = get_database()

    logger.info(
        "database_initialized"
    )

    # Initialize Redis connection & BullMQ worker
    application.state.redis = await init_redis()
    application.state.bullmq_worker = await start_python_worker()

    # -----------------------------------------------------------------------
    # Forecasting model
    # -----------------------------------------------------------------------

    model, feature_columns = _lazy_load_artifacts()

    application.state.forecast_model_loaded = (
        model is not None
        and feature_columns is not None
    )

    logger.info(
        "forecast_model_initialized",
        loaded=application.state.forecast_model_loaded,
    )

# -----------------------------------------------------------------------
    # LangGraph MongoDB Checkpointer
    #
    # The v0.2.0 MongoDBSaver implements both sync and async checkpoint
    # methods on the same object, so a single instance serves both
    # graph.invoke() and graph.astream_events().
    # -----------------------------------------------------------------------

    application.state.checkpointer = get_checkpointer()

    logger.info(
        "langgraph_checkpointer_initialized"
    )

    # -----------------------------------------------------------------------
    # Pre-warm LLM chains to eliminate cold-start delay on first requests
    # -----------------------------------------------------------------------
    try:
        from app.lib.llm import (
            get_planner_llm,
            get_reasoning_llm,
            get_intent_llm,
            get_critic_llm,
            get_small_llm,
            get_safeguard_llm,
        )
        get_planner_llm()
        get_reasoning_llm()
        get_intent_llm()
        get_critic_llm()
        get_small_llm()
        get_safeguard_llm()
        logger.info("llm_chains_prewarmed")
    except Exception as exc:
        logger.warning("llm_prewarm_failed", error=str(exc))

    # -----------------------------------------------------------------------
    # Application is ready
    # -----------------------------------------------------------------------

    logger.info(
        "application_ready"
    )

    try:

        yield

    finally:

        # -------------------------------------------------------------------
        # Shutdown
        # -------------------------------------------------------------------

        logger.info(
            "application_shutdown"
        )

        # Close main application database connection
        await close_connection()

        # Close BullMQ worker & Redis connection
        await stop_python_worker()
        await close_redis()

        # Close LangGraph checkpointer MongoDB client.
        await close_checkpointer()

        logger.info(
            "application_shutdown_complete"
        )


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    application = FastAPI(
        title="VyaparSakha AI Service",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # CORS Middleware
    # -----------------------------------------------------------------------
    from fastapi.middleware.cors import CORSMiddleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # API Routes & OAuth 2.0 Router
    # -----------------------------------------------------------------------

    application.include_router(
        api_router
    )

    from app.mcp.oauth import oauth_router
    application.include_router(
        oauth_router
    )

    # -----------------------------------------------------------------------
    # MCP (Model Context Protocol) Server Mount & Endpoints
    # -----------------------------------------------------------------------
    from app.mcp import mcp
    from app.mcp.http_handler import handle_mcp_jsonrpc
    from mcp.server.transport_security import TransportSecuritySettings

    # Configure transport security to permit remote hosts and tunnels (like ngrok)
    mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    # Direct JSON-RPC POST endpoints (used by ChatGPT, Claude, and Streamable HTTP clients)
    @application.post("/mcp/sse", tags=["mcp"])
    @application.post("/mcp", tags=["mcp"])
    @application.post("/sse", tags=["mcp"])
    async def mcp_post_jsonrpc(request: Request):
        """Handle incoming MCP JSON-RPC protocol requests from ChatGPT or Web MCP clients."""
        return await handle_mcp_jsonrpc(request)

    # Mount Starlette SSE app directly under /mcp (handles GET /mcp/sse and POST /mcp/messages/)
    # Note: Do NOT pass mount_path="/mcp" to sse_app() because Starlette's mount already prefixes /mcp,
    # which would otherwise cause duplicate '/mcp/mcp/messages/' resulting in 404 on action discovery.
    application.mount("/mcp", mcp.sse_app())

    @application.get("/mcp", tags=["mcp"])
    @application.get("/mcp-info", tags=["mcp"])
    async def mcp_status():
        """Returns metadata and available read-only tools on the Vyapar Sathi MCP server."""
        tools = await mcp.list_tools()
        return {
            "status": "online",
            "server": "VyaparSathi-Assistant",
            "mode": "read-only",
            "sse_endpoint": "/mcp/sse",
            "tool_count": len(tools),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description.strip().splitlines()[0] if t.description else "",
                }
                for t in tools
            ],
        }

    # -----------------------------------------------------------------------
    # Pure ASGI HTTP Logging Middleware (compatible with SSE streaming)
    # -----------------------------------------------------------------------
    class ASGILoggingMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.app(scope, receive, send)

            path = scope.get("path", "")
            method = scope.get("method", "GET")

            # SSE streams (/mcp/sse) stay open continuously; pass directly through without buffering
            if path.startswith("/mcp"):
                return await self.app(scope, receive, send)

            start_time = time.time()
            logger = structlog.get_logger("vyaparsathi.ai.http")
            logger.info("http_request_start", method=method, path=path)

            status_code = [200]

            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    status_code[0] = message.get("status", 200)
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.info(
                    "http_request_complete",
                    method=method,
                    path=path,
                    status_code=status_code[0],
                    duration_ms=duration_ms,
                )
            except Exception:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.exception("http_request_failed", method=method, path=path, duration_ms=duration_ms)
                raise

    application.add_middleware(ASGILoggingMiddleware)

    # -----------------------------------------------------------------------
    # Prometheus Metrics
    # -----------------------------------------------------------------------

    @application.get(
        "/metrics",
        tags=["monitoring"],
    )
    async def metrics() -> Response:
        """
        Prometheus metrics endpoint.
        """

        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return application


# ---------------------------------------------------------------------------
# ASGI Application
# ---------------------------------------------------------------------------

app = create_app()

