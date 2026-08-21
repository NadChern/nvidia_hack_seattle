"""FastAPI assembly; the conversational backend is selected once at startup."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import __version__
from agent.api import health, query, status
from agent.config import Settings, get_settings
from agent.errors import AgentServiceError
from agent.logging import configure_logging
from agent.metrics import AgentMetrics
from agent.reply import ReplyTransport
from agent.stub import QueryBackend, StubLlm
from agent.tools.assist import AssistTool
from agent.tools.memory import MemoryTool
from agent.tools.register import RegisterTool
from agent.workflow import RegistrationWorkflow

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings
    configure_logging(level=settings.log_level, service=settings.service_name, version=__version__)
    logger.info(
        "agent service starting",
        extra={
            "backend": settings.agent_backend,
            "model": settings.llm_model,
            "endpoint_host": settings.endpoint_host,
            "endpoint_scope": settings.endpoint_scope,
            "hands_free": settings.hands_free_enabled,
        },
    )

    listener_task: asyncio.Task[None] | None = None
    if settings.hands_free_enabled:
        from agent.listener import HandsFreeListener

        listener = HandsFreeListener(settings, app.state.backend, metrics=app.state.metrics)
        listener_task = asyncio.create_task(listener.run(), name="hands-free-listener")
        app.state.listener_task = listener_task

    register_task: asyncio.Task[None] | None = None
    workflow: RegistrationWorkflow | None = app.state.registration_workflow
    # Speech-free: the register button's poll runs independently of hands-free
    # STT, so a headset with no speech stack still registers objects.
    if settings.register_button_enabled and workflow is not None:
        from agent.register_listener import RegisterTriggerListener

        register_listener = RegisterTriggerListener(settings, workflow)
        register_task = asyncio.create_task(
            register_listener.run(), name="register-trigger-listener"
        )
        app.state.register_task = register_task

    try:
        yield
    finally:
        for task in (listener_task, register_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if workflow is not None:
            await workflow.aclose()


def _select_backend(
    settings: Settings, metrics: AgentMetrics
) -> tuple[QueryBackend, RegistrationWorkflow]:
    memory = MemoryTool(settings)
    assist = AssistTool(settings)
    registration = RegistrationWorkflow(
        RegisterTool(settings),
        ReplyTransport(settings),
        metrics=metrics,
    )
    if settings.agent_backend == "stub":
        return StubLlm(memory, registration, assist), registration

    # Imported only on the model path so health/config tooling for the stub
    # does not initialize ADK or LiteLLM as a side effect.
    from agent.agent import create_agent
    from agent.runner import AdkRunnerBackend

    return (
        AdkRunnerBackend(settings, create_agent(settings, memory, registration, assist)),
        registration,
    )


def create_app(
    settings: Settings | None = None,
    *,
    backend: QueryBackend | None = None,
) -> FastAPI:
    resolved = settings or get_settings()

    hide_schema = resolved.environment == "deploy"
    app = FastAPI(
        title=resolved.service_name,
        version=__version__,
        lifespan=lifespan,
        openapi_url=None if hide_schema else "/openapi.json",
        docs_url=None if hide_schema else "/docs",
        redoc_url=None if hide_schema else "/redoc",
    )
    app.state.settings = resolved
    metrics = AgentMetrics()
    if backend is None:
        selected_backend, registration_workflow = _select_backend(resolved, metrics)
    else:
        selected_backend, registration_workflow = backend, None
    app.state.backend = selected_backend
    app.state.metrics = metrics
    app.state.registration_workflow = registration_workflow
    app.state.listener_task = None
    app.state.register_task = None

    async def handle(_: Request, exc: AgentServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.add_exception_handler(AgentServiceError, handle)  # type: ignore[arg-type]
    for module in (health, status, query):
        app.include_router(module.router)
    return app


app = create_app()
