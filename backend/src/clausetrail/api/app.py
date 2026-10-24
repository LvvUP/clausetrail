import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from clausetrail import __version__
from clausetrail.api.routes import chat, documents, health, sessions
from clausetrail.chat import ChatService
from clausetrail.config import get_settings
from clausetrail.services import Services, build_services

API_PREFIX = "/api"


def create_app(services: Services | None = None) -> FastAPI:
    settings = services.settings if services else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = services is None
        svc = services or build_services(settings)
        app.state.services = svc
        app.state.chat = ChatService(svc)
        try:
            yield
        finally:
            await app.state.chat.drain()
            if owned:
                await svc.aclose()
                svc.close()

    app = FastAPI(title="ClauseTrail API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    for module in (chat, sessions, documents, health):
        app.include_router(module.router, prefix=API_PREFIX)
    return app


def run() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app()
