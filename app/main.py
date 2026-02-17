from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.search import router as search_router
from app.api.routes.ui import router as ui_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.dependencies import Container

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = Container()
    try:
        yield
    finally:
        await app.state.container.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(search_router)
app.include_router(ui_router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": settings.app_name, "status": "running"}




