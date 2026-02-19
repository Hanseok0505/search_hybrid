from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
router = APIRouter(tags=["ui"])


@router.get("/ui", response_class=HTMLResponse)
async def ui_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )
