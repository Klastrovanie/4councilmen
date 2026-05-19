"""
main.py — 4 Councilmen Model (4CM) FastAPI Backend
===================================================
Hybrid v2.0: Claude + Grok agents, Grok judge

Endpoints:
  GET  /health
  GET  /fourCM/agents              — list agent sets
  GET  /fourCM/agents/{set}        — load agent set
  POST /fourCM                     — SSE streaming run
  POST /fourCM/validate            — Grok orthogonality check

PhD Dissertation, 2011. Prototype, 2026.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fourCM_router import router as fourCM_router
from document_parser import cleanup_old_upload_sessions, ensure_upload_root

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="4 Councilmen Model API",
    description="Hybrid multi-agent convergence engine — Claude + Grok",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# 도커 내부에서 nginx가 프록시하므로 실질적으로 외부 노출 없음
# 개발 중에는 localhost:5173 (Vite dev server) 허용 필요
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:4173"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # X-Claude-Key, X-Grok-Key 헤더 허용
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(fourCM_router)

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "engine": "4CM Hybrid",
    }

# ── Startup log ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("  4CM API v2.0 starting")
    logger.info(f"  ANGRY_AGENTS_PATH : {os.environ.get('ANGRY_AGENTS_PATH', './angry_agents')}")
    logger.info(f"  ANTHROPIC_API_KEY : {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set (use X-Claude-Key header)'}")
    ensure_upload_root()
    cleanup_old_upload_sessions()
    logger.info(f"  XAI_API_KEY       : {'set' if os.environ.get('XAI_API_KEY') else 'not set (use X-Grok-Key header)'}")
    logger.info(f"  FOURCM_UPLOAD_ROOT: {os.environ.get('FOURCM_UPLOAD_ROOT', '/app/tmp_uploads')}")
    logger.info("=" * 60)
