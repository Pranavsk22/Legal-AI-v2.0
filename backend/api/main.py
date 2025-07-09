# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from dotenv import load_dotenv
import os

load_dotenv()

# add near the top, after imports
HF_PROXY_ROOT = os.getenv("HF_SPACE_PATH", "")           # e.g. "/proxy/7860"
# …
app = FastAPI(
    title="Comprehensive Legal AI",
    version="0.1.0",
    description="Upload legal docs, then ask questions with Groq‑powered summaries.",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    root_path=HF_PROXY_ROOT      # <── KEY CHANGE
)

# -------------------------------------------------------------------------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://comprehensive-legal-ai.vercel.app",
    "https://skpranav22-legal-ai.hf.space",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/ping")
def ping():
    return {"pong": True}

app.include_router(router)
