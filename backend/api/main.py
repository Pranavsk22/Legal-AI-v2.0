# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from dotenv import load_dotenv
import os

load_dotenv()

PROXY_PREFIX = "/proxy/7860"                    # ⬅️  **NEW**

app = FastAPI(
    title="Comprehensive Legal AI",
    version="0.1.0",
    description=(
        "Upload legal docs, then ask questions with Groq‑powered summaries."
    ),
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    root_path=PROXY_PREFIX                      # ⬅️  **KEY LINE**
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

app.include_router(router)
