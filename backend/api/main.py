# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from backend.api.routes import router
from dotenv import load_dotenv
import pathlib

load_dotenv()

HF_PROXY_PREFIX = "/proxy/7860"          # ← one single place

app = FastAPI(
    title="Comprehensive Legal AI",
    version="0.1.0",
    description="Upload legal docs, then ask questions with Groq‑powered summaries.",
    docs_url="/docs",                    # keep **relative**
    openapi_url="/openapi.json",
    redoc_url=None,
    root_path=HF_PROXY_PREFIX,           # FastAPI adds this in front
)

# ------------------------------------------------------------------
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

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def read_root():
    html_path = pathlib.Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")

@app.get("/healthz", tags=["health"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}

app.include_router(router)

