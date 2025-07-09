# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from dotenv import load_dotenv
load_dotenv()

#####################################################################
# 👇  THIS is the critical bit: tell FastAPI the external root path
HF_PROXY_PREFIX = "/proxy/7860"          # Hugging Face Docker Spaces proxy
#####################################################################

app = FastAPI(
    title="Comprehensive Legal AI",
    version="0.1.0",
    description="Upload legal docs, then ask questions with Groq‑powered summaries.",
    docs_url=f"{HF_PROXY_PREFIX}/docs",          # swagger at full path
    redoc_url=None,
    openapi_url=f"{HF_PROXY_PREFIX}/openapi.json",
    root_path=HF_PROXY_PREFIX                    # ←  **fix**
)

# ------------------------------------------------------------------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://comprehensive-legal-ai.vercel.app",
    "https://skpranav22-legal-ai.hf.space",      # front‑door domain
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz", tags=["health"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}

app.include_router(router)
