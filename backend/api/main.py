# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Comprehensive Legal AI",
    version="0.1.0",
    description="Upload legal docs, then ask questions with Groq‑powered summaries.",
    docs_url="/docs",          # <- expose Swagger UI
    redoc_url=None,
    openapi_url="/openapi.json",
    root_path=""               # <- needed on HF Spaces reverse‑proxy
)

# --- CORS -----------------------------------------------------------------
origins = [
    # local dev
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # production front‑end (Vercel)
    "https://comprehensive-legal-ai.vercel.app",
    # the Space itself (for /docs testing)
    "https://skpranav22-legal-ai.hf.space",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # use ["*"] during early testing if easier
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- health ----------------------------------------------------------------
@app.get("/", tags=["health"])
def health() -> dict[str, str]:
    """Simple health‑check so '/' doesn’t 404 on HF Spaces."""
    return {"status": "ok"}

# --- routes ----------------------------------------------------------------
app.include_router(router)      # /upload, /ask, etc.
