from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Comprehensive Legal AI",
    version="0.1.0",
    description="Upload legal docs, then ask questions with Groq-powered summaries."
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] while developing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)
