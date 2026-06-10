import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, logs

load_dotenv()

app = FastAPI(title="RAG Customer Support API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(logs.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
