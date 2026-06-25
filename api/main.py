from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, logs
from api.config import get_allowed_origins, ALLOW_METHODS, ALLOW_HEADERS

load_dotenv()

app = FastAPI(title="RAG Customer Support API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=ALLOW_METHODS,
    allow_headers=ALLOW_HEADERS,
    allow_credentials=False,
)

app.include_router(chat.router, prefix="/api")
app.include_router(logs.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
