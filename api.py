import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag.config import MODEL_NAME, TEMPERATURE, TOP_K, WEAK_SCORE_THRESHOLD, AGENT_ROUNDS
from rag.agent import agent_answer
from rag.query import guess_category, rewrite_query_for_search
from rag.vectorstore import open_vectorstore, hybrid_retrieve_with_score

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
persist_dir = BASE_DIR / "storage" / "chroma"

resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY が設定されていません")
    if not persist_dir.exists():
        raise RuntimeError("ベクトルDBがありません。先に build_index.py を実行してください")
    resources["db"] = open_vectorstore(persist_dir)
    resources["llm"] = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    yield
    resources.clear()


app = FastAPI(
    title="RAG Customer Support API",
    description="社内PDFを知識源とした問い合わせ対応RAGエージェントのAPIです。",
    version="1.0.0",
    lifespan=lifespan,
)


# ── リクエスト / レスポンス スキーマ ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class Citation(BaseModel):
    source: str
    page: int | None
    category: str
    quote: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    category: str
    citations: list[Citation]
    accuracy: int
    completeness: int


# ── エンドポイント ────────────────────────────────────────────────────────────

@app.get("/", summary="ヘルスチェック")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, summary="問い合わせに回答する")
def chat(request: ChatRequest):
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=422, detail="message が空です")

    db = resources["db"]
    llm = resources["llm"]

    search_query = rewrite_query_for_search(user_text, llm=llm)
    category = guess_category(user_text, llm=llm)

    search_results = hybrid_retrieve_with_score(
        db=db, query=search_query, k=TOP_K, category=category
    )

    if not search_results:
        return ChatResponse(
            answer="資料に記載がありません。",
            category=category,
            citations=[],
            accuracy=0,
            completeness=0,
        )

    scores = [score for _, score in search_results]
    best_score = min(scores)
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in search_results])

    citations = []
    for doc, score in search_results:
        page = doc.metadata.get("page", None)
        text = doc.page_content.strip().replace("\n", " ")
        citations.append(Citation(
            source=doc.metadata.get("source", ""),
            page=(page + 1) if isinstance(page, int) else None,
            category=doc.metadata.get("category", category),
            quote=text[:400] + ("..." if len(text) > 400 else ""),
            score=score,
        ))

    if best_score > WEAK_SCORE_THRESHOLD:
        return ChatResponse(
            answer="資料だけでは特定できませんでした。質問をより具体的に入力してください。",
            category=category,
            citations=citations,
            accuracy=0,
            completeness=0,
        )

    result = agent_answer(llm, user_text, context, rounds=AGENT_ROUNDS)

    return ChatResponse(
        answer=result["answer"],
        category=category,
        citations=citations,
        accuracy=result["accuracy"],
        completeness=result["completeness"],
    )
