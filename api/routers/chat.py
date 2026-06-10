import csv
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI

from rag.config import MODEL_NAME, TEMPERATURE, TOP_K, WEAK_SCORE_THRESHOLD, AGENT_ROUNDS
from rag.query import guess_category, rewrite_query_for_search
from rag.vectorstore import open_vectorstore, hybrid_retrieve_with_score
from rag.agent import agent_answer
from api.schemas import ChatRequest, ChatResponse, CitationItem

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PERSIST_DIR = BASE_DIR / "storage" / "chroma"

LOG_HEADERS = [
    "日時", "質問", "回答", "カテゴリ",
    "最高スコア", "正確性", "完全性",
    "エージェント実行回数", "使用トークン数", "参照資料",
]

_db = None
_llm = None


def _log_path() -> Path:
    filename = datetime.now().strftime("chat_log_%Y_%m_%d.csv")
    return BASE_DIR / "logs" / filename


def _get_db():
    global _db
    if _db is None:
        _db = open_vectorstore(PERSIST_DIR)
    return _db


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    return _llm


def _save_log(
    question: str,
    answer: str,
    category: str,
    best_score,
    accuracy: int,
    completeness: int,
    agent_loops: int,
    agent_tokens: int,
    citations: list,
) -> None:
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    sources = "; ".join({c["source"] for c in citations if c.get("source")})
    row = {
        "日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "質問": question,
        "回答": answer,
        "カテゴリ": category,
        "最高スコア": round(best_score, 4) if best_score is not None else "",
        "正確性": accuracy,
        "完全性": completeness,
        "エージェント実行回数": agent_loops,
        "使用トークン数": agent_tokens,
        "参照資料": sources,
    }
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_followup_questions() -> str:
    return """資料だけでは特定できませんでした。次のどれに近いですか？

1) 解約したい（いつ解約が有効になるか知りたい）
2) 返金できるか知りたい（返金条件を確認したい）
3) 請求・支払いについて知りたい
4) アカウント/ログインについて知りたい
5) その他（状況をもう少し具体的に教えてください）

たとえば「2) 返金。契約開始日が○月○日で、今○日目です」のように書いてください。
"""


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_text = request.question.strip()
    if not user_text:
        raise HTTPException(status_code=422, detail="質問が空です")

    db = _get_db()
    llm = _get_llm()

    category = ""
    best_score = None
    accuracy = 0
    completeness = 0
    agent_loops = 0
    agent_tokens = 0
    citations: list[dict] = []
    context = ""

    try:
        search_query = rewrite_query_for_search(user_text, llm=llm)
        category = guess_category(user_text, llm=llm)

        search_results = hybrid_retrieve_with_score(
            db=db,
            query=search_query,
            k=TOP_K,
            category=category,
        )

        if search_results:
            scores = [score for _, score in search_results]
            best_score = min(scores)
            context = "\n\n---\n\n".join(doc.page_content for doc, _ in search_results)

            for doc, score in search_results:
                src = doc.metadata.get("source", "")
                page = doc.metadata.get("page", None)
                cat = doc.metadata.get("category", category if category != "unknown" else "unknown")
                text = doc.page_content.strip().replace("\n", " ")
                quote = text[:400] + ("..." if len(text) > 400 else "")
                citations.append({
                    "category": cat,
                    "source": src,
                    "page": (page + 1) if isinstance(page, int) else None,
                    "quote": quote,
                    "score": score,
                })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

    if not context.strip():
        answer = "資料に記載がありません。該当するPDF名や用語（例：解約、返金、請求など）を少し具体的に教えてください。"
        citations = []
    elif best_score is not None and best_score > WEAK_SCORE_THRESHOLD:
        answer = _build_followup_questions()
    else:
        result = agent_answer(llm, user_text, context, rounds=AGENT_ROUNDS)
        answer = result["answer"]
        agent_loops = result["loops"]
        agent_tokens = result["tokens"]
        accuracy = result["accuracy"]
        completeness = result["completeness"]

    _save_log(
        question=user_text,
        answer=answer,
        category=category,
        best_score=best_score,
        accuracy=accuracy,
        completeness=completeness,
        agent_loops=agent_loops,
        agent_tokens=agent_tokens,
        citations=citations,
    )

    return ChatResponse(
        answer=answer,
        category=category,
        best_score=best_score,
        accuracy=accuracy,
        completeness=completeness,
        agent_loops=agent_loops,
        agent_tokens=agent_tokens,
        citations=[CitationItem(**c) for c in citations],
    )
