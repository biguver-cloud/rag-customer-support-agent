from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

PATCHES = [
    "api.routers.chat._get_db",
    "api.routers.chat._get_llm",
    "api.routers.chat.rewrite_query_for_search",
    "api.routers.chat.guess_category",
    "api.routers.chat._save_log",
]


def _make_doc(content="テスト内容", source="test.pdf", page=0, category="service"):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page, "category": category}
    return doc


def test_chat_returns_answer():
    """検索ヒット時に回答・引用情報・スコアが返る"""
    doc = _make_doc()
    with patch("api.routers.chat._get_db"), \
         patch("api.routers.chat._get_llm"), \
         patch("api.routers.chat.rewrite_query_for_search", return_value="解約"), \
         patch("api.routers.chat.guess_category", return_value="service"), \
         patch("api.routers.chat.hybrid_retrieve_with_score", return_value=[(doc, 0.3)]), \
         patch("api.routers.chat.agent_answer", return_value={
             "answer": "解約は月末までに申請が必要です。",
             "loops": 0, "tokens": 100, "accuracy": 90, "completeness": 85,
         }), \
         patch("api.routers.chat._save_log"):
        resp = client.post("/api/chat", json={"question": "解約したい"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "解約は月末までに申請が必要です。"
    assert data["accuracy"] == 90
    assert data["completeness"] == 85
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source"] == "test.pdf"


def test_chat_no_context_returns_fallback():
    """検索結果なしの場合はフォールバックメッセージを返す"""
    with patch("api.routers.chat._get_db"), \
         patch("api.routers.chat._get_llm"), \
         patch("api.routers.chat.rewrite_query_for_search", return_value="不明"), \
         patch("api.routers.chat.guess_category", return_value="unknown"), \
         patch("api.routers.chat.hybrid_retrieve_with_score", return_value=[]), \
         patch("api.routers.chat._save_log"):
        resp = client.post("/api/chat", json={"question": "全く関係ない質問"})

    assert resp.status_code == 200
    assert "資料に記載がありません" in resp.json()["answer"]


def test_chat_weak_score_returns_followup():
    """スコアが閾値超の場合は補助質問を返す"""
    doc = _make_doc()
    with patch("api.routers.chat._get_db"), \
         patch("api.routers.chat._get_llm"), \
         patch("api.routers.chat.rewrite_query_for_search", return_value="解約"), \
         patch("api.routers.chat.guess_category", return_value="unknown"), \
         patch("api.routers.chat.hybrid_retrieve_with_score", return_value=[(doc, 2.0)]), \
         patch("api.routers.chat._save_log"):
        resp = client.post("/api/chat", json={"question": "解約したい"})

    assert resp.status_code == 200
    assert "1)" in resp.json()["answer"]


def test_chat_empty_question_returns_422():
    """空文字の質問は 422 を返す"""
    resp = client.post("/api/chat", json={"question": ""})
    assert resp.status_code == 422


def test_chat_missing_field_returns_422():
    """question フィールドなしは 422 を返す"""
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 422
