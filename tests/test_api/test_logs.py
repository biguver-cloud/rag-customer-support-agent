import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_KEY = "test-secret-key"
VALID_HEADERS = {"x-api-key": VALID_KEY}


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", VALID_KEY)
    monkeypatch.setenv("ENABLE_LOG_API", "true")


# ── 認証テスト ────────────────────────────────────────────

def test_logs_no_api_key():
    """APIキーなし → 401"""
    resp = client.get("/api/logs")
    assert resp.status_code == 401


def test_logs_wrong_api_key():
    """APIキー不一致 → 401"""
    resp = client.get("/api/logs", headers={"x-api-key": "wrong-key"})
    assert resp.status_code == 401


def test_logs_valid_api_key(tmp_path, monkeypatch):
    """認証あり → 200"""
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs", headers=VALID_HEADERS)
    assert resp.status_code == 200


# ── ENABLE_LOG_API テスト ─────────────────────────────────

def test_logs_disabled(monkeypatch):
    """API無効化時 → 404"""
    monkeypatch.setenv("ENABLE_LOG_API", "false")
    resp = client.get("/api/logs", headers=VALID_HEADERS)
    assert resp.status_code == 404


def test_logs_download_disabled(monkeypatch):
    """API無効化時のダウンロード → 404"""
    monkeypatch.setenv("ENABLE_LOG_API", "false")
    resp = client.get("/api/logs/test.csv", headers=VALID_HEADERS)
    assert resp.status_code == 404


# ── 既存機能テスト（認証あり前提） ───────────────────────

def test_logs_empty_directory(tmp_path, monkeypatch):
    """ログが1件もない場合は空リストを返す"""
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs", headers=VALID_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == []


def test_logs_returns_file_list(tmp_path, monkeypatch):
    """CSVファイルがある場合はファイル情報の一覧を返す"""
    (tmp_path / "chat_log_2026_06_10.csv").write_text(
        "日時,質問\n2026-06-10,テスト", encoding="utf-8"
    )
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs", headers=VALID_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["filename"] == "chat_log_2026_06_10.csv"
    assert "size" in data[0]
    assert "modified" in data[0]


def test_logs_download_csv(tmp_path, monkeypatch):
    """指定ファイルをCSVとしてダウンロードできる"""
    content = "日時,質問\n2026-06-10,テスト"
    (tmp_path / "chat_log_2026_06_10.csv").write_text(content, encoding="utf-8")
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs/chat_log_2026_06_10.csv", headers=VALID_HEADERS)
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_logs_download_not_found(tmp_path, monkeypatch):
    """存在しないファイルは404を返す"""
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs/nonexistent.csv", headers=VALID_HEADERS)
    assert resp.status_code == 404


def test_logs_download_path_traversal(tmp_path, monkeypatch):
    """パストラバーサルは400または404を返す"""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", logs_dir)
    resp = client.get("/api/logs/%2E%2E%2Fsecret.csv", headers=VALID_HEADERS)
    assert resp.status_code in (400, 404)
