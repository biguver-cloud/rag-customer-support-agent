from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_logs_empty_directory(tmp_path, monkeypatch):
    """ログが1件もない場合は空リストを返す"""
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_logs_returns_file_list(tmp_path, monkeypatch):
    """CSVファイルがある場合はファイル情報の一覧を返す"""
    (tmp_path / "chat_log_2026_06_10.csv").write_text(
        "日時,質問\n2026-06-10,テスト", encoding="utf-8"
    )
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["filename"] == "chat_log_2026_06_10.csv"
    assert "size" in data[0]
    assert "modified" in data[0]


def test_logs_download_csv(tmp_path, monkeypatch):
    """指定ファイルを CSV としてダウンロードできる"""
    content = "日時,質問\n2026-06-10,テスト"
    (tmp_path / "chat_log_2026_06_10.csv").write_text(content, encoding="utf-8")
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs/chat_log_2026_06_10.csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_logs_download_not_found(tmp_path, monkeypatch):
    """存在しないファイルは 404 を返す"""
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", tmp_path)
    resp = client.get("/api/logs/nonexistent.csv")
    assert resp.status_code == 404


def test_logs_download_path_traversal(tmp_path, monkeypatch):
    """パストラバーサルは 400 または 404 を返す"""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr("api.routers.logs.LOGS_DIR", logs_dir)
    # %2E%2E%2F は URL エンコードされた ../
    resp = client.get("/api/logs/%2E%2E%2Fsecret.csv")
    assert resp.status_code in (400, 404)
