import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:3000")
    # 環境変数変更後にアプリを再生成
    import importlib
    import api.config
    import api.main
    importlib.reload(api.config)
    importlib.reload(api.main)
    from api.main import app
    return TestClient(app)


def test_allowed_origin(client_with_origin):
    """許可されたオリジンからアクセスできる"""
    resp = client_with_origin.get(
        "/health",
        headers={"Origin": "http://localhost:8080"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8080"


def test_another_allowed_origin(client_with_origin):
    """複数指定した場合、どちらのオリジンも許可される"""
    resp = client_with_origin.get(
        "/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_disallowed_origin(client_with_origin):
    """許可されていないオリジンはAccess-Control-Allow-Originが返らない"""
    resp = client_with_origin.get(
        "/health",
        headers={"Origin": "http://evil.example.com"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"


def test_preflight_allowed_origin(client_with_origin):
    """許可されたオリジンのプリフライトリクエストは200を返す"""
    resp = client_with_origin.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8080"


def test_preflight_disallowed_origin(client_with_origin):
    """許可されていないオリジンのプリフライトはAccess-Control-Allow-Originが返らない"""
    resp = client_with_origin.options(
        "/api/chat",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"
