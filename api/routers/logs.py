import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"


def _check_enabled():
    if os.getenv("ENABLE_LOG_API", "true").lower() != "true":
        raise HTTPException(status_code=404, detail="Log API is disabled")


def _verify_api_key(x_api_key: str | None = Header(default=None, alias="x-api-key")):
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


_deps = [Depends(_check_enabled), Depends(_verify_api_key)]


@router.get("/logs", dependencies=_deps)
def list_logs():
    if not LOGS_DIR.exists():
        return []
    files = sorted(LOGS_DIR.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        }
        for f in files
    ]


@router.get("/logs/{filename}", dependencies=_deps)
def download_log(filename: str):
    log_path = LOGS_DIR / filename
    resolved = log_path.resolve()
    if not resolved.is_relative_to(LOGS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="不正なファイルパスです")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="ログファイルが見つかりません")
    return FileResponse(path=str(resolved), filename=filename, media_type="text/csv")
