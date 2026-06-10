from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"


@router.get("/logs")
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


@router.get("/logs/{filename}")
def download_log(filename: str):
    log_path = LOGS_DIR / filename
    resolved = log_path.resolve()
    if not resolved.is_relative_to(LOGS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="不正なファイルパスです")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="ログファイルが見つかりません")
    return FileResponse(path=str(resolved), filename=filename, media_type="text/csv")
