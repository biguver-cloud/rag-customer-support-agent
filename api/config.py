import os


def get_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080")
    return [o.strip() for o in raw.split(",") if o.strip()]


# POSTはチャット、GETはログ取得・ヘルスチェックのみで十分
ALLOW_METHODS = ["GET", "POST"]

# Content-Type: POSTリクエストのJSONボディに必要
# x-api-key: ログAPIの認証ヘッダー
ALLOW_HEADERS = ["Content-Type", "x-api-key"]
